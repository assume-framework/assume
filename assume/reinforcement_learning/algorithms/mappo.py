# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging
import os

import numpy as np
import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import A2CAlgorithm
from assume.reinforcement_learning.neural_network_architecture import (
    ActorPPO,
    CriticPPO,
    LSTMActorPPO,
)

logger = logging.getLogger(__name__)


class PPO(A2CAlgorithm):
    """
    Proximal Policy Optimization (PPO) Algorithm.

    A policy gradient method that alternates between 
    sampling data through interaction with the environment, 
    and optimizing a surrogate objective function using 
    stochastic gradient ascent. It is an on-policy algorithm.

    Args:
        learning_role (LearningRole): The central learning role.
        clip_range (float): Clipping parameter epsilon.
        clip_range_vf (float, optional): Clipping parameter for the value function.
            If None, value function is not clipped.
        n_epochs (int): Number of epochs to optimize the surrogate loss per update.
        entropy_coef (float): Entropy coefficient for the loss calculation.
        vf_coef (float): Value function coefficient for the loss calculation.
        max_grad_norm (float): The maximum value for the gradient clipping.
    """

    def __init__(
        self,
        learning_role,
        clip_range=0.2,
        clip_range_vf=0.1,
        n_epochs=50,
        entropy_coef=0.05,
        vf_coef=1.0,
        max_grad_norm=0.5,
    ):
        """
        Initialize PPO algorithm with specific hyperparameters.

        Args:
            learning_role (LearningRole): The learning role object.
            clip_range (float, optional): The epsilon parameter for PPO clipping. 
            clip_range_vf (float, optional): The epsilon parameter for value function clipping.
            n_epochs (int, optional): Number of optimization epochs per rollout.
            entropy_coef (float, optional): Coefficient for entropy term in loss.
            vf_coef (float, optional): Coefficient for value function term in loss.
            max_grad_norm (float, optional): Maximum gradient norm for clipping.
        """
        super().__init__(learning_role)

        # Set PPO-specific architecture classes
        self.actor_architecture_class = ActorPPO
        self.critic_architecture_class = CriticPPO

        config = self.learning_config
        ppo_config = getattr(config, "ppo", None)

        # Use PPO-specific config if available, otherwise use defaults
        self.clip_range = clip_range if clip_range is not None else getattr(ppo_config, "clip_ratio", 0.2)
        self.clip_range_vf = clip_range_vf if clip_range_vf is not None else getattr(ppo_config, "clip_range_vf", None)
        self.n_epochs = n_epochs if n_epochs is not None else getattr(ppo_config, "n_epochs", 10)
        self.entropy_coef = entropy_coef if entropy_coef is not None else getattr(ppo_config, "entropy_coef", 0.01)
        self.vf_coef = vf_coef if vf_coef is not None else getattr(ppo_config, "vf_coef", 0.5)
        self.max_grad_norm = max_grad_norm if max_grad_norm is not None else getattr(ppo_config, "max_grad_norm", 0.5)

        # Update counter
        self.n_updates = 0

    # =========================================================================
    # CHECKPOINT SAVING METHODS
    # =========================================================================

    uses_target_networks: bool = False

    # Note: save_params, save_critic_params, save_actor_params, load_params,
    # load_critic_params, load_actor_params, initialize_policy are inherited from A2CAlgorithm



    def create_actors(self) -> None:
        """
        Creates stochastic actor networks for all agents.
        Initializes the ActorPPO network and its optimizer for each agent strategy.
        """
        config = self.learning_config
        ppo_config = getattr(config, "ppo", None)
        actor_architecture = getattr(ppo_config, "actor_architecture", "mlp")

        for strategy in self.learning_role.rl_strats.values():
            # Create PPO Actor
            if actor_architecture == "lstm":
                strategy.actor = LSTMActorPPO(
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    float_type=self.float_type,
                    unique_obs_dim=self.unique_obs_dim,
                    num_timeseries_obs_dim=strategy.num_timeseries_obs_dim,
                ).to(self.device)
            else:
                strategy.actor = ActorPPO(
                    obs_dim=self.obs_dim,
                    act_dim=self.act_dim,
                    float_type=self.float_type,
                ).to(self.device)

            # Create Optimizer
            strategy.actor.optimizer = AdamW(
                strategy.actor.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

            strategy.actor.loaded = False

    def create_critics(self) -> None:
        """
        Creates value networks for all agents.
        Initializes the CriticPPO network (Centralized Critic) and its optimizer.
        """
        n_agents = len(self.learning_role.rl_strats)

        for strategy in self.learning_role.rl_strats.values():
            # Create value network
            strategy.critics = CriticPPO(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            # Create optimizer
            strategy.critics.optimizer = AdamW(
                strategy.critics.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

    def extract_policy(self) -> dict:
        """
        Extract all actor and critic networks into a dictionary.

        Returns:
            dict: Dictionary with keys 'actors', 'critics', and dimension information.
        """
        actors = {}
        critics = {}

        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            critics[u_id] = strategy.critics

        return {
            "actors": actors,
            "critics": critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
        }

    # =========================================================================
    # CORE TRAINING: POLICY UPDATE
    # =========================================================================

    def update_policy(self) -> None:
        """
        Update actor and critic networks using proximal policy optimization (PPO).
        Checks if enough data is collected (batch_size).
        Computes Generalized Advantage Estimation (GAE) and Returns using the last value estimate.
        Updates the Actor and Critic networks over multiple epochs (n_epochs) using mini-batches.
        Calculates the surrogate objective with clipping (clip_range).
        Calculates value function loss (MSE) and entropy bonus.
        Logs metrics and gradients.
        Clears the on-policy buffer after the update.
        """
        logger.debug("Updating Policy")

        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        # Get buffer (will be RolloutBuffer for on-policy algorithms)
        rollout_buffer = self.learning_role.buffer

        # Check if rollout buffer has data
        if rollout_buffer is None or rollout_buffer.pos == 0:
            logger.debug("Rollout buffer is empty, skipping policy update")
            return

        # Accumulate data if we don't have enough for a full batch
        # This decouples train_freq from the required rollout length
        if rollout_buffer.pos < self.learning_role.learning_config.batch_size:
            logger.debug(
                f"Rollout buffer has {rollout_buffer.pos} samples, "
                f"waiting for {self.learning_role.learning_config.batch_size} (batch_size). "
                "Skipping update to accumulate more on-policy data."
            )
            return

        # Update learning rate
        progress_remaining = self.learning_role.get_progress_remaining()
        learning_rate = self.learning_role.calc_lr_from_progress(progress_remaining)

        for strategy in strategies:
            for param_group in strategy.critics.optimizer.param_groups:
                param_group["lr"] = learning_rate
            for param_group in strategy.actor.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Get last values for advantage computation
        last_values = np.zeros(n_rl_agents)
        dones = np.zeros(n_rl_agents)

        # Get the buffer size to index into the last stored state
        buffer_size = rollout_buffer.pos if not rollout_buffer.full else rollout_buffer.buffer_size

        if buffer_size > 0:
            # Use the LAST observation as the bootstrap for the REST of the buffer.
            # We sacrifice the last step (pos-1) to serve as s_{t+1} for the step before it.
            # This ensures V(s_{t+1}) is calculating using the REAL next state, not self-referential.

            last_idx = buffer_size - 1
            last_obs = rollout_buffer.observations[last_idx]
            last_dones = rollout_buffer.dones[last_idx]

            # Reduce buffer size by 1 so as to not train on the bootstrap step
            rollout_buffer.pos -= 1
            if rollout_buffer.full:
                rollout_buffer.full = False  # If it was full, it's not anymore

            # Prepare unique observations for centralized critic
            last_unique_obs = last_obs[:, self.obs_dim - self.unique_obs_dim :]

            with th.no_grad():
                for i, strategy in enumerate(strategies):
                    # Construct centralized observation
                    obs_i = last_obs[i : i + 1]
                    other_unique = np.concatenate(
                        (last_unique_obs[:i], last_unique_obs[i + 1 :]), axis=0
                    )
                    centralized_obs = np.concatenate(
                        (obs_i, other_unique.reshape(1, -1)), axis=1
                    )

                    obs_tensor = th.as_tensor(
                        centralized_obs,
                        device=self.device,
                        dtype=self.float_type,
                    )
                    # Get value estimate from critic
                    last_values[i] = strategy.critics(obs_tensor).cpu().numpy().flatten()[0]
                    dones[i] = last_dones[i]

        # Compute advantages and returns
        rollout_buffer.compute_returns_and_advantages(last_values, dones)

        # Initialize metrics storage
        all_actor_losses = []
        all_critic_losses = []
        all_entropy_losses = []

        # Initialize unit_params for gradient logging
        # Use an empty list that will be dynamically extended
        unit_params = []
        step_count = 0

        # Helper to create a new step entry
        def create_step_entry():
            return {
                u_id: {
                    "actor_loss": None,
                    "actor_total_grad_norm": None,
                    "actor_max_grad_norm": None,
                    "critic_loss": None,
                    "critic_total_grad_norm": None,
                    "critic_max_grad_norm": None,
                }
                for u_id in self.learning_role.rl_strats.keys()
            }

        for epoch in range(self.n_epochs):
            for batch in rollout_buffer.get(self.learning_config.batch_size):
                current_batch_size = batch.observations.shape[0]

                # Precompute unique observation parts for centralized critic
                unique_obs_from_others = batch.observations[
                    :, :, self.obs_dim - self.unique_obs_dim :
                ].reshape(current_batch_size, n_rl_agents, -1)

                for i, strategy in enumerate(strategies):
                    actor = strategy.actor
                    critic = strategy.critics

                    obs_i = batch.observations[:, i, :]

                    # Construct centralized state
                    other_unique_obs = th.cat(
                        (unique_obs_from_others[:, :i], unique_obs_from_others[:, i + 1 :]),
                        dim=1,
                    )
                    all_states = th.cat(
                        (
                            obs_i.reshape(current_batch_size, -1),
                            other_unique_obs.reshape(current_batch_size, -1),
                        ),
                        dim=1,
                    )

                    actions_i = batch.actions[:, i, :]
                    old_log_probs_i = batch.old_log_probs[:, i]
                    advantages_i = batch.advantages[:, i]
                    returns_i = batch.returns[:, i]
                    old_values_i = batch.old_values[:, i]

                    # Normalize advantages across the entire batch, not per-mini-batch
                    # This provides more stable training
                    advantages_flat = advantages_i.flatten()
                    advantages_i = (advantages_i - advantages_flat.mean()) / (
                        advantages_flat.std() + 1e-8
                    )

                    log_probs, entropy = actor.evaluate_actions(obs_i, actions_i)
                    values = critic(all_states).flatten()

                    # Importance sampling ratio
                    ratio = th.exp(log_probs - old_log_probs_i)

                    # Clipped surrogate objective
                    policy_loss_1 = advantages_i * ratio
                    policy_loss_2 = advantages_i * th.clamp(
                        ratio, 1 - self.clip_range, 1 + self.clip_range
                    )
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Entropy loss
                    entropy_loss = -self.entropy_coef * entropy.mean()

                    if self.clip_range_vf is not None:
                        # Clipped value function loss
                        values_clipped = old_values_i + th.clamp(
                            values - old_values_i,
                            -self.clip_range_vf,
                            self.clip_range_vf,
                        )
                        value_loss_1 = F.mse_loss(values, returns_i)
                        value_loss_2 = F.mse_loss(values_clipped, returns_i)
                        value_loss = th.max(value_loss_1, value_loss_2)
                    else:
                        value_loss = F.mse_loss(values, returns_i)

                    loss = policy_loss + entropy_loss + self.vf_coef * value_loss

                    # Actor update
                    actor.optimizer.zero_grad()
                    critic.optimizer.zero_grad()
                    loss.backward()

                    # Calculate gradient norms BEFORE clipping
                    actor_params = list(actor.parameters())
                    critic_params = list(critic.parameters())

                    actor_max_grad_norm = max(
                        (p.grad.norm().item() for p in actor_params if p.grad is not None),
                        default=0.0,
                    )
                    critic_max_grad_norm = max(
                        (p.grad.norm().item() for p in critic_params if p.grad is not None),
                        default=0.0,
                    )

                    # Gradient clipping
                    actor_total_grad_norm = th.nn.utils.clip_grad_norm_(
                        actor.parameters(), self.max_grad_norm
                    )
                    critic_total_grad_norm = th.nn.utils.clip_grad_norm_(
                        critic.parameters(), self.max_grad_norm
                    )

                    actor.optimizer.step()
                    critic.optimizer.step()

                    # Store metrics
                    all_actor_losses.append(policy_loss.item())
                    all_critic_losses.append(value_loss.item())
                    all_entropy_losses.append(entropy_loss.item())

                    # Ensure we have an entry for this step
                    if step_count >= len(unit_params):
                        unit_params.append(create_step_entry())

                    # Store per-unit gradient params for this step
                    unit_params[step_count][strategy.unit_id]["actor_loss"] = policy_loss.item()
                    unit_params[step_count][strategy.unit_id]["critic_loss"] = value_loss.item()
                    unit_params[step_count][strategy.unit_id]["actor_total_grad_norm"] = (
                        actor_total_grad_norm.item()
                        if isinstance(actor_total_grad_norm, th.Tensor)
                        else actor_total_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id]["actor_max_grad_norm"] = actor_max_grad_norm
                    unit_params[step_count][strategy.unit_id]["critic_total_grad_norm"] = (
                        critic_total_grad_norm.item()
                        if isinstance(critic_total_grad_norm, th.Tensor)
                        else critic_total_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id]["critic_max_grad_norm"] = critic_max_grad_norm

                step_count += 1

        self.n_updates += 1

        # Write gradient params to output
        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)

        # Clear rollout buffer
        rollout_buffer.reset()

        logger.debug(
            f"PPO update complete. Actor loss: {np.mean(all_actor_losses):.4f}, "
            f"Value loss: {np.mean(all_critic_losses):.4f}"
        )