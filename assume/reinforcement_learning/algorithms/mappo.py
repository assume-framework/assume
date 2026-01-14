# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import logging

import numpy as np
import torch as th
from torch.nn import functional as F

from assume.reinforcement_learning.algorithms.base_algorithm import A2CAlgorithm
from assume.reinforcement_learning.neural_network_architecture import (
    ActorPPO,
    CriticPPO,
)

logger = logging.getLogger(__name__)


class PPO(A2CAlgorithm):
    """
    Proximal Policy Optimization (PPO) Algorithm.
    """

    def __init__(
        self,
        learning_role,
        clip_range=0.1,  # Epsilon clipping constant preventing the policy from changing too much in a single update.
        clip_range_vf=0.1,  # preventing the value function from changing too much from previous estimates
        n_epochs=30,  # sample efficiency
        entropy_coef=0.02,  # encourages exploration by rewarding "randomness"
        vf_coef=1.0,  # balances the importance of training the Critic and training the Actor
        max_grad_norm=0.5,  # Gradient clipping
    ):
        """Initialize PPO algorithm."""
        super().__init__(learning_role)

        config = self.learning_config

        self.clip_range = (
            clip_range
            if clip_range is not None
            else getattr(config, "ppo_clip_range", 0.2)
        )
        self.clip_range_vf = (
            clip_range_vf
            if clip_range_vf is not None
            else getattr(config, "ppo_clip_range_vf", None)
        )
        self.n_epochs = (
            n_epochs if n_epochs is not None else getattr(config, "ppo_n_epochs", 10)
        )
        self.entropy_coef = (
            entropy_coef
            if entropy_coef is not None
            else getattr(config, "ppo_entropy_coef", 0.01)
        )
        self.vf_coef = (
            vf_coef if vf_coef is not None else getattr(config, "ppo_vf_coef", 0.5)
        )
        self.max_grad_norm = max_grad_norm

        self.actor_architecture_class = ActorPPO
        self.critic_architecture_class = CriticPPO

        # Update counter
        self.n_updates = 0

    def get_actions(self, next_observation):
        """
        Determines actions based on the current observation, applying noise for exploration if in learning mode.

        Args
        ----
        next_observation : torch.Tensor
            Observation data influencing bid price and direction.

        Returns
        -------
        torch.Tensor
            Actions that include bid price and direction.
        torch.Tensor
            Noise component which is already added to actions for exploration, if applicable.

        Notes
        -----
        In learning mode, actions incorporate noise for exploration. Initial exploration relies
        solely on noise to cover the action space broadly.
        For PPO, we also store log_prob and value estimates for later use.
        """

        # distinction whether we are in learning mode or not to handle exploration realised with noise
        if self.learning_mode and not self.evaluation_mode:
            # if we are in learning mode the first x episodes we want to explore the entire action space
            # to get a good initial experience, in the area around the costs of the agent
            if self.collect_initial_experience_mode:
                # define current action as solely noise
                noise = th.normal(
                    mean=0.0,
                    std=self.exploration_noise_std,
                    size=(self.act_dim,),
                    dtype=self.float_type,
                    device=self.device,
                )

                # =============================================================================
                # 2.1 Get Actions and handle exploration
                # =============================================================================
                # only use noise as the action to enforce exploration
                curr_action = noise

                self._last_log_prob = th.tensor(0.0, device=self.device)
                self._last_value = th.tensor(0.0, device=self.device)

            else:
                # PPO: use get_action_and_log_prob for proper stochastic sampling
                curr_action, log_prob = self.actor.get_action_and_log_prob(
                    next_observation.unsqueeze(0)
                )
                curr_action = curr_action.squeeze(0).detach()
                self._last_log_prob = log_prob.squeeze(0).detach()

                # Get value estimate from critic (if available)
                if (
                    hasattr(self.learning_role, "critics")
                    and self.unit_id in self.learning_role.critics
                ):
                    critic = self.learning_role.critics[self.unit_id]
                    self._last_value = (
                        critic(next_observation.unsqueeze(0)).squeeze().detach()
                    )
                else:
                    self._last_value = th.tensor(0.0, device=self.device)

                # PPO uses stochastic policy, no external noise needed
                noise = th.zeros_like(curr_action, dtype=self.float_type)

                # make sure that noise adding does not exceed the actual output of the NN as it pushes results in a direction that actor can't even reach
                curr_action = th.clamp(
                    curr_action, self.actor.min_output, self.actor.max_output
                )
        else:
            # if we are not in learning mode we just use the actor neural net to get the action without adding noise

            # For PPO evaluation, use deterministic action (mean)
            curr_action = self.actor(next_observation, deterministic=True).detach()

            # noise is an tensor with zeros, because we are not in learning mode
            noise = th.zeros_like(curr_action, dtype=self.float_type)

        return curr_action, noise

    def update_policy(self) -> None:
        """
        Update actor and critic networks.
        """
        logger.debug("Updating Policy")

        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        # Get rollout buffer
        rollout_buffer = self.learning_role.rollout_buffer

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
            for param_group in strategy.critic.optimizer.param_groups:
                param_group["lr"] = learning_rate
            for param_group in strategy.actor.optimizer.param_groups:
                param_group["lr"] = learning_rate

        # Get last values for advantage computation
        last_values = np.zeros(n_rl_agents)
        dones = np.zeros(n_rl_agents)

        # Get the buffer size to index into the last stored state
        buffer_size = (
            rollout_buffer.pos
            if not rollout_buffer.full
            else rollout_buffer.buffer_size
        )

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
                    last_values[i] = (
                        strategy.critic(obs_tensor).cpu().numpy().flatten()[0]
                    )
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
                    critic = strategy.critic

                    obs_i = batch.observations[:, i, :]

                    # Construct centralized state
                    other_unique_obs = th.cat(
                        (
                            unique_obs_from_others[:, :i],
                            unique_obs_from_others[:, i + 1 :],
                        ),
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

                    advantages_i = (advantages_i - advantages_i.mean()) / (
                        advantages_i.std() + 1e-8
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
                        (
                            p.grad.norm().item()
                            for p in actor_params
                            if p.grad is not None
                        ),
                        default=0.0,
                    )
                    critic_max_grad_norm = max(
                        (
                            p.grad.norm().item()
                            for p in critic_params
                            if p.grad is not None
                        ),
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
                    unit_params[step_count][strategy.unit_id]["actor_loss"] = (
                        policy_loss.item()
                    )
                    unit_params[step_count][strategy.unit_id]["critic_loss"] = (
                        value_loss.item()
                    )
                    unit_params[step_count][strategy.unit_id][
                        "actor_total_grad_norm"
                    ] = (
                        actor_total_grad_norm.item()
                        if isinstance(actor_total_grad_norm, th.Tensor)
                        else actor_total_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id]["actor_max_grad_norm"] = (
                        actor_max_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id][
                        "critic_total_grad_norm"
                    ] = (
                        critic_total_grad_norm.item()
                        if isinstance(critic_total_grad_norm, th.Tensor)
                        else critic_total_grad_norm
                    )
                    unit_params[step_count][strategy.unit_id][
                        "critic_max_grad_norm"
                    ] = critic_max_grad_norm

                step_count += 1

        self.n_updates += 1

        # Log average metrics
        # Log average metrics
        # if self.learning_role.tensor_board_logger:
        #     self.learning_role.tensor_board_logger.log_scalar(
        #         "ppo/actor_loss", np.mean(all_actor_losses), self.n_updates
        #     )
        #     self.learning_role.tensor_board_logger.log_scalar(
        #         "ppo/critic_loss", np.mean(all_critic_losses), self.n_updates
        #     )
        #     self.learning_role.tensor_board_logger.log_scalar(
        #         "ppo/entropy_loss", np.mean(all_entropy_losses), self.n_updates
        #     )
        # if all_actor_losses:
        #     logger.info(
        #         f"PPO Update {self.n_updates} - Actor loss: {np.mean(all_actor_losses):.4f}, "
        #         f"Critic loss: {np.mean(all_critic_losses):.4f}, "
        #         f"Entropy loss: {np.mean(all_entropy_losses):.4f}"
        #     )

        # Write gradient params to output
        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)

        # Clear rollout buffer
        rollout_buffer.reset()

        logger.debug(
            f"PPO update complete. Actor loss: {np.mean(all_actor_losses):.4f}, "
            f"Value loss: {np.mean(all_critic_losses):.4f}"
        )
