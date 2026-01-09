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

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import polyak_update
from assume.reinforcement_learning.neural_network_architecture import (
    ActorPPO,
    CriticPPO
)
from assume.reinforcement_learning.rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)

class PPO(RLAlgorithm):
    """
    Proximal Policy Optimization (PPO) Algorithm.
    """

    def __init__(
        self, 
        learning_role,
        clip_range = 0.2, # Clipping parameter 
        clip_range_vf = None, 
        n_epochs = 10, # Number of epochs per update
        entropy_coef = 0.01, # Entropy bonus coefficient
        vf_coef = 0.5, # Value function loss coefficient
        max_grad_norm = 0.5, # Gradient clipping
    ):
        """Initialize PPO algorithm."""
        super().__init__(learning_role)

        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Update counter
        self.n_updates = 0

    def save_params(self, directory: str) -> None:
        """Save all actor and critic network parameters to disk."""
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    def save_critic_params(self, directory: str) -> None:
        """Save value network parameters for all agents."""
        os.makedirs(directory, exist_ok=True)
        
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critic.state_dict(),
                "critic_optimizer": strategy.critic.optimizer.state_dict(),
            }
            path = f"{directory}/critic_{u_id}.pt"
            th.save(obj, path)

        # Save unit ID order
        u_id_list = [str(u) for u in self.learning_role.rl_strats.keys()]
        mapping = {"u_id_order": u_id_list}
        map_path = os.path.join(directory, "u_id_order.json")
        with open(map_path, "w") as f:
            json.dump(mapping, f, indent=2)

    def save_actor_params(self, directory: str) -> None:
        """Save actor network parameters for all agents."""
        os.makedirs(directory, exist_ok=True)
        
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "actor": strategy.actor.state_dict(),
                "actor_optimizer": strategy.actor.optimizer.state_dict(),
            }
            path = f"{directory}/actor_{u_id}.pt"
            th.save(obj, path)

    def load_params(self, directory: str) -> None:
        """Load all actor and critic parameters from disk."""
        self.load_critic_params(directory)
        self.load_actor_params(directory)

    def load_critic_params(self, directory: str) -> None:
        """Load critic parameters."""
        logger.info("Loading PPO critic parameters...")

        if not os.path.exists(directory):
            logger.warning(
                "Specified directory does not exist. Using randomly initialized critics."
            )
            return

        for u_id, strategy in self.learning_role.rl_strats.items():
            critic_path = os.path.join(directory, "critics", f"critic_{u_id}.pt")
            if not os.path.exists(critic_path):
                logger.warning(f"No saved critic for {u_id}; skipping.")
                continue

            try:
                critic_params = th.load(critic_path, weights_only=True)
                strategy.critic.load_state_dict(critic_params["critic"])
                strategy.critic.optimizer.load_state_dict(critic_params["critic_optimizer"])
            except Exception as e:
                logger.warning(f"Failed to load critic for {u_id}: {e}")

    def load_actor_params(self, directory: str) -> None:
        """Load actor network parameters from disk."""
        logger.info("Loading PPO actor parameters...")
        
        if not os.path.exists(directory):
            logger.warning(
                "Specified directory for actors does not exist! "
                "Starting with randomly initialized values!"
            )
            return

        for u_id, strategy in self.learning_role.rl_strats.items():
            try:
                actor_params = self.load_obj(
                    directory=f"{directory}/actors/actor_{str(u_id)}.pt"
                )
                
                strategy.actor.load_state_dict(actor_params["actor"])
                strategy.actor.optimizer.load_state_dict(actor_params["actor_optimizer"])
                strategy.actor.loaded = True
                
            except Exception:
                logger.warning(f"No actor values loaded for agent {u_id}")

    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Initialize actor and critic networks for all agents.
        
        Args:
            actors_and_critics: Optional pre-existing networks to assign
        """
        if actors_and_critics is None:
            self.check_strategy_dimensions()
            self.create_actors()
            self.create_critics()
        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.actor = actors_and_critics["actors"][u_id]
                strategy.critic = actors_and_critics["critics"][u_id]

            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

    def check_strategy_dimensions(self) -> None:
        """Validate that all agents have consistent dimensions."""
        foresight_list = []
        obs_dim_list = []
        act_dim_list = []
        unique_obs_dim_list = []
        num_timeseries_obs_dim_list = []

        for strategy in self.learning_role.rl_strats.values():
            foresight_list.append(strategy.foresight)
            obs_dim_list.append(strategy.obs_dim)
            act_dim_list.append(strategy.act_dim)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
            num_timeseries_obs_dim_list.append(strategy.num_timeseries_obs_dim)
        
        if len(set(foresight_list)) > 1:
            raise ValueError(
                f"All foresight values must be the same for all RL agents. THe defined learning strategies have the following foresight values: {foresight_list}"
            )
        else:
            self.foresight = foresight_list[0]
            
        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                f"All observation dimensions must be the same. Got: {obs_dim_list}"
            )
        else:
            self.obs_dim = obs_dim_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError(
                f"All action dimensions must be the same. Got: {act_dim_list}"
            )
        else:
            self.act_dim = act_dim_list[0]

        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                f"All unique_obs_dim values must be the same. Got: {unique_obs_dim_list}"
            )
        else:
            self.unique_obs_dim = unique_obs_dim_list[0]

        if len(set(num_timeseries_obs_dim_list)) > 1:
            raise ValueError(
                f"All num_timeseries_obs_dim values must be the same. "
                f"Got: {num_timeseries_obs_dim_list}"
            )
        else:
            self.num_timeseries_obs_dim = num_timeseries_obs_dim_list[0]

    def create_actors(self) -> None:
        """Create stochastic actor networks for all agents."""
        for strategy in self.learning_role.rl_strats.values():
            # Create PPO Actor
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
        Create value networks for all agents.
        """
        n_agents = len(self.learning_role.rl_strats)

        for strategy in self.learning_role.rl_strats.values():
            # Create value network
            strategy.critic = CriticPPO(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            # Create optimizer
            strategy.critic.optimizer = AdamW(
                strategy.critic.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

    def extract_policy(self) -> dict:
        """Extract all actor and critic networks into a dictionary."""
        actors = {}
        critics = {}

        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            critics[u_id] = strategy.critic

        return {
            "actors": actors,
            "critics": critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
        }

    def update_policy(self) -> None:
        """
        Update actor and critic networks.
        """
        logger.debug("Updating Policy")

        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        # Get rollout buffer
        rollout_buffer = self.learning_role.rollout_buffer

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
        buffer_size = rollout_buffer.pos if not rollout_buffer.full else rollout_buffer.buffer_size

        if buffer_size > 0:
            # Get the last observation from the buffer
            last_obs = rollout_buffer.observations[buffer_size-1]
            last_dones = rollout_buffer.dones[buffer_size-1]

            with th.no_grad():
                for i, strategy in enumerate(strategies):
                    obs_tensor = th.as_tensor(
                        last_obs[i:i+1],
                        device = self.device,
                        dtype = self.float_type
                    )
                    # Get value estimate from critic
                    last_values[i] = strategy.critic(obs_tensor).cpu().numpy().flatten()[0]
                    dones[i] = last_dones[i]

        # Compute advantages and returns
        rollout_buffer.compute_returns_and_advantages(last_values, dones)

        # Initialize metrics storage
        all_actor_losses = []
        all_critic_losses = []
        all_entropy_losses = []

        for epoch in range(self.n_epochs):
            for batch in rollout_buffer.get(self.learning_config.batch_size):
                for i, strategy in enumerate(strategies):
                    actor = strategy.actor
                    critic = strategy.critic

                    obs_i = batch.observations[:, i, :]
                    actions_i = batch.actions[:, i, :]
                    old_log_probs_i = batch.old_log_probs[:, i]
                    advantages_i = batch.advantages[:, i]
                    returns_i = batch.returns[:, i]
                    old_values_i = batch.old_values[:, i]

                    advantages_i = (advantages_i - advantages_i.mean()) / (
                        advantages_i.std() + 1e-8
                    )

                    log_probs, entropy = actor.evaluate_actions(
                        obs_i,
                        actions_i
                    )
                    values = critic(obs_i).flatten()

                    # Importance sampling ratio
                    ratio = th.exp(log_probs - old_log_probs_i)

                    # Clipped surrogate objective
                    policy_loss_1 = advantages_i * ratio
                    policy_loss_2 = advantages_i * th.clamp(
                        ratio, 1 - self.clip_range, 1 + self.clip_range
                    )
                    policy_loss = -th.min(policy_loss_1, policy_loss_2)

                    # Entropy loss
                    entropy_loss = -self.entropy_coef * entropy.mean()

                    if self.clip_rnage_vf is not None:
                        # Clipped value function loss
                        values_clipped = old_values_i + th.clamp(
                            values - old_values_i,
                            -self.clip_range_vf,
                            self.clip_range_vf
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

                    # Gradient clipping
                    th.nn.utils.clip_grad_norm_(
                        actor.parameters(), self.max_grad_norm
                    )
                    th.nn.utils.clip_grad_norm_(
                        critic.parameters(), self.max_grad_norm
                    )

                    actor.optimizer.step()
                    critic.optimizer.step()

                    # Store metrics
                    all_actor_losses.append(policy_loss.item())
                    all_critic_losses.append(value_loss.item())
                    all_entropy_losses.append(entropy_loss.item())

        self.n_updates += 1

        # Log average metrics
        if self.learning_role.tensor_board_logger:
            self.learning_role.tensor_board_logger.log_scalar(
                "ppo/actor_loss", np.mean(all_actor_losses), self.n_updates
            )
            self.learning_role.tensor_board_logger.log_scalar(
                "ppo/critic_loss", np.mean(all_critic_losses), self.n_updates
            )
            self.learning_role.tensor_board_logger.log_scalar(
                "ppo/entropy_loss", np.mean(all_entropy_losses), self.n_updates
            )

        # Clear rollout buffer
        rollout_buffer.reset()

        logger.debug(
            f"PPO update complete. Actor loss: {np.mean(all_actor_losses):.4f}, "
            f"Value loss: {np.mean(all_critic_losses):.4f}"
        )