# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
MADDPG - Multi-Agent Deep Deterministic Policy Gradient

This module implements the DDPG algorithm for multi-agent settings (MADDPG).

DDPG vs TD3 Comparison:
-----------------------
| Feature           | DDPG (this)     | TD3              |
|-------------------|-----------------|------------------|
| Critics           | 1 (single)      | 2 (twin)         |
| Policy Updates    | Every step      | Delayed (1:2)    |
| Target Noise      | No              | Yes (smoothing)  |
| Overestimation    | Can occur       | Reduced          |
| Complexity        | Simpler         | More complex     |

MADDPG extends DDPG to multi-agent settings using:
- Centralized Training: Critic sees all agents' observations and actions
- Decentralized Execution: Each actor only uses its own observation
"""

import json
import logging
import os

import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import (
    polyak_update,
    transfer_weights,
)
from assume.reinforcement_learning.neural_network_architecture import CriticDDPG

logger = logging.getLogger(__name__)


class DDPG(RLAlgorithm):
    """
    Deep Deterministic Policy Gradient (DDPG) Algorithm.
    
    Extended to multi-agent settings (MADDPG) for electricity market simulations.
    
    Key Features:
    - Single critic network (vs twin critics in TD3)
    - Updates actor every step (no policy delay)
    - No target action smoothing noise
    - Centralized training with decentralized execution
    """

    def __init__(self, learning_role):
        """Initialize DDPG algorithm."""
        super().__init__(learning_role)
        
        # Gradient step counter
        self.n_updates = 0
        
        # Gradient clipping threshold
        self.grad_clip_norm = 1.0

    # =========================================================================
    # CHECKPOINT SAVING METHODS
    # =========================================================================

    def save_params(self, directory: str) -> None:
        """Save all actor and critic network parameters to disk."""
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    def save_critic_params(self, directory: str) -> None:
        """Save critic network parameters for all agents."""
        os.makedirs(directory, exist_ok=True)
        
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critic.state_dict(),
                "critic_target": strategy.target_critic.state_dict(),
                "critic_optimizer": strategy.critic.optimizer.state_dict(),
            }
            path = f"{directory}/critic_{u_id}.pt"
            th.save(obj, path)

        # Save unit ID order for weight transfer
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
                "actor_target": strategy.actor_target.state_dict(),
                "actor_optimizer": strategy.actor.optimizer.state_dict(),
            }
            path = f"{directory}/actor_{u_id}.pt"
            th.save(obj, path)

    # =========================================================================
    # CHECKPOINT LOADING METHODS
    # =========================================================================

    def load_params(self, directory: str) -> None:
        """Load all actor and critic parameters from disk."""
        self.load_critic_params(directory)
        self.load_actor_params(directory)

    def load_critic_params(self, directory: str) -> None:
        """Load critic parameters with support for agent count changes."""
        logger.info("Loading critic parameters...")

        if not os.path.exists(directory):
            logger.warning(
                "Specified directory does not exist. Using randomly initialized critics."
            )
            return

        # Load saved unit ID order
        map_path = os.path.join(directory, "critics", "u_id_order.json")
        if os.path.exists(map_path):
            with open(map_path) as f:
                loaded_id_order = json.load(f).get("u_id_order", [])
        else:
            logger.warning("No u_id_order.json: assuming same order as current.")
            loaded_id_order = [str(u) for u in self.learning_role.rl_strats.keys()]

        new_id_order = [str(u) for u in self.learning_role.rl_strats.keys()]
        direct_load = loaded_id_order == new_id_order

        if direct_load:
            logger.info("Agents order unchanged. Loading critic weights directly.")
        else:
            logger.info(
                f"Agents mismatch: n_old={len(loaded_id_order)}, "
                f"n_new={len(new_id_order)}. Transferring weights."
            )

        for u_id, strategy in self.learning_role.rl_strats.items():
            critic_path = os.path.join(directory, "critics", f"critic_{u_id}.pt")
            if not os.path.exists(critic_path):
                logger.warning(f"No saved critic for {u_id}; skipping.")
                continue

            try:
                critic_params = th.load(critic_path, weights_only=True)
                
                for key in ("critic", "critic_target", "critic_optimizer"):
                    if key not in critic_params:
                        logger.warning(f"Missing {key} in critic params for {u_id}.")
                        continue

                if direct_load:
                    strategy.critic.load_state_dict(critic_params["critic"])
                    strategy.target_critic.load_state_dict(critic_params["critic_target"])
                    strategy.critic.optimizer.load_state_dict(critic_params["critic_optimizer"])
                else:
                    # Weight transfer for agent count changes
                    critic_weights = transfer_weights(
                        model=strategy.critic,
                        loaded_state=critic_params["critic"],
                        loaded_id_order=loaded_id_order,
                        new_id_order=new_id_order,
                        obs_base=strategy.obs_dim,
                        act_dim=strategy.act_dim,
                        unique_obs=strategy.unique_obs_dim,
                    )
                    target_critic_weights = transfer_weights(
                        model=strategy.target_critic,
                        loaded_state=critic_params["critic_target"],
                        loaded_id_order=loaded_id_order,
                        new_id_order=new_id_order,
                        obs_base=strategy.obs_dim,
                        act_dim=strategy.act_dim,
                        unique_obs=strategy.unique_obs_dim,
                    )

                    if critic_weights is None or target_critic_weights is None:
                        logger.warning(f"Weights transfer failed for {u_id}.")
                        continue

                    strategy.critic.load_state_dict(critic_weights)
                    strategy.target_critic.load_state_dict(target_critic_weights)

            except Exception as e:
                logger.warning(f"Failed to load critic for {u_id}: {e}")

    def load_actor_params(self, directory: str) -> None:
        """Load actor network parameters from disk."""
        logger.info("Loading actor parameters...")
        
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
                strategy.actor_target.load_state_dict(actor_params["actor_target"])
                strategy.actor.optimizer.load_state_dict(actor_params["actor_optimizer"])
                strategy.actor.loaded = True
                
            except Exception:
                logger.warning(f"No actor values loaded for agent {u_id}")

    # =========================================================================
    # NETWORK INITIALIZATION
    # =========================================================================

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
                strategy.actor_target = actors_and_critics["actor_targets"][u_id]
                strategy.critic = actors_and_critics["critics"][u_id]
                strategy.target_critic = actors_and_critics["target_critics"][u_id]

            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

    def check_strategy_dimensions(self) -> None:
        """Validate that all agents have consistent dimensions."""
        obs_dim_list = []
        act_dim_list = []
        unique_obs_dim_list = []
        num_timeseries_obs_dim_list = []

        for strategy in self.learning_role.rl_strats.values():
            obs_dim_list.append(strategy.obs_dim)
            act_dim_list.append(strategy.act_dim)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
            num_timeseries_obs_dim_list.append(strategy.num_timeseries_obs_dim)

        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                f"All observation dimensions must be the same. "
                f"Got: {obs_dim_list}"
            )
        else:
            self.obs_dim = obs_dim_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError(
                f"All action dimensions must be the same. "
                f"Got: {act_dim_list}"
            )
        else:
            self.act_dim = act_dim_list[0]

        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                f"All unique_obs_dim values must be the same. "
                f"Got: {unique_obs_dim_list}"
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
        """Create actor (policy) networks for all agents."""
        for strategy in self.learning_role.rl_strats.values():
            # Create main actor network
            strategy.actor = self.actor_architecture_class(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
            ).to(self.device)

            # Create target actor network
            strategy.actor_target = self.actor_architecture_class(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
            ).to(self.device)

            # Initialize target with same weights
            strategy.actor_target.load_state_dict(strategy.actor.state_dict())
            strategy.actor_target.train(mode=False)

            # Create optimizer
            strategy.actor.optimizer = AdamW(
                strategy.actor.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

            strategy.actor.loaded = False

    def create_critics(self) -> None:
        """
        Create critic (Q-function) networks for all agents.
        
        Key difference from TD3: Uses single critic instead of twin critics.
        """
        n_agents = len(self.learning_role.rl_strats)

        for strategy in self.learning_role.rl_strats.values():
            # Create main critic (single Q-network, not twin)
            strategy.critic = CriticDDPG(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            # Create target critic
            strategy.target_critic = CriticDDPG(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            # Initialize target with same weights
            strategy.target_critic.load_state_dict(strategy.critic.state_dict())
            strategy.target_critic.train(mode=False)

            # Create optimizer
            strategy.critic.optimizer = AdamW(
                strategy.critic.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

    def extract_policy(self) -> dict:
        """Extract all actor and critic networks into a dictionary."""
        actors = {}
        actor_targets = {}
        critics = {}
        target_critics = {}

        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            actor_targets[u_id] = strategy.actor_target
            critics[u_id] = strategy.critic
            target_critics[u_id] = strategy.target_critic

        return {
            "actors": actors,
            "actor_targets": actor_targets,
            "critics": critics,
            "target_critics": target_critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
        }

    # =========================================================================
    # CORE TRAINING: POLICY UPDATE
    # =========================================================================

    def update_policy(self) -> None:
        """
        Update actor and critic networks using the DDPG algorithm.
        
        Key differences from TD3:
        1. Uses single critic (no twin Q-learning)
        2. Updates actor every step (no policy delay)
        3. No target action smoothing noise
        """
        logger.debug("Updating Policy (MADDPG/DDPG)")

        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        # Initialize metrics storage
        unit_params = [
            {
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
            for _ in range(self.learning_config.gradient_steps)
        ]

        # Update noise and learning rate schedules
        progress_remaining = self.learning_role.get_progress_remaining()
        updated_noise_decay = self.learning_role.calc_noise_from_progress(progress_remaining)
        learning_rate = self.learning_role.calc_lr_from_progress(progress_remaining)

        for strategy in strategies:
            self.update_learning_rate(
                [strategy.critic.optimizer, strategy.actor.optimizer],
                learning_rate=learning_rate,
            )
            strategy.action_noise.update_noise_decay(updated_noise_decay)

        # Main gradient step loop
        for step in range(self.learning_config.gradient_steps):
            self.n_updates += 1

            # Sample from replay buffer
            transitions = self.learning_role.buffer.sample(
                self.learning_config.batch_size
            )
            
            states, actions, next_states, rewards = (
                transitions.observations,
                transitions.actions,
                transitions.next_observations,
                transitions.rewards,
            )

            # Compute target actions (no smoothing noise in DDPG)
            with th.no_grad():
                next_actions = th.stack([
                    strategy.actor_target(next_states[:, i, :]).clamp(-1, 1)
                    for i, strategy in enumerate(strategies)
                ])
                next_actions = next_actions.transpose(0, 1).contiguous()
                next_actions = next_actions.view(-1, n_rl_agents * self.act_dim)

            all_actions = actions.view(self.learning_config.batch_size, -1)

            # Precompute observation slices
            unique_obs_from_others = states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)
            
            next_unique_obs_from_others = next_states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)

            # =================================================================
            # CRITIC UPDATE
            # =================================================================
            for strategy in strategies:
                strategy.critic.optimizer.zero_grad(set_to_none=True)

            total_critic_loss = 0.0

            for i, strategy in enumerate(strategies):
                critic = strategy.critic
                critic_target = strategy.target_critic

                # Build centralized observation
                other_unique_obs = th.cat(
                    (unique_obs_from_others[:, :i], unique_obs_from_others[:, i + 1 :]),
                    dim=1,
                )
                other_next_unique_obs = th.cat(
                    (next_unique_obs_from_others[:, :i], next_unique_obs_from_others[:, i + 1 :]),
                    dim=1,
                )

                all_states = th.cat(
                    (
                        states[:, i, :].reshape(self.learning_config.batch_size, -1),
                        other_unique_obs.reshape(self.learning_config.batch_size, -1),
                    ),
                    dim=1,
                )
                all_next_states = th.cat(
                    (
                        next_states[:, i, :].reshape(self.learning_config.batch_size, -1),
                        other_next_unique_obs.reshape(self.learning_config.batch_size, -1),
                    ),
                    dim=1,
                )

                # Compute target Q-value (single critic, no min)
                with th.no_grad():
                    next_q_value = critic_target(all_next_states, next_actions)
                    target_Q_value = (
                        rewards[:, i].unsqueeze(1)
                        + self.learning_config.gamma * next_q_value
                    )

                # Compute current Q-value
                current_Q_value = critic(all_states, all_actions)

                # MSE loss (single critic)
                critic_loss = F.mse_loss(current_Q_value, target_Q_value)

                unit_params[step][strategy.unit_id]["critic_loss"] = critic_loss.item()
                total_critic_loss += critic_loss

            # Backward pass for critics
            total_critic_loss.backward()

            for strategy in strategies:
                parameters = list(strategy.critic.parameters())
                max_grad_norm = max(p.grad.norm() for p in parameters)
                total_norm = th.nn.utils.clip_grad_norm_(
                    parameters, max_norm=self.grad_clip_norm
                )
                strategy.critic.optimizer.step()

                unit_params[step][strategy.unit_id]["critic_total_grad_norm"] = total_norm
                unit_params[step][strategy.unit_id]["critic_max_grad_norm"] = max_grad_norm

            # =================================================================
            # ACTOR UPDATE (every step, no delay in DDPG)
            # =================================================================
            for strategy in strategies:
                strategy.actor.optimizer.zero_grad(set_to_none=True)

            total_actor_loss = 0.0

            for i, strategy in enumerate(strategies):
                actor = strategy.actor
                critic = strategy.critic

                state_i = states[:, i, :]
                action_i = actor(state_i)

                other_unique_obs = th.cat(
                    (unique_obs_from_others[:, :i], unique_obs_from_others[:, i + 1 :]),
                    dim=1,
                )
                all_states_i = th.cat(
                    (
                        state_i.reshape(self.learning_config.batch_size, -1),
                        other_unique_obs.reshape(self.learning_config.batch_size, -1),
                    ),
                    dim=1,
                )

                all_actions_clone = actions.clone().detach()
                all_actions_clone[:, i, :] = action_i
                all_actions_clone = all_actions_clone.view(
                    self.learning_config.batch_size, -1
                )

                # Actor loss: maximize Q-value
                actor_loss = -critic(all_states_i, all_actions_clone).mean()

                unit_params[step][strategy.unit_id]["actor_loss"] = actor_loss.item()
                total_actor_loss += actor_loss

            # Backward pass for actors
            total_actor_loss.backward()

            for strategy in strategies:
                parameters = list(strategy.actor.parameters())
                max_grad_norm = max(p.grad.norm() for p in parameters)
                total_norm = th.nn.utils.clip_grad_norm_(
                    parameters, max_norm=self.grad_clip_norm
                )
                strategy.actor.optimizer.step()

                unit_params[step][strategy.unit_id]["actor_total_grad_norm"] = total_norm
                unit_params[step][strategy.unit_id]["actor_max_grad_norm"] = max_grad_norm

            # =================================================================
            # TARGET NETWORK UPDATES (Polyak averaging)
            # =================================================================
            all_critic_params = []
            all_target_critic_params = []
            all_actor_params = []
            all_target_actor_params = []

            for strategy in strategies:
                all_critic_params.extend(strategy.critic.parameters())
                all_target_critic_params.extend(strategy.target_critic.parameters())
                all_actor_params.extend(strategy.actor.parameters())
                all_target_actor_params.extend(strategy.actor_target.parameters())

            polyak_update(
                all_critic_params,
                all_target_critic_params,
                self.learning_config.tau,
            )
            polyak_update(
                all_actor_params,
                all_target_actor_params,
                self.learning_config.tau,
            )

        # Log metrics
        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)
