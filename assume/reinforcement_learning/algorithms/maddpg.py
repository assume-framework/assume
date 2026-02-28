# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import torch as th
from torch.nn import functional as F

from assume.reinforcement_learning.algorithms.base_algorithm import A2CAlgorithm
from assume.reinforcement_learning.learning_utils import (
    polyak_update,
)
from assume.reinforcement_learning.neural_network_architecture import CriticDDPG

logger = logging.getLogger(__name__)


class DDPG(A2CAlgorithm):
    """Deep Deterministic Policy Gradient (DDPG) Algorithm.
    
    An off-policy actor-critic algorithm that uses deterministic policy gradients
    for continuous action spaces. DDPG combines Q-learning with policy gradients,
    using:
    
    - A single critic network to estimate Q-values
    - Deterministic actor networks that map states to actions
    - Target networks updated via Polyak averaging for stability
    - Replay buffer for sample efficiency and decorrelation
    
    Attributes:
        n_updates: Counter for gradient updates performed.
        grad_clip_norm: Maximum gradient norm for clipping.
        critic_architecture_class: Critic network architecture (CriticDDPG).
    
    Example:
        >>> ddpg = DDPG(learning_role)
        >>> ddpg.update_policy()  # Performs one training iteration
    """

    def __init__(self, learning_role) -> None:
        """Initialize the DDPG algorithm.
        
        Sets up the algorithm with gradient counters, clipping parameters,
        and critic architecture.
        
        Args:
            learning_role: Learning role object managing agents and replay buffer.
                Must have off-policy configuration.
        """
        super().__init__(learning_role)
        
        # Gradient step counter
        self.n_updates = 0
        
        # Gradient clipping threshold
        self.grad_clip_norm = 1.0

        # Define the critic architecture class for DDPG (single critic)
        self.critic_architecture_class = CriticDDPG

    def update_policy(self) -> None:
        """Update actor and critic networks using DDPG algorithm.
        
        Performs one complete training iteration consisting of:
        1. Sampling batches from replay buffer
        2. Updating critic networks using MSE loss
        3. Updating actor networks using policy gradient
        4. Updating target networks via Polyak averaging
        
        """
        logger.debug("Updating Policy (MADDPG/DDPG)")

        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        # Initialize metrics storage for gradient logging
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
            for _ in range(self.learning_config.off_policy.gradient_steps)
        ]

        # Update noise decay and learning rate based on training progress
        progress_remaining = self.learning_role.get_progress_remaining()
        updated_noise_decay = self.learning_role.calc_noise_from_progress(progress_remaining)
        learning_rate = self.learning_role.calc_lr_from_progress(progress_remaining)

        # Update learning rates and noise schedules for all strategies
        for strategy in strategies:
            self.update_learning_rate(
                [strategy.critics.optimizer, strategy.actor.optimizer],
                learning_rate=learning_rate,
            )
            strategy.action_noise.update_noise_decay(updated_noise_decay)

        # Perform gradient updates for specified number of steps
        for step in range(self.learning_config.off_policy.gradient_steps):
            self.n_updates += 1

            # Sample transition batch from replay buffer
            transitions = self.learning_role.buffer.sample(
                self.learning_config.batch_size
            )
            
            states, actions, next_states, rewards = (
                transitions.observations,
                transitions.actions,
                transitions.next_observations,
                transitions.rewards,
            )

            # Compute target actions using target actors
            with th.no_grad():
                next_actions = th.stack([
                    strategy.actor_target(next_states[:, i, :]).clamp(-1, 1)
                    for i, strategy in enumerate(strategies)
                ])
                next_actions = next_actions.transpose(0, 1).contiguous()
                next_actions = next_actions.view(-1, n_rl_agents * self.act_dim)

            all_actions = actions.view(self.learning_config.batch_size, -1)

            # Extract unique observations for centralized critic construction
            unique_obs_from_others = states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)
            
            next_unique_obs_from_others = next_states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)

            # ------------------------------------------------------------
            # CRITIC UPDATE PHASE
            # ------------------------------------------------------------
            for strategy in strategies:
                strategy.critics.optimizer.zero_grad(set_to_none=True)

            total_critic_loss = 0.0

            for i, strategy in enumerate(strategies):
                critic = strategy.critics
                critic_target = strategy.target_critics

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
                parameters = list(strategy.critics.parameters())
                max_grad_norm = max(p.grad.norm() for p in parameters)
                total_norm = th.nn.utils.clip_grad_norm_(
                    parameters, max_norm=self.grad_clip_norm
                )
                strategy.critics.optimizer.step()

                unit_params[step][strategy.unit_id]["critic_total_grad_norm"] = total_norm
                unit_params[step][strategy.unit_id]["critic_max_grad_norm"] = max_grad_norm

            # ------------------------------------------------------------
            # ACTOR UPDATE PHASE (updated every step)
            # ------------------------------------------------------------
            for strategy in strategies:
                strategy.actor.optimizer.zero_grad(set_to_none=True)

            total_actor_loss = 0.0

            for i, strategy in enumerate(strategies):
                actor = strategy.actor
                critic = strategy.critics

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

            # ------------------------------------------------------------
            # TARGET NETWORK UPDATE PHASE (Polyak averaging)
            # ------------------------------------------------------------
            all_critic_params = []
            all_target_critic_params = []
            all_actor_params = []
            all_target_actor_params = []

            for strategy in strategies:
                all_critic_params.extend(strategy.critics.parameters())
                all_target_critic_params.extend(strategy.target_critics.parameters())
                all_actor_params.extend(strategy.actor.parameters())
                all_target_actor_params.extend(strategy.actor_target.parameters())

            polyak_update(
                all_critic_params,
                all_target_critic_params,
                self.learning_config.off_policy.tau,
            )
            polyak_update(
                all_actor_params,
                all_target_actor_params,
                self.learning_config.off_policy.tau,
            )

        # Log gradient parameters and metrics to output
        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)
