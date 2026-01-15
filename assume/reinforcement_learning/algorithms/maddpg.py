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

        self.actor_architecture_class = None  # define actor class here with kwargs pop or something similar so that parent class storage etc works properly
        self.critic_architecture_class = CriticDDPG

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

            else:
                # and add noise to the action
                curr_action = self.actor(next_observation).detach()
                noise = self.action_noise.noise(
                    device=self.device, dtype=self.float_type
                )
                curr_action += noise

                # make sure that noise adding does not exceed the actual output of the NN as it pushes results in a direction that actor can't even reach
                curr_action = th.clamp(
                    curr_action, self.actor.min_output, self.actor.max_output
                )
        else:
            curr_action = self.actor(next_observation).detach()
            # noise is an tensor with zeros, because we are not in learning mode
            noise = th.zeros_like(curr_action, dtype=self.float_type)

        return curr_action, noise

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
        updated_noise_decay = self.learning_role.calc_noise_from_progress(
            progress_remaining
        )
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
                next_actions = th.stack(
                    [
                        strategy.actor_target(next_states[:, i, :]).clamp(-1, 1)
                        for i, strategy in enumerate(strategies)
                    ]
                )
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
                    (
                        next_unique_obs_from_others[:, :i],
                        next_unique_obs_from_others[:, i + 1 :],
                    ),
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
                        next_states[:, i, :].reshape(
                            self.learning_config.batch_size, -1
                        ),
                        other_next_unique_obs.reshape(
                            self.learning_config.batch_size, -1
                        ),
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

                unit_params[step][strategy.unit_id]["critic_total_grad_norm"] = (
                    total_norm
                )
                unit_params[step][strategy.unit_id]["critic_max_grad_norm"] = (
                    max_grad_norm
                )

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

                unit_params[step][strategy.unit_id]["actor_total_grad_norm"] = (
                    total_norm
                )
                unit_params[step][strategy.unit_id]["actor_max_grad_norm"] = (
                    max_grad_norm
                )

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
