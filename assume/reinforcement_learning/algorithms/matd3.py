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
from assume.reinforcement_learning.neural_network_architecture import CriticTD3

logger = logging.getLogger(__name__)


class TD3(A2CAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradients (TD3).
    Addressing Function Approximation Error in Actor-Critic Methods.
    TD3 is a direct successor of DDPG and improves it using three major tricks:
    clipped double Q-Learning, delayed policy update and target policy smoothing.

    Open AI Spinning guide: https://spinningup.openai.com/en/latest/algorithms/td3.html

    Original paper: https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(self, learning_role):
        super().__init__(learning_role)

        self.n_updates = 0
        self.grad_clip_norm = 1.0

        # Define the critic architecture class for TD3
        self.critic_architecture_class = CriticTD3

    def update_policy(self):
        """
        Update the policy of the reinforcement learning agent using the Twin Delayed Deep Deterministic Policy Gradients (TD3) algorithm.

        Note:
            This function performs the policy update step, which involves updating the actor (policy) and critic (Q-function) networks
            using TD3 algorithm. It iterates over the specified number of gradient steps and performs the following steps for each
            learning strategy:

            1. Sample a batch of transitions from the replay buffer.
            2. Calculate the next actions with added noise using the actor target network.
            3. Compute the target Q-values based on the next states, rewards, and the target critic network.
            4. Compute the critic loss as the mean squared error between current Q-values and target Q-values.
            5. Optimize the critic network by performing a gradient descent step.
            6. Update the actor network if the specified policy delay is reached.
            7. Apply Polyak averaging to update target networks.

        """

        logger.debug("Updating Policy (TD3)")

        # Stack strategies for easier access
        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

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

        # update noise decay and learning rate
        updated_noise_decay = self.learning_role.calc_noise_from_progress(
            self.learning_role.get_progress_remaining()
        )

        learning_rate = self.learning_role.calc_lr_from_progress(
            self.learning_role.get_progress_remaining()
        )

        # loop over all units to avoid update call for every gradient step, as it will be ambiguous
        for strategy in strategies:
            self.update_learning_rate(
                [
                    strategy.critics.optimizer,
                    strategy.actor.optimizer,
                ],
                learning_rate=learning_rate,
            )
            strategy.action_noise.update_noise_decay(updated_noise_decay)

        for step in range(self.learning_config.gradient_steps):
            self.n_updates += 1

            transitions = self.learning_role.buffer.sample(
                self.learning_config.batch_size
            )
            states, actions, next_states, rewards = (
                transitions.observations,
                transitions.actions,
                transitions.next_observations,
                transitions.rewards,
            )

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = (
                    th.randn_like(actions) * self.learning_config.off_policy.target_policy_noise
                )
                noise = noise.clamp(
                    -self.learning_config.off_policy.target_noise_clip,
                    self.learning_config.off_policy.target_noise_clip,
                )

                # Select next actions for all agents
                next_actions = th.stack(
                    [
                        (
                            strategy.actor_target(next_states[:, i, :]) + noise[:, i, :]
                        ).clamp(-1, 1)
                        for i, strategy in enumerate(strategies)
                    ]
                )
                next_actions = next_actions.transpose(0, 1).contiguous()
                next_actions = next_actions.view(-1, n_rl_agents * self.act_dim)

            all_actions = actions.view(self.learning_config.batch_size, -1)

            # Precompute unique observation parts for all agents
            unique_obs_from_others = states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)
            next_unique_obs_from_others = next_states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.learning_config.batch_size, n_rl_agents, -1)

            #####################################################################
            # CRITIC UPDATE: Accumulate losses for all agents, then backprop once
            #####################################################################

            # Zero-grad for all critics before accumulation
            for strategy in strategies:
                strategy.critics.optimizer.zero_grad(set_to_none=True)

            total_critic_loss = 0.0

            # Loop over all agents and accumulate critic loss
            for i, strategy in enumerate(strategies):
                actor = strategy.actor
                critic = strategy.critics
                critic_target = strategy.target_critics

                # Efficiently extract unique observations from all other agents
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

                # Construct final state representations
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

                # Compute the next Q-values: min over all critics targets
                with th.no_grad():
                    next_q_values = th.cat(
                        critic_target(all_next_states, next_actions), dim=1
                    )
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_Q_values = (
                        rewards[:, i].unsqueeze(1)
                        + self.learning_config.gamma * next_q_values
                    )

                # Get current Q-values estimates for each critic network
                current_Q_values = critic(all_states, all_actions)

                # Accumulate critic loss for this agent
                critic_loss = sum(
                    F.mse_loss(current_q, target_Q_values)
                    for current_q in current_Q_values
                )

                # Store the critic loss for this unit ID
                unit_params[step][strategy.unit_id]["critic_loss"] = critic_loss.item()
                total_critic_loss += critic_loss

            # Single backward pass for all agents' critics
            total_critic_loss.backward()

            # Clip the gradients and step each critic optimizer
            for strategy in strategies:
                parameters = list(strategy.critics.parameters())

                # Determine clipping statistics
                max_grad_norm = max(p.grad.norm() for p in parameters)

                # Perform clipping
                total_norm = th.nn.utils.clip_grad_norm_(
                    parameters, max_norm=self.grad_clip_norm
                )
                strategy.critics.optimizer.step()

                # Store clipping statistics
                unit_params[step][strategy.unit_id]["critic_total_grad_norm"] = (
                    total_norm
                )
                unit_params[step][strategy.unit_id]["critic_max_grad_norm"] = (
                    max_grad_norm
                )

            ######################################################################
            # ACTOR UPDATE (DELAYED): Accumulate losses for all agents in one pass
            ######################################################################
            if self.n_updates % self.learning_config.off_policy.policy_delay == 0:
                # Zero-grad for all actors first
                for strategy in strategies:
                    strategy.actor.optimizer.zero_grad(set_to_none=True)

                total_actor_loss = 0.0

                # We'll compute each agent's actor loss, accumulate, then do one backprop
                for i, strategy in enumerate(strategies):
                    actor = strategy.actor
                    critic = strategy.critics

                    # Build local state for actor i
                    state_i = states[:, i, :]
                    action_i = actor(state_i)

                    # Construct final state representation for agent i
                    other_unique_obs = th.cat(
                        (
                            unique_obs_from_others[:, :i],
                            unique_obs_from_others[:, i + 1 :],
                        ),
                        dim=1,
                    )
                    all_states_i = th.cat(
                        (
                            state_i.reshape(self.learning_config.batch_size, -1),
                            other_unique_obs.reshape(
                                self.learning_config.batch_size, -1
                            ),
                        ),
                        dim=1,
                    )

                    # Replace the i-th agent's action in the batch
                    all_actions_clone = actions.clone().detach()
                    all_actions_clone[:, i, :] = action_i

                    # Flatten again for the critic
                    all_actions_clone = all_actions_clone.view(
                        self.learning_config.batch_size, -1
                    )

                    # Calculate actor loss (negative Q1 of the updated action)
                    actor_loss = -critic.q1_forward(
                        all_states_i, all_actions_clone
                    ).mean()

                    # Store the actor loss for this unit ID
                    unit_params[step][strategy.unit_id]["actor_loss"] = (
                        actor_loss.item()
                    )
                    # Accumulate actor losses
                    total_actor_loss += actor_loss

                # Single backward pass for all actors
                total_actor_loss.backward()

                # Clip and step each actor optimizer
                for strategy in strategies:
                    parameters = list(strategy.actor.parameters())

                    # Determine clipping statistics
                    max_grad_norm = max(p.grad.norm() for p in parameters)

                    # Perform clipping
                    total_norm = th.nn.utils.clip_grad_norm_(
                        parameters, max_norm=self.grad_clip_norm
                    )

                    strategy.actor.optimizer.step()

                    # Store clipping statistics
                    unit_params[step][strategy.unit_id]["actor_total_grad_norm"] = (
                        total_norm
                    )
                    unit_params[step][strategy.unit_id]["actor_max_grad_norm"] = (
                        max_grad_norm
                    )

                # Perform batch-wise Polyak update at the end (instead of inside the loop)
                all_critic_params = []
                all_target_critic_params = []

                all_actor_params = []
                all_target_actor_params = []

                for strategy in strategies:
                    all_critic_params.extend(strategy.critics.parameters())
                    all_target_critic_params.extend(
                        strategy.target_critics.parameters()
                    )

                    all_actor_params.extend(strategy.actor.parameters())
                    all_target_actor_params.extend(strategy.actor_target.parameters())

                # Perform batch-wise Polyak update (NO LOOPS)
                polyak_update(
                    all_critic_params,
                    all_target_critic_params,
                    self.learning_config.off_policy.tau,
                )
                polyak_update(
                    all_actor_params, all_target_actor_params, self.learning_config.off_policy.tau
                )

        self.learning_role.write_rl_grad_params_to_output(learning_rate, unit_params)
