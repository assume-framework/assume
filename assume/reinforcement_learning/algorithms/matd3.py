import logging

import torch as th
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import polyak_update


class TD3(RLAlgorithm):
    """
    Twin Delayed Deep Deterministic Policy Gradients (TD3).
    Addressing Function Approximation Error in Actor-Critic Methods.
    TD3 is a direct successor of DDPG and improves it using three major tricks:
    clipped double Q-Learning, delayed policy update and target policy smoothing.

    Open AI Spinning guide: https://spinningup.openai.com/en/latest/algorithms/td3.html

    Original paper: https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(
        self,
        learning_role,
        learning_rate=1e-4,
        episodes_collecting_initial_experience=100,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        gradient_steps=-1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    ):
        super().__init__(
            learning_role,
            learning_rate,
            episodes_collecting_initial_experience,
            batch_size,
            tau,
            gamma,
            gradient_steps,
            policy_delay,
            target_policy_noise,
            target_noise_clip,
        )
        self.n_updates = 0

    def update_policy(self):
        """
        Update the policy of the reinforcement learning agent using the Twin Delayed Deep Deterministic Policy Gradients (TD3) algorithm.

        This function performs the policy update step, which involves updating the actor (policy) and critic (Q-function) networks
        using TD3 algorithm. It iterates over the specified number of gradient steps and performs the following steps for each
        learning strategy:

        1. Sample a batch of transitions from the replay buffer.
        2. Calculate the next actions with added noise using the actor target network.
        3. Compute the target Q-values based on the next states, rewards, and the target critic network.
        4. Compute the critic loss as the mean squared error between current Q-values and target Q-values.
        5. Optimize the critic network by performing a gradient descent step.
        6. Optionally, update the actor network if the specified policy delay is reached.
        7. Apply Polyak averaging to update target networks.

        This function implements the TD3 algorithm's key step for policy improvement and exploration.
        """

        logger.info(f"Updating Policy")
        n_rl_agents = len(self.learning_role.rl_strats.keys())
        for _ in range(self.gradient_steps):
            self.n_updates += 1
            i = 0
            strategy: LearningStrategy

            for u_id, strategy in self.learning_role.rl_strats.items():
                critic_target = self.learning_role.target_critics[u_id]
                critic = self.learning_role.critics[u_id]
                actor = self.learning_role.rl_strats[u_id].actor
                actor_target = self.learning_role.rl_strats[u_id].actor_target

                if i % 100 == 0:
                    transitions = self.learning_role.buffer.sample(self.batch_size)
                    states = transitions.observations
                    actions = transitions.actions
                    next_states = transitions.next_observations
                    rewards = transitions.rewards

                    with th.no_grad():
                        # Select action according to policy and add clipped noise
                        noise = actions.clone().data.normal_(
                            0, self.target_policy_noise
                        )
                        noise = noise.clamp(
                            -self.target_noise_clip, self.target_noise_clip
                        )
                        next_actions = [
                            (actor_target(next_states[:, i, :]) + noise[:, i, :]).clamp(
                                -1, 1
                            )
                            for i in range(n_rl_agents)
                        ]
                        next_actions = th.stack(next_actions)

                        next_actions = next_actions.transpose(0, 1).contiguous()
                        next_actions = next_actions.view(-1, n_rl_agents * self.act_dim)

                all_actions = actions.view(self.batch_size, -1)

                # temp = th.cat((states[:, :i, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1),
                #                 states[:, i+1:, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1)), axis=1)

                # all_states = th.cat((states[:, i, :].reshape(self.batch_size, -1), temp), axis = 1).view(self.batch_size, -1)
                all_states = states[:, i, :].reshape(self.batch_size, -1)

                # temp = th.cat((next_states[:, :i, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1),
                #                 next_states[:, i+1:, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1)), axis=1)

                # all_next_states = th.cat((next_states[:, i, :].reshape(self.batch_size, -1), temp), axis = 1).view(self.batch_size, -1)
                all_next_states = next_states[:, i, :].reshape(self.batch_size, -1)

                with th.no_grad():
                    # Compute the next Q-values: min over all critics targets
                    next_q_values = th.cat(
                        critic_target(all_next_states, next_actions), dim=1
                    )
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    target_Q_values = (
                        rewards[:, i].unsqueeze(1) + self.gamma * next_q_values
                    )

                # Get current Q-values estimates for each critic network
                current_Q_values = critic(all_states, all_actions)

                # Compute critic loss
                critic_loss = sum(
                    F.mse_loss(current_q, target_Q_values)
                    for current_q in current_Q_values
                )

                # Optimize the critics
                critic.optimizer.zero_grad()
                critic_loss.backward()
                critic.optimizer.step()

                # Delayed policy updates
                if self.n_updates % self.policy_delay == 0:
                    # Compute actor loss
                    state_i = states[:, i, :]
                    action_i = actor(state_i)

                    all_actions_clone = actions.clone()
                    all_actions_clone[:, i, :] = action_i
                    all_actions_clone = all_actions_clone.view(self.batch_size, -1)

                    actor_loss = -critic.q1_forward(
                        all_states, all_actions_clone
                    ).mean()

                    actor.optimizer.zero_grad()
                    actor_loss.backward()
                    actor.optimizer.step()

                    polyak_update(
                        critic.parameters(), critic_target.parameters(), self.tau
                    )
                    polyak_update(
                        actor.parameters(), actor_target.parameters(), self.tau
                    )

                i += 1
