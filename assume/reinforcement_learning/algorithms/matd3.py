import logging

import torch as th
from torch.nn import functional as F

logger = logging.getLogger(__name__)

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import polyak_update


class TD3(RLAlgorithm):
    def __init__(
        self,
        learning_role,
        learning_rate=1e-4,
        learning_starts=100,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=-1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    ):
        self.learning_role = learning_role
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.train_freq = train_freq
        self.gradient_steps = (
            self.train_freq if gradient_steps == -1 else gradient_steps
        )

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.obs_dim = self.learning_role.obs_dim
        self.act_dim = self.learning_role.act_dim

        self.device = self.learning_role.device
        self.float_type = self.learning_role.float_type

        self.unique_obs_len = 8

    def update_policy(self):
        logger.info(f"Updating Policy")
        n_rl_agents = len(self.learning_role.rl_units)
        for _ in range(self.gradient_steps):
            i = 0
            strategy: LearningStrategy
            for u_id, strategy in self.learning_role.rl_units.items():
                critic_target = self.learning_role.target_critics[u_id]
                critic = self.learning_role.critics[u_id]
                actor = self.learning_role.rl_units[u_id].actor
                actor_target = self.learning_role.rl_units[u_id].actor_target
                transitions = None
                next_actions = None
                if i % 100 == 0:
                    transitions = self.learning_role.buffer.sample(self.batch_size)
                    states = transitions.observations
                    actions = transitions.actions
                    next_states = transitions.next_observations
                    rewards = transitions.rewards

                    with th.no_grad():
                        # Select action according to policy and add clipped noise
                        noise = actions.clone().data.normal_(0, self.target_policy_noise)
                        noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                        next_actions = [(actor_target(next_states[:, i, :]) + noise[:, i, :]).clamp(-1, 1) for i, agent in enumerate(self.learning_role.rl_units.values())]
                        next_actions = th.stack(next_actions)

                        next_actions = (next_actions.transpose(0,1).contiguous())
                        next_actions = next_actions.view(-1, n_rl_agents * self.act_dim)
                i+=1

                all_actions = actions.view(self.batch_size, -1)

                temp = th.cat((states[:, :i, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1),
                                states[:, i+1:, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1)), axis=1)

                all_states = th.cat((states[:, i, :].reshape(self.batch_size, -1), temp), axis = 1).view(self.batch_size, -1)

                temp = th.cat((next_states[:, :i, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1),
                                next_states[:, i+1:, self.obs_dim-self.unique_obs_len:].reshape(self.batch_size, -1)), axis=1)

                all_next_states = th.cat((next_states[:, i, :].reshape(self.batch_size, -1), temp), axis = 1).view(self.batch_size, -1)
                with th.no_grad():
                    # Compute the next Q-values: min over all critics targets
                    next_q_values = th.cat(critic_target(all_next_states, next_actions), dim = 1)
                    next_q_values, _ = th.min(next_q_values, dim = 1, keepdim=True)
                    target_Q_values = rewards[:, i].unsqueeze(1) + self.gamma * next_q_values

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

                    actor_loss = -critic.q1_forward(all_states,all_actions_clone).mean()

                    # TODO Optimize the actor
                    #actor.optimizer.zero_grad()
                    actor_loss.backward()
                    #actor.optimizer.step()

                    polyak_update(critic.parameters(), critic_target.parameters(), self.tau)
                    polyak_update(actor.parameters(), actor_target.parameters(), self.tau)

                # TODO return a transition back to the strategy
                strategy.update_transition(transitions)