import torch as th
from buffer import ReplayBuffer
from misc import polyak_update
from torch.nn import functional as F
from torch.optim import Adam

from assume.reinforcement_learning.algorithms.base_algorithm import BaseAlgorithm
from assume.reinforcement_learning.learning_utils import CriticTD3 as Critic


class TD3(BaseAlgorithm):
    def __init__(
        self,
        env=None,
        learning_rate=1e-4,
        buffer_size=1e6,
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
        super().__init__()

        self.env = env
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.train_freq = len(env.snapshots) if train_freq == -1 else train_freq
        self.gradient_steps = (
            self.train_freq if gradient_steps == -1 else gradient_steps
        )
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.rl_agents = [
            agent for agent in env.rl_powerplants + env.rl_storages if agent.learning
        ]
        self.n_rl_agents = len(self.rl_agents)

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim

        self.device = env.device
        self.float_type = env.float_type

        self.buffer = ReplayBuffer(
            buffer_size=int(buffer_size),
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            n_rl_units=self.n_rl_agents,
            device=self.device,
        )

        self.unique_obs_len = 8

        for agent in self.rl_agents:
            agent.critic = Critic(
                self.n_rl_agents,
                self.obs_dim,
                self.act_dim,
                self.float_type,
                self.unique_obs_len,
            )

            agent.critic_target = Critic(
                self.n_rl_agents,
                self.obs_dim,
                self.act_dim,
                self.float_type,
                self.unique_obs_len,
            )

            agent.critic.optimizer = Adam(
                agent.critic.parameters(), lr=self.learning_rate
            )

            agent.critic_target.load_state_dict(agent.critic.state_dict())
            agent.critic_target.train(mode=False)

            agent.critic = agent.critic.to(self.device)
            agent.critic_target = agent.critic_target.to(self.device)

        if self.env.load_params:
            self.load_params(self.env.load_params)

        self.steps_done = 0
        self.n_updates = 0

    def update_policy(self):
        self.steps_done += 1

        if (self.steps_done % self.train_freq == 0) and (
            self.env.episodes_done + 1 > self.learning_starts
        ):
            super().set_training_mode(True)

            for _ in range(self.gradient_steps):
                self.n_updates += 1

                for i, agent in enumerate(self.rl_agents):
                    if i % 100 == 0:
                        # sample replay buffer
                        transitions = self.buffer.sample(self.batch_size)
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
                                (
                                    agent.actor_target(next_states[:, i, :])
                                    + noise[:, i, :]
                                ).clamp(-1, 1)
                                for i, agent in enumerate(self.rl_agents)
                            ]
                            next_actions = th.stack(next_actions)

                            next_actions = next_actions.transpose(0, 1).contiguous()
                            next_actions = next_actions.view(
                                -1, self.n_rl_agents * self.act_dim
                            )

                        all_actions = actions.view(self.batch_size, -1)

                    all_states = th.cat(
                        (states[:, i, :].reshape(self.batch_size, -1), temp), axis=1
                    ).view(self.batch_size, -1)

                    all_next_states = th.cat(
                        (next_states[:, i, :].reshape(self.batch_size, -1), temp),
                        axis=1,
                    ).view(self.batch_size, -1)

                    with th.no_grad():
                        # Compute the next Q-values: min over all critics targets
                        next_q_values = th.cat(
                            agent.critic_target(all_next_states, next_actions), dim=1
                        )
                        next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                        target_Q_values = (
                            rewards[:, i].unsqueeze(1) + self.gamma * next_q_values
                        )

                    # Get current Q-values estimates for each critic network
                    current_Q_values = agent.critic(all_states, all_actions)

                    # Compute critic loss
                    critic_loss = sum(
                        F.mse_loss(current_q, target_Q_values)
                        for current_q in current_Q_values
                    )

                    # Optimize the critics
                    agent.critic.optimizer.zero_grad()
                    critic_loss.backward()
                    agent.critic.optimizer.step()

                    # Delayed policy updates
                    if self.n_updates % self.policy_delay == 0:
                        # Compute actor loss
                        state_i = states[:, i, :]
                        action_i = agent.actor(state_i)

                        all_actions_clone = actions.clone()
                        all_actions_clone[:, i, :] = action_i
                        all_actions_clone = all_actions_clone.view(self.batch_size, -1)

                        actor_loss = -agent.critic.q1_forward(
                            all_states, all_actions_clone
                        ).mean()

                        # Optimize the actor
                        agent.actor.optimizer.zero_grad()
                        actor_loss.backward()
                        agent.actor.optimizer.step()

                        polyak_update(
                            agent.critic.parameters(),
                            agent.critic_target.parameters(),
                            self.tau,
                        )
                        polyak_update(
                            agent.actor.parameters(),
                            agent.actor_target.parameters(),
                            self.tau,
                        )

            super().set_training_mode(False)
