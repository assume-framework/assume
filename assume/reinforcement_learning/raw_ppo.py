import gym
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

class MLPActorCritic(nn.Module):
    """
    Simple MLP Actor-Critic network with separate actor and critic heads.
    """
    def __init__(self, obs_dim, act_dim):
        super(MLPActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        # Actor head
        self.actor = nn.Linear(64, act_dim)
        # Critic head
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        shared_out = self.shared(obs)
        return self.actor(shared_out), self.critic(shared_out)

    def act(self, obs):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        logits, values = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_log_probs, torch.squeeze(values, dim=-1), dist_entropy


class PPO:
    """
    Proximal Policy Optimization (PPO) implementation in PyTorch.
    """
    def __init__(self, env, actor_critic, clip_param=0.2, entcoeff=0.01, optim_stepsize=1e-3, optim_epochs=4, gamma=0.99, lam=0.95, batch_size=64):
        self.env = env
        self.actor_critic = actor_critic
        self.clip_param = clip_param
        self.entcoeff = entcoeff
        self.optim_epochs = optim_epochs
        self.optim_stepsize = optim_stepsize
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.optim_stepsize)

    def discount_rewards(self, rewards, dones, gamma):
        """
        Compute discounted rewards.
        """
        discounted_rewards = []
        r = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                r = 0
            r = reward + gamma * r
            discounted_rewards.insert(0, r)
        return discounted_rewards

    def compute_gae(self, rewards, values, dones, gamma, lam):
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        adv = 0
        advantages = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            adv = delta + gamma * lam * adv * (1 - dones[t])
            advantages.insert(0, adv)
        return advantages

    def rollout(self, timesteps_per_actorbatch):
        """
        Collect trajectories by running the policy in the environment.
        """
        # Reset env
        obs = self.env.reset()
        obs_list, actions_list, rewards_list, dones_list, log_probs_list, values_list = [], [], [], [], [], []
        for _ in range(timesteps_per_actorbatch):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action, log_prob, value = self.actor_critic.act(obs_tensor)

            obs_list.append(obs_tensor)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)

            next_obs, reward, done, _ = self.env.step(action.item())
            rewards_list.append(reward)
            dones_list.append(done)

            obs = next_obs
            if done:
                obs = self.env.reset()

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        _, _, last_value = self.actor_critic.act(obs_tensor)

        values_list.append(last_value)

        return {
            "observations": torch.cat(obs_list),
            "actions": torch.cat(actions_list),
            "log_probs": torch.cat(log_probs_list),
            "values": torch.cat(values_list),
            "rewards": rewards_list,
            "dones": dones_list,
        }

    def ppo_update(self, batch, clip_param, entcoeff):
        """
        Update the policy using PPO objective.
        """
        observations, actions, old_log_probs, returns, advantages = batch

        for _ in range(self.optim_epochs):
            new_log_probs, values, entropy = self.actor_critic.evaluate_actions(observations, actions)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (returns - values).pow(2).mean()
            entropy_loss = entropy.mean()

            loss = policy_loss + 0.5 * value_loss - entcoeff * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, total_timesteps, timesteps_per_actorbatch, log_interval=100):
        """
        Main training loop.
        """
        total_timesteps_done = 0
        reward_history = deque(maxlen=100)

        while total_timesteps_done < total_timesteps:
            # Rollout
            batch = self.rollout(timesteps_per_actorbatch)
            observations = batch["observations"]
            actions = batch["actions"]
            old_log_probs = batch["log_probs"]
            rewards = batch["rewards"]
            dones = batch["dones"]
            values = batch["values"].detach()

            # Compute discounted rewards and advantages
            returns = torch.FloatTensor(self.discount_rewards(rewards, dones, self.gamma))
            advantages = torch.FloatTensor(self.compute_gae(rewards, values.numpy(), dones, self.gamma, self.lam))

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Update the policy using PPO
            batch_data = (observations, actions, old_log_probs, returns, advantages)
            self.ppo_update(batch_data, self.clip_param, self.entcoeff)

            total_timesteps_done += timesteps_per_actorbatch
            avg_reward = np.mean(rewards)
            reward_history.append(avg_reward)

            if total_timesteps_done % log_interval == 0:
                print(f"Timesteps: {total_timesteps_done}, Avg Reward: {np.mean(reward_history)}")


# Example usage with CartPole environment
env = gym.make('CartPole-v1')
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

actor_critic = MLPActorCritic(obs_dim, act_dim)
ppo = PPO(env, actor_critic, clip_param=0.2, entcoeff=0.01, optim_stepsize=1e-3, optim_epochs=4, gamma=0.99, lam=0.95)

ppo.train(total_timesteps=10000, timesteps_per_actorbatch=256)
