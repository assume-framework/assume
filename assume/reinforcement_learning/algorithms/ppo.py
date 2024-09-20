# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import torch as th
from torch.nn import functional as F
from torch.optim import Adam

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.neural_network_architecture import ActorPPO, CriticPPO

logger = logging.getLogger(__name__)


class PPO(RLAlgorithm):
    """
    Proximal Policy Optimization (PPO) is a robust and efficient policy gradient method for reinforcement learning. 
    It strikes a balance between trust-region methods and simpler approaches by using clipped objective functions. 
    PPO avoids large updates to the policy by restricting changes to stay within a specified range, which helps stabilize training. 
    The key improvements include the introduction of a surrogate objective that limits policy updates and ensures efficient learning, 
    as well as the use of multiple epochs of stochastic gradient descent on batches of data.

    Open AI Spinning guide: https://spinningup.openai.com/en/latest/algorithms/ppo.html#

    Original paper: https://arxiv.org/pdf/1802.09477.pdf
    """
    
    # Change order and mandatory parameters in the superclass, removed and newly added parameters
    def __init__(
        self,
        learning_role,
        learning_rate=1e-4,
        batch_size=1024,
        gamma=0.99,
        epochs=10,  # PPO specific
        clip_ratio=0.2,  # PPO specific
        vf_coef=0.5,  # PPO specific
        entropy_coef=0.01,  # PPO specific
        max_grad_norm=0.5,  # PPO specific
        gae_lambda=0.95,  # PPO specific
        actor_architecture="mlp",
    ):
        super().__init__(
            learning_role,
            learning_rate,
            batch_size,
            gamma,
            actor_architecture,
        )
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        

    # Unchanged method from MATD3
    def save_params(self, directory):
        """
        This method saves the parameters of both the actor and critic networks associated with the learning role. It organizes the
        saved parameters into separate directories for critics and actors within the specified base directory.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    # Removed critic_target in comparison to MATD3
    def save_critic_params(self, directory):
        """
        Save the parameters of critic networks.

        This method saves the parameters of the critic networks, including the critic's state_dict, critic_target's state_dict. It organizes the saved parameters into a directory structure specific to the critic
        associated with each learning   strategy.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        os.makedirs(directory, exist_ok=True)
        for u_id in self.learning_role.rl_strats.keys():
            obj = {
                "critic": self.learning_role.critics[u_id].state_dict(),
                # "critic_target": self.learning_role.target_critics[u_id].state_dict(),
                "critic_optimizer": self.learning_role.critics[
                    u_id
                ].optimizer.state_dict(),
            }
            path = f"{directory}/critic_{u_id}.pt"
            th.save(obj, path)

    # Removed actor_target in comparison to MATD3 (Actor network = policy network)
    def save_actor_params(self, directory):
        """
        Save the parameters of actor networks.

        This method saves the parameters of the actor networks, including the actor's state_dict, actor_target's state_dict, and
        the actor's optimizer state_dict. It organizes the saved parameters into a directory structure specific to the actor
        associated with each learning strategy.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        os.makedirs(directory, exist_ok=True)
        for u_id in self.learning_role.rl_strats.keys():
            obj = {
                "actor": self.learning_role.rl_strats[u_id].actor.state_dict(),
                # "actor_target": self.learning_role.rl_strats[
                #     u_id
                # ].actor_target.state_dict(),
                "actor_optimizer": self.learning_role.rl_strats[
                    u_id
                ].actor.optimizer.state_dict(),
            }
            path = f"{directory}/actor_{u_id}.pt"
            th.save(obj, path)

    # Unchanged method from MATD3
    def load_params(self, directory: str) -> None:
        """
        Load the parameters of both actor and critic networks.

        This method loads the parameters of both the actor and critic networks associated with the learning role from the specified
        directory. It uses the `load_critic_params` and `load_actor_params` methods to load the respective parameters.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        self.load_critic_params(directory)
        self.load_actor_params(directory)

    # Removed critic_target in comparison to MATD3 (critic network = value function network)
    def load_critic_params(self, directory: str) -> None:
        """
        Load the parameters of critic networks from a specified directory.

        This method loads the parameters of critic networks, including the critic's state_dict, critic_target's state_dict, and
        the critic's optimizer state_dict, from the specified directory. It iterates through the learning strategies associated
        with the learning role, loads the respective parameters, and updates the critic and target critic networks accordingly.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        logger.info("Loading critic parameters...")

        if not os.path.exists(directory):
            logger.warning(
                "Specified directory for loading the critics does not exist! Starting with randomly initialized values!"
            )
            return

        for u_id in self.learning_role.rl_strats.keys():
            try:
                critic_params = self.load_obj(
                    directory=f"{directory}/critics/critic_{str(u_id)}.pt"
                )
                self.learning_role.critics[u_id].load_state_dict(
                    critic_params["critic"]
                )
                # self.learning_role.target_critics[u_id].load_state_dict(
                #     critic_params["critic_target"]
                # )
                self.learning_role.critics[u_id].optimizer.load_state_dict(
                    critic_params["critic_optimizer"]
                )
            except Exception:
                logger.warning(f"No critic values loaded for agent {u_id}")

    # Removed actor_target in comparison to MATD3
    def load_actor_params(self, directory: str) -> None:
        """
        Load the parameters of actor networks from a specified directory.

        This method loads the parameters of actor networks, including the actor's state_dict, actor_target's state_dict, and
        the actor's optimizer state_dict, from the specified directory. It iterates through the learning strategies associated
        with the learning role, loads the respective parameters, and updates the actor and target actor networks accordingly.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        logger.info("Loading actor parameters...")
        if not os.path.exists(directory):
            logger.warning(
                "Specified directory for loading the actors does not exist! Starting with randomly initialized values!"
            )
            return

        for u_id in self.learning_role.rl_strats.keys():
            try:
                actor_params = self.load_obj(
                    directory=f"{directory}/actors/actor_{str(u_id)}.pt"
                )
                self.learning_role.rl_strats[u_id].actor.load_state_dict(
                    actor_params["actor"]
                )
                # self.learning_role.rl_strats[u_id].actor_target.load_state_dict(
                #     actor_params["actor_target"]
                # )
                self.learning_role.rl_strats[u_id].actor.optimizer.load_state_dict(
                    actor_params["actor_optimizer"]
                )
            except Exception:
                logger.warning(f"No actor values loaded for agent {u_id}")


    # Removed target_critics and actor_target in comparison to MATD3
    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Create actor and critic networks for reinforcement learning.

        If `actors_and_critics` is None, this method creates new actor and critic networks.
        If `actors_and_critics` is provided, it assigns existing networks to the respective attributes.

        Args:
            actors_and_critics (dict): The actor and critic networks to be assigned.

        """
        if actors_and_critics is None:
            self.create_actors()
            self.create_critics()

        else:
            self.learning_role.critics = actors_and_critics["critics"]
            # self.learning_role.target_critics = actors_and_critics["target_critics"]
            for u_id, unit_strategy in self.learning_role.rl_strats.items():
                unit_strategy.actor = actors_and_critics["actors"][u_id]
                # unit_strategy.actor_target = actors_and_critics["actor_targets"][u_id]

            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

    # Removed actor_target in comparison to MATD3
    def create_actors(self) -> None:
        """
        Create actor networks for reinforcement learning for each unit strategy.

        This method initializes actor networks and their corresponding target networks for each unit strategy.
        The actors are designed to map observations to action probabilities in a reinforcement learning setting.

        The created actor networks are associated with each unit strategy and stored as attributes.

        Notes:
            The observation dimension need to be the same, due to the centralized criic that all actors share.
            If you have units with different observation dimensions. They need to have different critics and hence learning roles.

        """

        obs_dim_list = []
        act_dim_list = []

        for _, unit_strategy in self.learning_role.rl_strats.items():
            unit_strategy.actor = ActorPPO(
                obs_dim=unit_strategy.obs_dim,
                act_dim=unit_strategy.act_dim,
                float_type=self.float_type,
            ).to(self.device)

            # unit_strategy.actor_target = Actor(
            #     obs_dim=unit_strategy.obs_dim,
            #     act_dim=unit_strategy.act_dim,
            #     float_type=self.float_type,
            # ).to(self.device)
            # unit_strategy.actor_target.load_state_dict(unit_strategy.actor.state_dict())
            # unit_strategy.actor_target.train(mode=False)

            unit_strategy.actor.optimizer = Adam(
                unit_strategy.actor.parameters(), lr=self.learning_rate
            )

            obs_dim_list.append(unit_strategy.obs_dim)
            act_dim_list.append(unit_strategy.act_dim)

        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                "All observation dimensions must be the same for all RL agents"
            )
        else:
            self.obs_dim = obs_dim_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError("All action dimensions must be the same for all RL agents")
        else:
            self.act_dim = act_dim_list[0]

    # Removed target_critics in comparison to MATD3
    # Changed initialization of CriticPPO compared to MATD3 
    def create_critics(self) -> None:
        """
        Create critic networks for reinforcement learning.

        This method initializes critic networks for each agent in the reinforcement learning setup.

        Notes:
            The observation dimension need to be the same, due to the centralized critic that all actors share.
            If you have units with different observation dimensions. They need to have different critics and hence learning roles.
        """
        n_agents = len(self.learning_role.rl_strats)
        strategy: LearningStrategy
        unique_obs_dim_list = []

        for u_id, strategy in self.learning_role.rl_strats.items():

            self.learning_role.critics[u_id] = CriticPPO(
                n_agents=n_agents,
                obs_dim=strategy.obs_dim,
                unique_obs_dim=strategy.unique_obs_dim,
                float_type=self.float_type,
            )

            self.learning_role.critics[u_id].optimizer = Adam(
                self.learning_role.critics[u_id].parameters(), lr=self.learning_rate
            )

            self.learning_role.target_critics[u_id].load_state_dict(
                self.learning_role.critics[u_id].state_dict()
            )
            self.learning_role.target_critics[u_id].train(mode=False)

            self.learning_role.critics[u_id] = self.learning_role.critics[u_id].to(
                self.device
            )
            self.learning_role.target_critics[u_id] = self.learning_role.target_critics[
                u_id
            ].to(self.device)

            unique_obs_dim_list.append(strategy.unique_obs_dim)

        # check if all unique_obs_dim are the same and raise an error if not
        # if they are all the same, set the unique_obs_dim attribute
        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                "All unique_obs_dim values must be the same for all RL agents"
            )
        else:
            self.unique_obs_dim = unique_obs_dim_list[0]

    def extract_policy(self) -> dict:
        """
        Extract actor and critic networks.

        This method extracts the actor and critic networks associated with each learning strategy and organizes them into a
        dictionary structure. The extracted networks include actors, actor_targets, critics, and target_critics. The resulting
        dictionary is typically used for saving and sharing these networks.

        Returns:
            dict: The extracted actor and critic networks.
        """
        actors = {}
        actor_targets = {}

        for u_id, unit_strategy in self.learning_role.rl_strats.items():
            actors[u_id] = unit_strategy.actor
            actor_targets[u_id] = unit_strategy.actor_target

        actors_and_critics = {
            "actors": actors,
            # "actor_targets": actor_targets,
            "critics": self.learning_role.critics,
            "target_critics": self.learning_role.target_critics,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
        }

        return actors_and_critics
    
    def update_policy(self):
        """
        Perform policy updates using PPO with the clipped objective.
        """
        logger.debug("Updating Policy")

        for epoch in range(self.epochs):
            # Sample a batch from the replay buffer
            transitions = self.learning_role.buffer.sample(self.batch_size)
            states, actions, log_probs_old, returns, advantages = (
                transitions.observations,
                transitions.actions,
                transitions.log_probs,
                transitions.returns,
                transitions.advantages,
            )

            # Update the policy (actor)
            log_probs_new, entropy = self.learning_role.actor.evaluate_actions(states, actions)

            # Calculate the ratio of new policy probability to old policy probability
            # This represents how much the new policy has changed compared to the old policy
            ratio = (log_probs_new - log_probs_old).exp()

            # Compute the surrogate loss without clipping
            # This is the raw loss term based on the advantage function
            surrogate1 = ratio * advantages

            # Apply the clipping function to the ratio to prevent large policy updates
            # The clipping function limits the ratio to be within the range [1 - clip_ratio, 1 + clip_ratio]
            # This prevents the policy from deviating too much from the old policy
            surrogate2 = th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

            # Calculate the final policy loss by taking the minimum of the unclipped and clipped surrogate losses
            # The idea is to prevent large changes in policy and ensure stability during training
            # The final policy loss is the negative mean of this minimum value
            policy_loss = -th.min(surrogate1, surrogate2).mean()

            surrogate1 = ratio * advantages
            surrogate2 = th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
            policy_loss = -th.min(surrogate1, surrogate2).mean()

            # Update the critic (value function)
            values = self.learning_role.critic(states).squeeze()
            value_loss = F.mse_loss(returns, values)

            # Total loss
            loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy.mean()

            # Optimize actor and critic
            self.learning_role.actor.optimizer.zero_grad()
            self.learning_role.critic.optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            th.nn.utils.clip_grad_norm_(self.learning_role.actor.parameters(), self.max_grad_norm)
            th.nn.utils.clip_grad_norm_(self.learning_role.critic.parameters(), self.max_grad_norm)

            self.learning_role.actor.optimizer.step()
            self.learning_role.critic.optimizer.step()


    # def update_policy(self):
    #     """
    #     Update the policy of the reinforcement learning agent using the Twin Delayed Deep Deterministic Policy Gradients (TD3) algorithm.

    #     Notes:
    #         This function performs the policy update step, which involves updating the actor (policy) and critic (Q-function) networks
    #         using TD3 algorithm. It iterates over the specified number of gradient steps and performs the following steps for each
    #         learning strategy:

    #         1. Sample a batch of transitions from the replay buffer.
    #         2. Calculate the next actions with added noise using the actor target network.
    #         3. Compute the target Q-values based on the next states, rewards, and the target critic network.
    #         4. Compute the critic loss as the mean squared error between current Q-values and target Q-values.
    #         5. Optimize the critic network by performing a gradient descent step.
    #         6. Optionally, update the actor network if the specified policy delay is reached.
    #         7. Apply Polyak averaging to update target networks.

    #         This function implements the TD3 algorithm's key step for policy improvement and exploration.
    #     """

    #     logger.debug("Updating Policy")
    #     n_rl_agents = len(self.learning_role.rl_strats.keys())
    #     for _ in range(self.gradient_steps):
    #         self.n_updates += 1
    #         i = 0

    #         for u_id in self.learning_role.rl_strats.keys():
    #             critic_target = self.learning_role.target_critics[u_id]
    #             critic = self.learning_role.critics[u_id]
    #             actor = self.learning_role.rl_strats[u_id].actor
    #             actor_target = self.learning_role.rl_strats[u_id].actor_target

    #             if i % 100 == 0:
    #                 transitions = self.learning_role.buffer.sample(self.batch_size)
    #                 states = transitions.observations
    #                 actions = transitions.actions
    #                 next_states = transitions.next_observations
    #                 rewards = transitions.rewards

    #                 with th.no_grad():
    #                     # Select action according to policy and add clipped noise
    #                     noise = actions.clone().data.normal_(
    #                         0, self.target_policy_noise
    #                     )
    #                     noise = noise.clamp(
    #                         -self.target_noise_clip, self.target_noise_clip
    #                     )
    #                     next_actions = [
    #                         (actor_target(next_states[:, i, :]) + noise[:, i, :]).clamp(
    #                             -1, 1
    #                         )
    #                         for i in range(n_rl_agents)
    #                     ]
    #                     next_actions = th.stack(next_actions)

    #                     next_actions = next_actions.transpose(0, 1).contiguous()
    #                     next_actions = next_actions.view(-1, n_rl_agents * self.act_dim)

    #             all_actions = actions.view(self.batch_size, -1)

    #             # this takes the unique observations from all other agents assuming that
    #             # the unique observations are at the end of the observation vector
    #             temp = th.cat(
    #                 (
    #                     states[:, :i, self.obs_dim - self.unique_obs_dim :].reshape(
    #                         self.batch_size, -1
    #                     ),
    #                     states[
    #                         :, i + 1 :, self.obs_dim - self.unique_obs_dim :
    #                     ].reshape(self.batch_size, -1),
    #                 ),
    #                 axis=1,
    #             )

    #             # the final all_states vector now contains the current agent's observation
    #             # and the unique observations from all other agents
    #             all_states = th.cat(
    #                 (states[:, i, :].reshape(self.batch_size, -1), temp), axis=1
    #             ).view(self.batch_size, -1)
    #             # all_states = states[:, i, :].reshape(self.batch_size, -1)

    #             # this is the same as above but for the next states
    #             temp = th.cat(
    #                 (
    #                     next_states[
    #                         :, :i, self.obs_dim - self.unique_obs_dim :
    #                     ].reshape(self.batch_size, -1),
    #                     next_states[
    #                         :, i + 1 :, self.obs_dim - self.unique_obs_dim :
    #                     ].reshape(self.batch_size, -1),
    #                 ),
    #                 axis=1,
    #             )

    #             # the final all_next_states vector now contains the current agent's observation
    #             # and the unique observations from all other agents
    #             all_next_states = th.cat(
    #                 (next_states[:, i, :].reshape(self.batch_size, -1), temp), axis=1
    #             ).view(self.batch_size, -1)
    #             # all_next_states = next_states[:, i, :].reshape(self.batch_size, -1)

    #             with th.no_grad():
    #                 # Compute the next Q-values: min over all critics targets
    #                 next_q_values = th.cat(
    #                     critic_target(all_next_states, next_actions), dim=1
    #                 )
    #                 next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
    #                 target_Q_values = (
    #                     rewards[:, i].unsqueeze(1) + self.gamma * next_q_values
    #                 )

    #             # Get current Q-values estimates for each critic network
    #             current_Q_values = critic(all_states, all_actions)

    #             # Compute critic loss
    #             critic_loss = sum(
    #                 F.mse_loss(current_q, target_Q_values)
    #                 for current_q in current_Q_values
    #             )

    #             # Optimize the critics
    #             critic.optimizer.zero_grad()
    #             critic_loss.backward()
    #             critic.optimizer.step()

    #             # Delayed policy updates
    #             if self.n_updates % self.policy_delay == 0:
    #                 # Compute actor loss
    #                 state_i = states[:, i, :]
    #                 action_i = actor(state_i)

    #                 all_actions_clone = actions.clone()
    #                 all_actions_clone[:, i, :] = action_i
    #                 all_actions_clone = all_actions_clone.view(self.batch_size, -1)

    #                 actor_loss = -critic.q1_forward(
    #                     all_states, all_actions_clone
    #                 ).mean()

    #                 actor.optimizer.zero_grad()
    #                 actor_loss.backward()
    #                 actor.optimizer.step()

    #                 polyak_update(
    #                     critic.parameters(), critic_target.parameters(), self.tau
    #                 )
    #                 polyak_update(
    #                     actor.parameters(), actor_target.parameters(), self.tau
    #                 )

    #             i += 1



#     def save_params(self, directory):
#         """ Save the parameters of the actor and critic networks """
#         self.save_actor_params(directory=f"{directory}/actors")
#         self.save_critic_params(directory=f"{directory}/critics")

#     def save_actor_params(self, directory):
#         """ Save actor parameters. """
#         os.makedirs(directory, exist_ok=True)
#         for u_id in self.learning_role.rl_strats.keys():
#             obj = {
#                 "actor": self.learning_role.rl_strats[u_id].actor.state_dict(),
#                 "actor_optimizer": self.learning_role.rl_strats[u_id].actor.optimizer.state_dict(),
#             }
#             path = f"{directory}/actor_{u_id}.pt"
#             th.save(obj, path)

#     def save_critic_params(self, directory):
#         """ Save critic parameters. """
#         os.makedirs(directory, exist_ok=True)
#         for u_id in self.learning_role.rl_strats.keys():
#             obj = {
#                 "critic": self.learning_role.critics[u_id].state_dict(),
#                 "critic_optimizer": self.learning_role.critics[u_id].optimizer.state_dict(),
#             }
#             path = f"{directory}/critic_{u_id}.pt"
#             th.save(obj, path)

#     def load_params(self, directory: str) -> None:
#         """ Load actor and critic parameters """
#         self.load_actor_params(directory)
#         self.load_critic_params(directory)

#     def load_actor_params(self, directory: str) -> None:
#         """ Load actor parameters from a directory """
#         if not os.path.exists(directory):
#             logger.warning("Actor directory does not exist! Initializing randomly.")
#             return

#         for u_id in self.learning_role.rl_strats.keys():
#             try:
#                 actor_params = self.load_obj(f"{directory}/actors/actor_{str(u_id)}.pt")
#                 self.learning_role.rl_strats[u_id].actor.load_state_dict(actor_params["actor"])
#                 self.learning_role.rl_strats[u_id].actor.optimizer.load_state_dict(actor_params["actor_optimizer"])
#             except Exception:
#                 logger.warning(f"No actor values loaded for agent {u_id}")

#     def load_critic_params(self, directory: str) -> None:
#         """ Load critic parameters from a directory """
#         if not os.path.exists(directory):
#             logger.warning("Critic directory does not exist! Initializing randomly.")
#             return

#         for u_id in self.learning_role.rl_strats.keys():
#             try:
#                 critic_params = self.load_obj(f"{directory}/critics/critic_{str(u_id)}.pt")
#                 self.learning_role.critics[u_id].load_state_dict(critic_params["critic"])
#                 self.learning_role.critics[u_id].optimizer.load_state_dict(critic_params["critic_optimizer"])
#             except Exception:
#                 logger.warning(f"No critic values loaded for agent {u_id}")
