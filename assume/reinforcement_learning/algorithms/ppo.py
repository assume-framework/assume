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
from assume.reinforcement_learning.neural_network_architecture import CriticPPO

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
        learning_rate: float,
        gamma: float,  # Discount factor for future rewards
        epochs: int,  # Number of epochs for updating the policy
        clip_ratio: float,  # Clipping parameter for policy updates
        vf_coef: float,  # Value function coefficient in the loss function
        entropy_coef: float,  # Entropy coefficient for exploration
        max_grad_norm: float,  # Gradient clipping value
        gae_lambda: float,  # GAE lambda for advantage estimation
        actor_architecture: str,
    ):
        super().__init__(
            learning_role=learning_role,
            learning_rate=learning_rate,
            gamma=gamma,
            actor_architecture=actor_architecture,
        )
        self.epochs = epochs
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.n_updates = 0  # Number of updates performed

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
    # Decentralized
    # def save_critic_params(self, directory):
    #     """
    #     Save the parameters of critic networks.

    #     This method saves the parameters of the critic networks, including the critic's state_dict and the critic's optimizer state_dict. 
    #     It organizes the saved parameters into a directory structure specific to the critic associated with each learning strategy.

    #     Args:
    #         directory (str): The base directory for saving the parameters.
    #     """
    #     os.makedirs(directory, exist_ok=True)
    #     for u_id in self.learning_role.rl_strats.keys():
    #         obj = {
    #             "critic": self.learning_role.rl_strats[u_id].critic.state_dict(),
    #             "critic_optimizer": self.learning_role.rl_strats[u_id].critic.optimizer.state_dict(),
    #         }
    #         path = f"{directory}/critic_{u_id}.pt"
    #         th.save(obj, path)


    # Centralized
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

    # Removed actor_target in comparison to MATD3
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
    # Decentralized
    # def load_critic_params(self, directory: str) -> None:
    #     """
    #     Load the parameters of critic networks from a specified directory.

    #     This method loads the parameters of critic networks, including the critic's state_dict and
    #     the critic's optimizer state_dict, from the specified directory. It iterates through the learning strategies associated
    #     with the learning role, loads the respective parameters, and updates the critic networks accordingly.

    #     Args:
    #         directory (str): The directory from which the parameters should be loaded.
    #     """
    #     logger.info("Loading critic parameters...")

    #     if not os.path.exists(directory):
    #         logger.warning(
    #             "Specified directory for loading the critics does not exist! Starting with randomly initialized values!"
    #         )
    #         return

    #     for u_id in self.learning_role.rl_strats.keys():
    #         try:
    #             critic_params = self.load_obj(
    #                 directory=f"{directory}/critics/critic_{str(u_id)}.pt"
    #             )
    #             self.learning_role.rl_strats[u_id].critic.load_state_dict(
    #                 critic_params["critic"]
    #             )
    #             self.learning_role.rl_strats[u_id].critic.optimizer.load_state_dict(
    #                 critic_params["critic_optimizer"]
    #             )
    #         except Exception:
    #             logger.warning(f"No critic values loaded for agent {u_id}")


    # Centralized
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
    # Decentralized
    # def initialize_policy(self, actors_and_critics: dict = None) -> None:
    #     """
    #     Create actor and critic networks for reinforcement learning.

    #     If `actors_and_critics` is None, this method creates new actor and critic networks.
    #     If `actors_and_critics` is provided, it assigns existing networks to the respective attributes.

    #     Args:
    #         actors_and_critics (dict): The actor and critic networks to be assigned.
    #     """
    #     if actors_and_critics is None:
    #         self.create_actors()
    #         self.create_critics()
    #     else:
    #         # Decentralized initialization of actors and critics
    #         for u_id, unit_strategy in self.learning_role.rl_strats.items():
    #             unit_strategy.actor = actors_and_critics["actors"][u_id]
    #             # unit_strategy.actor_target = actors_and_critics["actor_targets"][u_id]
    #             unit_strategy.critic = actors_and_critics["critics"][u_id]
    #             # unit_strategy.critic_target = actors_and_critics["critic_targets"][u_id]

    #         # Assign shared dimensions
    #         self.obs_dim = actors_and_critics["obs_dim"]
    #         self.act_dim = actors_and_critics["act_dim"]
    #         self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

    # Centralized
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
            unit_strategy.actor = self.actor_architecture_class(
                obs_dim=unit_strategy.obs_dim,
                act_dim=unit_strategy.act_dim,
                float_type=self.float_type,
                unique_obs_dim=unit_strategy.unique_obs_dim,
                num_timeseries_obs_dim=unit_strategy.num_timeseries_obs_dim,
            ).to(self.device)

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
    # Decentralized
    # def create_critics(self) -> None:
    #     """
    #     Create decentralized critic networks for reinforcement learning.

    #     This method initializes a separate critic network for each agent in the reinforcement learning setup.
    #     Each critic learns to predict the value function based on the individual agent's observation.

    #     Notes:
    #         Each agent has its own critic, so the critic is no longer shared among all agents.
    #     """

    #     unique_obs_dim_list = []

    #     for _, unit_strategy in self.learning_role.rl_strats.items():
    #         unit_strategy.critic = CriticPPO(
    #             obs_dim=unit_strategy.obs_dim,
    #             float_type=self.float_type,
    #         ).to(self.device)

    #         unit_strategy.critic.optimizer = Adam(
    #             unit_strategy.critic.parameters(), lr=self.learning_rate
    #         )

    #         unique_obs_dim_list.append(unit_strategy.unique_obs_dim)

    #     # Check if all unique_obs_dim are the same and raise an error if not
    #     # If they are all the same, set the unique_obs_dim attribute
    #     if len(set(unique_obs_dim_list)) > 1:
    #         raise ValueError(
    #             "All unique_obs_dim values must be the same for all RL agents"
    #         )
    #     else:
    #         self.unique_obs_dim = unique_obs_dim_list[0]



    # Centralized
    def create_critics(self) -> None:
        """
        Create decentralized critic networks for reinforcement learning.

        This method initializes a separate critic network for each agent in the reinforcement learning setup.
        Each critic learns to predict the value function based on the individual agent's observation.

        Notes:
            Each agent has its own critic, so the critic is no longer shared among all agents.
        """

        n_agents = len(self.learning_role.rl_strats)
        strategy: LearningStrategy
        unique_obs_dim_list = []

        for u_id, strategy in self.learning_role.rl_strats.items():
            self.learning_role.critics[u_id] = CriticPPO(
                n_agents=n_agents,
                obs_dim=strategy.obs_dim,
                act_dim=strategy.act_dim,
                unique_obs_dim=strategy.unique_obs_dim,
                float_type=self.float_type,
            )

            self.learning_role.critics[u_id].optimizer = Adam(
                self.learning_role.critics[u_id].parameters(), lr=self.learning_rate
            )

            self.learning_role.critics[u_id] = self.learning_role.critics[u_id].to(
                self.device
            )

            unique_obs_dim_list.append(strategy.unique_obs_dim)

        # check if all unique_obs_dim are the same and raise an error if not
        # if they are all the same, set the unique_obs_dim attribute
        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                "All unique_obs_dim values must be the same for all RL agents"
            )
        else:
            self.unique_obs_dim = unique_obs_dim_list[0]

    # Decentralized
    # def extract_policy(self) -> dict:
    #     """
    #     Extract actor and critic networks.

    #     This method extracts the actor and critic networks associated with each learning strategy and organizes them into a
    #     dictionary structure. The extracted networks include actors and critics. The resulting
    #     dictionary is typically used for saving and sharing these networks.

    #     Returns:
    #         dict: The extracted actor and critic networks.
    #     """
    #     actors = {}
    #     critics = {}

    #     for u_id, unit_strategy in self.learning_role.rl_strats.items():
    #         actors[u_id] = unit_strategy.actor
    #         critics[u_id] = unit_strategy.critic

    #     actors_and_critics = {
    #         "actors": actors,
    #         "critics": critics,
    #         "obs_dim": self.obs_dim,
    #         "act_dim": self.act_dim,
    #         "unique_obs_dim": self.unique_obs_dim,
    #     }

    #     return actors_and_critics

    # Centralized
    def extract_policy(self) -> dict:
        """
        Extract actor and critic networks.

        This method extracts the actor and critic networks associated with each learning strategy and organizes them into a
        dictionary structure. The extracted networks include actors, and critics. The resulting
        dictionary is typically used for saving and sharing these networks.

        Returns:
            dict: The extracted actor and critic networks.
        """
        actors = {}

        for u_id, unit_strategy in self.learning_role.rl_strats.items():
            actors[u_id] = unit_strategy.actor

        actors_and_critics = {
            "actors": actors,
            "critics": self.learning_role.critics,
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
        # We will iterate for multiple epochs to update both the policy (actor) and value (critic) networks
        # The number of epochs controls how many times we update using the same collected data (from the buffer).

        for _ in range(self.epochs):
            self.n_updates += 1

            # Iterate through over each agent's strategy
            # Each agent has its own actor. Critic (value network) is centralized.
            for u_id in self.learning_role.rl_strats.keys():
                
                # Centralized
                critic = self.learning_role.critics[u_id]
                # Decentralized
                actor = self.learning_role.rl_strats[u_id].actor

                # Retrieve experiences from the buffer
                # The collected experiences (observations, actions, rewards, log_probs) are stored in the buffer.
                transitions = self.learning_role.buffer.get()
                states = transitions.observations
                actions = transitions.actions
                rewards = transitions.rewards
                log_probs = transitions.log_probs

                # STARTING FROM HERE, THE IMPLEMENTATION NEEDS TO BE FIXED
                # Potentially, it could be useful to source some functionality out into methods stored in buffer.py

                # Pass the current states through the critic network to get value estimates.
                values = critic(states, actions).squeeze(dim=2)

                logger.debug(f"Values: {values}")

                # Store the calculated values in the rollout buffer
                # These values are used later to calculate the advantage estimates (for policy updates).
                self.learning_role.buffer.values = values.detach().cpu().numpy()

                # Compute advantages using Generalized Advantage Estimation (GAE)
                advantages = []
                last_advantage = 0
                returns = []

                # Iterate through the collected experiences in reverse order to calculate advantages and returns
                for t in reversed(range(len(rewards))):
                    
                    logger.debug(f"Reward: {t}")    

                    if t == len(rewards) - 1:
                        next_value = 0
                    else:
                        next_value = values[t + 1]

                    # Temporal difference delta
                    delta = (
                        rewards[t] + self.gamma * next_value - values[t]
                    )  # Use self.gamma for discount factor

                    logger.debug(f"Delta: {delta}")

                    # GAE advantage
                    last_advantage = (
                        delta + self.gamma * self.gae_lambda * last_advantage
                    )  # Use self.gae_lambda for advantage estimation

                    logger.debug(f"Last_advantage: {last_advantage}")

                    advantages.insert(0, last_advantage)
                    returns.insert(0, last_advantage + values[t])

                # Convert advantages and returns to tensors
                advantages = th.tensor(advantages, dtype=th.float32, device=self.device)
                returns = th.tensor(returns, dtype=th.float32, device=self.device)

                # Evaluate the new log-probabilities and entropy under the current policy
                action_means = actor(states)
                action_stddev = th.ones_like(
                    action_means
                )  # Assuming fixed standard deviation for simplicity
                dist = th.distributions.Normal(action_means, action_stddev)
                new_log_probs = dist.log_prob(actions).sum(-1)
                entropy = dist.entropy().sum(-1)

                # Compute the ratio of new policy to old policy
                ratio = (new_log_probs - log_probs).exp()

                logger.debug(f"Ratio: {ratio}")

                # Surrogate loss calculation
                surrogate1 = ratio * advantages
                surrogate2 = (
                    th.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                    * advantages
                )  # Use self.clip_ratio

                logger.debug(f"surrogate1: {surrogate1}")
                logger.debug(f"surrogate2: {surrogate2}")

                # Final policy loss (clipped surrogate loss)
                policy_loss = -th.min(surrogate1, surrogate2).mean()

                logger.debug(f"policy_loss: {policy_loss}")

                # Value loss (mean squared error between the predicted values and returns)
                value_loss = F.mse_loss(returns, values.squeeze())

                logger.debug(f"value loss: {value_loss}")

                # Total loss: policy loss + value loss - entropy bonus
                total_loss = (
                    policy_loss
                    + self.vf_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )  # Use self.vf_coef and self.entropy_coef

                logger.debug(f"total loss: {total_loss}")

                # Zero the gradients and perform backpropagation for both actor and critic
                actor.optimizer.zero_grad()
                critic.optimizer.zero_grad()
                total_loss.backward()

                # Clip gradients to prevent gradient explosion
                th.nn.utils.clip_grad_norm_(
                    actor.parameters(), self.max_grad_norm
                )  # Use self.max_grad_norm
                th.nn.utils.clip_grad_norm_(
                    critic.parameters(), self.max_grad_norm
                )  # Use self.max_grad_norm

                # Perform optimization steps
                actor.optimizer.step()
                critic.optimizer.step()


def get_actions(rl_strategy, next_observation):
    """
    Gets actions for a unit based on the observation using PPO.

    Args:
        rl_strategy (RLStrategy): The strategy containing relevant information.
        next_observation (torch.Tensor): The observation.

    Returns:
        torch.Tensor: The sampled actions.
        torch.Tensor: The log probability of the sampled actions.
    """
    logger.debug("ppo.py: Get_actions method")

    actor = rl_strategy.actor
    device = rl_strategy.device

    # Pass observation through the actor network to get action logits (mean of action distribution)
    action_logits = actor(next_observation.to(device))

    logger.debug(f"Action logits: {action_logits}")

    # Create a normal distribution for continuous actions (with assumed standard deviation of 1.0)
    action_distribution = th.distributions.Normal(action_logits, 1.0)

    logger.debug(f"Action distribution: {action_distribution}")

    # Sample an action from the distribution
    sampled_action = action_distribution.sample()

    logger.debug(f"Sampled action: {sampled_action}")

    # Get the log probability of the sampled action (for later PPO loss calculation)
    log_prob_action = action_distribution.log_prob(sampled_action).sum(dim=-1)

    # Detach the log probability tensor to stop gradient tracking (since we only need the value for later)
    log_prob_action = log_prob_action.detach()

    logger.debug(f"Detached log probability of the sampled action: {log_prob_action}")

    # PREVIOUSLY SET TO (-1, 1)
    # Bound actions to [0, 1] range
    sampled_action = sampled_action.clamp(0, 1)

    logger.debug(f"Clamped sampled action: {sampled_action}")

    return sampled_action, log_prob_action
