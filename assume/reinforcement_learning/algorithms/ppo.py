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
from assume.reinforcement_learning.learning_utils import collect_obs_for_central_critic


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
        gradient_steps: int,  # Number of steps for updating the policy
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
        self.gradient_steps = gradient_steps
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lambda
        self.n_updates = 0  # Number of updates performed
        self.batch_size = learning_role.batch_size 

        # write error if different actor_architecture than dist is used
        if actor_architecture != "dist":
            raise ValueError(
                "PPO only supports the 'dist' actor architecture. Please define 'dist' as actor architecture in config."
            )

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
    
    def get_values(self, states, actions):
        """
        Gets values for a unit based on the observation using PPO.

        Args:
            rl_strategy (RLStrategy): The strategy containing relevant information.
            next_observation (torch.Tensor): The observation.

        Returns:
            torch.Tensor: The value of the observation.
        """
        #counter iterating over all agents for dynamic buffer slice
        i=0

        #get length of all states to pass it on as batch size, since the entire buffer is used for the PPO
        buffer_length = len(states)
        all_actions = actions.view(buffer_length, -1)
        # Initialize an empty tensor to store all values
        all_values = th.empty(0, buffer_length, 1)
         
        for u_id in self.learning_role.rl_strats.keys():

            all_states = collect_obs_for_central_critic(states, i, self.obs_dim, self.unique_obs_dim, buffer_length)
        
            critic = self.learning_role.critics[u_id]

            # Pass the current states through the critic network to get value estimates.
            values = critic(all_states, all_actions)

            if  all_values.numel() == 0:
                all_values = values
            else:
                all_values = th.cat((all_values, values), dim=1)

            i=i+1

        return all_values

    def get_advantages(self, rewards, values):

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages = []
        advantage = 0
        returns = []

        # Iterate through the collected experiences in reverse order to calculate advantages and returns
        for t in reversed(range(len(rewards))):
            
            logger.debug(f"Reward: {t}")    

            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            # Temporal difference delta Equation 12 from PPO paper
            delta = (
                - values[t] + rewards[t] + self.gamma * next_value 
            )  # Use self.gamma for discount factor

            logger.debug(f"Delta: {delta}")

            # GAE advantage Equation 11 from PPO paper
            advantage = (
                delta + self.gamma * self.gae_lambda * advantage
            )  # Use self.gae_lambda for advantage estimation

            logger.debug(f"Last_advantage: {advantage}")

            advantages.insert(0, advantage)
            returns.insert(0, advantage + values[t])
        
        # Convert advantages and returns to tensors
        advantages = th.tensor(advantages, dtype=th.float32, device=self.device)
        returns = th.tensor(returns, dtype=th.float32, device=self.device)

        #Normalize advantages
        #in accordance with spinning up and mappo version of PPO
        mean_advantages = th.nanmean(advantages)
        std_advantages = th.std(advantages)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        #TODO: Should we detach here? I though becaus of normalisation not being included in backward
        # but unsure if this is correct
        return advantages, returns


    def update_policy(self):
        """
        Perform policy updates using PPO with the clipped objective.
        """

        logger.debug("Updating Policy")
        # We will iterate for multiple epochs to update both the policy (actor) and value (critic) networks
        # The number of epochs controls how many times we update using the same collected data (from the buffer).


        # Retrieve experiences from the buffer
        # The collected experiences (observations, actions, rewards, log_probs) are stored in the buffer.
        full_transitions = self.learning_role.buffer.get()
        full_values = self.get_values(full_transitions.observations, full_transitions.actions)
        full_advantages, full_returns = self.get_advantages(full_transitions.rewards, full_values)
        


        for _ in range(self.gradient_steps):
            self.n_updates += 1

            transitions, batch_inds = self.learning_role.buffer.sample(self.batch_size)
            states = transitions.observations
            actions = transitions.actions
            log_probs = transitions.log_probs
            advantages = full_advantages[batch_inds]
            returns = full_returns[batch_inds]
            values = self.get_values(states, actions)  # always use updated values --> check later if best

            # Iterate through over each agent's strategy
            # Each agent has its own actor. Critic (value network) is centralized.
            for u_id in self.learning_role.rl_strats.keys():
                
                # Centralized
                critic = self.learning_role.critics[u_id]
                # Decentralized
                actor = self.learning_role.rl_strats[u_id].actor


                # Evaluate the new log-probabilities and entropy under the current policy
                action_distribution = actor(states)[1]
                new_log_probs = action_distribution.log_prob(actions).sum(-1)
                
                
                entropy = action_distribution.entropy().sum(-1)

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

                # Final policy loss (clipped surrogate loss) equation 7 from PPO paper
                policy_loss = th.min(surrogate1, surrogate2).mean()

                logger.debug(f"policy_loss: {policy_loss}")

                # Value loss (mean squared error between the predicted values and returns)
                value_loss = F.mse_loss(returns, values.squeeze())

                logger.debug(f"value loss: {value_loss}")

                # Total loss: policy loss + value loss - entropy bonus
                # euqation 9 from PPO paper multiplied with -1 to enable minimizing
                total_loss = (
                    - policy_loss
                    + self.vf_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )  # Use self.vf_coef and self.entropy_coef

                logger.debug(f"total loss: {total_loss}")

                # Zero the gradients and perform backpropagation for both actor and critic
                actor.optimizer.zero_grad()
                critic.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)

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
    learning_mode = rl_strategy.learning_mode
    perform_evaluation = rl_strategy.perform_evaluation

    # Pass observation through the actor network to get action logits (mean of action distribution)
    action_logits, action_distribution = actor(next_observation)
    action_logits = action_logits.detach()
    logger.debug(f"Action logits: {action_logits}")

    logger.debug(f"Action distribution: {action_distribution}")

    if learning_mode and not perform_evaluation:

        # Sample an action from the distribution
        sampled_action = action_distribution.sample().to(device)

    else:
        # If not in learning mode or during evaluation, use the mean of the action distribution
        sampled_action = action_logits.detach()

    logger.debug(f"Sampled action: {sampled_action}")

    # Get the log probability of the sampled actions (for later PPO loss calculation)
    # Sum the log probabilities across all action dimensions TODO: Why sum?
    log_prob_action = action_distribution.log_prob(sampled_action).sum(dim=-1)

    # Detach the log probability tensor to stop gradient tracking (since we only need the value for later)
    log_prob_action = log_prob_action.detach()

    logger.debug(f"Detached log probability of the sampled action: {log_prob_action}")


    return sampled_action, log_prob_action



