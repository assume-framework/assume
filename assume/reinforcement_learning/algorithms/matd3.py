# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import torch as th
from torch.nn import functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import Actor, CriticTD3, polyak_update


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

    def save_params(self, directory):
        """
        This method saves the parameters of both the actor and critic networks associated with the learning role. It organizes the
        saved parameters into separate directories for critics and actors within the specified base directory.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    def save_critic_params(self, directory):
        """
        Save the parameters of critic networks.

        This method saves the parameters of the critic networks, including the critic's state_dict, critic_target's state_dict,
        and the critic's optimizer state_dict. It organizes the saved parameters into a directory structure specific to the critic
        associated with each learning strategy.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        os.makedirs(directory, exist_ok=True)
        for u_id in self.learning_role.rl_strats.keys():
            obj = {
                "critic": self.learning_role.critics[u_id].state_dict(),
                "critic_target": self.learning_role.target_critics[u_id].state_dict(),
                "critic_optimizer": self.learning_role.critics[
                    u_id
                ].optimizer.state_dict(),
            }
            path = f"{directory}/critic_{u_id}.pt"
            th.save(obj, path)

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
                "actor_target": self.learning_role.rl_strats[
                    u_id
                ].actor_target.state_dict(),
                "actor_optimizer": self.learning_role.rl_strats[
                    u_id
                ].actor.optimizer.state_dict(),
            }
            path = f"{directory}/actor_{u_id}.pt"
            th.save(obj, path)

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
                self.learning_role.target_critics[u_id].load_state_dict(
                    critic_params["critic_target"]
                )
                self.learning_role.critics[u_id].optimizer.load_state_dict(
                    critic_params["critic_optimizer"]
                )
            except Exception:
                logger.warning(f"No critic values loaded for agent {u_id}")

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
                self.learning_role.rl_strats[u_id].actor_target.load_state_dict(
                    actor_params["actor_target"]
                )
                self.learning_role.rl_strats[u_id].actor.optimizer.load_state_dict(
                    actor_params["actor_optimizer"]
                )
            except Exception:
                logger.warning(f"No actor values loaded for agent {u_id}")

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
            self.learning_role.target_critics = actors_and_critics["target_critics"]
            for u_id, unit_strategy in self.learning_role.rl_strats.items():
                unit_strategy.actor = actors_and_critics["actors"][u_id]
                unit_strategy.actor_target = actors_and_critics["actor_targets"][u_id]

    def create_actors(self) -> None:
        """
        Create actor networks for reinforcement learning for each unit strategy.

        This method initializes actor networks and their corresponding target networks for each unit strategy.
        The actors are designed to map observations to action probabilities in a reinforcement learning setting.

        The created actor networks are associated with each unit strategy and stored as attributes.
        """
        for _, unit_strategy in self.learning_role.rl_strats.items():
            unit_strategy.actor = Actor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
            ).to(self.device)

            unit_strategy.actor_target = Actor(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
            ).to(self.device)
            unit_strategy.actor_target.load_state_dict(unit_strategy.actor.state_dict())
            unit_strategy.actor_target.train(mode=False)

            unit_strategy.actor.optimizer = Adam(
                unit_strategy.actor.parameters(), lr=self.learning_rate
            )

    def create_critics(self) -> None:
        """
        Create critic networks for reinforcement learning.

        This method initializes critic networks for each agent in the reinforcement learning setup.
        """
        n_agents = len(self.learning_role.rl_strats)
        strategy: LearningStrategy

        for u_id, strategy in self.learning_role.rl_strats.items():
            self.learning_role.critics[u_id] = CriticTD3(
                n_agents, strategy.obs_dim, strategy.act_dim, self.float_type
            )
            self.learning_role.target_critics[u_id] = CriticTD3(
                n_agents, strategy.obs_dim, strategy.act_dim, self.float_type
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
            "actor_targets": actor_targets,
            "critics": self.learning_role.critics,
            "target_critics": self.learning_role.target_critics,
        }

        return actors_and_critics

    def update_policy(self):
        """
        Update the policy of the reinforcement learning agent using the Twin Delayed Deep Deterministic Policy Gradients (TD3) algorithm.

        Notes:

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
