# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging
import os
from collections import defaultdict

import numpy as np
import torch as th
from sklearn.cluster import KMeans
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import polyak_update
from assume.reinforcement_learning.neural_network_architecture import CriticTD3

logger = logging.getLogger(__name__)


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
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        gradient_steps=100,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        actor_architecture="mlp",
        actor_clustering_method=None,
        clustering_method_kwargs={},
    ):
        super().__init__(
            learning_role,
            learning_rate,
            batch_size,
            tau,
            gamma,
            gradient_steps,
            policy_delay,
            target_policy_noise,
            target_noise_clip,
            actor_architecture,
        )
        self.n_updates = 0
        self.grad_clip_norm = 1.0

        self.use_shared_actor = (
            True if actor_architecture == "contextual_mlp" else False
        )
        if self.use_shared_actor:
            self.actor_clustering_method = actor_clustering_method
            self.clustering_method_kwargs = clustering_method_kwargs

            self.shared_actors = {}
            self.shared_actor_targets = {}
            self.clusters = defaultdict(list)

    def save_params(self, directory):
        """
        This method saves the parameters of both the actor and critic networks associated with the learning role. It organizes the
        saved parameters into separate directories for critics and actors within the specified base directory.

        Args:
            directory (str): The base directory for saving the parameters.
        """
        self.save_critic_params(directory=f"{directory}/critics")

        if self.use_shared_actor:
            self.save_shared_actor_params(directory=f"{directory}/actors")
        else:
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
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critics.state_dict(),
                "critic_target": strategy.target_critics.state_dict(),
                "critic_optimizer": strategy.critics.optimizer.state_dict(),
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
        # save individual actors for each agent
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "actor": strategy.actor.state_dict(),
                "actor_target": strategy.actor_target.state_dict(),
                "actor_optimizer": strategy.actor.optimizer.state_dict(),
            }
            path = f"{directory}/actor_{u_id}.pt"
            th.save(obj, path)

    def save_shared_actor_params(self, directory):
        os.makedirs(directory, exist_ok=True)

        # iterate over the shared_actors and save them
        for cluster_index in self.clusters.keys():
            obj = {
                "actor": self.shared_actors[cluster_index].state_dict(),
                "actor_target": self.shared_actor_targets[cluster_index].state_dict(),
                "actor_optimizer": self.shared_actors[
                    cluster_index
                ].optimizer.state_dict(),
            }
            path = f"{directory}/shared_actor_{cluster_index}.pt"
            th.save(obj, path)

        # also store the cluster mapping
        path = f"{directory}/cluster_mapping.json"
        with open(path, "w") as f:
            json.dump(self.cluster_mapping, f)

    def load_params(self, directory: str) -> None:
        """
        Load the parameters of both actor and critic networks.

        This method loads the parameters of both the actor and critic networks associated with the learning role from the specified
        directory. It uses the `load_critic_params` and `load_actor_params` methods to load the respective parameters.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        self.load_critic_params(directory)

        if self.use_shared_actor:
            self.load_shared_actor_params(directory)
        else:
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

        for u_id, strategy in self.learning_role.rl_strats.items():
            try:
                critic_params = self.load_obj(
                    directory=f"{directory}/critics/critic_{str(u_id)}.pt"
                )
                strategy.critics.load_state_dict(critic_params["critic"])
                strategy.target_critics.load_state_dict(critic_params["critic_target"])
                strategy.critics.optimizer.load_state_dict(
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

        for u_id, strategy in self.learning_role.rl_strats.items():
            try:
                actor_params = self.load_obj(
                    directory=f"{directory}/actors/actor_{str(u_id)}.pt"
                )
                strategy.actor.load_state_dict(actor_params["actor"])
                strategy.actor_target.load_state_dict(actor_params["actor_target"])
                strategy.actor.optimizer.load_state_dict(
                    actor_params["actor_optimizer"]
                )
            except Exception:
                logger.warning(f"No actor values loaded for agent {u_id}")

    def load_shared_actor_params(self, directory: str) -> None:
        """
        Load the parameters of shared actor networks from a specified directory.

        This method loads the parameters of shared actor networks, including the shared actor's state_dict, shared actor_target's
        state_dict, and the shared actor's optimizer state_dict, from the specified directory. It updates the shared actor and
        target actor networks accordingly.

        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        logger.info("Loading shared actor parameters...")
        if not os.path.exists(directory):
            logger.warning(
                "Specified directory for loading the actors does not exist! Starting with randomly initialized values!"
            )
            return

        try:
            for cluster_index in self.clusters.keys():
                actor_params = self.load_obj(
                    directory=f"{directory}/actors/shared_actor_{cluster_index}.pt"
                )
                self.shared_actors[cluster_index].load_state_dict(actor_params["actor"])
                self.shared_actor_targets[cluster_index].load_state_dict(
                    actor_params["actor_target"]
                )
                self.shared_actors[cluster_index].optimizer.load_state_dict(
                    actor_params["actor_optimizer"]
                )

        except Exception:
            logger.warning("No shared actor values loaded")

    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Create actor and critic networks for reinforcement learning.

        If `actors_and_critics` is None, this method creates new actor and critic networks.
        If `actors_and_critics` is provided, it assigns existing networks to the respective attributes.

        Args:
            actors_and_critics (dict): The actor and critic networks to be assigned.

        """
        if actors_and_critics is None:
            self.check_policy_dimensions()

            if self.use_shared_actor:
                self.create_shared_actors()
            else:
                self.create_individual_actors()

            self.create_critics()

        else:
            if self.use_shared_actor:
                self.cluster_mapping = actors_and_critics["cluster_mapping"]

                for strategy in self.learning_role.rl_strats.values():
                    cluster_index = self.cluster_mapping[strategy.unit_id]
                    self.clusters[cluster_index].append(strategy)

                for cluster_index, strategies in self.clusters.items():
                    self.shared_actors[cluster_index] = actors_and_critics["actors"][
                        cluster_index
                    ]
                    self.shared_actor_targets[cluster_index] = actors_and_critics[
                        "actor_targets"
                    ][cluster_index]

                    for strategy in strategies:
                        strategy.actor = self.shared_actors[cluster_index]
                        strategy.actor_target = self.shared_actor_targets[cluster_index]

            else:
                for u_id, strategy in self.learning_role.rl_strats.items():
                    strategy.actor = actors_and_critics["actors"][u_id]
                    strategy.actor_target = actors_and_critics["actor_targets"][u_id]

            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.critics = actors_and_critics["critics"][u_id]
                strategy.target_critics = actors_and_critics["target_critics"][u_id]

            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

            # check for consistency and raise error if not
            if not self.verify_actor_sharing():
                raise ValueError(
                    "Actor sharing is not consistent across strategies in the same cluster."
                )

    def check_policy_dimensions(self) -> None:
        obs_dim_list = []
        context_dim_list = []
        act_dim_list = []
        unique_obs_dim_list = []
        num_timeseries_obs_dim_list = []

        for strategy in self.learning_role.rl_strats.values():
            obs_dim_list.append(strategy.obs_dim)
            context_dim_list.append(strategy.context_dim)
            act_dim_list.append(strategy.act_dim)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
            num_timeseries_obs_dim_list.append(strategy.num_timeseries_obs_dim)

        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                "All observation dimensions must be the same for all RL agents"
            )
        else:
            self.obs_dim = obs_dim_list[0]

        if len(set(context_dim_list)) > 1:
            raise ValueError(
                "All context dimensions must be the same for all RL agents"
            )
        else:
            self.context_dim = context_dim_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError("All action dimensions must be the same for all RL agents")
        else:
            self.act_dim = act_dim_list[0]

        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                "All unique_obs_dim values must be the same for all RL agents"
            )
        else:
            self.unique_obs_dim = unique_obs_dim_list[0]

        if len(set(num_timeseries_obs_dim_list)) > 1:
            raise ValueError(
                "All num_timeseries_obs_dim values must be the same for all RL agents"
            )
        else:
            self.num_timeseries_obs_dim = num_timeseries_obs_dim_list[0]

    def create_individual_actors(self) -> None:
        """
        Create actor networks for reinforcement learning for each unit strategy.

        This method initializes actor networks and their corresponding target networks for each unit strategy.
        The actors are designed to map observations to action probabilities in a reinforcement learning setting.

        The created actor networks are associated with each unit strategy and stored as attributes.

        Notes:
            The observation dimension need to be the same, due to the centralized criic that all actors share.
            If you have units with different observation dimensions. They need to have different critics and hence learning roles.

        """

        for strategy in self.learning_role.rl_strats.values():
            strategy.actor = self.actor_architecture_class(
                obs_dim=strategy.obs_dim,
                act_dim=strategy.act_dim,
                float_type=self.float_type,
                unique_obs_dim=strategy.unique_obs_dim,
                num_timeseries_obs_dim=strategy.num_timeseries_obs_dim,
            ).to(self.device)

            strategy.actor_target = self.actor_architecture_class(
                obs_dim=strategy.obs_dim,
                act_dim=strategy.act_dim,
                float_type=self.float_type,
                unique_obs_dim=strategy.unique_obs_dim,
                num_timeseries_obs_dim=strategy.num_timeseries_obs_dim,
            ).to(self.device)

            strategy.actor_target.load_state_dict(strategy.actor.state_dict())
            strategy.actor_target.train(mode=False)

            strategy.actor.optimizer = AdamW(
                strategy.actor.parameters(),
                lr=self.learning_role.calc_lr_from_progress(
                    1
                ),  # 1=100% of simulation remaining, uses learning_rate from config as starting point
            )

    def create_shared_actors(self):
        if self.actor_clustering_method:
            self.clusters = self.create_clusters()
        else:
            # If no clustering method is specified, use a single cluster for all strategies
            self.clusters = {0: list(self.learning_role.rl_strats.values())}

        # make a mapping from the cluster index to the unit IDs
        self.cluster_mapping = {}
        for cluster_index, strategies in self.clusters.items():
            for strategy in strategies:
                self.cluster_mapping[strategy.unit_id] = cluster_index

        self.shared_actors = {}
        self.shared_actor_targets = {}

        for cluster_index, strategies in self.clusters.items():
            shared_actor = self.actor_architecture_class(
                obs_dim=self.obs_dim,
                context_dim=self.context_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
            ).to(self.device)

            shared_actor_target = self.actor_architecture_class(
                obs_dim=self.obs_dim,
                context_dim=self.context_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
            ).to(self.device)

            shared_actor_target.load_state_dict(shared_actor.state_dict())
            shared_actor_target.train(mode=False)

            shared_actor.optimizer = AdamW(
                shared_actor.parameters(),
                lr=self.learning_role.calc_lr_from_progress(1),
            )

            for strategy in strategies:
                strategy.actor = shared_actor
                strategy.actor_target = shared_actor_target

            self.shared_actors[cluster_index] = shared_actor
            self.shared_actor_targets[cluster_index] = shared_actor_target

    def create_critics(self) -> None:
        """
        Create critic networks for reinforcement learning.

        This method initializes critic networks for each agent in the reinforcement learning setup.

        Notes:
            The observation dimension need to be the same, due to the centralized criic that all actors share.
            If you have units with different observation dimensions. They need to have different critics and hence learning roles.
        """
        n_agents = len(self.learning_role.rl_strats)

        for strategy in self.learning_role.rl_strats.values():
            strategy.critics = CriticTD3(
                n_agents=n_agents,
                obs_dim=strategy.obs_dim,
                act_dim=strategy.act_dim,
                unique_obs_dim=strategy.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            strategy.target_critics = CriticTD3(
                n_agents=n_agents,
                obs_dim=strategy.obs_dim,
                act_dim=strategy.act_dim,
                unique_obs_dim=strategy.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            strategy.target_critics.load_state_dict(strategy.critics.state_dict())
            strategy.target_critics.train(mode=False)

            strategy.critics.optimizer = AdamW(
                strategy.critics.parameters(),
                lr=self.learning_role.calc_lr_from_progress(
                    1
                ),  # 1 = 100% of simulation remaining, uses learning_rate from config as starting point
            )

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

        critics = {}
        target_critics = {}

        if self.use_shared_actor:
            actors = self.shared_actors
            actor_targets = self.shared_actor_targets

        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                actors[u_id] = strategy.actor
                actor_targets[u_id] = strategy.actor_target

        for u_id, strategy in self.learning_role.rl_strats.items():
            critics[u_id] = strategy.critics
            target_critics[u_id] = strategy.target_critics

        actors_and_critics = {
            "actors": actors,
            "actor_targets": actor_targets,
            "critics": critics,
            "target_critics": target_critics,
            "cluster_mapping": self.cluster_mapping,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "unique_obs_dim": self.unique_obs_dim,
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
            6. Update the actor network if the specified policy delay is reached.
            7. Apply Polyak averaging to update target networks.

        """

        logger.debug("Updating Policy")

        # Stack strategies for easier access
        strategies = list(self.learning_role.rl_strats.values())
        n_rl_agents = len(strategies)

        unit_params = [
            {
                u_id: {"loss": None, "total_grad_norm": None, "max_grad_norm": None}
                for u_id in self.learning_role.rl_strats.keys()
            }
            for _ in range(self.gradient_steps)
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
                ],
                learning_rate=learning_rate,
            )
            strategy.action_noise.update_noise_decay(updated_noise_decay)

        if self.use_shared_actor:
            self.update_learning_rate(
                [actor.optimizer for actor in self.shared_actors.values()],
                learning_rate=learning_rate,
            )

        else:
            for strategy in strategies:
                self.update_learning_rate(
                    [
                        strategy.actor.optimizer,
                    ],
                    learning_rate=learning_rate,
                )

        for step in range(self.gradient_steps):
            self.n_updates += 1

            transitions = self.learning_role.buffer.sample(self.batch_size)
            states, actions, next_states, rewards = (
                transitions.observations,
                transitions.actions,
                transitions.next_observations,
                transitions.rewards,
            )

            # Get next actions with added noise
            with th.no_grad():
                # Compute noise for the target actions (shape: [batch_size, n_rl_agents, act_dim])
                noise = th.randn_like(actions) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

                if self.use_shared_actor:
                    # Use context if self.use_shared_actor is True
                    contexts = th.stack([strategy.context for strategy in strategies])
                    next_actions = th.stack(
                        [
                            strategy.actor_target(next_states[:, i, :], contexts[i])
                            + noise[:, i, :]
                            for i, strategy in enumerate(strategies)
                        ]
                    ).clamp(-1, 1)
                else:
                    # Without context
                    next_actions = th.stack(
                        [
                            strategy.actor_target(next_states[:, i, :]) + noise[:, i, :]
                            for i, strategy in enumerate(strategies)
                        ]
                    ).clamp(-1, 1)

                # Reformat tensor as per original intention
                next_actions = next_actions.transpose(0, 1).contiguous()
                next_actions = next_actions.reshape(-1, len(strategies) * self.act_dim)

            all_actions = actions.view(self.batch_size, -1)

            # Precompute unique observation parts for all agents
            unique_obs_from_others = states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.batch_size, n_rl_agents, -1)
            next_unique_obs_from_others = next_states[
                :, :, self.obs_dim - self.unique_obs_dim :
            ].reshape(self.batch_size, n_rl_agents, -1)

            #####################################################################
            # CRITIC UPDATE: Accumulate losses for all agents, then backprop once
            #####################################################################

            # Zero-grad for all critics before accumulation
            for strategy in strategies:
                strategy.critics.optimizer.zero_grad(set_to_none=True)

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
                        states[:, i, :].reshape(self.batch_size, -1),
                        other_unique_obs.reshape(self.batch_size, -1),
                    ),
                    dim=1,
                )
                all_next_states = th.cat(
                    (
                        next_states[:, i, :].reshape(self.batch_size, -1),
                        other_next_unique_obs.reshape(self.batch_size, -1),
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
                        rewards[:, i].unsqueeze(1) + self.gamma * next_q_values
                    )

                # Get current Q-values estimates for each critic network
                current_Q_values = critic(all_states, all_actions)

                # Accumulate critic loss for this agent
                critic_loss = sum(
                    F.mse_loss(current_q, target_Q_values)
                    for current_q in current_Q_values
                )

                # Store the critic loss for this unit ID
                unit_params[step][strategy.unit_id]["loss"] = critic_loss.item()
                critic_loss.backward()

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
                unit_params[step][strategy.unit_id]["total_grad_norm"] = total_norm
                unit_params[step][strategy.unit_id]["max_grad_norm"] = max_grad_norm

            ######################################################################
            # ACTOR UPDATE (DELAYED): Accumulate losses for all agents in one pass
            ######################################################################
            if self.n_updates % self.policy_delay == 0:
                # Zero-grad for all actors first
                if self.use_shared_actor:
                    for actor in self.shared_actors.values():
                        actor.optimizer.zero_grad(set_to_none=True)
                else:
                    for strategy in strategies:
                        strategy.actor.optimizer.zero_grad(set_to_none=True)

                total_actor_loss = 0.0

                # We'll compute each agent's actor loss, accumulate, then do one backprop
                for i, strategy in enumerate(strategies):
                    actor = strategy.actor
                    critic = strategy.critics

                    # Build local state for actor i
                    state_i = states[:, i, :]
                    if self.use_shared_actor:
                        action_i = actor(state_i, strategy.context)
                    else:
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
                            state_i.reshape(self.batch_size, -1),
                            other_unique_obs.reshape(self.batch_size, -1),
                        ),
                        dim=1,
                    )

                    # Replace the i-th agent's action in the batch
                    all_actions_clone = actions.clone().detach()
                    all_actions_clone[:, i, :] = action_i

                    # Flatten again for the critic
                    all_actions_clone = all_actions_clone.view(self.batch_size, -1)

                    # Calculate actor loss (negative Q1 of the updated action)
                    actor_loss = -critic.q1_forward(
                        all_states_i, all_actions_clone
                    ).mean()

                    # Accumulate actor losses
                    total_actor_loss += actor_loss

                # Single backward pass for all actors
                total_actor_loss.backward()

                # Clip and step each actor optimizer
                if self.use_shared_actor:
                    for shared_actor in self.shared_actors.values():
                        th.nn.utils.clip_grad_norm_(
                            shared_actor.parameters(), max_norm=self.grad_clip_norm
                        )
                        shared_actor.optimizer.step()
                else:
                    for strategy in strategies:
                        th.nn.utils.clip_grad_norm_(
                            strategy.actor.parameters(), max_norm=self.grad_clip_norm
                        )
                        strategy.actor.optimizer.step()

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

                    # Only add individual actor params if not shared
                    if not self.use_shared_actor:
                        all_actor_params.extend(strategy.actor.parameters())
                        all_target_actor_params.extend(
                            strategy.actor_target.parameters()
                        )

                # If shared, add the shared actor params only once
                if self.use_shared_actor:
                    for cluster_index in self.clusters.keys():
                        all_actor_params.extend(
                            self.shared_actors[cluster_index].parameters()
                        )
                        all_target_actor_params.extend(
                            self.shared_actor_targets[cluster_index].parameters()
                        )

                # Perform batch-wise Polyak update (NO LOOPS)
                polyak_update(all_critic_params, all_target_critic_params, self.tau)
                polyak_update(all_actor_params, all_target_actor_params, self.tau)

        self.learning_role.write_rl_critic_params_to_output(learning_rate, unit_params)

    def cluster_strategies_k_means(
        self, strategies: list, n_groups: int, random_state: int
    ):
        contexts = np.array(
            [strategy.context.detach().cpu().numpy() for strategy in strategies]
        )

        kmeans = KMeans(n_clusters=n_groups, random_state=random_state)
        labels = kmeans.fit_predict(contexts)

        clusters = {i: [] for i in range(n_groups)}
        for strategy, label in zip(strategies, labels):
            clusters[label].append(strategy)

        return clusters

    def cluster_strategies_max_size(
        self, strategies: list, max_size: int, random_state: int
    ):
        """
        Clusters strategies such that no cluster exceeds `max_size`.

        Args:
            strategies (list): List of strategy objects with `.context` attributes.
            max_size (int): Maximum allowed number of strategies per cluster.
            random_state (int): Seed for reproducible KMeans clustering.

        Returns:
            dict[int, list]: Dictionary of clustered strategies.

        Raises:
            ValueError: If a cluster still exceeds `max_size` after processing.
            RecursionError: If too many recursive splits occur.
        """

        n_clusters = max(1, len(strategies) // max_size)

        clusters = self.cluster_strategies_k_means(
            strategies=strategies, n_groups=n_clusters, random_state=random_state
        )

        def split_clusters(clusters_dict):
            final_clusters = {}
            cluster_id_counter = 0

            for cluster in clusters_dict.values():
                if len(cluster) <= max_size:
                    final_clusters[cluster_id_counter] = cluster
                    cluster_id_counter += 1
                else:
                    # Check for identical contexts before attempting split
                    sub_contexts = np.array(
                        [s.context.detach().cpu().numpy() for s in cluster]
                    )
                    unique_sub_contexts = np.unique(sub_contexts, axis=0)

                    if len(unique_sub_contexts) <= 1:
                        logger.warning(
                            f"Cannot split cluster of size {len(cluster)} further "
                            f"as contexts are identical. Keeping oversized cluster (violates max_size)."
                        )
                        final_clusters[cluster_id_counter] = cluster
                        cluster_id_counter += 1
                        continue

                    # Split cluster further
                    n_subclusters = max(2, len(cluster) // max_size)
                    sub_clusters = self.cluster_strategies_k_means(
                        strategies=cluster,
                        n_groups=n_subclusters,
                        random_state=random_state
                        + len(
                            cluster
                        ),  # add cluster size to random state for uniqueness
                    )

                    # Recursively process sub-clusters
                    sub_result = split_clusters(sub_clusters)
                    for sub_cluster in sub_result.values():
                        final_clusters[cluster_id_counter] = sub_cluster
                        cluster_id_counter += 1

            return final_clusters

        try:
            final_clusters = split_clusters(clusters)
        except RecursionError:
            logger.error(
                "Recursion limit reached while splitting clusters. "
                "Consider increasing the max_number_of_actors parameter"
            )
            raise

        # Final safety check
        for cluster in final_clusters.values():
            if len(cluster) > max_size:
                raise ValueError(
                    f"Cluster size exceeds max size: {len(cluster)} > {max_size}"
                )

        return final_clusters

    def create_clusters(self):
        """
        Create clusters of strategies based on the specified clustering method and return them.

        Supports two clustering methods:
        - 'max-size': Uses Agglomerative Clustering to create groups with a maximum number of strategies.
        - 'k-means': Uses KMeans to create a predefined number of clusters.

        Returns:
            dict[int, list]: A dictionary where keys are cluster indices and values are lists of strategies.

        Raises:
            ValueError: If the required parameters are missing or if the clustering method is unknown.
        """
        strategies = list(self.learning_role.rl_strats.values())
        n_strategies = len(strategies)

        if not strategies:
            raise ValueError("No strategies available for clustering.")

        all_contexts = np.array([s.context.detach().cpu().numpy() for s in strategies])
        if len(np.unique(all_contexts, axis=0)) == 1:
            logger.warning(
                "All strategy contexts are identical — clustering is not possible."
            )
            return {0: strategies}  # Return a single cluster with all strategies

        if self.actor_clustering_method == "max-size":
            # Validate 'max-size' method
            max_number_of_actors = self.clustering_method_kwargs.get(
                "max_number_of_actors"
            )
            if not isinstance(max_number_of_actors, int) or max_number_of_actors <= 0:
                raise ValueError(
                    "'max_number_of_actors' must be a positive integer when using 'max-size' clustering method."
                )

            clusters = self.cluster_strategies_max_size(
                strategies=strategies,
                max_size=max_number_of_actors,
                random_state=self.clustering_method_kwargs.get("random_state", 42),
            )

        elif self.actor_clustering_method == "k-means":
            # Validate 'k-means' method
            n_clusters = self.clustering_method_kwargs.get("n_clusters")
            if not isinstance(n_clusters, int) or n_clusters <= 0:
                raise ValueError(
                    "'n_clusters' must be a positive integer when using 'k-means' clustering method."
                )

            if n_clusters > n_strategies:
                raise ValueError(
                    f"'n_clusters' ({n_clusters}) cannot be greater than the number of strategies ({n_strategies})."
                )

            clusters = self.cluster_strategies_k_means(
                strategies=strategies,
                n_groups=n_clusters,
                random_state=self.clustering_method_kwargs.get("random_state", 42),
            )

        else:
            raise ValueError(
                f"Unknown clustering method '{self.actor_clustering_method}'. Supported methods are 'max-size' and 'k-means'."
            )

        return clusters

    def verify_actor_sharing(self):
        """
        Verifies that all strategies within the same cluster point to the
        exact same actor network instance in memory.

        Should be called after initialize_policy when using shared actors.

        Returns:
            bool: True if actor sharing is consistent, False otherwise.
        """
        if not self.use_shared_actor:
            logger.debug("Verification skipped: Not in shared actor mode.")
            return True  # Nothing to verify

        if not hasattr(self, "clusters") or not self.clusters:
            logger.error(
                "Verification failed: `self.clusters` attribute not found or empty."
            )
            return False

        logger.debug("Starting actor sharing verification...")
        overall_consistent = True

        for cluster_index, strategies_in_cluster in self.clusters.items():
            if not strategies_in_cluster:
                logger.debug(f"Cluster {cluster_index}: Skipping empty cluster.")
                continue

            # Get the actor instance from the first strategy as the reference
            try:
                reference_actor = strategies_in_cluster[0].actor
                reference_actor_id = id(reference_actor)  # Get memory address (ID)
                logger.debug(
                    f"Cluster {cluster_index}: Reference actor ID is {reference_actor_id}"
                )
            except AttributeError:
                logger.error(
                    f"Cluster {cluster_index}: First strategy {strategies_in_cluster[0].unit_id} missing 'actor' attribute."
                )
                overall_consistent = False
                continue  # Cannot check this cluster

            # Compare all other strategies in the cluster to the reference
            is_cluster_consistent = True
            for i, strategy in enumerate(strategies_in_cluster):
                try:
                    current_actor = strategy.actor
                    current_actor_id = id(current_actor)

                    # Check if they are the *exact same object* in memory
                    if current_actor is not reference_actor:
                        logger.error(
                            f"Inconsistency in Cluster {cluster_index}! "
                            f"Strategy '{strategy.unit_id}' (index {i}) has actor ID {current_actor_id}, "
                            f"expected {reference_actor_id}."
                        )
                        is_cluster_consistent = False
                        overall_consistent = False
                        # Optional: break here if you only care about the first mismatch per cluster
                        # break
                except AttributeError:
                    logger.error(
                        f"Cluster {cluster_index}: Strategy '{strategy.unit_id}' missing 'actor' attribute during verification."
                    )
                    is_cluster_consistent = False
                    overall_consistent = False
                    # Optional: break here

            if is_cluster_consistent:
                logger.debug(
                    f"Cluster {cluster_index}: OK - All {len(strategies_in_cluster)} strategies share the same actor instance (ID: {reference_actor_id})."
                )

        if overall_consistent:
            logger.debug(
                "Actor sharing verification finished: All clusters consistent."
            )
        else:
            logger.warning(
                "Actor sharing verification finished: Inconsistencies detected!"
            )

        return overall_consistent
