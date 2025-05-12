# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging
import os

import torch as th
from torch.nn import functional as F
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.learning_utils import (
    polyak_update,
    transfer_weights,
)
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
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critics.state_dict(),
                "critic_target": strategy.target_critics.state_dict(),
                "critic_optimizer": strategy.critics.optimizer.state_dict(),
            }
            path = f"{directory}/critic_{u_id}.pt"
            th.save(obj, path)

        # record the exact order of u_ids and save it with critics to ensure that the same order is used when loading the parameters
        u_id_list = [str(u) for u in self.learning_role.rl_strats.keys()]
        mapping = {"u_id_order": u_id_list}
        map_path = os.path.join(directory, "u_id_order.json")
        with open(map_path, "w") as f:
            json.dump(mapping, f, indent=2)

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
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "actor": strategy.actor.state_dict(),
                "actor_target": strategy.actor_target.state_dict(),
                "actor_optimizer": strategy.actor.optimizer.state_dict(),
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
        Load critic, target_critic, and optimizer states for each agent strategy.
        If agent count differs between saved and current model, performs weight transfer for both networks.
        Args:
            directory (str): The directory from which the parameters should be loaded.
        """
        logger.info("Loading critic parameters...")

        if not os.path.exists(directory):
            logger.warning(
                "Specified directory does not exist. Using randomly initialized critics."
            )
            return

        map_path = os.path.join(directory, "critics", "u_id_order.json")
        if os.path.exists(map_path):
            # read the saved order of u_ids from critics save directory
            with open(map_path) as f:
                loaded_id_order = json.load(f).get("u_id_order", [])
        else:
            logger.warning("No u_id_order.json: assuming same order as current.")
            loaded_id_order = [str(u) for u in self.learning_role.rl_strats.keys()]

        new_id_order = [str(u) for u in self.learning_role.rl_strats.keys()]
        direct_load = loaded_id_order == new_id_order

        if direct_load:
            logger.info("Agents order unchanged. Loading critic weights directly.")
        else:
            logger.info(
                f"Agents length and/or order mismatch: n_old={len(loaded_id_order)}, n_new={len(new_id_order)}. Transferring weights for critics and target critics."
            )

        for u_id, strategy in self.learning_role.rl_strats.items():
            critic_path = os.path.join(directory, "critics", f"critic_{u_id}.pt")
            if not os.path.exists(critic_path):
                logger.warning(f"No saved critic for {u_id}; skipping.")
                continue

            try:
                critic_params = th.load(critic_path, weights_only=True)
                for key in ("critic", "critic_target", "critic_optimizer"):
                    if key not in critic_params:
                        logger.warning(
                            f"Missing {key} in critic params for {u_id}; skipping."
                        )
                        continue

                if direct_load:
                    strategy.critics.load_state_dict(critic_params["critic"])
                    strategy.target_critics.load_state_dict(
                        critic_params["critic_target"]
                    )
                    strategy.critics.optimizer.load_state_dict(
                        critic_params["critic_optimizer"]
                    )
                    logger.debug(f"Loaded critic for {u_id} directly.")
                else:
                    critic_weights = transfer_weights(
                        model=strategy.critics,
                        loaded_state=critic_params["critic"],
                        loaded_id_order=loaded_id_order,
                        new_id_order=new_id_order,
                        obs_base=strategy.obs_dim,
                        act_dim=strategy.act_dim,
                        unique_obs=strategy.unique_obs_dim,
                    )
                    target_critic_weights = transfer_weights(
                        model=strategy.target_critics,
                        loaded_state=critic_params["critic_target"],
                        loaded_id_order=loaded_id_order,
                        new_id_order=new_id_order,
                        obs_base=strategy.obs_dim,
                        act_dim=strategy.act_dim,
                        unique_obs=strategy.unique_obs_dim,
                    )

                    if critic_weights is None or target_critic_weights is None:
                        logger.warning(
                            f"Critic weights transfer failed for {u_id}; skipping."
                        )
                        continue

                    strategy.critics.load_state_dict(critic_weights)
                    strategy.target_critics.load_state_dict(target_critic_weights)
                    logger.debug(f"Critic weights transferred for {u_id}.")

            except Exception as e:
                logger.warning(f"Failed to load critic for {u_id}: {e}")

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

                # add a tag to the strategy to indicate that the actor was loaded
                strategy.actor.loaded = True
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
            self.check_strategy_dimensions()
            self.create_actors()
            self.create_critics()

        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.actor = actors_and_critics["actors"][u_id]
                strategy.actor_target = actors_and_critics["actor_targets"][u_id]

                strategy.critics = actors_and_critics["critics"][u_id]
                strategy.target_critics = actors_and_critics["target_critics"][u_id]

            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

    def check_strategy_dimensions(self) -> None:
        """
        Iterate over all learning strategies and check if the dimensions of observations and actions are the same.
        Also check if the unique observation dimensions are the same. If not, raise a ValueError.
        This is important for the TD3 algorithm, as it uses a centralized critic that requires consistent dimensions across all agents.
        """
        obs_dim_list = []
        act_dim_list = []
        unique_obs_dim_list = []
        num_timeseries_obs_dim_list = []

        for strategy in self.learning_role.rl_strats.values():
            obs_dim_list.append(strategy.obs_dim)
            act_dim_list.append(strategy.act_dim)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
            num_timeseries_obs_dim_list.append(strategy.num_timeseries_obs_dim)

        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                f"All observation dimensions must be the same for all RL agents. The defined learning strategies have the following observation dimensions: {obs_dim_list}"
            )
        else:
            self.obs_dim = obs_dim_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError(
                f"All action dimensions must be the same for all RL agents. The defined learning strategies have the following action dimensions: {act_dim_list}"
            )
        else:
            self.act_dim = act_dim_list[0]

        if len(set(unique_obs_dim_list)) > 1:
            raise ValueError(
                f"All unique_obs_dim values must be the same for all RL agents. The defined learning strategies have the following unique_obs_dim values: {unique_obs_dim_list}"
            )
        else:
            self.unique_obs_dim = unique_obs_dim_list[0]

        if len(set(num_timeseries_obs_dim_list)) > 1:
            raise ValueError(
                f"All num_timeseries_obs_dim values must be the same for all RL agents. The defined learning strategies have the following num_timeseries_obs_dim values: {num_timeseries_obs_dim_list}"
            )
        else:
            self.num_timeseries_obs_dim = num_timeseries_obs_dim_list[0]

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

        for strategy in self.learning_role.rl_strats.values():
            strategy.actor = self.actor_architecture_class(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
            ).to(self.device)

            strategy.actor_target = self.actor_architecture_class(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                float_type=self.float_type,
                unique_obs_dim=self.unique_obs_dim,
                num_timeseries_obs_dim=self.num_timeseries_obs_dim,
            ).to(self.device)

            strategy.actor_target.load_state_dict(strategy.actor.state_dict())
            strategy.actor_target.train(mode=False)

            strategy.actor.optimizer = AdamW(
                strategy.actor.parameters(),
                lr=self.learning_role.calc_lr_from_progress(
                    1
                ),  # 1=100% of simulation remaining, uses learning_rate from config as starting point
            )

            strategy.actor.loaded = False

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
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            strategy.target_critics = CriticTD3(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
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

        for u_id, strategy in self.learning_role.rl_strats.items():
            actors[u_id] = strategy.actor
            actor_targets[u_id] = strategy.actor_target

            critics[u_id] = strategy.critics
            target_critics[u_id] = strategy.target_critics

        actors_and_critics = {
            "actors": actors,
            "actor_targets": actor_targets,
            "critics": critics,
            "target_critics": target_critics,
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
                    strategy.actor.optimizer,
                ],
                learning_rate=learning_rate,
            )
            strategy.action_noise.update_noise_decay(updated_noise_decay)

        for step in range(self.gradient_steps):
            self.n_updates += 1

            transitions = self.learning_role.buffer.sample(self.batch_size)
            states, actions, next_states, rewards = (
                transitions.observations,
                transitions.actions,
                transitions.next_observations,
                transitions.rewards,
            )

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = th.randn_like(actions) * self.target_policy_noise
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

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
                unit_params[step][strategy.unit_id]["total_grad_norm"] = total_norm
                unit_params[step][strategy.unit_id]["max_grad_norm"] = max_grad_norm

            ######################################################################
            # ACTOR UPDATE (DELAYED): Accumulate losses for all agents in one pass
            ######################################################################
            if self.n_updates % self.policy_delay == 0:
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

                    all_actor_params.extend(strategy.actor.parameters())
                    all_target_actor_params.extend(strategy.actor_target.parameters())

                # Perform batch-wise Polyak update (NO LOOPS)
                polyak_update(all_critic_params, all_target_critic_params, self.tau)
                polyak_update(all_actor_params, all_target_actor_params, self.tau)

        self.learning_role.write_rl_critic_params_to_output(learning_rate, unit_params)
