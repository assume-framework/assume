# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import json
import logging
import os

import torch as th
from torch.optim import AdamW

from assume.reinforcement_learning.algorithms import actor_architecture_aliases
from assume.reinforcement_learning.learning_utils import (
    transfer_weights,
)

logger = logging.getLogger(__name__)


class RLAlgorithm:
    """
    The base RL model class. To implement your own RL algorithm, you need to subclass this class and implement the `update_policy` method.

    Args:
        learning_role (Learning Role object): Learning object
    """

    def __init__(
        self,
        # init learning_role as object of Learning class
        learning_role,
    ):
        super().__init__()

        self.learning_role = learning_role
        self.learning_config = learning_role.learning_config

        if self.learning_config.actor_architecture in actor_architecture_aliases.keys():
            self.actor_architecture_class = actor_architecture_aliases[
                self.learning_config.actor_architecture
            ]
        else:
            raise ValueError(
                f"Policy '{self.learning_config.actor_architecture}' unknown. Supported architectures are {list(actor_architecture_aliases.keys())}"
            )

        self.device = self.learning_role.device
        self.float_type = self.learning_role.float_type

    def update_learning_rate(
        self,
        optimizers: list[th.optim.Optimizer] | th.optim.Optimizer,
        learning_rate: float,
    ) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule and the current progress remaining (from 1 to 0).

        Args:
            optimizers (List[th.optim.Optimizer] | th.optim.Optimizer): An optimizer or a list of optimizers.

        Note:
            Adapted from SB3:
            - https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/base_class.py#L286
            - https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L68

        """

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

    def update_policy(self):
        logger.error(
            "No policy update function of the used Rl algorithm was defined. Please define how the policies should be updated in the specific algorithm you use"
        )

    def load_obj(self, directory: str):
        """
        Load an object from a specified directory.

        This method loads an object, typically saved as a checkpoint file, from the specified
        directory and returns it. It uses the `torch.load` function and specifies the device for loading.

        Args:
            directory (str): The directory from which the object should be loaded.

        Returns:
            object: The loaded object.
        """
        return th.load(directory, map_location=self.device, weights_only=True)

    def load_params(self, directory: str) -> None:
        """
        Load learning params - abstract method to be implemented by the Learning Algorithm
        """


class A2CAlgorithm(RLAlgorithm):
    """
    The base A2C model class. To implement your own A2C algorithm, you need to subclass this class and implement the `update_policy` method.

    Args:
        learning_role (Learning Role object): Learning object
    """

    def __init__(
        self,
        # init learning_role as object of Learning class
        learning_role,
    ):
        super().__init__(
            learning_role,
        )

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
        foresight_list = []
        obs_dim_list = []
        act_dim_list = []
        unique_obs_dim_list = []
        num_timeseries_obs_dim_list = []

        for strategy in self.learning_role.rl_strats.values():
            foresight_list.append(strategy.foresight)
            obs_dim_list.append(strategy.obs_dim)
            act_dim_list.append(strategy.act_dim)
            unique_obs_dim_list.append(strategy.unique_obs_dim)
            num_timeseries_obs_dim_list.append(strategy.num_timeseries_obs_dim)

        if len(set(foresight_list)) > 1:
            raise ValueError(
                f"All foresight values must be the same for all RL agents. The defined learning strategies have the following foresight values: {foresight_list}"
            )
        else:
            self.foresight = foresight_list[0]

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

        # Check last, as other cases should fail before!
        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                f"All observation dimensions must be the same for all RL agents. The defined learning strategies have the following observation dimensions: {obs_dim_list}"
            )
        else:
            self.obs_dim = obs_dim_list[0]

    def create_actors(self) -> None:
        """
        Create actor networks for reinforcement learning for each unit strategy.

        This method initializes actor networks and their corresponding target networks for each unit strategy.
        The actors are designed to map observations to action probabilities in a reinforcement learning setting.

        The created actor networks are associated with each unit strategy and stored as attributes.

        Note:
            The observation dimension need to be the same, due to the centralized critic that all actors share.
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

        Note:
            The observation dimension need to be the same, due to the centralized criic that all actors share.
            If you have units with different observation dimensions. They need to have different critics and hence learning roles.
        """
        n_agents = len(self.learning_role.rl_strats)

        for strategy in self.learning_role.rl_strats.values():
            strategy.critics = self.critic_architecture_class(
                n_agents=n_agents,
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                unique_obs_dim=self.unique_obs_dim,
                float_type=self.float_type,
            ).to(self.device)

            strategy.target_critics = self.critic_architecture_class(
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
