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
    """Base reinforcement learning algorithm class.
    
    This is the foundation class for all Reinforcement Learning algorithms in the framework. 
    To implement a custom RL algorithm, subclass this class and override the `update_policy` method.
    
    The class provides common functionality for:
    - Learning rate scheduling
    - Parameter saving/loading
    - Device management
    
    Attributes:
        learning_role: The learning role object containing configuration and strategies.
        learning_config: Configuration parameters from the learning role.
        device: The computation device (CPU/GPU) for tensors.
        float_type: The floating point precision type for computations.
        actor_architecture_class: The actor network architecture class.
    
    Example:
        >>> class CustomAlgorithm(RLAlgorithm):
        ...     def update_policy(self):
        ...         # Custom policy update logic
        ...         pass
    """

    def __init__(self, learning_role):
        """Initialize the RL algorithm.
        
        Args:
            learning_role: Learning role object containing configuration and strategies.
                Must be an instance of the Learning class.
        """
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
        """Update optimizer learning rates.
        
        Sets the learning rate for one or more optimizers. Handles both single
        optimizers and lists of optimizers uniformly.
        
        Args:
            optimizers: A single optimizer or list of optimizers to update.
            learning_rate: The new learning rate value to set.
        
        Note:
            Adapted from Stable Baselines 3:
            - https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/base_class.py#L286
            - https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L68
        
        Example:
            >>> optimizer = AdamW(model.parameters(), lr=0.001)
            >>> algorithm.update_learning_rate(optimizer, 0.0001)
        """

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

    def update_policy(self) -> None:
        """Update the policy parameters.
        
        This method must be overridden by subclasses to implement the specific
        policy update logic for each RL algorithm. The base implementation raises
        an error to enforce this requirement.
        
        Raises:
            NotImplementedError: If called on the base class without override.
        
        Example:
            >>> class CustomAlgorithm(RLAlgorithm):
            ...     def update_policy(self):
            ...         # Implement algorithm-specific policy update
            ...         pass
        """
        logger.error(
            "No policy update function of the used RL algorithm was defined. "
            "Please define how the policies should be updated in the specific "
            "algorithm you use."
        )

    def load_obj(self, directory: str):
        """Load a serialized object from directory.
        
        Loads a PyTorch serialized object from the specified directory path.
        The object is loaded onto the device specified by the algorithm's configuration.
        
        Args:
            directory: Path to the directory containing the serialized object.
                Should point to a valid .pt file.
        
        Returns:
            object: The deserialized Python object.
        
        Example:
            >>> model_state = algorithm.load_obj('/path/to/checkpoint.pt')
        """
        return th.load(directory, map_location=self.device, weights_only=True)

    def load_params(self, directory: str) -> None:
        """Load learning parameters from disk.
        
        Abstract method that should be implemented by subclasses to load
        algorithm-specific parameters from the specified directory.
        
        Args:
            directory: Path to the directory containing saved parameters.
        
        Note:
            This is an abstract method that must be overridden by subclasses.
        """


class A2CAlgorithm(RLAlgorithm):
    """Base actor-critic algorithm class.
    
    Provides shared functionality for actor-critic reinforcement learning algorithms
    including parameter management, network initialization, and saving/loading utilities.
    This serves as the foundation for algorithms like MATD3, MADDPG, and MAPPO.
    
    The class handles:
    - Actor and critic network creation and management
    - Target network synchronization (when applicable)
    - Parameter saving and loading
    - Weight transfer between different agent configurations
    
    Attributes:
        uses_target_networks: Whether this algorithm uses target networks.
            TD3 and DDPG use target networks (True), PPO does not (False).
    
    Example:
        >>> class ActorCriticAlgorithm(A2CAlgorithm):
        ...     def update_policy(self):
        ...         # Custom actor-critic update logic
        ...         pass
    """

    #: Whether this algorithm uses target networks for stability.
    #: TD3 and DDPG use target networks (True), PPO does not (False).
    uses_target_networks: bool = True

    def __init__(self, learning_role):
        """Initialize the actor-critic algorithm.
        
        Args:
            learning_role: Learning role object containing configuration and strategies.
        """
        super().__init__(learning_role)

    def save_params(self, directory: str) -> None:
        """Save actor and critic network parameters.
        
        Saves both actor and critic network parameters to separate subdirectories.
        Creates the directory structure if it doesn't exist.
        
        Args:
            directory: Base directory path where parameters will be saved.
                Will create 'actors/' and 'critics/' subdirectories.
        
        Example:
            >>> algorithm.save_params('/path/to/save/directory')
            # Creates:
            # /path/to/save/directory/actors/
            # /path/to/save/directory/critics/
        """
        self.save_critic_params(directory=f"{directory}/critics")
        self.save_actor_params(directory=f"{directory}/actors")

    def save_critic_params(self, directory: str) -> None:
        """Save critic network parameters.
        
        Saves critic networks, their optimizers, and target critics (if applicable)
        for all registered learning strategies. Also saves agent ID ordering information
        to ensure proper loading.
        
        Args:
            directory: Directory path where critic parameters will be saved.
                Will be created if it doesn't exist.
        
        Example:
            >>> algorithm.save_critic_params('/path/to/critics/')
        """
        os.makedirs(directory, exist_ok=True)
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "critic": strategy.critics.state_dict(),
                "critic_optimizer": strategy.critics.optimizer.state_dict(),
            }
            # Only save target critic if this algorithm uses target networks
            if self.uses_target_networks:
                obj["critic_target"] = strategy.target_critics.state_dict()
            
            path = f"{directory}/critic_{u_id}.pt"
            th.save(obj, path)

        # record the exact order of u_ids and save it with critics to ensure that the same order is used when loading the parameters
        u_id_list = [str(u) for u in self.learning_role.rl_strats.keys()]
        mapping = {"u_id_order": u_id_list}
        map_path = os.path.join(directory, "u_id_order.json")
        with open(map_path, "w") as f:
            json.dump(mapping, f, indent=2)

    def save_actor_params(self, directory: str) -> None:
        """Save actor network parameters.
        
        Saves actor networks, their optimizers, and target actors (if applicable)
        for all registered learning strategies.
        
        Args:
            directory: Directory path where actor parameters will be saved.
                Will be created if it doesn't exist.
        
        Example:
            >>> algorithm.save_actor_params('/path/to/actors/')
        """
        os.makedirs(directory, exist_ok=True)
        for u_id, strategy in self.learning_role.rl_strats.items():
            obj = {
                "actor": strategy.actor.state_dict(),
                "actor_optimizer": strategy.actor.optimizer.state_dict(),
            }
            # Only save target actor if this algorithm uses target networks
            if self.uses_target_networks:
                obj["actor_target"] = strategy.actor_target.state_dict()
            
            path = f"{directory}/actor_{u_id}.pt"
            th.save(obj, path)

    def load_params(self, directory: str) -> None:
        """Load actor and critic network parameters.
        
        Loads both actor and critic parameters from the specified directory.
        Calls load_critic_params() and load_actor_params() sequentially.
        
        Args:
            directory: Base directory containing 'actors/' and 'critics/' subdirectories.
        
        Example:
            >>> algorithm.load_params('/path/to/saved/parameters/')
        """
        self.load_critic_params(directory)
        self.load_actor_params(directory)

    def load_critic_params(self, directory: str) -> None:
        """Load critic network parameters.
        
        Loads critic networks, target critics (if applicable), and optimizer states
        for each registered agent strategy. Handles cases where the number of agents
        differs between saved and current models by performing intelligent weight transfer.
        
        Args:
            directory: Base directory containing the 'critics/' subdirectory.
        
        Note:
            Automatically handles agent count mismatches through weight transfer.
            Preserves the order of agents using saved mapping information.
        
        Example:
            >>> algorithm.load_critic_params('/path/to/saved/parameters/')
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
                
                # Required keys depend on whether algorithm uses target networks
                required_keys = ["critic", "critic_optimizer"]
                if self.uses_target_networks:
                    required_keys.append("critic_target")
                
                for key in required_keys:
                    if key not in critic_params:
                        logger.warning(
                            f"Missing {key} in critic params for {u_id}; skipping."
                        )
                        continue

                if direct_load:
                    strategy.critics.load_state_dict(critic_params["critic"])
                    strategy.critics.optimizer.load_state_dict(
                        critic_params["critic_optimizer"]
                    )
                    # Only load target critic if this algorithm uses target networks
                    if self.uses_target_networks and "critic_target" in critic_params:
                        strategy.target_critics.load_state_dict(
                            critic_params["critic_target"]
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
                    
                    strategy.critics.load_state_dict(critic_weights)
                    
                    # Only transfer target critic weights if this algorithm uses target networks
                    if self.uses_target_networks and "critic_target" in critic_params:
                        target_critic_weights = transfer_weights(
                            model=strategy.target_critics,
                            loaded_state=critic_params["critic_target"],
                            loaded_id_order=loaded_id_order,
                            new_id_order=new_id_order,
                            obs_base=strategy.obs_dim,
                            act_dim=strategy.act_dim,
                            unique_obs=strategy.unique_obs_dim,
                        )
                        if target_critic_weights is not None:
                            strategy.target_critics.load_state_dict(target_critic_weights)
                    
                    logger.debug(f"Critic weights transferred for {u_id}.")

            except Exception as e:
                logger.warning(f"Failed to load critic for {u_id}: {e}")

    def load_actor_params(self, directory: str) -> None:
        """Load actor network parameters.
        
        Loads actor networks, target actors (if applicable), and optimizer states
        for each registered agent strategy from the specified directory.
        
        Args:
            directory: The directory containing the 'actors/' subdirectory where the parameters should be loaded.
        
        Example:
            >>> algorithm.load_actor_params('/path/to/saved/parameters/')
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
                strategy.actor.optimizer.load_state_dict(
                    actor_params["actor_optimizer"]
                )

                # Only load target actor if this algorithm uses target networks
                if self.uses_target_networks and "actor_target" in actor_params:
                    strategy.actor_target.load_state_dict(actor_params["actor_target"])

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
            actors_and_critics: Optional dictionary containing pre-trained networks.
                If None, creates new networks. If provided, assigns existing networks.
                Expected format includes 'actors', 'critics', and optionally
                'actor_targets' and 'target_critics' keys.
        
        Example:
            >>> # Create new networks
            >>> algorithm.initialize_policy()
            >>> 
            >>> # Assign existing networks
            >>> algorithm.initialize_policy(existing_networks_dict)
        """
        if actors_and_critics is None:
            self.check_strategy_dimensions()
            self.create_actors()
            self.create_critics()

        else:
            for u_id, strategy in self.learning_role.rl_strats.items():
                strategy.actor = actors_and_critics["actors"][u_id]
                strategy.critics = actors_and_critics["critics"][u_id]
                
                if self.uses_target_networks:
                    strategy.actor_target = actors_and_critics["actor_targets"][u_id]
                    strategy.target_critics = actors_and_critics["target_critics"][u_id]

            self.obs_dim = actors_and_critics["obs_dim"]
            self.act_dim = actors_and_critics["act_dim"]
            self.unique_obs_dim = actors_and_critics["unique_obs_dim"]

    def check_strategy_dimensions(self) -> None:
        """Validate learning strategy dimensions.
        
        Ensures all registered learning strategies have consistent dimensional
        properties required for centralized critic algorithms. Checks:
        - Observation dimensions
        - Action dimensions
        - Unique observation dimensions
        - Timeseries observation dimensions
        - Foresight parameters
        If not consistent, raises a ValueError. This is important for centralized
        critic algorithms, as it uses a centralized critic that requires consistent
        dimensions across all agents.
        
        Raises:
            ValueError: If any dimension mismatch is detected across strategies.
        
        Note:
            This validation is crucial for centralized critic algorithms where
            all agents must have compatible observation and action spaces.
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
        """Create actor networks for all learning strategies.
        
        This method initializes actor networks and their corresponding target networks for
        each registered unit strategy. Actors map observations to actions.
        
        Note:
            All strategies must have the same observation dimension due to the
            centralized critic architecture. Units with different observation
            dimensions require separate learning roles with different critics.
        
        Example:
            >>> algorithm.create_actors()
            >>> # Creates actor and actor_target for each strategy
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
        """Create critic networks for all learning strategies.
        
        Initializes critic networks and their corresponding target networks for
        each registered agent strategy. Critics evaluate state-action pairs.
        
        Note:
            All strategies must have the same observation dimension due to the
            centralized critic architecture. Units with different observation
            dimensions require separate learning roles with different critics.
        
        Example:
            >>> algorithm.create_critics()
            >>> # Creates critics and target_critics for each strategy
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
        """Extract all policy networks.
        
        Collects actor and critic networks from all learning strategies into
        a structured dictionary. Includes both primary and target networks.
        
        Returns:
            Dictionary containing all network components organized by type:
                - 'actors': Primary actor networks
                - 'actor_targets': Target actor networks
                - 'critics': Primary critic networks
                - 'target_critics': Target critic networks
                - Dimension information for reconstruction
        
        Example:
            >>> policy_dict = algorithm.extract_policy()
            >>> # Contains all networks ready for saving or transfer
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
