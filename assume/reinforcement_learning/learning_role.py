# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
from datetime import datetime

import torch as th
from dateutil import rrule as rr
from mango import Role

from assume.common.base import LearningConfig, LearningStrategy
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.algorithms.matd3 import TD3
from assume.reinforcement_learning.buffer import ReplayBuffer

logger = logging.getLogger(__name__)


class Learning(Role):
    """
    This class manages the learning process of reinforcement learning agents, including initializing key components such as
    neural networks, replay buffer, and learning hyperparameters. It handles both training and evaluation modes based on
    the provided learning configuration.

    Args:
        simulation_start (datetime): The start of the simulation.
        simulation_end (datetime): The end of the simulation.
        learning_config (LearningConfig): The configuration for the learning process.

    """

    def __init__(
        self,
        learning_config: LearningConfig,
        start: datetime,
        end: datetime,
    ):
        self.simulation_start = start
        self.simulation_end = end

        # how many learning roles do exist and how are they named
        self.buffer: ReplayBuffer = None
        self.obs_dim = learning_config["observation_dimension"]
        self.act_dim = learning_config["action_dimension"]
        self.episodes_done = 0
        self.rl_strats: dict[int, LearningStrategy] = {}
        self.rl_algorithm = learning_config["algorithm"]
        self.critics = {}
        self.target_critics = {}

        # define whether we train model or evaluate it
        self.training_episodes = learning_config["training_episodes"]
        self.learning_mode = learning_config["learning_mode"]
        self.continue_learning = learning_config["continue_learning"]
        self.trained_actors_path = learning_config["trained_actors_path"]

        cuda_device = (
            learning_config["device"]
            if "cuda" in learning_config.get("device", "cpu")
            else "cpu"
        )
        self.device = th.device(cuda_device if th.cuda.is_available() else "cpu")

        # future: add option to choose between float16 and float32
        # float_type = learning_config.get("float_type", "float32")
        self.float_type = th.float

        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True

        self.learning_rate = learning_config.get("learning_rate", 1e-4)
        self.episodes_collecting_initial_experience = learning_config.get(
            "episodes_collecting_initial_experience", 5
        )
        self.train_freq = learning_config.get("train_freq", 1)
        self.gradient_steps = (
            self.train_freq
            if learning_config.get("gradient_steps", -1) == -1
            else learning_config["gradient_steps"]
        )
        self.batch_size = learning_config.get("batch_size", 128)
        self.gamma = learning_config.get("gamma", 0.99)

        self.eval_episodes_done = 0
        self.max_eval_reward = -1e9
        self.max_eval_regret = 1e9
        self.max_eval_profit = -1e9

        # function that initializes learning, needs to be an extra function so that it can be called after buffer is given to Role
        self.create_learning_algorithm(self.rl_algorithm)

        # store evaluation values
        self.rl_eval_rewards = []
        self.rl_eval_profits = []
        self.rl_eval_regrets = []

    def setup(self) -> None:
        """
        Set up the learning role for reinforcement learning training.

        Notes:
            This method prepares the learning role for the reinforcement learning training process. It subscribes to relevant messages
            for handling the training process and schedules recurrent tasks for policy updates based on the specified training frequency.
        """
        # subscribe to messages for handling the training process
        self.context.subscribe_message(
            self,
            self.handle_message,
            lambda content, meta: content.get("context") == "rl_training",
        )

        recurrency_task = rr.rrule(
            freq=rr.HOURLY,
            interval=self.train_freq,
            dtstart=self.simulation_start,
            until=self.simulation_end,
            cache=True,
        )

        self.context.schedule_recurrent_task(self.update_policy, recurrency_task)

    def handle_message(self, content: dict, meta: dict) -> None:
        """
        Handles the incoming messages and performs corresponding actions.

        Args:
            content (dict): The content of the message.
            meta (dict): The metadata associated with the message. (not needed yet)
        """

        if content.get("type") == "replay_buffer":
            data = content["data"]
            self.buffer.add(
                obs=data[0],
                actions=data[1],
                reward=data[2],
            )

    def turn_off_initial_exploration(self) -> None:
        """
        Disable initial exploration mode for all learning strategies.

        Notes:
            This method turns off the initial exploration mode for all learning strategies associated with the learning role. Initial
            exploration is often used to collect initial experience before training begins. Disabling it can be useful when the agent
            has collected sufficient initial data and is ready to focus on training.
        """
        for _, unit in self.rl_strats.items():
            unit.collect_initial_experience_mode = False

    def create_learning_algorithm(self, algorithm: RLAlgorithm):
        """
        Create and initialize the reinforcement learning algorithm.

        This method creates and initializes the reinforcement learning algorithm based on the specified algorithm name. The algorithm
        is associated with the learning role and configured with relevant hyperparameters.

        Args:
            algorithm (RLAlgorithm): The name of the reinforcement learning algorithm.
        """
        if algorithm == "matd3":
            self.rl_algorithm = TD3(
                learning_role=self,
                learning_rate=self.learning_rate,
                episodes_collecting_initial_experience=self.episodes_collecting_initial_experience,
                gradient_steps=self.gradient_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
            )
        else:
            logger.error(f"Learning algorithm {algorithm} not implemented!")

    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Initialize the policy of the reinforcement learning agent considering the respective algorithm.

        This method initializes the policy (actor) of the reinforcement learning agent. It tests if we want to continue the learning process with
        stored policies from a former training process. If so, it loads the policies from the specified directory. Otherwise, it initializes the
        respective new policies.
        """

        self.rl_algorithm.initialize_policy(actors_and_critics)

        if self.continue_learning == True and actors_and_critics == None:
            load_directory = self.trained_actors_path
            self.load_policies(load_directory)

    def load_policies(self, load_directory) -> None:
        """
        Load the policies of the reinforcement learning agent.

        This method loads the entire policies (actor and critics) of the reinforcement learning agent from the specified directory.
        This is used if we want to continue learning from already learned strategies.

        Args:
            load_directory (str): The directory from which to load the policies.
        """
        if load_directory is None:
            logger.warning(
                "You have specified continue learning as True but no trained_actors_path was given!"
            )
            logger.info("Restart learning process!")
        else:
            logger.info(f"Loading pretrained policies from {load_directory}!")
            self.rl_algorithm.load_params(load_directory)

    async def update_policy(self) -> None:
        """
        Update the policy of the reinforcement learning agent.

        This method is responsible for updating the policy (actor) of the reinforcement learning agent asynchronously. It checks if
        the number of episodes completed is greater than the number of episodes required for initial experience collection. If so,
        it triggers the policy update process by calling the `update_policy` method of the associated reinforcement learning algorithm.

        Notes:
            This method is typically scheduled to run periodically during training to continuously improve the agent's policy.
        """
        if self.episodes_done > self.episodes_collecting_initial_experience:
            self.rl_algorithm.update_policy()

    # TODO: add evaluation function
    def compare_and_save_policies(self) -> None:
        """
        Compare evaluation metrics and save policies based on the best achieved performance.

        This method compares the evaluation metrics, such as reward, profit, and regret, and saves the policies if they achieve the
        best performance in their respective categories. It iterates through the specified modes, compares the current evaluation
        value with the previous best, and updates the best value if necessary. If an improvement is detected, it saves the policy
        and associated parameters.

        Notes:
            This method is typically used during the evaluation phase to save policies that achieve superior performance.
        """
        modes = ["reward", "profit", "regret"]
        for mode in modes:
            value = None

            if not self.rl_eval_rewards:
                # TODO?
                return

            if mode == "reward" and self.rl_eval_rewards[-1] > self.max_eval_reward:
                self.max_eval_reward = self.rl_eval_rewards[-1]
                dir_name = "highest_reward"
                value = self.max_eval_reward
            elif mode == "profit" and self.rl_eval_profits[-1] > self.max_eval_profit:
                self.max_eval_profit = self.rl_eval_profits[-1]
                dir_name = "highest_profit"
                value = self.max_eval_profit
            elif (
                mode == "regret"
                and self.rl_eval_regrets[-1] < self.max_eval_regret
                and self.rl_eval_regrets[-1] != 0
            ):
                self.max_eval_regret = self.rl_eval_regrets[-1]
                dir_name = "lowest_regret"
                value = self.max_eval_regret

            if value is not None:
                self.rl_algorithm.save_params(dir_name=dir_name)
                for unit in self.rl_powerplants + self.rl_storages:
                    if unit.learning:
                        unit.save_params(dir_name=dir_name)

                logger.info(
                    f"Policies saved, episode: {self.eval_episodes_done + 1}, mode: {mode}, value: {value:.2f}"
                )
