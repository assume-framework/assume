# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch as th
from mango import Role

from assume.common.base import LearningConfig, LearningStrategy
from assume.common.utils import (
    create_rrule,
    datetime2timestamp,
    timestamp2datetime,
)
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.algorithms.matd3 import TD3
from assume.reinforcement_learning.buffer import ReplayBuffer
from assume.reinforcement_learning.learning_utils import (
    linear_schedule_func,
    transform_buffer_data,
)
from assume.reinforcement_learning.tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


class Learning(Role):
    """
    This class manages the learning process of reinforcement learning agents, including initializing key components such as
    neural networks, replay buffer, and learning hyperparameters. It handles both training and evaluation modes based on
    the provided learning configuration.

    Args:
        learning_config (LearningConfig): The configuration for the learning process.
        start (datetime.datetime): The start datetime for the simulation.
        end (datetime.datetime): The end datetime for the simulation.

    """

    def __init__(
        self,
        learning_config: LearningConfig,
        start: datetime,
        end: datetime,
    ):
        super().__init__()

        # how many learning roles do exist and how are they named
        self.buffer: ReplayBuffer = None
        self.episodes_done = 0
        self.rl_strats: dict[int, LearningStrategy] = {}
        self.learning_config = learning_config
        self.critics = {}
        self.target_critics = {}

        self.device = th.device(
            self.learning_config.device
            if (
                self.learning_config
                and "cuda" in self.learning_config.device
                and th.cuda.is_available()
            )
            else "cpu"
        )
        # future: add option to choose between float16 and float32
        # float_type = learning_config.float_type
        self.float_type = th.float

        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True

        self.start = datetime2timestamp(start)
        self.start_datetime = start
        self.end = datetime2timestamp(end)
        self.end_datetime = end

        self.datetime = None
        if self.learning_config.learning_mode:
            # configure additional learning parameters if we are in learning or evaluation mode
            if self.learning_config.learning_rate_schedule == "linear":
                self.calc_lr_from_progress = linear_schedule_func(
                    self.learning_config.learning_rate
                )
            else:
                self.calc_lr_from_progress = (
                    lambda x: self.learning_config.learning_rate
                )

            if self.learning_config.action_noise_schedule == "linear":
                self.calc_noise_from_progress = linear_schedule_func(
                    self.learning_config.noise_dt
                )
            else:
                self.calc_noise_from_progress = lambda x: self.learning_config.noise_dt

            self.eval_episodes_done = 0

            # function that initializes learning, needs to be an extra function so that it can be called after buffer is given to Role
            self.create_learning_algorithm(self.learning_config.algorithm)

            # store evaluation values
            self.max_eval = defaultdict(lambda: -1e9)
            self.rl_eval = defaultdict(list)
            # list of avg_changes
            self.avg_rewards = []

            self.tensor_board_logger = None
            self.update_steps = None

            # init dictionaries for all learning instances in this role
            # Note: we use atomic-swaps later to ensure no overwrites while we write the data into the buffer
            # this works since we do not use multi-threading, otherwise threading.locks would be needed here.
            self.all_obs = defaultdict(lambda: defaultdict(list))
            self.all_actions = defaultdict(lambda: defaultdict(list))
            self.all_noises = defaultdict(lambda: defaultdict(list))
            self.all_rewards = defaultdict(lambda: defaultdict(list))
            self.all_regrets = defaultdict(lambda: defaultdict(list))
            self.all_profits = defaultdict(lambda: defaultdict(list))

    def on_ready(self):
        """
        Set up the learning role for reinforcement learning training.

        Notes:
            This method prepares the learning role for the reinforcement learning training process. It subscribes to relevant messages
            for handling the training process and schedules recurrent tasks for policy updates based on the specified training frequency.
            This cannot happen in the init since the context (compare mango agents) is not yet available there.To avoid inconsistent replay buffer states (e.g. observation and action has been stored but not the reward), this
            slightly shifts the timing of the buffer updates.
        """
        super().on_ready()

        shifted_start = self.start_datetime + pd.Timedelta(
            self.learning_config.train_freq
        )  # shift start by hours in time frequency

        recurrency_task = create_rrule(
            start=shifted_start,
            end=self.end_datetime,
            freq=self.learning_config.train_freq,
        )

        self.context.schedule_recurrent_task(
            self.store_to_buffer_and_update,
            recurrency_task,
            src="no_wait",
        )

    def sync_train_freq_with_simulation_horizon(self) -> str | None:
        """
        Ensure self.train_freq evenly divides the simulation length.
        If not, adjust self.train_freq (in-place) and return the new string, otherwise return None.
        Uses self.start_datetime/self.end_datetime when available, otherwise falls back to timestamp fields.
        """

        # ensure train_freq evenly divides simulation length (may adjust self.train_freq)

        if not self.learning_config.learning_mode:
            return None

        train_freq_str = str(self.learning_config.train_freq)
        try:
            train_freq = pd.Timedelta(train_freq_str)
        except Exception:
            logger.warning(
                f"Invalid train_freq '{train_freq_str}' — skipping adjustment."
            )
            return None
        total_length = self.end_datetime - self.start_datetime
        quotient, remainder = divmod(total_length, train_freq)

        if remainder != pd.Timedelta(0):
            n_intervals = int(quotient) + 1
            new_train_freq_hours = int(
                (total_length / n_intervals).total_seconds() / 3600
            )
            new_train_freq_str = f"{new_train_freq_hours}h"
            self.learning_config.train_freq = new_train_freq_str

            logger.warning(
                f"Simulation length ({total_length}) is not divisible by train_freq ({train_freq_str}). "
                f"Adjusting train_freq to {new_train_freq_str}."
            )

        return self.learning_config.train_freq

    def determine_validation_interval(self) -> int:
        """
        Compute and validate validation_interval.

        Returns:
            validation_interval (int)
        Raises:
            ValueError if training_episodes is too small.
        """
        default_interval = self.learning_config.validation_episodes_interval
        training_episodes = self.learning_config.training_episodes
        validation_interval = min(training_episodes, default_interval)

        min_required_episodes = (
            self.learning_config.episodes_collecting_initial_experience
            + validation_interval
        )

        if self.learning_config.training_episodes < min_required_episodes:
            raise ValueError(
                f"Training episodes ({training_episodes}) must be greater than the sum of initial experience episodes ({self.learning_config.episodes_collecting_initial_experience}) and evaluation interval ({validation_interval})."
            )

        return validation_interval

    def register_strategy(self, strategy: LearningStrategy) -> None:
        """
        Register a learning strategy with this learning role.

        Args:
            strategy (LearningStrategy): The learning strategy to register.

        """

        self.rl_strats[strategy.unit_id] = strategy

    async def store_to_buffer_and_update(self) -> None:
        # Atomic dict operations - create new references
        current_obs = self.all_obs
        current_actions = self.all_actions
        current_rewards = self.all_rewards
        current_noises = self.all_noises
        current_regrets = self.all_regrets
        current_profits = self.all_profits

        # Reset cache dicts immediately with new defaultdicts
        self.all_obs = defaultdict(lambda: defaultdict(list))
        self.all_actions = defaultdict(lambda: defaultdict(list))
        self.all_rewards = defaultdict(lambda: defaultdict(list))
        self.all_noises = defaultdict(lambda: defaultdict(list))
        self.all_regrets = defaultdict(lambda: defaultdict(list))
        self.all_profits = defaultdict(lambda: defaultdict(list))

        # Get timestamps from cache we took
        all_timestamps = sorted(current_obs.keys())
        if len(all_timestamps) > 1:
            # Remove last timestamp that has no reward yet
            timestamps_to_process = all_timestamps[:-1]

            # Create filtered cache (only complete timesteps)
            cache = {
                "obs": {t: current_obs[t] for t in timestamps_to_process},
                "actions": {t: current_actions[t] for t in timestamps_to_process},
                "rewards": {t: current_rewards[t] for t in timestamps_to_process},
                "noises": {t: current_noises[t] for t in timestamps_to_process},
                "regret": {t: current_regrets[t] for t in timestamps_to_process},
                "profit": {t: current_profits[t] for t in timestamps_to_process},
            }

            # write data to output agent
            self.write_rl_params_to_output(cache)

            # if we are training also update the policy and write data into buffer
            if not self.learning_config.evaluation_mode:
                # Process cache in background
                await self._store_to_buffer_and_update_sync(cache, self.device)
        else:
            logger.warning("No experience retrieved to store in buffer at update step!")

    async def _store_to_buffer_and_update_sync(self, cache, device) -> None:
        """
        This function takes all the information that the strategies wrote into the learning_role cache dicts and post_processes them to fit into the buffer.
        Further triggers the next policy update

        """
        first_start = next(iter(cache["obs"]))
        for name, buffer in [
            ("observations", cache["obs"]),
            ("actions", cache["actions"]),
            ("rewards", cache["rewards"]),
        ]:
            # check if all entries for the buffers have the same number of unit_ids as rl_strats
            if len(buffer[first_start]) != len(self.rl_strats):
                logger.error(
                    f"Number of unit_ids with {name} in learning role ({len(buffer[first_start])}) does not match number of rl_strats ({len(self.rl_strats)}). "
                    "It seems like some learning_instances are not reporting experience. Cannot store to buffer and update policy!"
                )
                return

        # rewrite dict so that obs.shape == (n_rl_units, obs_dim) and sorted by keys and store in buffer
        self.buffer.add(
            obs=transform_buffer_data(cache["obs"], device),
            actions=transform_buffer_data(cache["actions"], device),
            reward=transform_buffer_data(cache["rewards"], device),
        )

        if (
            self.episodes_done
            >= self.learning_config.episodes_collecting_initial_experience
        ):
            self.rl_algorithm.update_policy()

    def add_observation_to_cache(self, unit_id, start, observation) -> None:
        """
        Add the observation to the cache dict, per unit_id.

        Args:
            unit_id (str): The id of the unit.
            observation (torch.Tensor): The observation to be added.

        """
        self.all_obs[start][unit_id].append(observation)

    def add_actions_to_cache(self, unit_id, start, action, noise) -> None:
        """
        Add the action and noise to the cache dict, per unit_id.

        Args:
            unit_id (str): The id of the unit.
            action (torch.Tensor): The action to be added.
            noise (torch.Tensor): The noise to be added.

        """

        # Add validation to catch unexpected unit_ids
        if unit_id == 0 or unit_id is None:
            logger.warning(
                f"Got invalid unit_id while storing learning experience: {unit_id}"
            )
            return

        self.all_actions[start][unit_id].append(action)
        self.all_noises[start][unit_id].append(noise)

    def add_reward_to_cache(self, unit_id, start, reward, regret, profit) -> None:
        """
        Add the reward to the cache dict, per unit_id.

        Args:
            unit_id (str): The id of the unit.
            reward (float): The reward to be added.

        """
        self.all_rewards[start][unit_id].append(reward)
        self.all_regrets[start][unit_id].append(regret)
        self.all_profits[start][unit_id].append(profit)

    def load_inter_episodic_data(self, inter_episodic_data):
        """
        Load the inter-episodic data from the dict stored across simulation runs.

        Args:
            inter_episodic_data (dict): The inter-episodic data to be loaded.

        """
        self.episodes_done = inter_episodic_data["episodes_done"]
        self.eval_episodes_done = inter_episodic_data["eval_episodes_done"]
        self.max_eval = inter_episodic_data["max_eval"]
        self.rl_eval = inter_episodic_data["all_eval"]
        self.avg_rewards = inter_episodic_data["avg_all_eval"]
        self.buffer = inter_episodic_data["buffer"]

        self.initialize_policy(inter_episodic_data["actors_and_critics"])

        # Disable initial exploration if initial experience collection is complete
        if (
            self.episodes_done
            >= self.learning_config.episodes_collecting_initial_experience
        ):
            self.turn_off_initial_exploration()

        # In continue_learning mode, disable it only for loaded strategies
        elif self.learning_config.continue_learning:
            self.turn_off_initial_exploration(loaded_only=True)

    def get_inter_episodic_data(self):
        """
        Dump the inter-episodic data to a dict for storing across simulation runs.

        Returns:
            dict: The inter-episodic data to be stored.
        """

        return {
            "episodes_done": self.episodes_done,
            "eval_episodes_done": self.eval_episodes_done,
            "max_eval": self.max_eval,
            "all_eval": self.rl_eval,
            "avg_all_eval": self.avg_rewards,
            "buffer": self.buffer,
            "actors_and_critics": self.rl_algorithm.extract_policy(),
        }

    def turn_off_initial_exploration(self, loaded_only=False) -> None:
        """
        Disable initial exploration mode.

        If `loaded_only=True`, only turn off exploration for strategies that were loaded (used in continue_learning mode).
        If `loaded_only=False`, turn it off for all strategies.

        Args:
            loaded_only (bool): Whether to disable exploration only for loaded strategies.
        """
        for strategy in self.rl_strats.values():
            if loaded_only:
                if strategy.actor.loaded:
                    strategy.collect_initial_experience_mode = False
            else:
                strategy.collect_initial_experience_mode = False

    def get_progress_remaining(self) -> float:
        """
        Get the remaining learning progress from the simulation run.

        """
        total_duration = self.end - self.start
        elapsed_duration = self.context.current_timestamp - self.start

        learning_episodes = (
            self.learning_config.training_episodes
            - self.learning_config.episodes_collecting_initial_experience
        )

        if (
            self.episodes_done
            < self.learning_config.episodes_collecting_initial_experience
        ):
            progress_remaining = 1
        else:
            progress_remaining = (
                1
                - (
                    (
                        self.episodes_done
                        - self.learning_config.episodes_collecting_initial_experience
                    )
                    / learning_episodes
                )
                - ((1 / learning_episodes) * (elapsed_duration / total_duration))
            )

        return progress_remaining

    def create_learning_algorithm(self, algorithm: RLAlgorithm):
        """
        Create and initialize the reinforcement learning algorithm.

        This method creates and initializes the reinforcement learning algorithm based on the specified algorithm name. The algorithm
        is associated with the learning role and configured with relevant hyperparameters.

        Args:
            algorithm (RLAlgorithm): The name of the reinforcement learning algorithm.
        """
        if algorithm == "matd3":
            self.rl_algorithm = TD3(learning_role=self)
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

        if (
            self.learning_config.continue_learning is True
            and actors_and_critics is None
        ):
            directory = self.learning_config.trained_policies_load_path
            if directory and Path(directory).is_dir():
                logger.info(f"Loading pretrained policies from {directory}!")
                self.rl_algorithm.load_params(directory)
            else:
                raise FileNotFoundError(
                    f"Directory {directory} does not exist! Cannot load pretrained policies from trained_policies_load_path!"
                )

    def compare_and_save_policies(self, metrics: dict) -> bool:
        """
        Compare evaluation metrics and save policies based on the best achieved performance according to the metrics calculated.

        This method compares the evaluation metrics, such as reward, profit, and regret, and saves the policies if they achieve the
        best performance in their respective categories. It iterates through the specified modes, compares the current evaluation
        value with the previous best, and updates the best value if necessary. If an improvement is detected, it saves the policy
        and associated parameters.

        metrics contain a metric key like "reward" and the current value.
        This function stores the policies with the highest metric.
        So if minimize is required one should add for example "minus_regret" which is then maximized.

        Returns:
            bool: True if early stopping criteria is triggered.

        Note:
            This method is typically used during the evaluation phase to save policies that achieve superior performance.
            Currently the best evaluation metric is still assessed by the development team and preliminary we use the average rewards.
        """
        if not metrics:
            logger.error("tried to save policies but did not get any metrics")
            return
        # if the current values are a new max in one of the metrics - we store them in the default folder

        # add current reward to list of all rewards
        for metric, value in metrics.items():
            self.rl_eval[metric].append(value)

            # check if the current value is the best value
            if self.rl_eval[metric][-1] > self.max_eval[metric]:
                self.max_eval[metric] = self.rl_eval[metric][-1]

                # use first metric as default
                if metric == list(metrics.keys())[0]:
                    # store the best for our current metric in its folder
                    self.rl_algorithm.save_params(
                        directory=f"{self.learning_config.trained_policies_save_path}/{metric}_eval_policies"
                    )

                    logger.info(
                        f"New best policy saved, episode: {self.eval_episodes_done + 1}, {metric=}, value={value:.2f}"
                    )
            else:
                logger.info(
                    f"Current policy not better than best policy, episode: {self.eval_episodes_done + 1}, {metric=}, value={value:.2f}"
                )

            # if we do not see any improvement in the last x evaluation runs we stop the training
            if len(self.rl_eval[metric]) >= self.learning_config.early_stopping_steps:
                self.avg_rewards.append(
                    sum(
                        self.rl_eval[metric][
                            -self.learning_config.early_stopping_steps :
                        ]
                    )
                    / self.learning_config.early_stopping_steps
                )

                if len(self.avg_rewards) >= self.learning_config.early_stopping_steps:
                    recent_rewards = self.avg_rewards[
                        -self.learning_config.early_stopping_steps :
                    ]
                    min_reward = min(recent_rewards)
                    max_reward = max(recent_rewards)

                    # Avoid division by zero or unexpected behavior with negative values
                    denominator = max(
                        abs(min_reward), 1e-8
                    )  # Use small value to avoid zero-division

                    avg_change = abs((max_reward - min_reward) / denominator)

                    if avg_change < self.learning_config.early_stopping_threshold:
                        logger.info(
                            f"Stopping training as no improvement above {self.learning_config.early_stopping_threshold * 100}% in last {self.learning_config.early_stopping_steps} evaluations for {metric}"
                        )
                        if (
                            self.learning_config.learning_rate_schedule
                            or self.learning_config.action_noise_schedule
                        ) is not None:
                            logger.info(
                                f"Learning rate schedule ({self.learning_config.learning_rate_schedule}) or action noise schedule ({self.learning_config.action_noise_schedule}) were scheduled to decay, further learning improvement can be possible. End value of schedule may not have been reached."
                            )

                        self.rl_algorithm.save_params(
                            directory=f"{self.learning_config.trained_policies_save_path}/last_policies"
                        )

                        return True
            return False

    def init_logging(
        self,
        simulation_id: str,
        episode: int,
        eval_episode: int,
        db_uri: str,
        output_agent_addr: str,
        train_start: str,
    ):
        """
        Initialize the logging for the reinforcement learning agent.

        This method initializes the tensor board logger for the reinforcement learning agent.
        It also initializes the parameters required for sending data to the output role.

        Args:
            simulation_id (str): The unique identifier for the simulation.
            episode (int): The current training episode number.
            eval_episode (int): The current evaluation episode number.
            db_uri (str): URI for connecting to the database.
            output_agent_addr (str): The address of the output agent.
            train_start (str): The start time of simulation.
        """

        self.tensor_board_logger = TensorBoardLogger(
            simulation_id=simulation_id,
            db_uri=db_uri,
            learning_mode=self.learning_config.learning_mode,
            evaluation_mode=self.learning_config.evaluation_mode,
            episode=episode,
            eval_episode=eval_episode,
            episodes_collecting_initial_experience=self.learning_config.episodes_collecting_initial_experience,
        )

        # Parameters required for sending data to the output role
        self.db_addr = output_agent_addr

        self.datetime = pd.to_datetime(train_start)

        self.update_steps = 0

    def write_rl_params_to_output(self, cache):
        """
        Sends the current rl_strategy update to the output agent.

        Args:
            products_index (pandas.DatetimeIndex): The index of all products.
            marketconfig (MarketConfig): The market configuration.
        """
        output_agent_list = []

        for unit_id in sorted(cache["obs"][next(iter(cache["obs"]))].keys()):
            starts = cache["obs"].keys()
            for idx, start in enumerate(starts):
                output_dict = {
                    "datetime": start,
                    "unit": unit_id,
                    "reward": cache["rewards"][start][unit_id][0],
                    "regret": cache["regret"][start][unit_id][0],
                    "profit": cache["profit"][start][unit_id][0],
                }

                action_tuple = cache["actions"][start][unit_id][0]

                noise_tuple = cache["noises"][start][unit_id][0]

                if action_tuple is not None:
                    action_dim = len(action_tuple)
                    for i in range(action_dim):
                        output_dict[f"actions_{i}"] = action_tuple[i]
                if noise_tuple is not None:
                    for i in range(len(noise_tuple)):
                        output_dict[f"exploration_noise_{i}"] = noise_tuple[i]

                output_agent_list.append(output_dict)

        if self.db_addr and output_agent_list:
            self.context.schedule_instant_message(
                receiver_addr=self.db_addr,
                content={
                    "context": "write_results",
                    "type": "rl_params",
                    "data": output_agent_list,
                },
            )

    def write_rl_grad_params_to_output(
        self, learning_rate: float, unit_params_list: list[dict]
    ) -> None:
        """
        Writes learning parameters and critic losses to output at specified time intervals.

        This function processes training metrics for each critic over multiple time steps and
        sends them to a database for storage. It tracks the learning rate and critic losses
        across training iterations, associating each record with a timestamp.

        Parameters
        ----------
        learning_rate : float
            The current learning rate used in training.
        unit_params_list : list[dict]
            A list of dictionaries containing critic losses for each time step.
            Each dictionary maps critic names to their corresponding loss values.
        """
        # gradient steps performed in previous training episodes
        gradient_steps_done = (
            max(
                self.episodes_done
                - self.learning_config.episodes_collecting_initial_experience,
                0,
            )
            * int(
                (timestamp2datetime(self.end) - timestamp2datetime(self.start))
                / pd.Timedelta(self.learning_config.train_freq)
            )
            * self.learning_config.gradient_steps
        )

        output_list = [
            {
                "step": gradient_steps_done
                + self.update_steps
                * self.learning_config.gradient_steps  # gradient steps performed in current training episode
                + gradient_step,
                "unit": u_id,
                "actor_loss": params["actor_loss"],
                "actor_total_grad_norm": params["actor_total_grad_norm"],
                "actor_max_grad_norm": params["actor_max_grad_norm"],
                "critic_loss": params["critic_loss"],
                "critic_total_grad_norm": params["critic_total_grad_norm"],
                "critic_max_grad_norm": params["critic_max_grad_norm"],
                "learning_rate": learning_rate,
            }
            for gradient_step in range(self.learning_config.gradient_steps)
            for u_id, params in unit_params_list[gradient_step].items()
        ]

        if self.db_addr:
            self.context.schedule_instant_message(
                receiver_addr=self.db_addr,
                content={
                    "context": "write_results",
                    "type": "rl_grad_params",
                    "data": output_list,
                },
            )

        # Number of network updates (with `self.gradient_steps`) during this episode
        self.update_steps += 1
