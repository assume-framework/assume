# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from mango import Role

from assume.common.base import (
    LearningConfig,
    LearningStrategy,
    is_off_policy,
    is_on_policy,
)
from assume.common.utils import (
    create_rrule,
    datetime2timestamp,
    timestamp2datetime,
)
from assume.reinforcement_learning.algorithms.base_algorithm import RLAlgorithm
from assume.reinforcement_learning.algorithms.matd3 import TD3
from assume.reinforcement_learning.algorithms.maddpg import DDPG
from assume.reinforcement_learning.algorithms.mappo import PPO
from assume.reinforcement_learning.buffer import (
    ReplayBuffer, 
    RolloutBuffer
)
from assume.reinforcement_learning.learning_utils import (
    linear_schedule_func,
    transform_buffer_data,
)
from assume.reinforcement_learning.tensorboard_logger import TensorBoardLogger

logger = logging.getLogger(__name__)


class Learning(Role):
    """Manages the learning process of reinforcement learning agents.
    
    This class handles the initialization of key components such as neural networks,
    replay buffer, and learning hyperparameters. It handles both training and evaluation 
    modes based on the provided learning configuration.

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

        # Single buffer that can be either ReplayBuffer (off-policy) or RolloutBuffer (on-policy)
        self.buffer = None
        self.episodes_done = 0
        self.rl_strats: dict[int, LearningStrategy] = {}
        self.learning_config = learning_config
        self.critics = {}
        self.target_critics = {}

        device = "cpu"
        if self.learning_config:
            if "cuda" in self.learning_config.device and th.cuda.is_available():
                device = self.learning_config.device
            elif (
                "mps" in self.learning_config.device and th.backends.mps.is_available()
            ):
                device = self.learning_config.device
        self.device = th.device(device)

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
            # Only set up noise schedule for off-policy algorithms
            if is_off_policy(self.learning_config.algorithm):
                if self.learning_config.off_policy.action_noise_schedule == "linear":
                    self.calc_noise_from_progress = linear_schedule_func(
                        self.learning_config.off_policy.noise_dt
                    )
                else:
                    self.calc_noise_from_progress = lambda x: self.learning_config.off_policy.noise_dt
            # For on-policy algorithms, no noise schedule needed

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
            # PPO algorithm specific caches for on-policy learning
            self.all_values = defaultdict(lambda: defaultdict(list))
            self.all_log_probs = defaultdict(lambda: defaultdict(list))
            self.all_dones = defaultdict(lambda: defaultdict(list))

    def on_ready(self):
        """Set up the learning role for reinforcement learning training.

        Note:
            This method prepares the learning role for the reinforcement learning training process.
            It subscribes to relevant messages for handling the training process and schedules 
            recurrent tasks for policy updates based on the specified training frequency.
            This cannot happen in the init since the context (compare mango agents) is not 
            yet available there. To avoid inconsistent replay buffer states (e.g. observation 
            and action has been stored but not the reward), this slightly shifts the timing 
            of the buffer updates.
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
        """Ensure self.train_freq evenly divides the simulation length.
        
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
        assert total_length >= train_freq, (
            f"Simulation length ({total_length}) must be at least as long as train_freq ({train_freq_str})"
        )
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
        """Compute and validate validation_interval.

        Returns:
            validation_interval (int)
        Raises:
            ValueError if training_episodes is too small.
        """
        default_interval = self.learning_config.validation_episodes_interval
        training_episodes = self.learning_config.training_episodes
        validation_interval = min(training_episodes, default_interval)

        # Only check initial experience episodes for off-policy algorithms
        if is_off_policy(self.learning_config.algorithm):
            min_required_episodes = (
                self.learning_config.off_policy.episodes_collecting_initial_experience
                + validation_interval
            )
            
            if self.learning_config.training_episodes < min_required_episodes:
                raise ValueError(
                    f"Training episodes ({training_episodes}) must be greater than the sum of initial experience episodes ({self.learning_config.off_policy.episodes_collecting_initial_experience}) and evaluation interval ({validation_interval})."
                )
        else:
            # For on-policy algorithms, no initial experience collection needed
            min_required_episodes = validation_interval
            
            if self.learning_config.training_episodes < min_required_episodes:
                raise ValueError(
                    f"Training episodes ({training_episodes}) must be greater than evaluation interval ({validation_interval})."
                )

        return validation_interval

    def register_strategy(self, strategy: LearningStrategy) -> None:
        """Register a learning strategy with this learning role.

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
        # PPO specific caches
        current_values = self.all_values
        current_log_probs = self.all_log_probs
        current_dones = self.all_dones

        # Reset cache dicts immediately with new defaultdicts
        self.all_obs = defaultdict(lambda: defaultdict(list))
        self.all_actions = defaultdict(lambda: defaultdict(list))
        self.all_rewards = defaultdict(lambda: defaultdict(list))
        self.all_noises = defaultdict(lambda: defaultdict(list))
        self.all_regrets = defaultdict(lambda: defaultdict(list))
        self.all_profits = defaultdict(lambda: defaultdict(list))
        # PPO specific resets
        self.all_values = defaultdict(lambda: defaultdict(list))
        self.all_log_probs = defaultdict(lambda: defaultdict(list))
        self.all_dones = defaultdict(lambda: defaultdict(list))

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
                "values": {t: current_values[t] for t in timestamps_to_process},
                "log_probs": {t: current_log_probs[t] for t in timestamps_to_process},
                "dones": {t: current_dones[t] for t in timestamps_to_process}
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
        """Process strategy data into the buffer and trigger policy update.
        
        This function takes all the information that the strategies wrote into the 
        learning_role cache dicts and post-processes them to fit into the buffer.
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

        # Add data to buffer - type depends on algorithm category
        if is_on_policy(self.learning_config.algorithm):
            # For on-policy algorithms (PPO/MAPPO), use RolloutBuffer
            for timestamp in sorted(cache["obs"].keys()):
                obs_data = transform_buffer_data(
                    {
                        timestamp: cache["obs"][timestamp]
                    },
                    device
                )
                actions_data = transform_buffer_data(
                    {
                        timestamp: cache["actions"][timestamp]
                    },
                    device
                )
                rewards_data = transform_buffer_data(
                    {
                        timestamp: cache["rewards"][timestamp]
                    },
                    device
                )
                
                values_data = transform_buffer_data(
                    {
                        timestamp: cache["values"][timestamp]
                    },
                    device
                )
                
                log_probs_data = transform_buffer_data(
                    {
                        timestamp: cache["log_probs"][timestamp]
                    },
                    device
                )

                dones_data = transform_buffer_data(
                    {
                        timestamp: cache["dones"][timestamp]
                    },
                    device
                )

                # Add to rollout buffer
                self.buffer.add(
                    obs = obs_data,
                    action = actions_data,
                    reward = rewards_data,
                    done = dones_data,
                    value = values_data,
                    log_prob = log_probs_data
                )
        else:
            # For off-policy algorithms (TD3/DDPG), use ReplayBuffer
            # rewrite dict so that obs.shape == (n_rl_units, obs_dim) and sorted by keys and store in buffer
            self.buffer.add(
                obs = transform_buffer_data(cache["obs"], device),
                actions = transform_buffer_data(cache["actions"], device),
                reward = transform_buffer_data(cache["rewards"], device),
            )

        # Only update policy after initial experience for off-policy algorithms
        if is_off_policy(self.learning_config.algorithm):
            if (
                self.episodes_done
                >= self.learning_config.off_policy.episodes_collecting_initial_experience
            ):
                self.rl_algorithm.update_policy()
        else:
            # For on-policy algorithms, update policy immediately
            self.rl_algorithm.update_policy()

    def add_observation_to_cache(self, unit_id, start, observation) -> None:
        """Add the observation to the cache dict, per unit_id.

        Args:
            unit_id (str): The id of the unit.
            start: The start time.
            observation (torch.Tensor): The observation to be added.

        """
        self.all_obs[start][unit_id].append(observation)

    def add_actions_to_cache(self, unit_id, start, action, noise) -> None:
        """Add the action and noise to the cache dict, per unit_id.

        Args:
            unit_id (str): The id of the unit.
            start: The start time.
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
        """Add the reward to the cache dict, per unit_id.

        Args:
            unit_id: The id of the unit.
            start: The start time.
            reward: The reward to be added.
            regret: The regret to be added.
            profit: The profit to be added.
        """
        self.all_rewards[start][unit_id].append(reward)
        self.all_regrets[start][unit_id].append(regret)
        self.all_profits[start][unit_id].append(profit)

    def add_ppo_data_to_cache(
        self,
        unit_id,
        start,
        value,
        log_prob,
        done=False
    ) -> None:
        """Add PPO specific data to the cache dict, per unit_id.

        Args:
            unit_id: The id of the unit.
            start: The start time.
            value: The value estimate V(s) from the critic.
            log_prob: The log probability of the action.
            done: Whether a terminal state or not.
        """
        self.all_values[start][unit_id].append(value)
        self.all_log_probs[start][unit_id].append(log_prob)
        self.all_dones[start][unit_id].append(float(done))

    def load_inter_episodic_data(self, inter_episodic_data):
        """Load the inter-episodic data from the dict stored across simulation runs.

        Args:
            inter_episodic_data: The inter-episodic data to be loaded.
        """
        self.episodes_done = inter_episodic_data["episodes_done"]
        self.eval_episodes_done = inter_episodic_data["eval_episodes_done"]
        self.max_eval = inter_episodic_data["max_eval"]
        self.rl_eval = inter_episodic_data["all_eval"]
        self.avg_rewards = inter_episodic_data["avg_all_eval"]
        self.buffer = inter_episodic_data["buffer"]

        self.initialize_policy(inter_episodic_data["actors_and_critics"])

        # Disable initial exploration if initial experience collection is complete
        # Only for off-policy algorithms
        if is_off_policy(self.learning_config.algorithm):
            if (
                self.episodes_done
                >= self.learning_config.off_policy.episodes_collecting_initial_experience
            ):
                self.turn_off_initial_exploration()
        # For on-policy algorithms, no initial exploration to disable

        # In continue_learning mode, disable it only for loaded strategies
        elif self.learning_config.continue_learning:
            self.turn_off_initial_exploration(loaded_only=True)

    def get_inter_episodic_data(self):
        """Dump the inter-episodic data to a dict for storing across simulation runs.

        Returns:
            The inter-episodic data to be stored.
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
        """Disable initial exploration mode.

        If `loaded_only=True`, only turn off exploration for strategies that were loaded (used in continue_learning mode).
        If `loaded_only=False`, turn it off for all strategies.

        Args:
            loaded_only: Whether to disable exploration only for loaded strategies.
        """
        for strategy in self.rl_strats.values():
            if loaded_only:
                if strategy.actor.loaded:
                    strategy.collect_initial_experience_mode = False
            else:
                strategy.collect_initial_experience_mode = False

    def get_progress_remaining(self) -> float:
        """Get the remaining learning progress from the simulation run.
        
        Returns:
            The remaining progress as a float between 0 and 1.
        """
        total_duration = self.end - self.start
        elapsed_duration = self.context.current_timestamp - self.start

        # Only calculate progress for off-policy algorithms
        if is_off_policy(self.learning_config.algorithm):
            initial_experience_episodes = self.learning_config.off_policy.episodes_collecting_initial_experience
            learning_episodes = (
                self.learning_config.training_episodes
                - initial_experience_episodes
            )

            if (
                self.episodes_done
                < initial_experience_episodes
            ):
                progress_remaining = 1
            else:
                progress_remaining = (
                    1
                    - (
                        (
                            self.episodes_done
                            - initial_experience_episodes
                        )
                        / learning_episodes
                    )
                    - ((1 / learning_episodes) * (elapsed_duration / total_duration))
                )
        else:
            # For on-policy algorithms, simpler progress calculation
            total_episodes = self.learning_config.training_episodes
            progress_remaining = 1 - (self.episodes_done / total_episodes) - (elapsed_duration / total_duration)

        return progress_remaining

    def create_learning_algorithm(self, algorithm: RLAlgorithm):
        """Create and initialize the reinforcement learning algorithm.

        This method creates and initializes the reinforcement learning algorithm based on 
        the specified algorithm name. The algorithm is associated with the learning role 
        and configured with relevant hyperparameters.

        Args:
            algorithm: The name of the reinforcement learning algorithm.
        """
        if algorithm == "matd3":
            self.rl_algorithm = TD3(learning_role=self)
        elif algorithm == "maddpg":
            self.rl_algorithm = DDPG(learning_role=self)
        elif algorithm == "mappo":
            self.rl_algorithm = PPO(learning_role=self)
        else:
            logger.error(f"Learning algorithm {algorithm} not implemented!")

    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Initialize the policy of the reinforcement learning agent considering the respective algorithm.

        This method initializes the policy (actor) of the reinforcement learning agent. It 
        tests if we want to continue the learning process with stored policies from a former 
        training process. If so, it loads the policies from the specified directory. 
        Otherwise, it initializes the respective new policies.
        
        Args:
            actors_and_critics: The pre-initialized actor and critic policies.
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
        """Compare evaluation metrics and save best performing policies.

        This method compares the evaluation metrics, such as reward, profit, and regret, 
        and saves the policies if they achieve the best performance in their respective 
        categories. It iterates through the specified modes, compares the current evaluation
        value with the previous best, and updates the best value if necessary. If an improvement 
        is detected, it saves the policy and associated parameters.

        Metrics contain a metric key like "reward" and the current value. This function 
        stores the policies with the highest metric. If minimize is required, one should 
        add for example "minus_regret" which is then maximized.

        Args:
            metrics: Dictionary of metrics evaluated.
            
        Returns:
            True if early stopping criteria is triggered, False otherwise.

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
                            or self.learning_config.off_policy.action_noise_schedule
                        ) is not None:
                            logger.info(
                                f"Learning rate schedule ({self.learning_config.learning_rate_schedule}) or action noise schedule ({self.learning_config.off_policy.action_noise_schedule}) were scheduled to decay, further learning improvement can be possible. End value of schedule may not have been reached."
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
        """Initialize the logging for the reinforcement learning agent.

        This method initializes the tensor board logger for the reinforcement learning agent.
        It also initializes the parameters required for sending data to the output role.

        Args:
            simulation_id: The unique identifier for the simulation.
            episode: The current training episode number.
            eval_episode: The current evaluation episode number.
            db_uri: URI for connecting to the database.
            output_agent_addr: The address of the output agent.
            train_start: The start time of simulation.
        """

        self.tensor_board_logger = TensorBoardLogger(
            simulation_id=simulation_id,
            db_uri=db_uri,
            learning_mode=self.learning_config.learning_mode,
            evaluation_mode=self.learning_config.evaluation_mode,
            episode=episode,
            eval_episode=eval_episode,
            episodes_collecting_initial_experience=(
                self.learning_config.off_policy.episodes_collecting_initial_experience
                if is_off_policy(self.learning_config.algorithm)
                else 0
            ),
        )

        # Parameters required for sending data to the output role
        self.db_addr = output_agent_addr

        self.datetime = pd.to_datetime(train_start)

        self.update_steps = 0

    def write_rl_params_to_output(self, cache):
        """Sends the current rl_strategy update to the output agent.

        Args:
            cache: The data cache from the strategies.
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
        """Writes learning parameters and critic losses to output at specified intervals.

        This function processes training metrics for each critic over multiple time steps and
        sends them to a database for storage. It tracks the learning rate and critic losses
        across training iterations, associating each record with a timestamp.

        Args:
            learning_rate: The current learning rate used in training.
            unit_params_list: A list of dictionaries containing critic losses for each 
                time step (mapping critic names to their losses in dict).
        """
        # gradient steps performed in previous training episodes
        if is_off_policy(self.learning_config.algorithm):
            gradient_steps_done = (
                max(
                    self.episodes_done
                    - self.learning_config.off_policy.episodes_collecting_initial_experience,
                    0,
                )
                * int(
                    (timestamp2datetime(self.end) - timestamp2datetime(self.start))
                    / pd.Timedelta(self.learning_config.train_freq)
                )
                * self.learning_config.off_policy.gradient_steps
            )
            current_gradient_steps = self.learning_config.off_policy.gradient_steps
        else:
            # For on-policy, no gradient steps concept - use 1 for calculation purposes
            gradient_steps_done = 0
            current_gradient_steps = 1

        # Handle different parameter structures for on-policy vs off-policy
        if self.learning_config.algorithm == "mappo":
            # For PPO/MAPPO: unit_params_list length equals actual update steps
            actual_gradient_steps = len(unit_params_list)
            gradient_step_range = range(actual_gradient_steps)
            # For on-policy, use simple step counting
            base_step = self.update_steps * actual_gradient_steps
        else:
            # For off-policy: use configured gradient_steps
            actual_gradient_steps = self.learning_config.off_policy.gradient_steps
            gradient_step_range = range(actual_gradient_steps)
            
            # gradient steps performed in previous training episodes
            gradient_steps_done = (
                max(
                    self.episodes_done
                    - self.learning_config.off_policy.episodes_collecting_initial_experience,
                    0,
                )
                * int(
                    (timestamp2datetime(self.end) - timestamp2datetime(self.start))
                    / pd.Timedelta(self.learning_config.train_freq)
                )
                * self.learning_config.off_policy.gradient_steps
            )
            base_step = gradient_steps_done + self.update_steps * actual_gradient_steps

        output_list = [
            {
                "step": base_step + gradient_step,
                "unit": u_id,
                "actor_loss": params["actor_loss"],
                "actor_total_grad_norm": params["actor_total_grad_norm"],
                "actor_max_grad_norm": params["actor_max_grad_norm"],
                "critic_loss": params["critic_loss"],
                "critic_total_grad_norm": params["critic_total_grad_norm"],
                "critic_max_grad_norm": params["critic_max_grad_norm"],
                "learning_rate": learning_rate,
            }
            for gradient_step in gradient_step_range
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
