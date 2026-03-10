# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections import defaultdict
from datetime import datetime
from functools import partial
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

# Default level name used for backward compatibility (single LearningConfig)
DEFAULT_LEVEL = "units"


def _new_cache():
    """Create a fresh nested defaultdict for experience caching."""
    return defaultdict(lambda: defaultdict(list))


class LevelView:
    """
    Provides a per-level view of the Learning role.

    Strategies and RL algorithms receive a LevelView instead of the full Learning role,
    so they transparently access only their level's data (strats, buffer, config, caches)
    while shared data (episodes, device, timestamps) is delegated to the actual Learning role.

    This means TD3 (and other RL algorithms) need NO code changes — they interact with
    LevelView exactly as they did with the old single-level Learning role.
    """

    def __init__(self, learning_role: "LearningRole", level: str):
        object.__setattr__(self, "_role", learning_role)
        object.__setattr__(self, "_level", level)

    @property
    def level(self) -> str:
        return self._level

    # --- Per-level properties ---

    @property
    def rl_strats(self):
        return self._role.rl_strats[self._level]

    @property
    def buffer(self):
        return self._role.buffer[self._level]

    @buffer.setter
    def buffer(self, value):
        self._role.buffer[self._level] = value

    @property
    def learning_config(self):
        return self._role.learning_config[self._level]

    @property
    def calc_lr_from_progress(self):
        return self._role.calc_lr_from_progress[self._level]

    @property
    def calc_noise_from_progress(self):
        return self._role.calc_noise_from_progress[self._level]

    @property
    def update_steps(self):
        return self._role.update_steps[self._level]

    @update_steps.setter
    def update_steps(self, value):
        self._role.update_steps[self._level] = value

    # --- Per-level methods (called by strategies) ---

    def register_strategy(self, strategy: LearningStrategy) -> None:
        self._role.rl_strats[self._level][strategy.unit_id] = strategy

    def add_observation_to_cache(self, unit_id, start, observation) -> None:
        self._role.all_obs[self._level][start][unit_id].append(observation)

    def add_actions_to_cache(self, unit_id, start, action, noise) -> None:
        if unit_id == 0 or unit_id is None:
            logger.warning(
                f"Got invalid unit_id while storing learning experience: {unit_id}"
            )
            return
        self._role.all_actions[self._level][start][unit_id].append(action)
        self._role.all_noises[self._level][start][unit_id].append(noise)

    def add_reward_to_cache(self, unit_id, start, reward, regret, profit) -> None:
        self._role.all_rewards[self._level][start][unit_id].append(reward)
        self._role.all_regrets[self._level][start][unit_id].append(regret)
        self._role.all_profits[self._level][start][unit_id].append(profit)

    def write_rl_grad_params_to_output(self, learning_rate, unit_params_list) -> None:
        """Forward to Learning role with level context."""
        self._role.write_rl_grad_params_to_output(
            learning_rate, unit_params_list, level=self._level
        )

    # --- Delegate everything else to the actual Learning role ---

    def __getattr__(self, name):
        return getattr(self._role, name)


class LearningRole(Role):
    """
    This class manages the learning process of reinforcement learning agents across
    multiple levels (e.g. "units", "units_operator", "markets", ...). Each level has its
    own strategies, replay buffer, RL algorithm, and experience caches. Episode counters
    and evaluation metrics are shared across levels.

    The set of levels is determined dynamically from the keys of the learning_config dict.

    Args:
        learning_config (dict[str, LearningConfig] | LearningConfig): Per-level learning
            configurations keyed by arbitrary level names. A single LearningConfig is
            accepted for backward compatibility and is treated as DEFAULT_LEVEL ("units").
        start (datetime.datetime): The start datetime for the simulation.
        end (datetime.datetime): The end datetime for the simulation.
        shared_config (LearningConfig | None): Shared configuration for episode-level
            settings (training_episodes, episodes_collecting_initial_experience,
            validation_episodes_interval, early_stopping_*, device, continue_learning).
            If None, falls back to the first per-level config (backward compatibility).
    """

    def __init__(
        self,
        learning_config: dict[str, LearningConfig] | LearningConfig,
        start: datetime,
        end: datetime,
        shared_config: LearningConfig | None = None,
    ):
        super().__init__()

        # Accept single LearningConfig for backward compatibility
        if isinstance(learning_config, LearningConfig):
            learning_config = {DEFAULT_LEVEL: learning_config}
        self.learning_config: dict[str, LearningConfig] = learning_config

        # Shared config holds episode-level settings (training_episodes, early_stopping, etc.)
        # Falls back to first per-level config for backward compatibility.
        self.shared_config: LearningConfig = (
            shared_config
            if shared_config is not None
            else next(iter(self.learning_config.values()))
        )

        # The set of levels is derived from the config keys
        self.levels: tuple[str, ...] = tuple(self.learning_config.keys())

        # Shared episode counters
        self.episodes_done = 0

        # Per-level: strategies, buffers, algorithms
        self.rl_strats: dict[str, dict[int, LearningStrategy]] = {
            level: {} for level in self.levels
        }
        self.buffer: dict[str, ReplayBuffer | None] = {
            level: None for level in self.levels
        }
        self.rl_algorithm: dict[str, RLAlgorithm] = {}
        self.critics = {level: {} for level in self.levels}
        self.target_critics = {level: {} for level in self.levels}

        # Level views: pass these to strategies/algorithms instead of `self`
        self.level_views: dict[str, LevelView] = {
            level: LevelView(self, level) for level in self.levels
        }

        # Device setup (from shared config)
        device = "cpu"
        if self.shared_config:
            if "cuda" in self.shared_config.device and th.cuda.is_available():
                device = self.shared_config.device
            elif "mps" in self.shared_config.device and th.backends.mps.is_available():
                device = self.shared_config.device
        self.device = th.device(device)

        # future: add option to choose between float16 and float32
        self.float_type = th.float

        th.backends.cuda.matmul.allow_tf32 = True
        th.backends.cudnn.allow_tf32 = True

        self.start = datetime2timestamp(start)
        self.start_datetime = start
        self.end = datetime2timestamp(end)
        self.end_datetime = end

        self.datetime = None

        # Check if any level has learning_mode enabled
        any_learning_mode = any(
            cfg.learning_mode for cfg in self.learning_config.values()
        )

        if any_learning_mode:
            # Per-level: learning rate schedules, noise schedules, algorithms
            self.calc_lr_from_progress: dict[str, callable] = {}
            self.calc_noise_from_progress: dict[str, callable] = {}

            for level, cfg in self.learning_config.items():
                if not cfg.learning_mode:
                    continue

                if cfg.learning_rate_schedule == "linear":
                    self.calc_lr_from_progress[level] = linear_schedule_func(
                        cfg.learning_rate
                    )
                else:
                    self.calc_lr_from_progress[level] = (
                        lambda x, lr=cfg.learning_rate: lr
                    )

                if cfg.action_noise_schedule == "linear":
                    self.calc_noise_from_progress[level] = linear_schedule_func(
                        cfg.noise_dt
                    )
                else:
                    self.calc_noise_from_progress[level] = lambda x, nd=cfg.noise_dt: nd

                # Create RL algorithm for this level (receives a LevelView)
                self.create_learning_algorithm(cfg.algorithm, level)

            self.eval_episodes_done = 0

            # Shared evaluation tracking
            self.max_eval = defaultdict(lambda: -1e9)
            self.rl_eval = defaultdict(list)
            self.avg_rewards = []

            self.tensor_board_logger = None
            self.update_steps: dict[str, int | None] = {
                level: None for level in self.levels
            }

            # Per-level experience cache dicts
            # We use atomic-swaps later to avoid overwrites while writing into the buffer.
            self.all_obs = {level: _new_cache() for level in self.levels}
            self.all_actions = {level: _new_cache() for level in self.levels}
            self.all_noises = {level: _new_cache() for level in self.levels}
            self.all_rewards = {level: _new_cache() for level in self.levels}
            self.all_regrets = {level: _new_cache() for level in self.levels}
            self.all_profits = {level: _new_cache() for level in self.levels}

    @property
    def active_levels(self) -> list[str]:
        """Return levels that have registered strategies."""
        return [level for level in self.levels if self.rl_strats[level]]

    def get_level_view(self, level: str) -> LevelView:
        """
        Get the LevelView for a given level.

        Pass this to strategies and RL algorithms instead of the Learning role itself,
        so they transparently access only their level's data.

        Args:
            level (str): The level name (must be a key in learning_config).

        Returns:
            LevelView: The per-level view.
        """
        if level not in self.level_views:
            raise KeyError(
                f"Unknown level '{level}'. Available levels: {list(self.levels)}"
            )
        return self.level_views[level]

    def on_ready(self):
        """
        Set up the learning role for reinforcement learning training.

        Schedules a recurrent buffer-update task for each active level based on its train_freq.
        This cannot happen in __init__ since the mango context is not yet available there.
        """
        super().on_ready()

        for level in self.active_levels:
            cfg = self.learning_config[level]
            shifted_start = self.start_datetime + pd.Timedelta(cfg.train_freq)

            recurrency_task = create_rrule(
                start=shifted_start,
                end=self.end_datetime,
                freq=cfg.train_freq,
            )

            self.context.schedule_recurrent_task(
                partial(self.store_to_buffer_and_update, level=level),
                recurrency_task,
                src="no_wait",
            )

    def sync_train_freq_with_simulation_horizon(self) -> dict[str, str | None]:
        """
        Ensure train_freq evenly divides the simulation length for each active level.
        If not, adjust train_freq (in-place) and return the (possibly adjusted) values.

        Returns:
            dict[str, str | None]: The train_freq per level (None if not in learning mode).
        """
        result = {}
        for level, cfg in self.learning_config.items():
            if not cfg.learning_mode:
                result[level] = None
                continue

            train_freq_str = str(cfg.train_freq)
            try:
                train_freq = pd.Timedelta(train_freq_str)
            except Exception:
                logger.warning(
                    f"Invalid train_freq '{train_freq_str}' for level '{level}' — skipping adjustment."
                )
                result[level] = None
                continue

            total_length = self.end_datetime - self.start_datetime
            assert total_length >= train_freq, (
                f"Simulation length ({total_length}) must be at least as long as "
                f"train_freq ({train_freq_str}) for level '{level}'"
            )
            quotient, remainder = divmod(total_length, train_freq)

            if remainder != pd.Timedelta(0):
                n_intervals = int(quotient) + 1
                new_train_freq_hours = int(
                    (total_length / n_intervals).total_seconds() / 3600
                )
                new_train_freq_str = f"{new_train_freq_hours}h"
                cfg.train_freq = new_train_freq_str

                logger.warning(
                    f"Simulation length ({total_length}) is not divisible by train_freq "
                    f"({train_freq_str}) for level '{level}'. "
                    f"Adjusting train_freq to {new_train_freq_str}."
                )

            result[level] = cfg.train_freq

        return result

    def determine_validation_interval(self) -> int:
        """
        Compute and validate validation_interval.

        Uses episode-related config from the first level (these should be identical across levels).

        Returns:
            validation_interval (int)
        Raises:
            ValueError if training_episodes is too small.
        """
        default_interval = self.shared_config.validation_episodes_interval
        training_episodes = self.shared_config.training_episodes
        validation_interval = min(training_episodes, default_interval)

        min_required_episodes = (
            self.shared_config.episodes_collecting_initial_experience
            + validation_interval
        )

        if self.shared_config.training_episodes < min_required_episodes:
            raise ValueError(
                f"Training episodes ({training_episodes}) must be greater than the sum "
                f"of initial experience episodes ({self.shared_config.episodes_collecting_initial_experience}) "
                f"and evaluation interval ({validation_interval})."
            )

        return validation_interval

    def register_strategy(
        self, strategy: LearningStrategy, level: str = DEFAULT_LEVEL
    ) -> None:
        """
        Register a learning strategy with this learning role at the specified level.

        Args:
            strategy (LearningStrategy): The learning strategy to register.
            level (str): The level name (must be a key in learning_config).
        """
        self.rl_strats[level][strategy.unit_id] = strategy

    async def store_to_buffer_and_update(self, level: str) -> None:
        """
        Collect cached experience for one level, store in buffer, and trigger policy update.

        Args:
            level (str): The level to process.
        """
        # Atomic dict operations - create new references
        current_obs = self.all_obs[level]
        current_actions = self.all_actions[level]
        current_rewards = self.all_rewards[level]
        current_noises = self.all_noises[level]
        current_regrets = self.all_regrets[level]
        current_profits = self.all_profits[level]

        # Reset cache dicts immediately with new defaultdicts
        self.all_obs[level] = _new_cache()
        self.all_actions[level] = _new_cache()
        self.all_rewards[level] = _new_cache()
        self.all_noises[level] = _new_cache()
        self.all_regrets[level] = _new_cache()
        self.all_profits[level] = _new_cache()

        # Get timestamps from cache we took
        all_timestamps = sorted(current_obs.keys())
        if len(all_timestamps) > 1:
            # Identify all incomplete timesteps (no reward yet)
            incomplete_timestamps = [
                ts for ts in all_timestamps if ts not in current_rewards
            ]

            # Process only complete timesteps
            timestamps_to_process = [
                ts for ts in all_timestamps if ts not in incomplete_timestamps
            ]
            # Carry over incomplete timesteps to new cache dicts
            for ts in incomplete_timestamps:
                self.all_obs[level][ts] = current_obs[ts]
                self.all_actions[level][ts] = current_actions[ts]
                self.all_noises[level][ts] = current_noises[ts]

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
            self.write_rl_params_to_output(cache, level)

            # if we are training also update the policy and write data into buffer
            cfg = self.learning_config[level]
            if not cfg.evaluation_mode:
                await self._store_to_buffer_and_update_sync(cache, self.device, level)
        else:
            logger.warning(
                f"No experience retrieved to store in buffer at update step for level '{level}'!"
            )

    async def _store_to_buffer_and_update_sync(self, cache, device, level: str) -> None:
        """
        Post-process cached experience for one level into the replay buffer
        and trigger the next policy update.
        """
        strats = self.rl_strats[level]
        first_start = next(iter(cache["obs"]))
        for name, buffer in [
            ("observations", cache["obs"]),
            ("actions", cache["actions"]),
            ("rewards", cache["rewards"]),
        ]:
            if len(buffer[first_start]) != len(strats):
                logger.error(
                    f"Number of unit_ids with {name} in learning role level '{level}' "
                    f"({len(buffer[first_start])}) does not match number of rl_strats "
                    f"({len(strats)}). It seems like some learning_instances are not "
                    "reporting experience. Cannot store to buffer and update policy!"
                )
                return

        # rewrite dict so that obs.shape == (n_rl_units, obs_dim) and store in buffer
        self.buffer[level].add(
            obs=transform_buffer_data(cache["obs"], device, strats.keys()),
            actions=transform_buffer_data(cache["actions"], device, strats.keys()),
            reward=transform_buffer_data(cache["rewards"], device, strats.keys()),
        )

        cfg = self.learning_config[level]
        if self.episodes_done >= cfg.episodes_collecting_initial_experience:
            self.rl_algorithm[level].update_policy()

    def add_observation_to_cache(
        self, unit_id, start, observation, level: str = DEFAULT_LEVEL
    ) -> None:
        """
        Add the observation to the cache dict for the given level.

        Args:
            unit_id (str): The id of the unit.
            observation (torch.Tensor): The observation to be added.
            level (str): The level name.
        """
        self.all_obs[level][start][unit_id].append(observation)

    def add_actions_to_cache(
        self, unit_id, start, action, noise, level: str = DEFAULT_LEVEL
    ) -> None:
        """
        Add the action and noise to the cache dict for the given level.

        Args:
            unit_id (str): The id of the unit.
            action (torch.Tensor): The action to be added.
            noise (torch.Tensor): The noise to be added.
            level (str): The level name.
        """
        if unit_id == 0 or unit_id is None:
            logger.warning(
                f"Got invalid unit_id while storing learning experience: {unit_id}"
            )
            return

        self.all_actions[level][start][unit_id].append(action)
        self.all_noises[level][start][unit_id].append(noise)

    def add_reward_to_cache(
        self, unit_id, start, reward, regret, profit, level: str = DEFAULT_LEVEL
    ) -> None:
        """
        Add the reward to the cache dict for the given level.

        Args:
            unit_id (str): The id of the unit.
            reward (float): The reward to be added.
            regret (float): The regret to be added.
            profit (float): The profit to be added.
            level (str): The level name.
        """
        self.all_rewards[level][start][unit_id].append(reward)
        self.all_regrets[level][start][unit_id].append(regret)
        self.all_profits[level][start][unit_id].append(profit)

    def load_inter_episodic_data(self, inter_episodic_data):
        """
        Load the inter-episodic data from the dict stored across simulation runs.

        Supports both the new per-level format (buffer/actors_and_critics keyed by level)
        and the old flat format for backward compatibility.

        Args:
            inter_episodic_data (dict): The inter-episodic data to be loaded.
        """
        self.episodes_done = inter_episodic_data["episodes_done"]
        self.eval_episodes_done = inter_episodic_data["eval_episodes_done"]
        self.max_eval = inter_episodic_data["max_eval"]
        self.rl_eval = inter_episodic_data["all_eval"]
        self.avg_rewards = inter_episodic_data["avg_all_eval"]

        # Load per-level buffers
        buffers = inter_episodic_data.get("buffer", {})
        if isinstance(buffers, ReplayBuffer):
            # Backward compat: single buffer -> assign to first active level
            first_level = (
                self.active_levels[0] if self.active_levels else self.levels[0]
            )
            self.buffer[first_level] = buffers
        elif isinstance(buffers, dict):
            for level in self.levels:
                if level in buffers:
                    self.buffer[level] = buffers[level]

        # Load per-level policies
        ac = inter_episodic_data.get("actors_and_critics")
        if ac is not None and "actors" in ac:
            # Backward compat: old flat format -> wrap to first active level
            first_level = (
                self.active_levels[0] if self.active_levels else self.levels[0]
            )
            ac = {first_level: ac}
        self.initialize_policy(ac)

        # Disable initial exploration if initial experience collection is complete
        if (
            self.episodes_done
            >= self.shared_config.episodes_collecting_initial_experience
        ):
            self.turn_off_initial_exploration()

        # In continue_learning mode, disable it only for loaded strategies
        elif self.shared_config.continue_learning:
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
            "buffer": {level: self.buffer[level] for level in self.levels},
            "actors_and_critics": {
                level: self.rl_algorithm[level].extract_policy()
                for level in self.active_levels
            },
        }

    def turn_off_initial_exploration(self, loaded_only=False) -> None:
        """
        Disable initial exploration mode for all active levels.

        If `loaded_only=True`, only turn off exploration for strategies that were loaded
        (used in continue_learning mode).
        If `loaded_only=False`, turn it off for all strategies at all active levels.

        Args:
            loaded_only (bool): Whether to disable exploration only for loaded strategies.
        """
        for level in self.active_levels:
            for strategy in self.rl_strats[level].values():
                if loaded_only:
                    if strategy.actor.loaded:
                        strategy.collect_initial_experience_mode = False
                else:
                    strategy.collect_initial_experience_mode = False

    def get_progress_remaining(self) -> float:
        """
        Get the remaining learning progress from the simulation run.
        Uses shared episode counters and shared_config for episode settings.
        """
        total_duration = self.end - self.start
        elapsed_duration = self.context.current_timestamp - self.start

        learning_episodes = (
            self.shared_config.training_episodes
            - self.shared_config.episodes_collecting_initial_experience
        )

        if (
            self.episodes_done
            < self.shared_config.episodes_collecting_initial_experience
        ):
            progress_remaining = 1
        else:
            progress_remaining = (
                1
                - (
                    (
                        self.episodes_done
                        - self.shared_config.episodes_collecting_initial_experience
                    )
                    / learning_episodes
                )
                - ((1 / learning_episodes) * (elapsed_duration / total_duration))
            )

        return progress_remaining

    def create_learning_algorithm(self, algorithm: str, level: str):
        """
        Create and initialize the reinforcement learning algorithm for a given level.

        The algorithm receives a LevelView so it transparently accesses only its level's
        strategies, buffer, and config — no changes needed in the algorithm code.

        Args:
            algorithm (str): The name of the reinforcement learning algorithm.
            level (str): The level name (must be a key in learning_config).
        """
        if algorithm == "matd3":
            self.rl_algorithm[level] = TD3(learning_role=self.get_level_view(level))
        else:
            logger.error(f"Learning algorithm {algorithm} not implemented!")

    def initialize_policy(self, actors_and_critics: dict = None) -> None:
        """
        Initialize the policies for all active levels.

        Args:
            actors_and_critics (dict | None): Per-level dict of actor/critic networks,
                keyed by level name. None to initialize fresh networks.
        """
        for level in self.active_levels:
            level_ac = None
            if actors_and_critics and level in actors_and_critics:
                level_ac = actors_and_critics[level]

            self.rl_algorithm[level].initialize_policy(level_ac)

            cfg = self.learning_config[level]
            if cfg.continue_learning is True and level_ac is None:
                directory = cfg.trained_policies_load_path
                if directory and Path(directory).is_dir():
                    logger.info(
                        f"Loading pretrained policies for level '{level}' from {directory}!"
                    )
                    self.rl_algorithm[level].load_params(directory)
                else:
                    raise FileNotFoundError(
                        f"Directory {directory} does not exist! Cannot load pretrained "
                        f"policies for level '{level}' from trained_policies_load_path!"
                    )

    def compare_and_save_policies(self, metrics: dict) -> bool:
        """
        Compare evaluation metrics and save policies for all active levels based on best
        achieved performance.

        Returns:
            bool: True if early stopping criteria is triggered.
        """
        if not metrics:
            logger.error("tried to save policies but did not get any metrics")
            return False

        for metric, value in metrics.items():
            self.rl_eval[metric].append(value)

            # check if the current value is the best value
            if self.rl_eval[metric][-1] > self.max_eval[metric]:
                self.max_eval[metric] = self.rl_eval[metric][-1]

                # use first metric as default
                if metric == list(metrics.keys())[0]:
                    # store the best for all active levels
                    for level in self.active_levels:
                        cfg = self.learning_config[level]
                        self.rl_algorithm[level].save_params(
                            directory=f"{cfg.trained_policies_save_path}/{metric}_eval_policies"
                        )

                    logger.info(
                        f"New best policy saved, episode: {self.eval_episodes_done + 1}, {metric=}, value={value:.2f}"
                    )
            else:
                logger.info(
                    f"Current policy not better than best policy, episode: {self.eval_episodes_done + 1}, {metric=}, value={value:.2f}"
                )

            # if we do not see any improvement in the last x evaluation runs we stop the training
            if len(self.rl_eval[metric]) >= self.shared_config.early_stopping_steps:
                self.avg_rewards.append(
                    sum(
                        self.rl_eval[metric][-self.shared_config.early_stopping_steps :]
                    )
                    / self.shared_config.early_stopping_steps
                )

                if len(self.avg_rewards) >= self.shared_config.early_stopping_steps:
                    recent_rewards = self.avg_rewards[
                        -self.shared_config.early_stopping_steps :
                    ]
                    min_reward = min(recent_rewards)
                    max_reward = max(recent_rewards)

                    # Avoid division by zero or unexpected behavior with negative values
                    denominator = max(
                        abs(min_reward), 1e-8
                    )  # Use small value to avoid zero-division

                    avg_change = abs((max_reward - min_reward) / denominator)

                    if avg_change < self.shared_config.early_stopping_threshold:
                        logger.info(
                            f"Stopping training as no improvement above "
                            f"{self.shared_config.early_stopping_threshold * 100}% in last "
                            f"{self.shared_config.early_stopping_steps} evaluations for {metric}"
                        )
                        if (
                            self.shared_config.learning_rate_schedule
                            or self.shared_config.action_noise_schedule
                        ) is not None:
                            logger.info(
                                f"Learning rate schedule ({self.shared_config.learning_rate_schedule}) "
                                f"or action noise schedule ({self.shared_config.action_noise_schedule}) "
                                f"were scheduled to decay, further learning improvement can "
                                f"be possible. End value of schedule may not have been reached."
                            )

                        for level in self.active_levels:
                            cfg = self.learning_config[level]
                            self.rl_algorithm[level].save_params(
                                directory=f"{cfg.trained_policies_save_path}/last_policies"
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
            learning_mode=self.shared_config.learning_mode,
            evaluation_mode=self.shared_config.evaluation_mode,
            episode=episode,
            eval_episode=eval_episode,
            episodes_collecting_initial_experience=self.shared_config.episodes_collecting_initial_experience,
        )

        # Parameters required for sending data to the output role
        self.db_addr = output_agent_addr

        self.datetime = pd.to_datetime(train_start)

        self.update_steps = {level: 0 for level in self.levels}

    def write_rl_params_to_output(self, cache, level: str = DEFAULT_LEVEL):
        """
        Sends the current rl_strategy update to the output agent for the given level.

        Args:
            cache (dict): The cached experience data.
            level (str): The level name.
        """
        output_agent_list = []

        for unit_id in sorted(cache["obs"][next(iter(cache["obs"]))].keys()):
            starts = cache["obs"].keys()
            for idx, start in enumerate(starts):
                output_dict = {
                    "datetime": start,
                    "unit": unit_id,
                    "level": level,
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
        self,
        learning_rate: float,
        unit_params_list: list[dict],
        level: str = DEFAULT_LEVEL,
    ) -> None:
        """
        Writes learning parameters and critic losses to output for a given level.

        Parameters
        ----------
        learning_rate : float
            The current learning rate used in training.
        unit_params_list : list[dict]
            A list of dictionaries containing critic losses for each time step.
        level : str
            The level name.
        """
        cfg = self.learning_config[level]

        # gradient steps performed in previous training episodes
        gradient_steps_done = (
            max(
                self.episodes_done - cfg.episodes_collecting_initial_experience,
                0,
            )
            * int(
                (timestamp2datetime(self.end) - timestamp2datetime(self.start))
                / pd.Timedelta(cfg.train_freq)
            )
            * cfg.gradient_steps
        )

        output_list = [
            {
                "step": gradient_steps_done
                + self.update_steps[level]
                * cfg.gradient_steps  # gradient steps performed in current training episode
                + gradient_step,
                "unit": u_id,
                "level": level,
                "actor_loss": params["actor_loss"],
                "actor_total_grad_norm": params["actor_total_grad_norm"],
                "actor_max_grad_norm": params["actor_max_grad_norm"],
                "critic_loss": params["critic_loss"],
                "critic_total_grad_norm": params["critic_total_grad_norm"],
                "critic_max_grad_norm": params["critic_max_grad_norm"],
                "learning_rate": learning_rate,
            }
            for gradient_step in range(cfg.gradient_steps)
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

        # Number of network updates (with `gradient_steps`) during this episode
        self.update_steps[level] += 1
