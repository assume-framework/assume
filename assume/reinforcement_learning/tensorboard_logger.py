# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import DataError, OperationalError, ProgrammingError
from torch.utils.tensorboard import SummaryWriter

# Turn off TF onednn optimizations to avoid memory leaks
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logger = logging.getLogger(__name__)


tensorboard_intro = """
# TensorBoard Guide: Parameter Visualization and Interpretation

Welcome to the TensorBoard visualization interface for the reinforcement learning training process. This guide will help you navigate and interpret the displayed parameters effectively.

## Accessing the Data

To view the training and evaluation results, navigate to the "SCALARS" page in TensorBoard. Here you'll find various metrics tracked during the training process, visualized as interactive plots.

## Available Parameters

The following parameters are being tracked and displayed:

### Training Metrics:
01_episode_reward: The sum of the rewards per episode averaged over all units
02_reward: The sum of the rewards per day averaged over all units
03_profit: The sum of the profits per day averaged over all units
04_regret: The sum of the regrets per day averaged over all units
05_noise: The average of the noises of all units per day
06_learning_rate: The learning rate over gradient steps
07_actor_loss: The average of the losses of all units per gradient step
08_actor_total_grad_norm: The average of the total gradient norm per gradient step
09_actor_max_grad_norm: The maximum gradient norm per gradient step
10_critic_loss: The average of the losses of all units per gradient step
11_critic_total_grad_norm: The average of the total gradient norm per gradient step
12_critic_max_grad_norm: The maximum gradient norm per gradient step


For the training metrics, the episode indices are displayed as negative values during the initial exploration phase,
as the results are random due to the explorative nature of the RL algorithm.

### Evaluation Metrics:
01_episode_reward: The sum of the rewards per episode averaged over all units
02_reward: The sum of the rewards per day averaged over all units
03_profit: The sum of the profits per day averaged over all units

## Visualization Settings

For optimal visualization of the training progress:

- Set the smoothing parameter to 0,999 for all metrics except the learning rate
- For the learning rate visualization, set smoothing to 0.0 to see the exact values
- The x-axis represents time in hours, displayed as consecutive integers over the episodes
- Data display begins after the initial exploration phase, as early results are random due to the exploration nature of the RL algorithm

## Interactive Features

The TensorBoard interface offers various interactive features to help you analyze the data:

- Zoom functionality for detailed inspection of specific time periods
- Clickable data points for detailed value inspection
- Additional data information available in the upper left corner of each plot
- Customizable display options for better visualization

## Interpreting the Results

To effectively analyze the training progress, focus on the learning trends and performance improvements over time:

- Monitor how reward, regret, and other metrics evolve over time
- Look for positive trends such as increasing rewards or decreasing regret
- Check whether the learning rate and noise parameters follow the trend you set in the config file

The data presentation is designed to help you track the algorithm's learning progress and performance improvements over time. Use the interactive features to focus on specific aspects or time periods of interest.
"""


class TensorBoardLogger:
    """
    Initializes an instance of the TensorBoardLogger class.

    Args:
    db_uri (str): URI for connecting to the database.
    simulation_id (str): The unique identifier for the simulation.
    tensorboard_path (str, optional): Path for storing tensorboard logs.
    learning_mode (bool, optional): Whether the simulation is in learning mode. Defaults to False.
    episodes_collecting_initial_experience (dict[str, int] | int, optional): Per-level number of
        episodes for initial experience collection, keyed by level name. A plain int is accepted
        for backward compatibility and applied to all levels. Defaults to 0.
    evaluation_mode (bool, optional): Whether the simulation is in evaluation mode. Defaults to False.

    """

    def __init__(
        self,
        db_uri: str,
        simulation_id: str,
        learning_mode: bool = False,
        evaluation_mode: bool = False,
        episode: int = 1,
        eval_episode: int = 1,
        episodes_collecting_initial_experience: dict[str, int] | int = 0,
    ):
        self.simulation_id = simulation_id
        self.learning_mode = learning_mode
        self.evaluation_mode = evaluation_mode
        # Accept both a per-level dict and a plain int (backward compat)
        if isinstance(episodes_collecting_initial_experience, int):
            self._episodes_collecting_initial_experience_default = (
                episodes_collecting_initial_experience
            )
            self.episodes_collecting_initial_experience: dict[str, int] = {}
        else:
            self._episodes_collecting_initial_experience_default = 0
            self.episodes_collecting_initial_experience = (
                episodes_collecting_initial_experience
            )

        self.writers: dict[str, SummaryWriter] = {}  # per-level writers
        self.db_uri = db_uri
        if self.db_uri:
            self.db = create_engine(self.db_uri)

        # get episode number if in learning or evaluation mode
        self.episode = episode if not evaluation_mode else eval_episode

    def _get_writer(self, level: str) -> SummaryWriter:
        """Get or create a SummaryWriter for the given level."""
        if level not in self.writers:
            self.writers[level] = SummaryWriter(
                log_dir=os.path.join("tensorboard", self.simulation_id, level)
            )
        return self.writers[level]

    def _get_rl_levels(self) -> list[str]:
        """Discover all levels from the level column in the rl_params table."""
        try:
            df = pd.read_sql(
                f"SELECT DISTINCT level FROM rl_params WHERE simulation = '{self.simulation_id}'",
                self.db,
            )
            levels = df["level"].tolist()
        except Exception:
            levels = []
        return levels if levels else ["units"]  # fallback for backward compat

    def _get_table_columns(self, table: str) -> list[str]:
        """Get column names for a table."""
        query_columns = (
            f"PRAGMA table_info({table})"
            if self.db.dialect.name == "sqlite"
            else f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = '{table}'
        """
        )
        columns_df = pd.read_sql(query_columns, self.db)
        return (
            columns_df["name"].tolist()
            if self.db.dialect.name == "sqlite"
            else columns_df["column_name"].tolist()
        )

    def update_tensorboard(self):
        """Store episodic evaluation data in tensorboard, per level."""
        if not self.learning_mode or not self.db_uri:
            return

        mode = "02_train" if not self.evaluation_mode else "01_eval"

        levels = self._get_rl_levels()

        for level in levels:
            self._update_tensorboard_for_level(level, mode)

    def _get_initial_experience_episodes(self, level: str) -> int:
        """Return the initial-experience episode count for level."""
        return self.episodes_collecting_initial_experience.get(
            level, self._episodes_collecting_initial_experience_default
        )

    def _update_tensorboard_for_level(self, level: str, mode: str):
        """Store episodic evaluation data for a single level."""
        rl_params_table = "rl_params"
        rl_grad_table = "rl_grad_params"
        writer = self._get_writer(level)

        ##############################
        # Values per simulation step #
        ##############################
        try:
            column_names = self._get_table_columns(rl_params_table)
        except Exception:
            logger.warning(
                f"Table {rl_params_table} not found, skipping level '{level}'"
            )
            return

        noise_columns = [
            col for col in column_names if col.startswith("exploration_noise_")
        ]
        noise_sql = (
            ", ".join([f"AVG({col}) AS {col}" for col in noise_columns])
            if noise_columns
            else ""
        )

        date_func = (
            "strftime('%Y-%m-%d', datetime)"
            if self.db.dialect.name == "sqlite"
            else "TO_CHAR(datetime, 'YYYY-MM-DD')"
        )

        # Dynamically build the SQL query, ensuring proper commas
        query_parts = [
            f"{date_func} AS dt",
            "unit",
            "SUM(profit) AS profit",
            "SUM(reward) AS reward",
        ]

        # Add regret column if it exists
        if "regret" in column_names:
            query_parts.append("SUM(regret) AS regret")

        # Add noise columns (if any)
        if noise_sql:
            query_parts.append(noise_sql)

        # Build the final SQL queries
        query_sim = f"""
            SELECT
                {", ".join(query_parts)}
            FROM {rl_params_table}
            WHERE episode = '{self.episode}'
            AND simulation = '{self.simulation_id}'
            AND evaluation_mode = {self.evaluation_mode}
            AND level = '{level}'
            GROUP BY dt, unit
            ORDER BY dt
        """

        ##############################
        #  Values per gradient step  #
        ##############################

        # Define the SQL query dynamically
        rl_grad_columns = """
            AVG(step) AS step,
            AVG(actor_loss) AS actor_loss,
            AVG(actor_total_grad_norm) AS actor_total_grad_norm,
            MAX(actor_max_grad_norm) AS actor_max_grad_norm,
            AVG(critic_loss) AS critic_loss,
            AVG(critic_total_grad_norm) AS critic_total_grad_norm,
            MAX(critic_max_grad_norm) AS critic_max_grad_norm,
            AVG(learning_rate) AS learning_rate
        """

        query_grad = (
            f"""
            SELECT {rl_grad_columns}
            FROM {rl_grad_table}
            WHERE episode = '{self.episode}'
            AND simulation = '{self.simulation_id}'
            AND evaluation_mode = {self.evaluation_mode}
            AND level = '{level}'
            GROUP BY step, unit
            ORDER BY step
            """
            if mode == "02_train"
            and self.episode > self._get_initial_experience_episodes(level)
            else None
        )

        try:
            # Add TensorBoard introduction text only in the first training episode
            if self.episode == 1 and mode == "02_train":
                writer.add_text("TensorBoard Introduction", tensorboard_intro)

            ##############################
            # Values per simulation step #
            ##############################
            # Load query results into a DataFrame
            df_sim = pd.read_sql(query_sim, self.db)
            df_sim["dt"] = pd.to_datetime(df_sim["dt"])

            # Drop all columns with only NaN values
            df_sim.dropna(axis=1, how="all", inplace=True)

            # adjust noise columns if some were not present in the database and dropped
            noise_columns = [col for col in noise_columns if col in df_sim.columns]

            # Fill missing numeric values
            df_sim.fillna(0.0, inplace=True)

            # Calculate x_index
            datetimes = df_sim["dt"].unique()
            x_index = (self.episode - 1) * len(datetimes)
            if mode == "02_train":
                x_index -= self._get_initial_experience_episodes(level) * len(datetimes)

            # Define metric order explicitly
            metric_order_sim = {
                "02_reward": "reward",
                "03_profit": "profit",
                "04_regret": "regret",
                "05_noise": "noise",
            }

            # Group data upfront instead of filtering repeatedly
            grouped_data_sim = df_sim.groupby("dt")

            # Pre-aggregate noise means if relevant (column-wise mean)
            if mode == "02_train" and noise_columns:
                noise_means = df_sim[noise_columns].abs().groupby(df_sim["dt"]).mean()
                noise_avg_per_dt = noise_means.mean(axis=1)
            else:
                noise_avg_per_dt = {}

            # Process metrics for each timestamp
            for i, (dt, time_df) in enumerate(grouped_data_sim):
                metric_dicts_sim = {
                    "reward": {"avg": time_df["reward"].mean()},
                    "profit": {"avg": time_df["profit"].mean()},
                }

                if mode == "02_train":
                    metric_dicts_sim["noise"] = {"avg": noise_avg_per_dt.get(dt, 0.0)}
                    metric_dicts_sim["regret"] = (
                        {"avg": time_df["regret"].mean()}
                        if "regret" in df_sim.columns
                        else {"avg": 0.0}
                    )

                # Log metrics in the specified order using prefixed names
                for prefixed_name, metric in metric_order_sim.items():
                    if metric in metric_dicts_sim:
                        writer.add_scalar(
                            f"{mode}/{prefixed_name}",
                            metric_dicts_sim[metric]["avg"],
                            x_index + i,
                        )

            ##############################
            #  Values per gradient step  #
            ##############################
            # Load query results into a DataFrame
            if query_grad:
                df_grad = pd.read_sql(query_grad, self.db)

                # Define metric order explicitly
                metric_order_grad = {
                    "06_learning_rate": "learning_rate",
                    "07_actor_loss": "actor_loss",
                    "08_actor_total_grad_norm": "actor_total_grad_norm",
                    "09_actor_max_grad_norm": "actor_max_grad_norm",
                    "10_critic_loss": "critic_loss",
                    "11_critic_total_grad_norm": "critic_total_grad_norm",
                    "12_critic_max_grad_norm": "critic_max_grad_norm",
                }

                # Group data upfront instead of filtering repeatedly
                grouped_data_grad = df_grad.groupby("step")

                if mode == "02_train":
                    # Precompute grouped means only for available columns
                    columns_to_group = ["step"] + [
                        col
                        for col in metric_order_grad.values()
                        if col in df_grad.columns
                    ]
                    grouped_data_grad = df_grad[columns_to_group].groupby("step").mean()

                    # Prepare default value dictionary for missing metrics
                    default_values = {col: 0.0 for col in metric_order_grad.values()}

                # Process metrics for each timestamp
                for step, row in grouped_data_grad.iterrows():
                    for prefixed_name, metric_col in metric_order_grad.items():
                        value = row.get(metric_col, default_values[metric_col])
                        writer.add_scalar(f"03_grad/{prefixed_name}", value, step)

                # Handle missing steps (i.e., if a column was missing entirely)
                all_steps = df_grad["step"].unique()
                logged_steps = grouped_data_grad.index.values
                missing_steps = set(all_steps) - set(logged_steps)

                for step in missing_steps:
                    for prefixed_name, metric_col in metric_order_grad.items():
                        writer.add_scalar(f"03_grad/{prefixed_name}", 0.0, step)

            episode_index = (
                self.episode - self._get_initial_experience_episodes(level)
                if mode == "02_train"
                else self.episode
            )
            # Log episode-level reward
            episode_reward_avg = df_sim.groupby("unit")["reward"].sum().mean()
            writer.add_scalar(
                f"{mode}/01_episode_reward",
                episode_reward_avg,
                episode_index,
            )

        except (ProgrammingError, OperationalError, DataError) as db_error:
            logger.error(f"Database error for level '{level}': {db_error}")
            return
        except Exception as e:
            logger.error(
                f"Unexpected error in update_tensorboard for level '{level}': {e}"
            )
            return

    def __del__(self):
        """
        Cleans up all SummaryWriter instances.
        """
        for writer in getattr(self, "writers", {}).values():
            writer.flush()
            writer.close()
