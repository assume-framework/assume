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
05_learning_rate: The daily learning rate
06_loss: The average of the losses of all units per day
07_total_grad_norm: The average of the total gradient norm per day
08_max_grad_norm: The maximum gradient norm per day
09_noise": The average of the noises of all units per day

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
    episodes_collecting_initial_experience (int, optional): Number of episodes for initial experience collection. Defaults to 0.
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
        episodes_collecting_initial_experience: int = 0,
    ):
        self.simulation_id = simulation_id
        self.learning_mode = learning_mode
        self.evaluation_mode = evaluation_mode
        self.episodes_collecting_initial_experience = (
            episodes_collecting_initial_experience
        )

        self.writer = None  # Delay creation of SummaryWriter
        self.db_uri = db_uri
        if self.db_uri:
            self.db = create_engine(self.db_uri)

        # get episode number if in learning or evaluation mode
        self.episode = episode if not evaluation_mode else eval_episode

    def update_tensorboard(self):
        """Store episodic evaluation data in tensorboard"""
        if not self.learning_mode or not self.db_uri:
            return

        if self.writer is None:
            self.writer = SummaryWriter(
                log_dir=os.path.join("tensorboard", self.simulation_id)
            )

        mode = "train" if not self.evaluation_mode else "eval"

        # Dynamically detect noise columns in database
        query_columns = (
            "PRAGMA table_info(rl_params)"
            if self.db.dialect.name == "sqlite"
            else """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'rl_params'
        """
        )
        columns_df = pd.read_sql(query_columns, self.db)

        column_names = (
            columns_df["name"].tolist()
            if self.db.dialect.name == "sqlite"
            else columns_df["column_name"].tolist()
        )
        noise_columns = [
            col for col in column_names if col.startswith("exploration_noise_")
        ]
        noise_sql = (
            ", ".join([f"AVG({col}) AS {col}" for col in noise_columns])
            if noise_columns
            else ""
        )

        # Define the SQL query dynamically
        critic_columns = (
            """AVG(critic_loss) AS loss,
            AVG(total_grad_norm) AS total_grad_norm,
            MAX(max_grad_norm) AS max_grad_norm,
            AVG(learning_rate) AS lr
        """
            if mode == "train"
            and self.episode > self.episodes_collecting_initial_experience
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

        # Add critic-related columns (if any)
        if critic_columns:
            query_parts.append(critic_columns)

        # Add noise columns (if any)
        if noise_sql:
            query_parts.append(noise_sql)

        # Build the final SQL query
        query = f"""
            SELECT
                {", ".join(query_parts)}
            FROM rl_params
            WHERE episode = '{self.episode}'
            AND simulation = '{self.simulation_id}'
            AND evaluation_mode = {self.evaluation_mode}
            GROUP BY dt, unit
            ORDER BY dt
        """

        try:
            # Add TensorBoard introduction text only in the first training episode
            if self.episode == 1 and mode == "train":
                self.writer.add_text("TensorBoard Introduction", tensorboard_intro)

            # Load query results into a DataFrame
            df = pd.read_sql(query, self.db)
            df["dt"] = pd.to_datetime(df["dt"])

            # Drop all columns with only NaN values
            df.dropna(axis=1, how="all", inplace=True)

            # adjust noise columns if some were not present in the database and dropped
            noise_columns = [col for col in noise_columns if col in df.columns]

            # Fill missing numeric values
            df.fillna(0.0, inplace=True)

            # Calculate x_index
            datetimes = df["dt"].unique()
            x_index = (self.episode - 1) * len(datetimes)
            if mode == "train":
                x_index -= self.episodes_collecting_initial_experience * len(datetimes)

            # Define metric order explicitly
            metric_order = {
                "02_reward": "reward",
                "03_profit": "profit",
                "04_regret": "regret",
                "05_learning_rate": "learning_rate",
                "06_loss": "loss",
                "07_total_grad_norm": "total_grad_norm",
                "08_max_grad_norm": "max_grad_norm",
                "09_noise": "noise",
            }

            # Group data upfront instead of filtering repeatedly
            grouped_data = df.groupby("dt")

            # Process metrics for each timestamp
            for i, (_, time_df) in enumerate(grouped_data):
                metric_dicts = {
                    "reward": {"avg": time_df["reward"].mean()},
                    "profit": {"avg": time_df["profit"].mean()},
                }

                if mode == "train":
                    # Compute noise dynamically
                    noise_values = [time_df[col].abs().mean() for col in noise_columns]
                    noise_avg = (
                        sum(noise_values) / len(noise_values) if noise_values else 0.0
                    )

                    # Update training-specific metrics
                    metric_dicts.update(
                        {
                            "noise": {"avg": noise_avg},
                            "regret": {"avg": time_df["regret"].mean()}
                            if "regret" in time_df
                            else {"avg": 0.0},
                            "learning_rate": {"avg": time_df["lr"].iat[0]}
                            if "lr" in time_df
                            else {"avg": 0.0},
                            "loss": {"avg": time_df["loss"].mean()}
                            if "loss" in time_df
                            else {"avg": 0.0},
                            "total_grad_norm": {
                                "avg": time_df["total_grad_norm"].mean()
                            }
                            if "total_grad_norm" in time_df
                            else {"avg": 0.0},
                            "max_grad_norm": {"avg": time_df["max_grad_norm"].mean()}
                            if "max_grad_norm" in time_df
                            else {"avg": 0.0},
                        }
                    )

                # Log metrics in the specified order using prefixed names
                for prefixed_name, metric in metric_order.items():
                    if metric in metric_dicts:
                        self.writer.add_scalar(
                            f"{mode}/{prefixed_name}",
                            metric_dicts[metric]["avg"],
                            x_index + i,
                        )

            episode_index = (
                self.episode - self.episodes_collecting_initial_experience
                if mode == "train"
                else self.episode
            )
            # Log episode-level reward
            episode_reward_avg = df.groupby("unit")["reward"].sum().mean()
            self.writer.add_scalar(
                f"{mode}/01_episode_reward",
                episode_reward_avg,
                episode_index,
            )

        except (ProgrammingError, OperationalError, DataError) as db_error:
            logger.error(f"Database error while reading query: {db_error}")
            return
        except Exception as e:
            logger.error(f"Unexpected error in update_tensorboard: {e}")
            return

    def __del__(self):
        """
        Deletes the WriteOutput instance.
        """
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.flush()
            self.writer.close()
