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
a) Reward
b) Profit
c) Regret
d) Loss
e) Learning Rate
f) Noise Parameters

### Evaluation Metrics:
a) Reward
b) Profit

## Visualization Settings

For optimal visualization of the training progress:

- Set the smoothing parameter to 0,999 for all metrics except the learning rate
- For the learning rate visualization, set smoothing to 0.0 to see the exact values
- The x-axis represents time in hours, displayed as consecutive integers over the episodes
- Data display begins after the initial exploration phase, as early results are random due to the exploration nature of the RL algorithm

## Using Regex Filters

To focus on specific metrics or units, use the regex filter option ion the left of the SCALARS page. This allows you to display only the data you're interested in, making it easier to analyze the results.
You can filter by unit name, display only averaged values or filter different study cases.

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
    perform_evaluation (bool, optional): Whether the simulation is in evaluation mode. Defaults to False.

    """

    def __init__(
        self,
        db_uri: str,
        simulation_id: str,
        tensorboard_path: str = "logs/tensorboard",
        learning_mode: bool = False,
        episodes_collecting_initial_experience: int = 0,
        perform_evaluation: bool = False,
    ):
        self.simulation_id = simulation_id
        self.learning_mode = learning_mode
        self.perform_evaluation = perform_evaluation
        self.episodes_collecting_initial_experience = (
            episodes_collecting_initial_experience
        )

        self.tensorboard_path = tensorboard_path
        self.writer = None  # Delay creation of SummaryWriter
        self.db_uri = db_uri
        if self.db_uri:
            self.db = create_engine(self.db_uri)

        # get episode number if in learning or evaluation mode
        self.episode = None
        if self.learning_mode or self.perform_evaluation:
            episode = self.simulation_id.split("_")[-1]
            if episode.isdigit():
                self.episode = int(episode)

    def update_tensorboard(self):
        """Store episodic evaluation data in tensorboard"""
        if not (self.episode and self.learning_mode):
            return

        if self.writer is None:
            sim_id = self.simulation_id.replace("_eval", "").rsplit("_", 1)[0]
            self.writer = SummaryWriter(log_dir=os.path.join("tensorboard", sim_id))

        mode = "train" if not self.perform_evaluation else "eval"

        # columns specific to the training mode.
        train_columns = """,
            AVG(regret) AS regret,
            AVG(critic_loss) AS loss,
            AVG(total_grad_norm) AS total_grad_norm,
            MAX(max_grad_norm) AS max_grad_norm,
            AVG(learning_rate) AS lr,
            AVG(exploration_noise_0) AS noise_0,
            AVG(exploration_noise_1) AS noise_1
            """

        # Use appropriate date function and parameter style based on database type
        if self.db.dialect.name == "sqlite":
            # To aggregate by hour instead of day, replace '%Y-%m-%d' with '%Y-%m-%d %H'
            # To aggregate by month instead of day, replace '%Y-%m-%d' with '%Y-%m'
            date_func = "strftime('%Y-%m-%d', datetime)"
        elif self.db.dialect.name == "postgresql":
            # To aggregate by hour instead of day, replace 'YYYY-MM-DD' with 'YYYY-MM-DD HH24'
            # To aggregate by month instead of day, replace 'YYYY-MM-DD' with 'YYYY-MM'
            date_func = "TO_CHAR(datetime, 'YYYY-MM-DD')"

        query = f"""
            SELECT
                {date_func} AS dt,
                unit,
                AVG(profit) AS profit,
                AVG(reward) AS reward
                {train_columns if mode == 'train' else ''}
            FROM rl_params
            WHERE episode = '{self.episode}'
            AND simulation = '{self.simulation_id}'
            AND perform_evaluation = {self.perform_evaluation}
            {'AND initial_exploration = False' if mode == 'train' else ''}
            GROUP BY dt, unit
            ORDER BY dt
        """
        try:
            # Add intro text for first episode
            if self.episode == 1 and mode == "train":
                self.writer.add_text("TensorBoard Introduction", tensorboard_intro)

            # Process dataframe
            df = pd.read_sql(query, self.db)
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.fillna(0.0)

            # Calculate x_index
            datetimes = df["dt"].unique()
            x_index = (self.episode - 1) * len(datetimes)
            if mode == "train":
                x_index -= self.episodes_collecting_initial_experience * len(datetimes)

            # Define the order of metrics explicitly with prefixes
            metric_order = {
                "01_reward": "reward",
                "02_profit": "profit",
                "03_regret": "regret",
                "04_learning_rate": "learning_rate",
                "05_loss": "loss",
                "06_total_grad_norm": "total_grad_norm",
                "07_max_grad_norm": "max_grad_norm",
                "08_noise": "noise",
            }

            # Process metrics for each timestamp
            for i, time in enumerate(datetimes):
                time_df = df[df["dt"] == time]

                # Define and compute metrics
                metric_dicts = {
                    "reward": {"avg": time_df["reward"].mean()},
                    "profit": {"avg": time_df["profit"].mean()},
                }

                if mode == "train":
                    # Dynamically detect noise columns
                    noise_columns = [
                        col for col in time_df.columns if col.startswith("noise_")
                    ]
                    noise_avg = (
                        sum(time_df[col].abs().mean() for col in noise_columns)
                        / len(noise_columns)
                        if noise_columns
                        else 0.0
                    )

                    metric_dicts.update(
                        {
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
                            "noise": {
                                "avg": noise_avg
                            },  # Dynamically computed noise average
                        }
                    )

                # Log metrics in the specified order using prefixed names
                for prefixed_name, metric in metric_order.items():
                    if (
                        metric in metric_dicts
                    ):  # Ensure the metric exists before logging
                        self.writer.add_scalar(
                            f"{mode}/{prefixed_name}",
                            metric_dicts[metric]["avg"],
                            x_index + i,
                        )

        except (ProgrammingError, OperationalError, DataError):
            return
        except Exception as e:
            logger.error("could not read query: %s", e)
            return

    def __del__(self):
        """
        Deletes the WriteOutput instance.
        """
        if hasattr(self, "writer") and self.writer is not None:
            self.writer.flush()
            self.writer.close()
