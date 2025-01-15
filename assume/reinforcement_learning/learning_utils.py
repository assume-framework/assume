# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

from collections.abc import Callable
from datetime import datetime
from typing import TypedDict
from sqlalchemy import create_engine
from torch.utils.tensorboard import SummaryWriter
from docs.tensorboard_info import tensor_board_intro
from sqlalchemy.exc import DataError, OperationalError, ProgrammingError

import numpy as np
import torch as th
import pandas as pd
class ObsActRew(TypedDict):
    observation: list[th.Tensor]
    action: list[th.Tensor]
    reward: list[th.Tensor]

logger = logging.getLogger(__name__)

observation_dict = dict[list[datetime], ObsActRew]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, action noise scale ...)
Schedule = Callable[[float], float]


# Ornstein-Uhlenbeck Noise
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    A class that implements Ornstein-Uhlenbeck noise.
    """

    def __init__(self, action_dimension, mu=0, sigma=0.5, theta=0.15, dt=1e-2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.noise_prev = np.zeros(self.action_dimension)
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else np.zeros(self.action_dimension)
        )

    def noise(self):
        noise = (
            self.noise_prev
            + self.theta * (self.mu - self.noise_prev) * self.dt
            + self.sigma
            * np.sqrt(self.dt)
            * np.random.normal(size=self.action_dimension)
        )
        self.noise_prev = noise

        return noise


class NormalActionNoise:
    """
    A gaussian action noise
    """

    def __init__(self, action_dimension, mu=0.0, sigma=0.1, scale=1.0, dt=0.9998):
        self.act_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.dt = dt

    def noise(self):
        noise = (
            self.dt
            * self.scale
            * np.random.normal(self.mu, self.sigma, self.act_dimension)
        )
        return noise

    def update_noise_decay(self, updated_decay: float):
        self.dt = updated_decay


def polyak_update(params, target_params, tau: float):
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    Args:
        params: parameters to use to update the target params
        target_params: parameters to update
        tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            th.add(target_param.data, param.data, alpha=tau, out=target_param.data)


def linear_schedule_func(
    start: float, end: float = 0, end_fraction: float = 1
) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = 1 - ``end_fraction``.

    Args:
        start: value to start with if ``progress_remaining`` = 1
        end: value to end with if ``progress_remaining`` = 0
        end_fraction: fraction of ``progress_remaining``
            where end is reached e.g 0.1 then end is reached after 10%
            of the complete training process.

    Returns:
        Linear schedule function.

    Note:
        Adapted from SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L100

    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_schedule(val: float) -> Schedule:
    """
    Create a function that returns a constant. It is useful for learning rate schedule (to avoid code duplication)

    Args:
        val: constant value
    Returns:
        Constant schedule function.

    Note:
        From SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L124

    """

    def func(_):
        return val

    return func

class TensorBoardLogger:
    """
    Initializes an instance of the TensorBoardLogger class.

    Args:
    learning_mode (bool, optional): Indicates if the simulation is in learning mode. Defaults to False.
    episodes_collecting_initial_experience: Number of episodes collecting initial experience. Defaults to 0.
    tensorboard_path (str, optional): The path for storing tensorboard logs. Defaults to "".
    perform_evaluation (bool, optional): Indicates if the simulation is in evaluation mode. Defaults to False.
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
        self.episodes_collecting_initial_experience = episodes_collecting_initial_experience
        
        self.writer = SummaryWriter(tensorboard_path)
        self.db_uri = db_uri
        if self.db_uri:
            self.db = create_engine(self.db_uri)

        # get episode number if in learning or evaluation mode
        self.episode = None
        if self.learning_mode or self.perform_evaluation:
            episode = self.simulation_id.split("_")[-1]
            if episode.isdigit():
                self.episode = int(episode)

    def update_tensorboard (self):
        """
        Store episodic evalution data in tensorboard
        """
        if self.episode and self.learning_mode:
            if self.episode == 1 and not self.perform_evaluation:
                self.writer.add_text("TensorBoard Introduction", tensor_board_intro)
            query = f"""
            SELECT
                datetime as dt,
                unit,
                profit,
                reward,
                regret,
                critic_loss as loss,
                learning_rate as lr,
                exploration_noise_0 as noise_0,
                exploration_noise_1 as noise_1
            FROM rl_params
            WHERE episode = '{self.episode}'
            AND simulation = '{self.simulation_id}'
            AND perform_evaluation = False
            AND initial_exploration = False
            """
            try:
                rl_params_df = pd.read_sql(query, self.db)
                rl_params_df["dt"] = pd.to_datetime(rl_params_df["dt"])
                # replace all NaN values with 0 to allow for plotting
                rl_params_df = rl_params_df.fillna(0.0)

                # loop over all datetimes as tensorboard does not allow to store time series
                datetimes = rl_params_df["dt"].unique()
                
                for i, time in enumerate(datetimes):
                    time_df = rl_params_df[rl_params_df["dt"] == time]
                    
                    # efficient implementation for unit specific metrics  
                    metrics = ['reward', 'profit', 'regret', 'loss']
                    dicts = {
                        metric: {**time_df.set_index('unit')[metric].to_dict(), 
                                'avg': time_df[metric].mean()}
                        for metric in metrics
                    }

                    # calculate the average noise
                    noise_0 = time_df["noise_0"].abs().mean()
                    noise_1 = time_df["noise_1"].abs().mean()

                    # get the learning rate
                    lr = time_df["lr"].values[0]

                    # store the data in tensorboard
                    x_index = (
                        self.episode - 1 - self.episodes_collecting_initial_experience
                    ) * len(datetimes) + i
                    self.writer.add_scalars("a) reward", dicts['reward'], x_index)
                    self.writer.add_scalars("b) profit", dicts['profit'], x_index)
                    self.writer.add_scalars("c) regret", dicts['regret'], x_index)
                    self.writer.add_scalars("d) loss", dicts['loss'], x_index)
                    self.writer.add_scalar("e) learning rate", lr, x_index)
                    self.writer.add_scalar("f) noise_0", noise_0, x_index)
                    self.writer.add_scalar("g) noise_1", noise_1, x_index)
            except (ProgrammingError, OperationalError, DataError):
                return
            except Exception as e:
                logger.error("could not read query: %s", e)
                return