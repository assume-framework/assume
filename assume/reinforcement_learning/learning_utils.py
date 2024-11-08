# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections.abc import Callable
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch as th


class ObsActRew(TypedDict):
    observation: list[th.Tensor]
    action: list[th.Tensor]
    reward: list[th.Tensor]


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
        # TODO: document changes to normal action noise and different usage of dt parameter?
        noise = (
            self.dt
            * self.scale
            * np.random.normal(self.mu, self.sigma, self.act_dimension)
        )
        # self.scale = self.dt * self.scale  # if self.scale >= 0.1 else self.scale
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


def get_schedule_fn(value_schedule: Schedule | float) -> Schedule:
    """
    Transform (if needed) values (e.g. learning rate, action noise scale, ...) to Schedule function.

    Adapted from SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L80

    :param value_schedule: Constant value of schedule function
    :return: Schedule function (can return constant value)
    """
    # If the passed schedule is a float
    # create a constant function
    if isinstance(value_schedule, float | int):
        # Cast to float to avoid errors
        value_schedule = constant_schedule(float(value_schedule))
    else:
        assert callable(value_schedule)
    # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
    # Some types are have odd behaviors when part of a Schedule, like numpy floats
    return lambda progress_remaining: float(value_schedule(progress_remaining))


def linear_schedule(start: float, end: float = 0, end_fraction: float = 1) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = 1 - ``end_fraction``.

    Adapted from SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L100

    :params start: value to start with if ``progress_remaining`` = 1
    :params end: value to end with if ``progress_remaining`` = 0
    :params end_fraction: fraction of ``progress_remaining``
        where end is reached e.g 0.1 then end is reached after 10%
        of the complete training process.
    :return: Linear schedule function.
    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_schedule(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)

    From SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L124

    :param val: constant value
    :return: Constant schedule function.
    """

    def func(_):
        return val

    return func
