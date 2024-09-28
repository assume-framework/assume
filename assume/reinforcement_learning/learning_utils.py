# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import timedelta
import pandas as pd
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch as th


# TD3 and PPO
class ObsActRew(TypedDict):
    observation: list[th.Tensor]
    action: list[th.Tensor]
    reward: list[th.Tensor]


# TD3 and PPO
observation_dict = dict[list[datetime], ObsActRew]



# TD3
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


# TD3
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
        noise = self.scale * np.random.normal(self.mu, self.sigma, self.act_dimension)
        self.scale = self.dt * self.scale  # if self.scale >= 0.1 else self.scale
        return noise


# TD3
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


# # For non-dynamic PPO buffer size calculation (remove if buffer stays dynamic)
# def convert_to_timedelta(time_str):
#     # Wenn bereits ein Timedelta-Objekt, direkt zurückgeben
#     if isinstance(time_str, pd.Timedelta):
#         return time_str

#     # Extrahiere den Zeitwert und die Einheit aus dem String
#     time_value, time_unit = int(time_str[:-1]), time_str[-1]
    
#     if time_unit == 'h':
#         return timedelta(hours=time_value)
#     elif time_unit == 'd':
#         return timedelta(days=time_value)
#     elif time_unit == 'm':
#         return timedelta(minutes=time_value)
#     else:
#         raise ValueError(f"Unsupported time unit: {time_unit}")

# # For non-dynamic PPO buffer size calculation (remove if buffer stays dynamic)
# def calculate_total_timesteps_per_episode(start_date, end_date, time_step):
#     # Wenn start_date und end_date bereits Timestamps sind, direkt nutzen
#     if isinstance(start_date, str):
#         start_dt = datetime.strptime(start_date, "%Y-%m-%d %H:%M")
#     else:
#         start_dt = start_date

#     if isinstance(end_date, str):
#         end_dt = datetime.strptime(end_date, "%Y-%m-%d %H:%M")
#     else:
#         end_dt = end_date
    
#     # Berechne den gesamten Zeitraum
#     total_time = end_dt - start_dt
    
#     # Konvertiere time_step in ein timedelta-Objekt, wenn es kein Timedelta ist
#     time_step_td = convert_to_timedelta(time_step)
    
#     # Berechne die Gesamtanzahl der Zeitschritte für die gesamte Dauer
#     total_timesteps = total_time // time_step_td
    
#     # print("Total timesteps:")
#     # print(total_timesteps)

#     return total_timesteps


