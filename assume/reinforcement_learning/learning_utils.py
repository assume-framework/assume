# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime
from typing import TypedDict

import numpy as np
import torch as th
from torch import nn
from torch.nn import functional as F


class ObsActRew(TypedDict):
    observation: list[th.Tensor]
    action: list[th.Tensor]
    reward: list[th.Tensor]


observation_dict = dict[list[datetime], ObsActRew]


class CriticTD3(nn.Module):
    """Initialize parameters and build model.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of each state
        act_dim (int): Dimension of each action
    """

    def __init__(self, n_agents, obs_dim, act_dim, float_type, unique_obs_len=16):
        super(CriticTD3, self).__init__()

        self.obs_dim = obs_dim  # + unique_obs_len * (n_agents - 1)
        self.act_dim = act_dim * n_agents

        # Q1 architecture
        # if n_agents <= 50:
        self.FC1_1 = nn.Linear(self.obs_dim + self.act_dim, 512, dtype=float_type)
        self.FC1_2 = nn.Linear(512, 256, dtype=float_type)
        self.FC1_3 = nn.Linear(256, 128, dtype=float_type)
        self.FC1_4 = nn.Linear(128, 1, dtype=float_type)
        # else:
        #     self.FC1_1 = nn.Linear(self.obs_dim + self.act_dim, 1024, dtype = float_type)
        #     self.FC1_2 = nn.Linear(1024, 512, dtype = float_type)
        #     self.FC1_3 = nn.Linear(512, 128, dtype = float_type)
        #     self.FC1_4 = nn.Linear(128, 1, dtype = float_type)

        # Q2 architecture
        # if n_agents <= 50:
        self.FC2_1 = nn.Linear(self.obs_dim + self.act_dim, 512, dtype=float_type)
        self.FC2_2 = nn.Linear(512, 256, dtype=float_type)
        self.FC2_3 = nn.Linear(256, 128, dtype=float_type)
        self.FC2_4 = nn.Linear(128, 1, dtype=float_type)
        # else:
        #     self.FC2_1 = nn.Linear(self.obs_dim + self.act_dim, 1024, dtype = float_type)
        #     self.FC2_2 = nn.Linear(1024, 512, dtype = float_type)
        #     self.FC2_3 = nn.Linear(512, 128, dtype = float_type)
        #     self.FC2_4 = nn.Linear(128, 1, dtype = float_type)

    def forward(self, obs, actions):
        """
        Forward pass through the network, from observation to actions.
        """
        xu = th.cat([obs, actions], 1)

        x1 = F.relu(self.FC1_1(xu))
        x1 = F.relu(self.FC1_2(x1))
        x1 = F.relu(self.FC1_3(x1))
        x1 = self.FC1_4(x1)

        x2 = F.relu(self.FC2_1(xu))
        x2 = F.relu(self.FC2_2(x2))
        x2 = F.relu(self.FC2_3(x2))
        x2 = self.FC2_4(x2)

        return x1, x2

    def q1_forward(self, obs, actions):
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).

        Args:
            obs (torch.Tensor): The observations
            actions (torch.Tensor): The actions

        """
        x = th.cat([obs, actions], 1)
        x = F.relu(self.FC1_1(x))
        x = F.relu(self.FC1_2(x))
        x = F.relu(self.FC1_3(x))
        x = self.FC1_4(x)

        return x


class Actor(nn.Module):
    """
    The neurnal network for the actor.
    """

    def __init__(self, obs_dim, act_dim, float_type):
        super(Actor, self).__init__()

        self.FC1 = nn.Linear(obs_dim, 256, dtype=float_type)
        self.FC2 = nn.Linear(256, 128, dtype=float_type)
        self.FC3 = nn.Linear(128, act_dim, dtype=float_type)

    def forward(self, obs):
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        x = F.softsign(self.FC3(x))
        # x = th.tanh(self.FC3(x))

        return x


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
        noise = self.scale * np.random.normal(self.mu, self.sigma, self.act_dimension)
        self.scale = self.dt * self.scale  # if self.scale >= 0.1 else self.scale
        return noise


def polyak_update(params, target_params, tau):
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
