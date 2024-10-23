# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch as th
from torch import nn
from torch.nn import functional as F


class CriticTD3(nn.Module):
    """Initialize parameters and build model.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of each state
        act_dim (int): Dimension of each action
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int = 0,
    ):
        super().__init__()

        self.obs_dim = obs_dim + unique_obs_dim * (n_agents - 1)
        self.act_dim = act_dim * n_agents

        # Q1 architecture
        if n_agents <= 50:
            self.FC1_1 = nn.Linear(self.obs_dim + self.act_dim, 512, dtype=float_type)
            self.FC1_2 = nn.Linear(512, 256, dtype=float_type)
            self.FC1_3 = nn.Linear(256, 128, dtype=float_type)
            self.FC1_4 = nn.Linear(128, 1, dtype=float_type)
        else:
            self.FC1_1 = nn.Linear(self.obs_dim + self.act_dim, 1024, dtype=float_type)
            self.FC1_2 = nn.Linear(1024, 512, dtype=float_type)
            self.FC1_3 = nn.Linear(512, 128, dtype=float_type)
            self.FC1_4 = nn.Linear(128, 1, dtype=float_type)

        # Q2 architecture
        if n_agents <= 50:
            self.FC2_1 = nn.Linear(self.obs_dim + self.act_dim, 512, dtype=float_type)
            self.FC2_2 = nn.Linear(512, 256, dtype=float_type)
            self.FC2_3 = nn.Linear(256, 128, dtype=float_type)
            self.FC2_4 = nn.Linear(128, 1, dtype=float_type)
        else:
            self.FC2_1 = nn.Linear(self.obs_dim + self.act_dim, 1024, dtype=float_type)
            self.FC2_2 = nn.Linear(1024, 512, dtype=float_type)
            self.FC2_3 = nn.Linear(512, 128, dtype=float_type)
            self.FC2_4 = nn.Linear(128, 1, dtype=float_type)

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


class CriticPPO(nn.Module):
    """Critic Network for Proximal Policy Optimization (PPO).

    Centralized critic.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of each state
        act_dim (int): Dimension of each action
    """

    def __init__(self, n_agents: int, obs_dim: int, act_dim: int, float_type, unique_obs_dim: int = 0):
        super().__init__()

        self.obs_dim = obs_dim + unique_obs_dim * (n_agents - 1)
        self.act_dim = act_dim * n_agents 

        if n_agents <= 50:
            self.FC_1 = nn.Linear(self.obs_dim + self.act_dim, 512, dtype=float_type)
            self.FC_2 = nn.Linear(512, 256, dtype=float_type)
            self.FC_3 = nn.Linear(256, 128, dtype=float_type)
            self.FC_4 = nn.Linear(128, 1, dtype=float_type)
        else:
            self.FC_1 = nn.Linear(self.obs_dim + self.act_dim, 1024, dtype=float_type)
            self.FC_2 = nn.Linear(1024, 512, dtype=float_type)
            self.FC_3 = nn.Linear(512, 128, dtype=float_type)
            self.FC_4 = nn.Linear(128, 1, dtype=float_type)

    def forward(self, obs, actions):
        """
        Args:
            obs (torch.Tensor): The observations
            actions (torch.Tensor): The actions

        """

        xu = th.cat([obs, actions], dim=-1)

        x = F.relu(self.FC_1(xu))
        x = F.relu(self.FC_2(x))
        x = F.relu(self.FC_3(x))
        value = self.FC_4(x)

        return value

class Actor(nn.Module):
    """
    Parent class for actor networks.
    """

    def __init__(self):
        super().__init__()


class MLPActor(Actor):
    """
    The neurnal network for the MLP actor.
    """

    def __init__(self, obs_dim: int, act_dim: int, float_type, *args, **kwargs):
        super().__init__()

        self.FC1 = nn.Linear(obs_dim, 256, dtype=float_type)
        self.FC2 = nn.Linear(256, 128, dtype=float_type)
        self.FC3 = nn.Linear(128, act_dim, dtype=float_type)

    def forward(self, obs):
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        # Works with MATD3, output of softsign: [-1, 1]
        x = F.softsign(self.FC3(x))
        
        # x = th.tanh(self.FC3(x))

        # Tested for PPO, scales the output to [0, 1] range
        #x = th.sigmoid(self.FC3(x))

        return x


class LSTMActor(Actor):
    """
    The LSTM recurrent neurnal network for the actor.

    Based on "Multi-Period and Multi-Spatial Equilibrium Analysis in Imperfect Electricity Markets"
    by Ye at al. (2019)

    Note: the original source code was not available, therefore this implementation was derived from the published paper.
    Adjustments to resemble final layers from MLPActor:
    - dense layer 2 was omitted
    - single output layer with softsign activation function to output actions directly instead of two output layers for mean and stddev
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int = 0,
        num_timeseries_obs_dim: int = 2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.float_type = float_type
        self.unique_obs_dim = unique_obs_dim
        self.num_timeseries_obs_dim = num_timeseries_obs_dim

        try:
            self.timeseries_len = int(
                (obs_dim - unique_obs_dim) / num_timeseries_obs_dim
            )
        except Exception as e:
            raise ValueError(
                f"Using LSTM but not providing correctly shaped timeseries: Expected integer as unique timeseries length, got {(obs_dim - unique_obs_dim) / num_timeseries_obs_dim} instead."
            ) from e

        self.LSTM1 = nn.LSTMCell(num_timeseries_obs_dim, 8, dtype=float_type)
        self.LSTM2 = nn.LSTMCell(8, 16, dtype=float_type)

        # input size defined by forecast horizon and concatenated with capacity and marginal cost values
        self.FC1 = nn.Linear(self.timeseries_len * 16 + 2, 128, dtype=float_type)
        self.FC2 = nn.Linear(128, act_dim, dtype=float_type)

    def forward(self, obs):
        if obs.dim() not in (1, 2):
            raise ValueError(
                f"LSTMCell: Expected input to be 1D or 2D, got {obs.dim()}D instead"
            )

        is_batched = obs.dim() == 2
        if not is_batched:
            obs = obs.unsqueeze(0)

        x1, x2 = obs.split(
            [obs.shape[1] - self.unique_obs_dim, self.unique_obs_dim], dim=1
        )
        x1 = x1.reshape(-1, self.num_timeseries_obs_dim, self.timeseries_len)

        h_t = th.zeros(x1.size(0), 8, dtype=self.float_type, device=obs.device)
        c_t = th.zeros(x1.size(0), 8, dtype=self.float_type, device=obs.device)

        h_t2 = th.zeros(x1.size(0), 16, dtype=self.float_type, device=obs.device)
        c_t2 = th.zeros(x1.size(0), 16, dtype=self.float_type, device=obs.device)

        outputs = []

        for time_step in x1.split(1, dim=2):
            time_step = time_step.reshape(-1, 2)
            h_t, c_t = self.LSTM1(time_step, (h_t, c_t))
            h_t2, c_t2 = self.LSTM2(h_t, (h_t2, c_t2))
            outputs += [h_t2]

        outputs = th.cat(outputs, dim=1)
        x = th.cat((outputs, x2), dim=1)

        x = F.relu(self.FC1(x))
        x = F.softsign(self.FC2(x))
        # x = th.tanh(self.FC3(x))

        if not is_batched:
            x = x.squeeze(0)

        return x
