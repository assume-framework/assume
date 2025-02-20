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

        # Select proper architecture based on `n_agents`
        if n_agents <= 20:
            hidden_sizes = [256, 128]  # Shallow network for small `n_agents`
        elif n_agents <= 50:
            hidden_sizes = [512, 256, 128]  # Medium network
        else:
            hidden_sizes = [1024, 512, 256, 128]  # Deeper network for large `n_agents`

        # First Q-network (Q1)
        self.q1_layers = self._build_q_network(hidden_sizes, float_type)

        # Second Q-network (Q2) for double Q-learning
        self.q2_layers = self._build_q_network(hidden_sizes, float_type)

        # Initialize weights properly
        self._init_weights()

    def _build_q_network(self, hidden_sizes, float_type):
        """
        Dynamically creates a Q-network given the chosen hidden layer sizes.
        """
        layers = nn.ModuleList()
        input_dim = (
            self.obs_dim + self.act_dim
        )  # Input includes all observations and actions

        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h, dtype=float_type))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1, dtype=float_type))  # Output Q-value

        return layers

    def _init_weights(self):
        """Apply Xavier initialization to all layers."""

        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_layer)

    def forward(self, obs, actions):
        """
        Forward pass through both Q-networks.
        """
        xu = th.cat([obs, actions], dim=1)  # Concatenate obs & actions

        # Compute Q1
        x1 = xu
        for layer in self.q1_layers[:-1]:  # All hidden layers
            x1 = F.relu(layer(x1))
        x1 = self.q1_layers[-1](x1)  # Output layer (no activation)

        # Compute Q2
        x2 = xu
        for layer in self.q2_layers[:-1]:  # All hidden layers
            x2 = F.relu(layer(x2))
        x2 = self.q2_layers[-1](x2)  # Output layer (no activation)

        return x1, x2

    def q1_forward(self, obs, actions):
        """
        Compute only Q1 (used during actor updates).
        """
        x = th.cat([obs, actions], dim=1)

        for layer in self.q1_layers[:-1]:  # All hidden layers
            x = F.relu(layer(x))

        x = self.q1_layers[-1](x)  # Output layer (no activation)

        return x


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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Apply Xavier initialization to all layers."""

        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_layer)

    def forward(self, obs):
        """Forward pass for action prediction."""
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        x = F.softsign(self.FC3(x))

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
