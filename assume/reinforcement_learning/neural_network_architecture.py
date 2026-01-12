# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch as th
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type, Optional, Union


class Critic(nn.Module):
    """
    Base Critic class handling architecture generation and initialization.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of observation per agent
        act_dim: Dimension of action per agent
        float_type: Data type for parameters
        unique_obs_dim: Dimension of agent-specific observations
    """
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int,
    ):
        super().__init__()

        # Calculate total (centralized) dimensions
        self.obs_dim = obs_dim + unique_obs_dim * (n_agents - 1)
        self.act_dim = act_dim * n_agents

        self.float_type = float_type

        # Dynamic Architecture Definition
        self.hidden_sizes = self._get_architecture(n_agents)

    def _get_architecture(
        self, n_agents: int
    ) -> List[int]:
        """Returns hidden layer sizes based on the number of agents."""
        if n_agents <= 20:
            hidden_sizes = [256, 128]  # Shallow network for small `n_agents`
        elif n_agents <= 50:
            hidden_sizes = [512, 256, 128]  # Medium network
        else:
            hidden_sizes = [1024, 512, 256, 128]  # Deeper network for large `n_agents`
        return hidden_sizes

    def _build_q_network(self) -> nn.ModuleList:
        """
        Dynamically create a Q-network given the chosen hidden layer sizes.
        """
        layers = nn.ModuleList()
        input_dim = (
            self.obs_dim + self.act_dim
        ) # Input includes all observations and actions

        for h in self.hidden_sizes:
            layers.append(nn.Linear(input_dim, h, dtype=self.float_type))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 1, dtype=self.float_type)) # Output Q-value

        return layers

    def _init_weights(self):
        """Apply Xavier initialization to all layers."""

        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(init_layer)


class CriticTD3(Critic):
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
        unique_obs_dim: int
    ):
        super().__init__(
            n_agents, 
            obs_dim, 
            act_dim, 
            float_type, 
            unique_obs_dim
        )

        # First Q-network (Q1)
        self.q1_layers = self._build_q_network()

        # Second Q-network (Q2) for double Q-learning
        self.q2_layers = self._build_q_network()

    def forward(
        self,
        obs: th.Tensor,
        actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass through both Q-networks.
        """
        xu = th.cat([obs, actions], dim=1) # Concatenate obs & actions

        # Compute Q1
        x1 = nn.Sequential(*self.q1_layers)(xu)

        # Compute Q2
        x2 = nn.Sequential(*self.q2_layers)(xu)

        return x1, x2

    def q1_forward(
        self,
        obs: th.Tensor,
        actions: th.Tensor
    ) -> th.Tensor:
        """
        Compute only Q1 (used during actor updates).
        """
        x = th.cat([obs, actions], dim=1)

        x = nn.Sequential(*self.q1_layers)(x)

        return x


class CriticDDPG(Critic):
    """Initialize parameters and build model.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of observation per agent
        act_dim: Dimension of action per agent
        float_type: Data type for parameters
        unique_obs_dim: Dimension of agent-specific observations
    """
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        float_type: th.dtype,
        unique_obs_dim: int,
    ):
        super().__init__(
            n_agents, 
            obs_dim, 
            act_dim, 
            float_type, 
            unique_obs_dim
        )

        # Q-network
        self.q_layers = self._build_q_network()
        
        # Initialize weights properly
        self._init_weights()

    def forward(
        self,
        obs: th.Tensor,
        actions: th.Tensor
    ) -> th.Tensor:
        """Returns Q value."""
        xu = th.cat([obs, actions], dim=1) # Concatenate obs & actions

        # Compute Q
        x = nn.Sequential(*self.q_layers)(xu)

        return x


class CriticPPO(Critic):
    """Initialize parameters and build PPO value network.

    Args:
        n_agents (int): Number of agents
        obs_dim (int): Dimension of observation per agent
        float_type: Data type for parameters
        unique_obs_dim: Dimension of agent-specific observations
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        float_type,
        unique_obs_dim: int
    ):
        super().__init__(
            n_agents=n_agents,
            obs_dim=obs_dim,
            act_dim=0,
            float_type=float_type,
            unique_obs_dim=unique_obs_dim
        )

        # V-network
        self.v_layers = self._build_q_network()

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Apply Orthogonal initialization.
        """
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        
        self.apply(init_layer)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Returns V value."""
        x = obs
        for layer in self.v_layers:
            x = layer(x)
        return x


class Actor(nn.Module):
    """
    Parent class for actor networks.
    """

    activation_function_limit = {
        "softsign": (-1, 1),
        "tanh": (-1, 1),
        "sigmoid": (0, 1),
        "relu": (0, float("inf")),
    }

    activation_function_map = {
        "softsign": F.softsign,
        "tanh": th.tanh,
        "sigmoid": th.sigmoid,
        "relu": F.relu
    }

    def __init__(self):
        super().__init__()

        self.activation = "softsign" # or "tanh", "sigmoid", "relu"

        if self.activation not in self.activation_function_limit:
            raise ValueError(
                f"Activation '{self.activation}' not supported! Supported: {list(self.activation_function_limit.keys())}"
            )
        
        self.min_output, self.max_output = self.activation_function_limit[
            self.activation
        ]

        self.activation_function = self.activation_function_map.get(self.activation)

        if self.activation_function is None:
            raise ValueError(
                f"Activation '{self.activation}' not implemented in forward pass!"
            )


class MLPActor(Actor):
    """
    The neural network for the MLP actor.
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
        x = self.activation_function(self.FC3(x))

        return x


class LSTMActor(Actor):
    """
    The LSTM recurrent neural network for the actor.

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
        unique_obs_dim: int,
        num_timeseries_obs_dim: int,
        *args,
        **kwargs
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
            time_step = time_step.reshape(-1, self.num_timeseris_obs_dim)
            h_t, c_t = self.LSTM1(time_step, (h_t, c_t))
            h_t2, c_t2 = self.LSTM2(h_t, (h_t2, c_t2))
            outputs += [h_t2]

        outputs = th.cat(outputs, dim=1)
        x = th.cat((outputs, x2), dim=1)
        
        x = F.relu(self.FC1(x))
        x = self.activation_function(self.FC2(x))

        if not is_batched:
            x = x.squeeze(0)

        return x


class ActorPPO(nn.Module):
    activation_function_limit = {
        "softsign": (-1, 1),
        "tanh": (-1, 1),
        "sigmoid": (0, 1),
        "relu": (0, float("inf")),
    }

    activation_function_map = {
        "softsign": F.softsign,
        "tanh": th.tanh,
        "sigmoid": th.sigmoid,
        "relu": F.relu
    }
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        float_type,
        log_std_init: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.act_dim = act_dim
        self.float_type = float_type

        self.activation = "softsign" # or "tanh", "sigmoid", "relu"

        if self.activation not in self.activation_function_limit:
            raise ValueError(
                f"Activation '{self.activation}' not supported! Supported: {list(self.activation_function_limit.keys())}"
            )
        
        self.min_output, self.max_output = self.activation_function_limit[
            self.activation
        ]

        # Policy network (outputs mean)
        self.FC1 = nn.Linear(obs_dim, 256, dtype=float_type)
        self.FC2 = nn.Linear(256, 128, dtype=float_type)
        self.mean_layer = nn.Linear(128, act_dim, dtype=float_type)

        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            th.ones(act_dim, dtype=float_type) * log_std_init
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization."""
        def init_layer(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        
        # Initialize hidden layers with larger gain
        nn.init.orthogonal_(self.FC1.weight, gain=1.0)
        nn.init.orthogonal_(self.FC2.weight, gain=1.0)
        nn.init.zeros_(self.FC1.bias)
        nn.init.zeros_(self.FC2.bias)
        
        # Initialize output layer with small gain
        nn.init.orthogonal_(self.mean_layer.weight, gain=0.01)
        nn.init.zeros_(self.mean_layer.bias)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        """Forward pass"""
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mean = th.tanh(self.mean_layer(x))  # Bounded to [-1, 1]
        
        if deterministic:
            return mean
        
        # Sample from Gaussian during training
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()
        noise = th.randn_like(mean)
        action = mean + std * noise
        
        # Clamp to valid range
        return th.clamp(action, -1.0, 1.0)

    def get_distribution(self, obs: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        Get the policy distribution parameters.
        """
        x = F.relu(self.FC1(obs))
        x = F.relu(self.FC2(x))
        mean = th.tanh(self.mean_layer(x))  # Bounded to [-1, 1]
        log_std = self.log_std.expand_as(mean)
        
        return mean, log_std

    def get_action_and_log_prob(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        """
        Sample action and compute log probability.
        
        Args:
            obs: Observations
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, log_prob)
        """
        mean, log_std = self.get_distribution(obs)
        std = log_std.exp()

        if deterministic:
            action = mean
        else:
            # Sample from Gaussian
            noise = th.randn_like(mean)
            action = mean + std * noise

        # Clamp action to valid range
        action = th.clamp(action, -1.0, 1.0)

        # Compute log probability
        log_prob = self._compute_log_prob(action, mean, std)

        return action, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate log probability and entropy for given actions.
        
        Used during PPO update to compute importance ratio.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (log_prob, entropy, values)
        """
        mean, log_std = self.get_distribution(obs)
        std = log_std.exp()

        # Log probability
        log_prob = self._compute_log_prob(actions, mean, std)

        # Entropy for exploration bonus
        entropy = 0.5 * (1.0 + th.log(2 * th.pi * std.pow(2))).sum(dim=-1)

        return log_prob, entropy

    def _compute_log_prob(
        self,
        actions: th.Tensor,
        mean: th.Tensor,
        std: th.Tensor,
    ) -> th.Tensor:
        """Compute log probability of actions under Gaussian distribution."""
        var = std.pow(2)
        log_prob = -0.5 * (
            ((actions - mean).pow(2) / var)
            + 2 * th.log(std)
            + th.log(th.tensor(2 * th.pi))
        )
        return log_prob.sum(dim=-1)