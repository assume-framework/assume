# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import torch as th
from torch import nn
from torch.nn import functional as F


class Critic(nn.Module):
    """Parent class for critic networks."""

    def __init__(self):
        super().__init__()

    def _build_q_network(self, input_dim, hidden_sizes, float_type):
        """
        Build a Q-network as a sequence of linear layers.
        """
        layers = nn.ModuleList()
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h, dtype=float_type))
            input_dim = h
        layers.append(nn.Linear(input_dim, 1, dtype=float_type))  # Output Q-value
        return layers

    def _init_weights(self):
        """
        Apply Xavier initialization to all linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


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

        input_dim = self.obs_dim + self.act_dim
        # First Q-network (Q1)
        self.q1_layers = self._build_q_network(input_dim, hidden_sizes, float_type)

        # Second Q-network (Q2) for double Q-learning
        self.q2_layers = self._build_q_network(input_dim, hidden_sizes, float_type)

        # Initialize weights properly
        self._init_weights()

    def forward(self, obs, actions, *args, **kwargs):
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

    def q1_forward(self, obs, actions, *args, **kwargs):
        """
        Compute only Q1 (used during actor updates).
        """
        x = th.cat([obs, actions], dim=1)

        for layer in self.q1_layers[:-1]:  # All hidden layers
            x = F.relu(layer(x))

        x = self.q1_layers[-1](x)  # Output layer (no activation)

        return x


class ContextualCriticTD3(Critic):
    """
    A centralized critic that incorporates both the global state (observation) and
    unit-specific context in separate branches before combining them with the actions.

    Args:
        n_agents (int): Number of agents.
        obs_dim (int): Dimension of each agent's observation.
        context_dim (int): Dimension of each agent's extra context.
        act_dim (int): Dimension of each agent's action.
        float_type: The torch data type (e.g., torch.float32).
        unique_obs_dim (int, optional): Extra observation dimensions from other agents (default=0).
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        context_dim: int,
        act_dim: int,
        float_type,
        unique_obs_dim: int = 0,
        hidden_sizes=[1024, 512, 256, 128],
    ):
        super().__init__()

        # Effective dimensions:
        # For the observation branch, we use the agent's own observation
        # plus the unique observations from the other agents.
        self.effective_obs_dim = obs_dim + unique_obs_dim * (n_agents - 1)
        # For the actions branch, we have all agents' actions concatenated.
        self.effective_act_dim = act_dim * n_agents
        # For the context branch, we assume that each agent has a context vector;
        # these are concatenated in a fixed order.
        self.effective_context_dim = context_dim

        # Build separate processing layers for observation and context.
        self.obs_fc = nn.Linear(
            self.effective_obs_dim, hidden_sizes[-1], dtype=float_type
        )
        self.context_fc = nn.Linear(
            self.effective_context_dim, hidden_sizes[-1], dtype=float_type
        )

        # After processing, we concatenate the two embeddings with the actions.
        # The combined input dimension becomes: 128 (obs) + 128 (context) + effective_act_dim.
        combined_input_dim = hidden_sizes[-1] * 2 + self.effective_act_dim

        # Build the two Q-networks (Q1 and Q2).
        self.q1_layers = self._build_q_network(
            combined_input_dim, hidden_sizes, float_type
        )
        self.q2_layers = self._build_q_network(
            combined_input_dim, hidden_sizes, float_type
        )

        self._init_weights()

    def forward(self, obs, context, actions):
        """
        Forward pass through both Q-networks.

        Args:
            obs (torch.Tensor): Global observation tensor with shape [batch, effective_obs_dim].
            context (torch.Tensor): Concatenated context tensor with shape [batch, effective_context_dim].
            actions (torch.Tensor): Concatenated actions with shape [batch, effective_act_dim].

        Returns:
            Tuple of Q-value estimates (Q1, Q2) each with shape [batch, 1].
        """

        # expand context to match batch size
        if obs.dim() != context.dim():
            context = context.unsqueeze(0).expand(obs.size(0), -1)

        # Process the observation and context through their respective branches.
        obs_emb = F.relu(self.obs_fc(obs))
        context_emb = F.relu(self.context_fc(context))

        # Concatenate the embeddings with the actions.
        xu = th.cat([obs_emb, context_emb, actions], dim=1)

        # Compute Q1.
        x1 = xu
        for layer in self.q1_layers[:-1]:
            x1 = F.relu(layer(x1))
        x1 = self.q1_layers[-1](x1)

        # Compute Q2.
        x2 = xu
        for layer in self.q2_layers[:-1]:
            x2 = F.relu(layer(x2))
        x2 = self.q2_layers[-1](x2)

        return x1, x2

    def q1_forward(self, obs, context, actions):
        """
        Compute only the Q1 estimate (useful for actor updates).
        """
        # expand context to match batch size
        if obs.dim() != context.dim():
            context = context.unsqueeze(0).expand(obs.size(0), -1)

        obs_emb = F.relu(self.obs_fc(obs))
        context_emb = F.relu(self.context_fc(context))
        xu = th.cat([obs_emb, context_emb, actions], dim=1)
        x = xu
        for layer in self.q1_layers[:-1]:
            x = F.relu(layer(x))
        x = self.q1_layers[-1](x)
        return x


class Actor(nn.Module):
    """
    Parent class for actor networks.
    """

    def __init__(self):
        super().__init__()

    def _build_mlp_layers(self, input_dim, act_dim, hidden_sizes, float_type):
        """
        Dynamically creates an MLP given the chosen hidden layer sizes.
        """
        layers = nn.ModuleList()
        for h in hidden_sizes:
            layers.append(nn.Linear(input_dim, h, dtype=float_type))
            input_dim = h

        layers.append(
            nn.Linear(input_dim, act_dim, dtype=float_type)
        )  # Final output layer

        return layers

    def _init_weights(self):
        # Apply Xavier initialization to all linear layers.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


class MLPActor(Actor):
    """
    The neurnal network for the MLP actor.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        float_type,
        hidden_sizes=[256, 128],
        *args,
        **kwargs,
    ):
        super().__init__()

        self.layers = self._build_mlp_layers(obs_dim, act_dim, hidden_sizes, float_type)

        self._init_weights()

    def forward(self, obs, *args, **kwargs):
        """Forward pass for action prediction."""
        x = obs
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Apply the final layer with softsign activation
        x = F.softsign(self.layers[-1](x))

        return x


class ContextualMLPActor(Actor):
    """
    A modified MLP actor that incorporates additional context.
    """

    def __init__(
        self,
        obs_dim: int,
        context_dim: int,
        act_dim: int,
        float_type,
        hidden_sizes=[512, 256, 128],
        *args,
        **kwargs,
    ):
        super().__init__()

        # Process general observation
        self.obs_fc = nn.Linear(obs_dim, 128, dtype=float_type)
        # Process unit-specific context
        self.context_fc = nn.Linear(context_dim, 128, dtype=float_type)

        combined_input_dim = 128 + 128
        # Build the MLP layers that output the final action
        self.layers = self._build_mlp_layers(
            combined_input_dim, act_dim, hidden_sizes, float_type
        )

        self._init_weights()

    def forward(self, obs, context):
        """
        Forward pass takes both the general observation and unit-specific context.
        """
        # Expand context to match batch size if necessary.
        if obs.dim() != context.dim():
            context = context.unsqueeze(0).expand(obs.size(0), -1)

        obs_emb = F.relu(self.obs_fc(obs))
        context_emb = F.relu(self.context_fc(context))

        # Concatenate the embeddings along the feature dimension.
        x = th.cat([obs_emb, context_emb], dim=-1)
        # Pass through the MLP hidden layers
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        # Apply the final layer with softsign activation.
        x = F.softsign(self.layers[-1](x))
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
