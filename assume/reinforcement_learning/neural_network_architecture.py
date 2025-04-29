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


class ContextualLateFusionActor(Actor):
    """
    Contextual MLP Actor where observations are processed first,
    then merged with processed context before final layers.
    """

    def __init__(
        self,
        obs_dim: int,
        context_dim: int,
        act_dim: int,
        float_type,
        obs_hidden_sizes=[512, 256, 128],  # Layers processing observation ONLY
        context_embedding_dim=64,  # Size of the context embedding
        merged_hidden_sizes=[128],  # Layers processing merged features
        *args,
        **kwargs,
    ):
        super().__init__()
        self.float_type = float_type
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        self.act_dim = act_dim

        # 1. Observation Processing Layers
        self.obs_encoder = nn.ModuleList()
        current_dim = obs_dim
        for h_size in obs_hidden_sizes:
            self.obs_encoder.append(nn.Linear(current_dim, h_size, dtype=float_type))
            current_dim = h_size
        # Output dimension after processing observations
        self.obs_feature_dim = current_dim

        # 2. Context Processing Layer (simple embedding)
        self.context_encoder = nn.Linear(
            context_dim, context_embedding_dim, dtype=float_type
        )
        self.context_feature_dim = context_embedding_dim

        # 3. Merged Processing Layers (using _build_mlp_layers helper)
        combined_input_dim = self.obs_feature_dim + self.context_feature_dim
        self.final_layers = self._build_mlp_layers(
            combined_input_dim, act_dim, merged_hidden_sizes, float_type
        )

        self._init_weights()  # Initialize all defined layers

    def forward(self, obs, context):
        """
        Forward pass: Process obs, process context, merge, process merged.
        """
        # Expand context to match batch size if necessary.
        if obs.dim() != context.dim():
            context = context.unsqueeze(0).expand(obs.size(0), -1)

        # 1. Process observation through its dedicated layers
        obs_features = obs
        for layer in self.obs_encoder:
            obs_features = F.relu(layer(obs_features))

        # 2. Process context through its dedicated layer
        context_features = F.relu(self.context_encoder(context))

        # 3. Concatenate the features
        merged_features = th.cat([obs_features, context_features], dim=-1)

        # 4. Pass merged features through the final layers
        final_output = merged_features
        for layer in self.final_layers[:-1]:
            final_output = F.relu(layer(final_output))

        # Apply the final layer with Tanh activation (common for TD3 actions)
        action = th.tanh(self.final_layers[-1](final_output))

        return action


class ContextualFiLMActor(Actor):
    """
    Contextual MLP Actor using FiLM layers.
    The context is used to generate modulation parameters (gamma, beta)
    which are applied to the processed observation features.
    """

    def __init__(
        self,
        obs_dim: int,
        context_dim: int,
        act_dim: int,
        float_type,
        obs_hidden_sizes=[512, 256, 128],  # Layers processing observation ONLY
        film_hidden_dim=64,  # Hidden layer size(s) for FiLM generator
        post_film_hidden_sizes=[128],  # Layers processing features AFTER FiLM
        *args,
        **kwargs,
    ):
        super().__init__()
        self.float_type = float_type
        self.obs_dim = obs_dim
        self.context_dim = context_dim
        self.act_dim = act_dim

        # 1. Observation Processing Layers (Encoder)
        self.obs_encoder = nn.ModuleList()
        current_dim = obs_dim
        if not obs_hidden_sizes:  # Handle no obs encoder layers
            self.obs_feature_dim = obs_dim  # FiLM will modulate raw obs
        else:
            for h_size in obs_hidden_sizes:
                self.obs_encoder.append(
                    nn.Linear(current_dim, h_size, dtype=float_type)
                )
                current_dim = h_size
            # Output dimension of the observation features to be modulated by FiLM
            self.obs_feature_dim = current_dim

        # 2. FiLM Generator Network
        # Takes context as input, outputs 2 * obs_feature_dim (for gamma and beta)
        self.film_generator = nn.Sequential(
            nn.Linear(context_dim, film_hidden_dim, dtype=float_type),
            nn.ReLU(),
            nn.Linear(film_hidden_dim, 2 * self.obs_feature_dim, dtype=float_type),
            # Note: No activation here, gamma/beta are produced directly.
            # Consider initializing the bias of the final layer near 1s (for gamma)
            # and 0s (for beta) for stability, e.g., via a hook or manual init.
        )

        # 3. Post-FiLM Processing Layers (using _build_mlp_layers helper)
        # Takes the modulated features (size: obs_feature_dim) as input
        self.post_film_layers = self._build_mlp_layers(
            self.obs_feature_dim, act_dim, post_film_hidden_sizes, float_type
        )

        self._init_weights()  # Initialize all defined layers (uses default Xavier)
        # Optional: Custom initialization for FiLM generator's final layer bias
        # Could initialize gamma biases to 1 and beta biases to 0
        nn.init.constant_(
            self.film_generator[-1].bias[: self.obs_feature_dim], 1.0
        )  # Gammas near 1
        nn.init.constant_(
            self.film_generator[-1].bias[self.obs_feature_dim :], 0.0
        )  # Betas near 0

    def forward(self, obs, context):
        """
        Forward pass: Process obs, generate FiLM params from context, modulate, process modulated.
        """
        # Expand context to match batch size if necessary.
        if obs.dim() != context.dim():
            context = context.unsqueeze(0).expand(obs.size(0), -1)

        # 1. Process observation through its dedicated layers
        obs_features = obs
        for layer in self.obs_encoder:
            obs_features = F.relu(layer(obs_features))
        # obs_features now has shape (batch_size, obs_feature_dim)

        # 2. Generate FiLM parameters (gamma, beta) from context
        # film_params has shape (batch_size, 2 * obs_feature_dim)
        film_params = self.film_generator(context)

        # Split into gamma and beta
        # gamma, beta each have shape (batch_size, obs_feature_dim)
        gamma, beta = th.chunk(film_params, 2, dim=-1)

        # 3. Apply FiLM modulation
        # Element-wise multiplication (gamma * features) and addition ( + beta)
        modulated_features = gamma * obs_features + beta

        # 4. Pass modulated features through the final layers
        final_output = modulated_features
        for layer in self.post_film_layers[:-1]:
            final_output = F.relu(layer(final_output))

        # Apply the final layer with Tanh activation (common for continuous actions)
        action = th.tanh(self.post_film_layers[-1](final_output))

        return action


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
