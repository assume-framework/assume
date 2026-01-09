# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import torch as th
from typing import NamedTuple, Generator


class RolloutBufferSamples(NamedTuple):
    """Container for rollout buffer samples."""
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_probs: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class RolloutBuffer:
    """
    On-policy rollout buffer for PPO algorithm.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        n_rl_units: int,
        device: str | th.device,
        float_type: th.dtype,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Initialize rollout buffer.
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_rl_units = n_rl_units
        self.device = device
        self.float_type = float_type
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Current position and full flag
        self.pos = 0
        self.full = False
        self.generator_ready = False

        # Allocate buffers
        self.reset()

    def reset(self) -> None:
        """Clear the buffer and allocate new storage."""
        self.observations = np.zeros(
            (self.buffer_size, self.n_rl_units, self.obs_dim),
            dtype=np.float32,
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_rl_units, self.act_dim),
            dtype=np.float32,
        )
        self.rewards = np.zeros(
            (self.buffer_size, self.n_rl_units),
            dtype=np.float32,
        )
        self.values = np.zeros(
            (self.buffer_size, self.n_rl_units),
            dtype=np.float32,
        )
        self.log_probs = np.zeros(
            (self.buffer_size, self.n_rl_units),
            dtype=np.float32,
        )
        self.dones = np.zeros(
            (self.buffer_size, self.n_rl_units),
            dtype=np.float32,
        )
        
        # Computed after rollout
        self.advantages = np.zeros(
            (self.buffer_size, self.n_rl_units),
            dtype=np.float32,
        )
        self.returns = np.zeros(
            (self.buffer_size, self.n_rl_units),
            dtype=np.float32,
        )

        self.pos = 0
        self.full = False
        self.generator_ready = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_prob: np.ndarray,
    ) -> None:
        """
        Add a transition to the buffer.
        """
        if self.pos >= self.buffer_size:
            self.full = True
            return

        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.values[self.pos] = np.array(value).copy()
        self.log_probs[self.pos] = np.array(log_prob).copy()

        self.pos += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """
        Compute GAE advantages and returns.
        """
        last_values = np.array(last_values).flatten()
        dones = np.array(dones).flatten()

        # GAE computation
        last_gae_lam = np.zeros(self.n_rl_units, dtype=np.float32)
        buffer_size = self.pos if not self.full else self.buffer_size

        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]

            # TD error
            delta = (
                self.rewards[step]
                + self.gamma * next_values * next_non_terminal
                - self.values[step]
            )
            
            # GAE advantage
            last_gae_lam = (
                delta
                + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[step] = last_gae_lam

        # Returns = advantages + values
        self.returns = self.advantages + self.values
        self.generator_ready = True

    def get(
        self,
        batch_size: int | None = None,
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        Generate batches of samples for training.
        """
        if not self.generator_ready:
            raise ValueError(
                "Must call compute_returns_and_advantages before sampling."
            )

        buffer_size = self.pos if not self.full else self.buffer_size
        indices = np.random.permutation(buffer_size)

        if batch_size is None:
            batch_size = buffer_size

        start_idx = 0
        while start_idx < buffer_size:
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield self._get_samples(batch_indices)
            start_idx += batch_size

    def _get_samples(self, indices: np.ndarray) -> RolloutBufferSamples:
        """Convert numpy arrays to torch tensors for given indices."""
        return RolloutBufferSamples(
            observations=th.as_tensor(
                self.observations[indices], device=self.device, dtype=self.float_type
            ),
            actions=th.as_tensor(
                self.actions[indices], device=self.device, dtype=self.float_type
            ),
            old_values=th.as_tensor(
                self.values[indices], device=self.device, dtype=self.float_type
            ),
            old_log_probs=th.as_tensor(
                self.log_probs[indices], device=self.device, dtype=self.float_type
            ),
            advantages=th.as_tensor(
                self.advantages[indices], device=self.device, dtype=self.float_type
            ),
            returns=th.as_tensor(
                self.returns[indices], device=self.device, dtype=self.float_type
            ),
        )

    def size(self) -> int:
        """Return current number of stored transitions."""
        return self.buffer_size if self.full else self.pos