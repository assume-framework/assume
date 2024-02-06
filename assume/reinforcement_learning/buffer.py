# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
from typing import NamedTuple

import numpy as np
import torch as th

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    rewards: th.Tensor


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        act_dim: int,
        n_rl_units: int,
        device: str,
        float_type,
    ):
        """
        A class that represents a replay buffer for storing observations, actions, and rewards.
        The replay buffer is implemented as a circular buffer, where the oldest experiences are discarded when the buffer is full.

        Args:
            buffer_size (int): The maximum size of the buffer.
            obs_dim (int): The dimension of the observation space.
            act_dim (int): The dimension of the action space.
            n_rl_units (int): The number of reinforcement learning units.
            device (str): The device to use for storing the data (e.g., 'cpu' or 'cuda').
            float_type (torch.dtype): The data type to use for the stored data.
            observations (np.array): The stored observations.
            actions (np.array): The stored actions.
            rewards (np.array): The stored rewards.
        """

        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.pos = 0
        self.full = False

        self.device = device

        # future: use float16 for GPU
        self.np_float_type = np.float16 if float_type == th.float16 else np.float32
        self.th_float_type = float_type

        self.n_rl_units = n_rl_units

        self.observations = np.zeros(
            (self.buffer_size, self.n_rl_units, self.obs_dim), dtype=self.np_float_type
        )
        self.actions = np.zeros(
            (self.buffer_size, self.n_rl_units, self.act_dim), dtype=self.np_float_type
        )
        self.rewards = np.zeros(
            (self.buffer_size, self.n_rl_units), dtype=self.np_float_type
        )

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

            total_memory_usage = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes
            )

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def size(self):
        # write docstring for this function
        """
        Return the current size of the buffer (i.e. number of transitions
        stored in the buffer).

        Returns:
            buffer_size(int): The current size of the buffer

        """
        return self.buffer_size if self.full else self.pos

    def to_torch(self, array: np.array, copy=True):
        """
        Converts a numpy array to a PyTorch tensor. Note: It copies the data by default.

        Args:
            array (np.array): The numpy array to convert.
            copy (bool, optional): Whether to copy or not the data
                (may be useful to avoid changing things by reference). Defaults to True.

        Returns:
            torch.Tensor: The converted PyTorch tensor.
        """

        if copy:
            return th.tensor(array, dtype=self.th_float_type, device=self.device)

        return th.as_tensor(array, dtype=self.th_float_type, device=self.device)

    def add(
        self,
        obs: np.array,
        actions: np.array,
        reward: np.array,
    ):
        """
        Adds an observation, action, and reward of all agents to the replay buffer.

        Args:
            obs (np.array): The observation to add.
            actions (np.array): The actions to add.
            reward (np.array): The reward to add.
        """
        # copying all to avoid modification
        self.observations[self.pos] = obs.copy()
        self.actions[self.pos] = actions.copy()
        self.rewards[self.pos] = reward.copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Samples a randome batch of experiences from the replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            ReplayBufferSamples: A named tuple containing the sampled observations, actions, and rewards.

        Raises:
            Exception: If there are less than two entries in the buffer.
        """

        upper_bound = self.buffer_size if self.full else self.pos
        if upper_bound < 2:
            raise Exception("at least two entries needed to sample")
        batch_inds = np.random.randint(0, upper_bound - 1, size=batch_size)

        data = (
            self.observations[batch_inds, :, :],
            self.actions[batch_inds, :, :],
            self.observations[batch_inds + 1, :, :],
            self.rewards[batch_inds],
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
