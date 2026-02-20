# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
from typing import NamedTuple, Generator

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
            observations (numpy.ndarray): The stored observations.
            actions (numpy.ndarray): The stored actions.
            rewards (numpy.ndarray): The stored rewards.
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
                    "This system apparently does not have enough memory to store the complete "
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
            array (numpy.ndarray): The numpy array to convert.
            copy (bool, optional): Whether to copy the data or not
                (may be useful to avoid changing things by reference). Defaults to True.

        Returns:
            torch.Tensor: The converted PyTorch tensor.
        """

        if copy:
            return th.tensor(array, dtype=self.th_float_type, device=self.device)

        return th.as_tensor(array, dtype=self.th_float_type, device=self.device)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        reward: np.ndarray,
    ):
        """
        Adds an observation, action, and reward of all agents to the replay buffer.

        Args:
            obs (numpy.ndarray): The observation to add.
            actions (numpy.ndarray): The actions to add.
            reward (numpy.ndarray): The reward to add.
        """
        # copying all to avoid modification
        len_obs = obs.shape[0]
        self.observations[self.pos : self.pos + len_obs] = obs.copy()
        self.actions[self.pos : self.pos + len_obs] = actions.copy()
        self.rewards[self.pos : self.pos + len_obs] = np.squeeze(reward.copy(), axis=-1)

        self.pos += len_obs
        if self.pos + len_obs >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Samples a random batch of experiences from the replay buffer.

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

class RolloutBufferSamples(NamedTuple):
    """
    Container for roll buffer samples. It holds one batch of training samples 
    from PPO's rollout buffer.
    """
    observations: th.Tensor     # states/observations the agent saw
    actions: th.Tensor          # actions the agent took
    old_values: th.Tensor       # critic's value estimates
    old_log_probs: th.Tensor    # log_probability of taking each action
    advantages: th.Tensor       # generalized advantage estimates
    returns: th.Tensor          # expected returns

class RolloutBuffer:
    """
    Rollout buffer is used in on-policy algorithms like PPO.

    It corresponds to the transitions collected using the current policy.
    This experience is discarded after the policy is updated.
    In order to use PPO, the current observations are needed to be stored.
    the observations include actions, rewards, values, log probabilities and done for each action.

    Args:
        buffer_size (int): Max number of elements allowed in the buffer
        obs_dim (int): Dimension of the observation space
        act_dim (int): Dimension of the action space
        n_rl_units (int): Number of RL agents
        device (str | th.device): PyTorch device config
        float_type (th.dtype): Data type for floating point numbers
        gamma (float): Discount factor
        gae_lambda (float): bias-variance trade-off factor for Generalized Advantage Estimator
    
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
        gae_lambda: float = 0.98
    ):
        """Initialize the rollout buffer."""
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
        """
        Reset the rollout buffer. 
        Clearing the buffer and allocating new storage.
        """
        self.observations = np.zeros(
            (
                self.buffer_size, 
                self.n_rl_units, 
                self.obs_dim
            ),
            dtype = np.float32
        )
        self.actions = np.zeros(
            (
                self.buffer_size,
                self.n_rl_units,
                self.act_dim
            ),
            dtype = np.float32
        )
        self.rewards = np.zeros(
            (
                self.buffer_size,
                self.n_rl_units
            ),
            dtype = np.float32
        )
        self.values = np.zeros(
            (
                self.buffer_size,
                self.n_rl_units
            ),
            dtype = np.float32
        )
        self.log_probs = np.zeros(
            (
                self.buffer_size,
                self.n_rl_units
            ),
            dtype = np.float32
        )
        self.dones = np.zeros(
            (
                self.buffer_size,
                self.n_rl_units
            ),
            dtype = np.float32
        )

        # Computed after rollout
        self.advantages = np.zeros(
            (
                self.buffer_size,
                self.n_rl_units
            ),
            dtype = np.float32
        )
        self.returns = np.zeros(
            (
                self.buffer_size,
                self.n_rl_units
            ),
            dtype = np.float32
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
        log_prob: np.ndarray
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            obs (np.ndarray): Observation of the agents
            action (np.ndarray): Action taken by the agents
            reward (np.ndarray): Reward obtained
            done (np.ndarray): Whether the episode ended
            value (np.ndarray): Value estimate from the critic
            log_prob (np.ndarray): Log probability of the action
        """
        if self.pos >= self.buffer_size:
            self.full = True
            return
        
        self.observations[self.pos] = np.array(obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).flatten().copy()
        self.dones[self.pos] = np.array(done).flatten().copy()
        self.values[self.pos] = np.array(value).flatten().copy()
        self.log_probs[self.pos] = np.array(log_prob).flatten().copy()
        # flattening the rewards, dones, values, log_probs array to (n_units,) size

        self.pos += 1

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        Uses Generalized Advantage Estimation to compute the advantage. 
        To obtain the lambda-return, the advantage is added to the value estiamte.

        Args:
            last_values (np.ndarray): value estimation for the last step
            dones (np.ndarray): whether the last step was terminal
        """
        # taking the final value estimates and episode-end flags,
        # and making them flat arrays providing one number per agent.
        last_values = np.array(last_values).flatten()
        dones = np.array(dones).flatten()

        # GAE computation
        # starting with running total of zero for each agent.
        last_gae_lam = np.zeros(self.n_rl_units, dtype=np.float32)
        buffer_size = self.pos if not self.full else self.buffer_size

        # backward loop
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                # if at the last step, use the last_vlaues given as input
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                # for all the other steps, get the next value and next episode flag.
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
        batch_size: int | None = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        """
        Generator for generating batches of transition samples for training.
        
        Args:
            batch_size (int | None): Number of samples to be accessed per batch.

        Yields:
            Generator[RolloutBufferSamples]: A generator yielding RolloutBufferSamples
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
        """
        Helper function to sample data from the buffer.
        Converts numpy arrays to torch tensors for given indices.
        
        Args:
            indices (np.ndarray): Indices of the samples to retrieve.
        
        Returns:
            RolloutBufferSamples: The batch of samples converted to PyTorch tensors.
        """
        return RolloutBufferSamples(
            observations = th.as_tensor(
                self.observations[indices],
                device = self.device,
                dtype = self.float_type
            ),
            actions = th.as_tensor(
                self.actions[indices],
                device = self.device,
                dtype = self.float_type
            ),
            old_values = th.as_tensor(
                self.values[indices],
                device = self.device,
                dtype = self.float_type
            ),
            old_log_probs = th.as_tensor(
                self.log_probs[indices],
                device = self.device,
                dtype = self.float_type
            ),
            advantages = th.as_tensor(
                self.advantages[indices],
                device = self.device,
                dtype = self.float_type
            ),
            returns = th.as_tensor(
                self.returns[indices],
                device = self.device,
                dtype = self.float_type
            )
        )
    
    def size(self) -> int:
        """
        Return the current number of stored transitions.

        Returns:
            int: The size of the buffer.
        """
        return self.buffer_size if self.full else self.pos
