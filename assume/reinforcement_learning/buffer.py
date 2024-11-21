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
            array (numpy.ndarray): The numpy array to convert.
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
            obs (numpy.ndarray): The observation to add.
            actions (numpy.ndarray): The actions to add.
            reward (numpy.ndarray): The reward to add.
        """
        # copying all to avoid modification
        len_obs = obs.shape[0]
        self.observations[self.pos : self.pos + len_obs] = obs.copy()
        self.actions[self.pos : self.pos + len_obs] = actions.copy()
        self.rewards[self.pos : self.pos + len_obs] = reward.copy()

        self.pos += len_obs

        # Circular buffer
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
            self.observations[batch_inds, :, :], # current observation
            self.actions[batch_inds, :, :],
            self.observations[batch_inds + 1, :, :], # next observation
            self.rewards[batch_inds],
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class RolloutBufferTransitions(NamedTuple):
    """
    A named tuple that represents the data stored in a rollout buffer for PPO.

    Attributes:
        observations (torch.Tensor): The observations of the agents.
        actions (torch.Tensor): The actions taken by the agents.
        log_probs (torch.Tensor): The log probabilities of the actions taken.
        advantages (torch.Tensor): The advantages calculated using GAE.
        returns (torch.Tensor): The returns (discounted rewards) calculated.
    """

    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    log_probs: th.Tensor


class RolloutBuffer:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_rl_units: int,
        device: str,
        float_type,
        buffer_size: int,
    ):
        """
        A class that represents a rollout buffer for storing observations, actions, and rewards.
        The buffer starts empty and is dynamically expanded when needed.

        Args:
            obs_dim (int): The dimension of the observation space.
            act_dim (int): The dimension of the action space.
            n_rl_units (int): The number of reinforcement learning units.
            device (str): The device to use for storing the data (e.g., 'cpu' or 'cuda').
            float_type (torch.dtype): The data type to use for the stored data.
            buffer_size (int): The maximal size of the buffer
        """

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_rl_units = n_rl_units
        self.device = device
        self.buffer_size = buffer_size

        # Start with no buffer (None), will be created dynamically when first data is added
        self.observations = (
            None  # Stores the agent's observations (states) at each timestep
        )
        self.actions = None  # Stores the actions taken by the agent
        self.rewards = None  # Stores the rewards received after each action
        self.log_probs = None  # Stores the log-probabilities of the actions, used to compute the ratio for policy update

        # self.values = (
        #     None  # Stores the value estimates (critic's predictions) of each state
        # )
        # self.advantages = None  # Stores the computed advantages using GAE (Generalized Advantage Estimation), central to PPO's policy updates
        # self.returns = None  # Stores the discounted rewards (also known as returns), used to compute the value loss for training the critic

        self.pos = 0
        self.full = False

        # Datatypes for numpy and PyTorch
        self.np_float_type = np.float16 if float_type == th.float16 else np.float32
        self.th_float_type = float_type

    def initialize_buffer(self, size):
        """Initializes the buffer with the given size."""
        self.observations = np.zeros(
            (size, self.n_rl_units, self.obs_dim), dtype=self.np_float_type
        )
        self.actions = np.zeros(
            (size, self.n_rl_units, self.act_dim), dtype=self.np_float_type
        )
        self.rewards = np.zeros((size, self.n_rl_units), dtype=self.np_float_type)
        self.log_probs = np.zeros((size, self.n_rl_units), dtype=np.float32)
        # self.values = np.zeros((size, self.n_rl_units), dtype=np.float32)
        # self.advantages = np.zeros((size, self.n_rl_units), dtype=np.float32)
        # self.returns = np.zeros((size, self.n_rl_units), dtype=np.float32)

    def expand_buffer(self, additional_size):
        """Expands the buffer by the given additional size and checks if there is enough memory available."""

        # Calculation of the memory requirement for all 7 arrays
        additional_memory_usage = (
            np.zeros(
                (additional_size, self.n_rl_units, self.obs_dim),
                dtype=self.np_float_type,
            ).nbytes
            + np.zeros(
                (additional_size, self.n_rl_units, self.act_dim),
                dtype=self.np_float_type,
            ).nbytes
            + np.zeros(
                (additional_size, self.n_rl_units), dtype=self.np_float_type
            ).nbytes  # rewards
            + np.zeros(
                (additional_size, self.n_rl_units), dtype=np.float32
            ).nbytes  # log_probs
            # + np.zeros(
            #     (additional_size, self.n_rl_units), dtype=np.float32
            # ).nbytes  # values
            # + np.zeros(
            #     (additional_size, self.n_rl_units), dtype=np.float32
            # ).nbytes  # advantages
            # + np.zeros(
            #     (additional_size, self.n_rl_units), dtype=np.float32
            # ).nbytes  # returns
        )

        # Check whether enough memory is available
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
            if additional_memory_usage > mem_available:
                # Conversion to GB
                additional_memory_usage_gb = additional_memory_usage / 1e9
                mem_available_gb = mem_available / 1e9
                raise MemoryError(
                    f"{additional_memory_usage_gb:.2f}GB required, but only {mem_available_gb:.2f}GB available."
                )

            if self.pos + additional_size > self.buffer_size:
                warnings.warn(
                    f"Expanding the buffer will exceed the maximum buffer size of {self.buffer_size}. "
                    f"Current position: {self.pos}, additional size: {additional_size}."
                )

            self.observations = np.concatenate(
                (
                    self.observations,
                    np.zeros(
                        (additional_size, self.n_rl_units, self.obs_dim),
                        dtype=self.np_float_type,
                    ),
                ),
                axis=0,
            )
            self.actions = np.concatenate(
                (
                    self.actions,
                    np.zeros(
                        (additional_size, self.n_rl_units, self.act_dim),
                        dtype=self.np_float_type,
                    ),
                ),
                axis=0,
            )
            self.rewards = np.concatenate(
                (
                    self.rewards,
                    np.zeros(
                        (additional_size, self.n_rl_units), dtype=self.np_float_type
                    ),
                ),
                axis=0,
            )
            self.log_probs = np.concatenate(
                (
                    self.log_probs,
                    np.zeros((additional_size, self.n_rl_units), dtype=np.float32),
                ),
                axis=0,
            )
            # self.values = np.concatenate(
            #     (
            #         self.values,
            #         np.zeros((additional_size, self.n_rl_units), dtype=np.float32),
            #     ),
            #     axis=0,
            # )
            # self.advantages = np.concatenate(
            #     (
            #         self.advantages,
            #         np.zeros((additional_size, self.n_rl_units), dtype=np.float32),
            #     ),
            #     axis=0,
            # )
            # self.returns = np.concatenate(
            #     (
            #         self.returns,
            #         np.zeros((additional_size, self.n_rl_units), dtype=np.float32),
            #     ),
            #     axis=0,
            # )

    def add(
        self,
        obs: np.array,
        actions: np.array,
        reward: np.array,
        log_probs: np.array,
    ):
        """
        Adds an observation, action, reward, and log probabilities of all agents to the rollout buffer.
        If the buffer does not exist, it will be initialized. If the buffer is full, it will be expanded.

        Args:
            obs (numpy.ndarray): The observation to add.
            actions (numpy.ndarray): The actions to add.
            reward (numpy.ndarray): The reward to add.
            log_probs (numpy.ndarray): The log probabilities of the actions taken.
        """
        len_obs = obs.shape[0]

        if self.observations is None:
            # Initialize buffer with initial size if it's the first add
            self.initialize_buffer(len_obs)

        elif self.pos + len_obs > self.observations.shape[0]:
            # If the buffer is full, expand it
            self.expand_buffer(len_obs)

        # Add data to the buffer
        self.observations[self.pos : self.pos + len_obs] = obs.copy()
        self.actions[self.pos : self.pos + len_obs] = actions.copy()
        self.rewards[self.pos : self.pos + len_obs] = reward.copy()
        self.log_probs[self.pos : self.pos + len_obs] = log_probs.squeeze(-1).copy()

        self.pos += len_obs

    def reset(self):
        """
        Resets the buffer, clearing all stored data.
        Might be needed if policy is changed within one episode, then it needs to be killed and initalized again.

        """
        self.observations = None
        self.actions = None
        self.rewards = None
        self.log_probs = None
        # self.values = None
        # self.advantages = None
        # self.returns = None
        self.pos = 0
        self.full = False 

    # def compute_returns_and_advantages(self, last_values, dones):
    #     """
    #     Compute the returns and advantages using Generalized Advantage Estimation (GAE).

    #     Args:
    #         last_values (np.array): Value estimates for the last observation.
    #         dones (np.array): Whether the episode has finished for each agent.
    #     """
    #     # Initialize the last advantage to 0. This will accumulate as we move backwards in time.
    #     last_advantage = 0

    #     # Loop backward through all the steps in the buffer to calculate returns and advantages.
    #     # This is because GAE (Generalized Advantage Estimation) relies on future rewards,
    #     # so we compute it from the last step back to the first step.
    #     for step in reversed(range(self.pos)):

    #         # If we are at the last step in the buffer
    #         if step == self.pos - 1:
    #             # If it's the last step, check whether the episode has finished using `dones`.
    #             # `next_non_terminal` is 0 if the episode has ended, 1 if it's ongoing.
    #             next_non_terminal = 1.0 - dones
    #             # Use the provided last values (value estimates for the final observation in the episode)
    #             next_values = last_values
    #         else:
    #             # For other steps, use the mask to determine if the episode is ongoing.
    #             # If `masks[step + 1]` is 1, the episode is ongoing; if it's 0, the episode has ended.
    #             next_non_terminal = self.masks[step + 1]
    #             # Use the value of the next time step to compute the future returns
    #             next_values = self.values[step + 1]

    #         # Temporal difference (TD) error, also known as delta:
    #         # This is the difference between the reward obtained at this step and the estimated value of this step
    #         # plus the discounted value of the next step (if the episode is ongoing).
    #         # This measures how "off" the value function is at predicting the future return.
    #         delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]

    #         # Compute the advantage for this step using GAE:
    #         # `delta` is the immediate advantage, and we add to it the discounted future advantage,
    #         # scaled by the factor `lambda` (from GAE). This allows for a more smooth approximation of advantage.
    #         # `next_non_terminal` ensures that if the episode has ended, the future advantage stops accumulating.
    #         self.advantages[step] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage

    #         # The return is the advantage plus the baseline value estimate.
    #         # This makes sure that the return includes both the immediate rewards and the learned value of future rewards.
    #         self.returns[step] = self.advantages[step] + self.values[step]

    def to_torch(self, array: np.array, copy=True):
        """
        Converts a numpy array to a PyTorch tensor. Note: It copies the data by default.

        Args:
            array (numpy.ndarray): The numpy array to convert.
            copy (bool, optional): Whether to copy or not the data
                (may be useful to avoid changing things by reference). Defaults to True.

        Returns:
            torch.Tensor: The converted PyTorch tensor.
        """

        if copy:
            return th.tensor(array, dtype=self.th_float_type, device=self.device)

        return th.as_tensor(array, dtype=self.th_float_type, device=self.device)

    def get(self) -> RolloutBufferTransitions:
        """
        Get all data stored in the buffer and convert it to PyTorch tensors.
        Returns the observations, actions, log_probs, advantages, returns, and masks.
        """
        data = (
            self.observations[: self.pos],
            self.actions[: self.pos],
            self.rewards[: self.pos],
            self.log_probs[: self.pos],
            # self.masks[:self.pos],
        )

        return RolloutBufferTransitions(*tuple(map(self.to_torch, data)))


    def sample(self, batch_size: int) -> RolloutBufferTransitions:
        """
        Samples a random batch of experiences from the rollout buffer.
        Unlike the replay buffer, this samples only from the current rollout data (up to self.pos)
        and includes log probabilities needed for PPO updates.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            RolloutBufferTransitions: A named tuple containing the sampled observations, actions, rewards,
                and log probabilities.

        Raises:
            Exception: If there are less than batch_size entries in the buffer.
        """
        if self.pos < batch_size:
            raise Exception(f"Not enough entries in buffer (need {batch_size}, have {self.pos})")
        
        batch_inds = np.random.randint(0, self.pos, size=batch_size)
        
        data = (
            self.observations[batch_inds, :, :],
            self.actions[batch_inds, :, :],
            self.rewards[batch_inds],
            self.log_probs[batch_inds],
        )
        
        return RolloutBufferTransitions(*tuple(map(self.to_torch, data))), batch_inds # also return the indices of the sampled minibatch episodes