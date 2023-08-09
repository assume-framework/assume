# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:03:55 2021

@author: Nick_SimPC
"""

import os
import pickle
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
    def __init__(self, buffer_size, obs_dim, act_dim, n_rl_units, device):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.pos = 0
        self.full = False

        self.device = device
        # self.np_float_type = np.float16 if self.device.type == "cuda" else np.float32
        # self.th_float_type = th.half if self.device.type == "cuda" else th.float
        self.np_float_type = np.float32
        self.th_float_type = th.float

        self.n_rl_units = n_rl_units

        self.observations = np.zeros(
            (self.buffer_size, self.n_rl_units, self.obs_dim), dtype=self.np_float_type
        )
        self.next_observations = np.zeros(
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
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.next_observations.nbytes
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
        if self.full:
            return self.buffer_size
        else:
            return self.pos

    def reset(self):
        self.pos = 0
        self.full = False

    def to_torch(self, array, copy=True):
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data
            (may be useful to avoid changing things be reference)
        :return:
        """
        if copy:
            return th.tensor(array, dtype=self.th_float_type, device=self.device)

        return th.as_tensor(array, dtype=self.th_float_type, device=self.device)

    def add(self, obs, actions, reward):
        # copying all to avoid modification
        obs = obs.cpu().numpy()
        actions = actions.cpu().numpy()

        self.observations[self.pos] = np.array(obs[0], dtype=self.np_float_type).copy()
        self.next_observations[self.pos] = np.array(
            obs[1], dtype=self.np_float_type
        ).copy()
        self.actions[self.pos] = np.array(actions, dtype=self.np_float_type).copy()
        self.rewards[self.pos] = np.array(reward, dtype=self.np_float_type).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)

        data = (
            self.observations[batch_inds, :, :],
            self.actions[batch_inds, :, :],
            self.next_observations[batch_inds, :, :],
            self.rewards[batch_inds],
        )

        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    def save_params(self, simulation_id):
        def save_obj(name, values, directory):
            with open(directory + name + ".pkl", "wb") as f:
                pickle.dump(values, f, pickle.HIGHEST_PROTOCOL)

        directory = "output/" + simulation_id + "/buffer/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        params = {
            "observations": self.observations[: self.pos],
            "next_observations": self.next_observations[: self.pos],
            "actions": self.actions[: self.pos],
            "rewards": self.rewards[: self.pos],
            "pos": self.pos,
            "full": self.full,
        }

        for name, values in params.items():
            save_obj(name, values, directory)

    def load_params(self, simulation_id):
        def load_obj(directory, name):
            with open(directory + name + ".pkl", "rb") as f:
                return pickle.load(f)

        directory = "output/" + simulation_id + "/buffer/"
        if not os.path.exists(directory):
            raise FileNotFoundError(
                "Specified directory for loading the buffer does not exist!"
            )

        self.pos = load_obj(directory, "pos")

        self.observations[: self.pos] = load_obj(directory, "observations")[: self.pos]
        self.next_observations[: self.pos] = load_obj(directory, "next_observations")[
            : self.pos
        ]
        self.actions[: self.pos] = load_obj(directory, "actions")[: self.pos]
        self.rewards[: self.pos] = load_obj(directory, "rewards")[: self.pos]

        self.full = load_obj(directory, "full")
