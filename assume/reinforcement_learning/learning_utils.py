# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import math
from collections.abc import Callable
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch as th
from torch import nn

logger = logging.getLogger(__name__)


class ObsActRew(TypedDict):
    observation: list[th.Tensor]
    action: list[th.Tensor]
    reward: list[th.Tensor]


observation_dict = dict[list[datetime], ObsActRew]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, action noise scale ...)
Schedule = Callable[[float], float]


# Ornstein-Uhlenbeck Noise
# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    """
    A class that implements Ornstein-Uhlenbeck noise.
    """

    def __init__(self, action_dimension, mu=0, sigma=0.5, theta=0.15, dt=1e-2):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.noise_prev = np.zeros(self.action_dimension)
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else np.zeros(self.action_dimension)
        )

    def noise(self):
        noise = (
            self.noise_prev
            + self.theta * (self.mu - self.noise_prev) * self.dt
            + self.sigma
            * np.sqrt(self.dt)
            * np.random.normal(size=self.action_dimension)
        )
        self.noise_prev = noise

        return noise


class NormalActionNoise:
    """
    A Gaussian action noise that supports direct tensor creation on a given device.
    """

    def __init__(self, action_dimension, mu=0.0, sigma=0.1, scale=1.0, dt=0.9998):
        self.act_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma
        self.scale = scale
        self.dt = dt

    def noise(self, device=None, dtype=th.float):
        """
        Generates noise using torch.normal(), ensuring efficient execution on GPU if needed.

        Args:
        - device (torch.device, optional): Target device (e.g., 'cuda' or 'cpu').
        - dtype (torch.dtype, optional): Data type of the tensor (default: torch.float32).

        Returns:
        - torch.Tensor: Noise tensor on the specified device.
        """
        return (
            self.dt
            * self.scale
            * th.normal(
                mean=self.mu,
                std=self.sigma,
                size=(self.act_dimension,),
                dtype=dtype,
                device=device,
            )
        )

    def update_noise_decay(self, updated_decay: float):
        self.dt = updated_decay


def polyak_update(params, target_params, tau: float):
    """
    Perform a Polyak average update on ``target_params`` using ``params``:
    target parameters are slowly updated towards the main parameters.
    ``tau``, the soft update coefficient controls the interpolation:
    ``tau=1`` corresponds to copying the parameters to the target ones whereas nothing happens when ``tau=0``.
    The Polyak update is done in place, with ``no_grad``, and therefore does not create intermediate tensors,
    or a computation graph, reducing memory cost and improving performance.  We scale the target params
    by ``1-tau`` (in-place), add the new weights, scaled by ``tau`` and store the result of the sum in the target
    params (in place).
    See https://github.com/DLR-RM/stable-baselines3/issues/93

    Args:
        params: parameters to use to update the target params
        target_params: parameters to update
        tau: the soft update coefficient ("Polyak update", between 0 and 1)
    """
    with th.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.lerp_(param, tau)  # More efficient in-place operation


def linear_schedule_func(
    start: float, end: float = 0, end_fraction: float = 1
) -> Schedule:
    """
    Create a function that interpolates linearly between start and end
    between ``progress_remaining`` = 1 and ``progress_remaining`` = 1 - ``end_fraction``.

    Args:
        start: value to start with if ``progress_remaining`` = 1
        end: value to end with if ``progress_remaining`` = 0
        end_fraction: fraction of ``progress_remaining``
            where end is reached e.g 0.1 then end is reached after 10%
            of the complete training process.

    Returns:
        Linear schedule function.

    Note:
        Adapted from SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L100

    """

    def func(progress_remaining: float) -> float:
        if (1 - progress_remaining) > end_fraction:
            return end
        else:
            return start + (1 - progress_remaining) * (end - start) / end_fraction

    return func


def constant_schedule(val: float) -> Schedule:
    """
    Create a function that returns a constant. It is useful for learning rate schedule (to avoid code duplication)

    Args:
        val: constant value
    Returns:
        Constant schedule function.

    Note:
        From SB3: https://github.com/DLR-RM/stable-baselines3/blob/512eea923afad6f6da4bb53d72b6ea4c6d856e59/stable_baselines3/common/utils.py#L124

    """

    def func(_):
        return val

    return func


def get_hidden_sizes(state_dict: dict, prefix: str) -> list[int]:
    sizes = []
    i = 0
    while f"{prefix}.{i}.weight" in state_dict:
        weight = state_dict[f"{prefix}.{i}.weight"]
        out_dim = weight.shape[0]
        sizes.append(out_dim)
        i += 1
    return sizes[:-1]  # exclude the final output layer if needed


def infer_n_agents(state_dict, obs_base, act_dim, unique_obs_dim):
    key = "q1_layers.0.weight"
    if key not in state_dict:
        raise KeyError(f"Missing expected key '{key}' in state_dict.")

    input_dim = state_dict[key].shape[1]

    # If both are 0, cannot infer
    denom = unique_obs_dim + act_dim
    if denom == 0:
        raise ValueError("Cannot infer N: unique_obs_dim and act_dim both zero.")

    # Proper formula derived from CriticTD3 logic
    num = input_dim - obs_base + unique_obs_dim
    n_est = num / denom
    n_rounded = round(n_est)

    # Reconstruct the original input_dim as CriticTD3 would
    reconstructed_input = (
        obs_base + unique_obs_dim * (n_rounded - 1) + act_dim * n_rounded
    )

    if not math.isclose(reconstructed_input, input_dim) or n_rounded <= 0:
        raise ValueError(
            f"Inferred N={n_rounded} is invalid (Recalculated input={reconstructed_input}, got={input_dim})"
        )

    return n_rounded


def copy_layer_data(dst, src):
    for k in dst:
        if k in src and dst[k].shape == src[k].shape:
            dst[k].data.copy_(src[k].data)


def transfer_weights(
    model: nn.Module,
    old_state: dict,
    old_n_agents: int,
    new_n_agents: int,
    obs_base: int,
    act_dim: int,
    unique_obs: int,
) -> dict | None:
    # Check architecture compatibility (extract hidden sizes from state_dicts)
    old_hidden = get_hidden_sizes(old_state, prefix="q1_layers")
    new_hidden = get_hidden_sizes(model.state_dict(), prefix="q1_layers")
    if old_hidden != new_hidden:
        logger.warning(
            f"Cannot transfer weights: architecture mismatch.\n"
            f"Old hidden sizes: {old_hidden}, New hidden sizes: {new_hidden}. Skipping transfer."
        )
        return None

    new_template = model.state_dict()
    new_state = {k: v.clone() for k, v in new_template.items()}

    # Calculate complete dimensions from both old and new agent counts
    old_obs = obs_base + unique_obs * max(0, old_n_agents - 1)
    new_obs = obs_base + unique_obs * max(0, new_n_agents - 1)

    # Determine how many agents’ data can be safely copied.
    copy_agent_count = min(old_n_agents, new_n_agents)
    # For the unique observations we copy only (copy_agent_count - 1) agents worth,
    # because the first agent’s observations are already counted in obs_base.
    copy_unique_obs = unique_obs * max(0, copy_agent_count - 1)
    # Thus the total observation columns to copy are:
    copy_obs_end = obs_base + copy_unique_obs
    # For actions, we copy act_dim columns per agent for copy_agent_count agents:
    copy_action_count = act_dim * copy_agent_count

    for prefix in ["q1_layers", "q2_layers"]:
        try:
            # Get input layer weights and biases
            w_old = old_state[f"{prefix}.0.weight"]
            b_old = old_state[f"{prefix}.0.bias"]
            w_new = new_state[f"{prefix}.0.weight"]
            b_new = new_state[f"{prefix}.0.bias"]

            # --- Copy the shared observation part ---
            w_new[:, :obs_base] = w_old[:, :obs_base]

            # --- Copy the agent-unique observation part ---
            # (Copy only if there is any agent-unique portion to copy.)
            if copy_obs_end > obs_base:
                w_new[:, obs_base:copy_obs_end] = w_old[:, obs_base:copy_obs_end]

            # --- Copy the action part ---
            # In the old state the action portion starts at old_obs,
            # and in the new state it starts at new_obs.
            w_new[:, new_obs : new_obs + copy_action_count] = w_old[
                :, old_obs : old_obs + copy_action_count
            ]

            # --- Copy bias from the input layer (assumed to have matching shape) ---
            b_new.copy_(b_old)

            # --- Copy deeper layers ---
            for i in range(1, len(new_hidden) + 1):
                w_key, b_key = f"{prefix}.{i}.weight", f"{prefix}.{i}.bias"
                copy_layer_data({w_key: new_state[w_key]}, old_state)
                copy_layer_data({b_key: new_state[b_key]}, old_state)

        except KeyError as e:
            logger.warning(f"Missing key for {prefix} during transfer: {e}")
            return None

    return new_state
