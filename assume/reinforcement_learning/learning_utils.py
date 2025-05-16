# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from collections.abc import Callable
from datetime import datetime
from typing import TypedDict

import numpy as np
import torch as th

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


def copy_layer_data(dst, src):
    for k in dst:
        if k in src and dst[k].shape == src[k].shape:
            dst[k].data.copy_(src[k].data)


def transfer_weights(
    model: th.nn.Module,
    loaded_state: dict,
    loaded_id_order: list[str],
    new_id_order: list[str],
    obs_base: int,
    act_dim: int,
    unique_obs: int,
) -> dict | None:
    """
    Transfer weights from loaded model to new model. Copy only those obs_ and action-slices for matching IDs.
    New IDs keep their original (random) weights. Function only works if the neural network architeczture remained stable besides the input layer, namely with the same hidden layers.

    Args:
        model (th.nn.Module): The model to transfer weights to.
        loaded_state (dict): The state dictionary of the loaded model.
        loaded_id_order (list[str]): The list of unit IDs from the loaded model that shows us the order of units.
        new_id_order (list[str]): The list of IDs from the new model, includes potentially different agents in comparison to the loaded model.
        obs_base (int): The base observation size.
        act_dim (int): The action dimension size.
        unique_obs (int): The unique observation size per agent, smaller than obs_base as these include also shared observation values.

    returns:
        dict | None: The updated state dictionary with transferred weights, or None if architecture mismatch.
    """

    # 1) Architecture check
    new_state = model.state_dict()
    loaded_hidden = get_hidden_sizes(loaded_state, prefix="q1_layers")
    new_hidden = get_hidden_sizes(new_state, prefix="q1_layers")
    if loaded_hidden != new_hidden:
        logger.warning(
            f"Cannot transfer weights: neural network architecture mismatch.\n"
            f"Loaded sizes: {loaded_hidden}, New sizes: {new_hidden}."
        )
        return None

    # 2) Compute total dims
    loaded_n = len(loaded_id_order)
    new_n = len(new_id_order)
    loaded_obs_tot = obs_base + unique_obs * max(0, loaded_n - 1)
    new_obs_tot = obs_base + unique_obs * max(0, new_n - 1)

    # 3) Clone new state
    new_state_copy = {k: v.clone() for k, v in new_state.items()}

    # 4) Transfer per-prefix
    for prefix in ("q1_layers", "q2_layers"):
        w_loaded = loaded_state[f"{prefix}.0.weight"]
        b_loaded = loaded_state[f"{prefix}.0.bias"]
        w_new = new_state_copy[f"{prefix}.0.weight"]
        b_new = new_state_copy[f"{prefix}.0.bias"]
        w_loaded_random = new_state[f"{prefix}.0.weight"].clone()

        # a) shared obs_base
        w_new[:, :obs_base] = w_loaded[:, :obs_base]

        # b) matched agents’ ID
        # copy weights from loaded to new model
        for new_idx, u in enumerate(new_id_order):
            if u not in loaded_id_order:
                continue
            loaded_idx = loaded_id_order.index(u)

            # unique_obs for agents beyond the first
            if new_idx > 0 and loaded_idx > 0:
                ns = obs_base + unique_obs * (new_idx - 1)
                os_ = obs_base + unique_obs * (loaded_idx - 1)
                w_new[:, ns : ns + unique_obs] = w_loaded[:, os_ : os_ + unique_obs]

            # action blocks for every agent
            new_act = new_obs_tot + act_dim * new_idx
            loaded_act = loaded_obs_tot + act_dim * loaded_idx
            w_new[:, new_act : new_act + act_dim] = w_loaded[
                :, loaded_act : loaded_act + act_dim
            ]

        # c) unmatched agents’ ID
        # use randomly initialized weights for unmatched agents
        for new_idx, u in enumerate(new_id_order):
            if new_idx == 0 or u in loaded_id_order:
                continue
            ns = obs_base + unique_obs * (new_idx - 1)
            w_new[:, ns : ns + unique_obs] = w_loaded_random[:, ns : ns + unique_obs]
            # actions untouched

        # d) bias and deeper layers
        # copy all other wigths and biases (besides input layer) from loaded to new model
        b_new.copy_(b_loaded)
        for i in range(1, len(new_hidden) + 1):
            new_state_copy[f"{prefix}.{i}.weight"].copy_(
                loaded_state[f"{prefix}.{i}.weight"]
            )
            new_state_copy[f"{prefix}.{i}.bias"].copy_(
                loaded_state[f"{prefix}.{i}.bias"]
            )

    return new_state_copy
