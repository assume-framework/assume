# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Read a trained critic: the value surface over actions, and its true gradient.

Kept separate from both ``run_benchmark`` (which records these during training)
and ``critic_landscape`` (which plots them afterwards), so neither has to import
the other.

The gradient here comes from ``torch.autograd`` -- it is the same quantity that
``actor_loss.backward()`` propagates into the actor, not a finite difference of a
sampled curve. On this landscape the two agree to ~0.1% in the flat regions but
diverge by up to ~4e-2 at the kink near 32 EUR/MWh, where the sampled curve has a
corner and a finite difference straddles it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR  # noqa: E402  (also puts the folders on sys.path)

#: Saved networks are searched for in the live output folder first, then in the
#: run archive, so an archived run keeps working in place.
MODEL_DIRS = (
    OUT_DIR / "models",
    OUT_DIR / "runs" / "data" / "02-critic" / "models",
)


def load_model(algo: str, seed: int, env, model_dir: Path | None = None):
    from stable_baselines3 import DDPG, SAC, TD3

    classes = {"TD3": TD3, "DDPG": DDPG, "SAC": SAC}
    if algo not in classes:
        raise ValueError(f"{algo} has no critic to inspect (off-policy only)")

    candidates = [model_dir] if model_dir else MODEL_DIRS
    for directory in candidates:
        path = directory / f"{algo}_seed{seed}.zip"
        if path.exists():
            return classes[algo].load(path, env=env, device="cpu")

    searched = "\n  ".join(str(d) for d in candidates)
    raise FileNotFoundError(
        f"no saved {algo} seed {seed} in:\n  {searched}\n"
        f"run: python run_benchmark.py --algos {algo} --save-models"
    )


def actor_objective(model, obs_batch: th.Tensor, act_batch: th.Tensor) -> th.Tensor:
    """The quantity the actor actually performs gradient ascent on.

    This differs by algorithm, and the difference is not cosmetic:

    * **TD3 / DDPG** (``td3.py:199``: ``-critic.q1_forward(...)``) climb **Q1
      alone** -- not ``min(Q1, Q2)``. The min forms the *critic target*, not the
      actor loss. Using the min instead shifts the apparent argmax by ~0.6
      EUR/MWh on the trained TD3 critic.
    * **SAC** (``sac.py:281``: ``(ent_coef * log_prob - min_qf_pi)``) climbs
      ``min(Q1, Q2)``. Its full objective also carries ``-alpha * log pi(a|s)``,
      a property of the policy rather than of a fixed action grid, so it is not
      included here: this is the Q-part of SAC's ascent direction only.
    """
    from stable_baselines3 import SAC

    if isinstance(model, SAC):
        qs = th.cat(model.critic(obs_batch, act_batch), dim=1)
        return qs.min(dim=1, keepdim=True).values
    return model.critic.q1_forward(obs_batch, act_batch)


def critic_curve(
    model, obs: np.ndarray, actions: np.ndarray, max_bid_price: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep the actor's objective and its autograd gradient over an action grid.

    Each row's Q depends only on that row's action, so differentiating the sum
    yields every per-row derivative in a single backward pass.

    Returns
    -------
    (q, dq_dbid)
        ``q`` in critic units; ``dq_dbid`` per EUR/MWh -- autograd's
        ``d(Q)/d(action)`` divided by ``max_bid_price``, since the action is the
        bid scaled into ``[-1, 1]``.
    """
    obs_batch = th.as_tensor(
        np.repeat(obs[None, :], len(actions), axis=0), dtype=th.float32
    )
    act_batch = th.as_tensor(actions[:, None], dtype=th.float32)
    act_batch.requires_grad_(True)

    q = actor_objective(model, obs_batch, act_batch)
    (grad,) = th.autograd.grad(q.sum(), act_batch)

    return q.detach().numpy().ravel(), grad.numpy().ravel() / max_bid_price


def greedy_action(model, obs: np.ndarray) -> float:
    action, _ = model.predict(obs, deterministic=True)
    return float(np.asarray(action).ravel()[0])
