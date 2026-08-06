# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Plot what the critics actually learned, against the true reward.

The actor update in DDPG/TD3/SAC is gradient ascent on the critic:
``L_actor = -Q(s, pi(s))``. So the actor never sees the reward function -- it only
ever climbs the critic's *approximation* of it. If the critic smooths the cliff
at 30 EUR/MWh into a ramp, the actor slides past the optimum and onto the zero
plateau, and no amount of extra exploration fixes that.

This script makes that visible. Run the benchmark with ``--save-models`` first:

    python run_benchmark.py --algos TD3 SAC --save-models
    python critic_landscape.py

Comparing Q against reward directly
-----------------------------------
Q and reward live on different scales (``Q ~ r / (1 - gamma)``), and a second
y-axis would be a lie. They are genuinely comparable after one shift, though: the
next state here does not depend on the action, so

    Q(s, a) = r(a) + gamma * V(s')      with s' independent of a
            = r(a) + const

A perfectly fitted critic is therefore the reward curve *plus a constant offset*.
Subtracting each curve's own mean puts both in reward units on one axis, and any
remaining difference in shape is real approximation error -- which is exactly what
the actor is misled by.

The lower panel plots ``dQ/da``, the actual gradient the actor follows. Where the
true reward is flat this should be zero; where it is not, the critic is inventing
a slope.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))

from incdec_env import IncDecEnv  # noqa: E402
from incdec_reward import PAPER_SMALL, reward_from_bid  # noqa: E402
from run_benchmark import COLORS, INK, MUTED  # noqa: E402

HERE = Path(__file__).parent


#: Where --save-models writes, and where the archived networks live. Both are
#: searched, newest working directory first.
OUT_DIR = HERE.parents[2] / "outputs" / HERE.parents[0].name / "rl_benchmark"
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


def critic_curve(model, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Evaluate the critic over a grid of actions at one fixed observation.

    Returns the value the actor actually climbs: for twin-critic algorithms
    (TD3, SAC) that is ``min(Q1, Q2)``, matching how the target is formed.
    """
    obs_batch = th.as_tensor(
        np.repeat(obs[None, :], len(actions), axis=0), dtype=th.float32
    )
    act_batch = th.as_tensor(actions[:, None], dtype=th.float32)

    with th.no_grad():
        qs = model.critic(obs_batch, act_batch)

    stacked = th.cat([q.reshape(1, -1) for q in qs], dim=0)
    return stacked.min(dim=0).values.numpy()


def greedy_action(model, obs: np.ndarray) -> float:
    action, _ = model.predict(obs, deterministic=True)
    return float(np.asarray(action).ravel()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algos", nargs="+", default=["TD3", "DDPG", "SAC"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=801)
    parser.add_argument(
        "--models",
        type=Path,
        default=None,
        help=f"directory of saved networks; default searches {', '.join(str(d) for d in MODEL_DIRS)}",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR / "critic_landscape.png")
    args = parser.parse_args()

    p = PAPER_SMALL
    env = IncDecEnv()
    obs, _ = env.reset(seed=args.seed)

    actions = np.linspace(-1.0, 1.0, args.grid)
    bids = actions * p.max_bid_price
    true_reward = reward_from_bid(bids, p)

    n = len(args.algos)
    fig, axes = plt.subplots(
        2, n, figsize=(5.2 * n, 8), sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [1.35, 1.0]},
    )

    for col, algo in enumerate(args.algos):
        model = load_model(algo, args.seed, env, args.models)
        q = critic_curve(model, obs, actions)
        a_greedy = greedy_action(model, obs)
        color = COLORS.get(algo, MUTED)

        # --- top: learned value vs true reward, both mean-centred -------------
        ax = axes[0][col]
        ax.axvspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.08, lw=0)
        ax.plot(
            bids,
            true_reward - true_reward.mean(),
            lw=2,
            color=INK,
            label="true reward",
        )
        ax.plot(bids, q - q.mean(), lw=2, color=color, label=f"{algo} critic")
        ax.axvline(a_greedy * p.max_bid_price, ls="-", lw=1.4, color=color, alpha=0.5)
        ax.annotate(
            f"actor bids\n{a_greedy * p.max_bid_price:.1f}",
            xy=(a_greedy * p.max_bid_price, ax.get_ylim()[1]),
            xytext=(-4 if a_greedy > 0 else 4, -8),
            textcoords="offset points",
            ha="right" if a_greedy > 0 else "left",
            va="top",
            fontsize=8.5,
            color=color,
        )
        ax.axvline(p.optimal_bid, ls="--", lw=1.2, color=INK, zorder=0)
        ax.set_title(algo, loc="left", fontsize=11, color=color)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        if col == 0:
            ax.set_ylabel("value, mean-centred (reward units)")

        # --- bottom: the gradient the actor actually follows -------------------
        ax = axes[1][col]
        ax.axvspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.08, lw=0)
        ax.axhline(0.0, lw=1, color=MUTED, zorder=0)
        ax.plot(bids, np.gradient(q, bids), lw=2, color=color)
        ax.axvline(p.optimal_bid, ls="--", lw=1.2, color=INK, zorder=0)
        ax.axvline(a_greedy * p.max_bid_price, ls="-", lw=1.4, color=color, alpha=0.5)
        ax.set_xlabel("bid price (EUR/MWh)")
        if col == 0:
            ax.set_ylabel("dQ/d(bid) -- the actor's ascent direction")

    for ax in axes.ravel():
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(
        "What the critic learned, and where it points the actor",
        x=0.006,
        y=1.005,
        ha="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.006,
        0.972,
        "critic is mean-centred: with an action-independent next state a perfect "
        "critic equals the reward plus a constant, so any shape difference is "
        "approximation error",
        fontsize=9,
        color=MUTED,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
