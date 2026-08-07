# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
How the critic's gradient field evolves, and why the actor stops following it.

Plots a run recorded with ``--critic-grid N``.

Why a small gradient is *not* by itself the problem
---------------------------------------------------
Adam updates by ``lr * m_hat / (sqrt(v_hat) + eps)``. For a consistently-signed
gradient of magnitude ``G`` both moments settle near ``G``, so the step is
``lr * G / (G + eps)``:

* ``G >> eps``  -> step ~ ``lr``, regardless of how small ``G`` is in absolute
  terms. A steadily-directed 1e-5 gradient moves the actor just as fast as a
  1e-1 one.
* ``G ~ eps``   -> steps are damped proportionally.
* ``G == 0``    -> nothing moves, and no ``eps`` can help.

So the quantity that decides the actor's fate is not ``dQ/da``; it is the full
chain-rule product that reaches the parameters, measured against ``eps``:

    dL/dtheta = dQ/da * (1 - a^2) * dz/dtheta        with a = tanh(z)

The ``1 - a^2`` factor comes from the tanh on the actor's output layer, and it is
exact from the recorded greedy action -- no extra instrumentation needed. In
float32 ``tanh(z)`` rounds to exactly 1.0 for ``|z| >~ 9``, at which point the
factor is not merely small but *identically zero*.

Panels
------
1. ``dQ/d(bid)`` over (bid, training step), with the actor's trajectory on top.
   Symmetric log colour scale: the field spans a +-0.1 spike at the cliff and a
   ~1e-5 background on the plateaus.
2. The chain-rule product against Adam's ``eps`` -- the "does the actor still
   move" test.
3. The *signed* gradient at the actor's own action -- because Adam only delivers
   full-sized steps while the direction stays consistent.

Usage::

    python run_benchmark.py --algos TD3 DDPG SAC --critic-grid 201 \\
        --results <out>/critic_evolution.npz
    python critic_evolution.py --results <out>/critic_evolution.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, resolve  # noqa: E402  (also sets sys.path)
from incdec_reward import PAPER_SMALL  # noqa: E402
from run_benchmark import COLORS, INK, MUTED  # noqa: E402

HERE = Path(__file__).parent

#: Adam's division guard, from the live optimizer (torch default, unchanged by SB3).
ADAM_EPS = 1e-8
#: Floor for log axes, since a saturated tanh gives exactly zero.
FLOOR = 1e-13


def headroom(action: np.ndarray, activation: str) -> np.ndarray:
    """``df/dz`` of the actor's output squashing, expressed in terms of its output.

    This is the factor that multiplies ``dQ/da`` on its way into the actor's
    weights, and the whole reason tanh and softsign behave differently:

    * ``tanh``:     ``a = tanh(z)``  ->  ``df/dz = 1 - a²``, which is *exactly*
      zero once float32 rounds ``tanh(z)`` to 1.0 (|z| >~ 9);
    * ``softsign``: ``a = z/(1+|z|)`` -> ``df/dz = 1/(1+|z|)² = (1 - |a|)²``,
      which is positive for every ``|a| < 1``.
    """
    a = np.abs(action)
    if activation == "tanh":
        return np.maximum(1 - a**2, 0.0)
    if activation == "softsign":
        return np.maximum(1 - a, 0.0) ** 2
    raise ValueError(f"unknown actor activation {activation!r}")

# Diverging ramp: two poles with a neutral -- not a hue -- at the midpoint, so
# "no gradient" reads as absence of colour rather than as a colour.
DIVERGING = LinearSegmentedColormap.from_list(
    "grad", ["#2a78d6", "#9dc2ec", "#f2f2f0", "#f4b79c", "#eb6834"]
)


def load(path: Path):
    data = np.load(path)
    if "critic_bids" not in data.files:
        raise SystemExit(
            f"{path} holds no critic sweeps -- re-run with --critic-grid 201"
        )
    algos = [k.split("/", 1)[1] for k in data.files if k.startswith("critic_grad/")]
    return data, algos


def at_actor(bids: np.ndarray, grad: np.ndarray, actor_bids: np.ndarray):
    """dQ/d(bid) sampled at the actor's own action, one value per probe."""
    return np.array([np.interp(b, bids, g) for b, g in zip(actor_bids, grad)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=resolve("critic_evolution.npz"))
    parser.add_argument("--seed", type=int, default=0, help="which seed row to draw")
    parser.add_argument(
        "--activation",
        choices=("tanh", "softsign"),
        default="tanh",
        help="the actor's output squashing, which sets the headroom factor in the "
        "lower-left panel. SB3 runs are tanh unless --actor-activation said "
        "otherwise; ASSUME's own Actor is softsign",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR / "critic_evolution.png")
    args = parser.parse_args()

    p = PAPER_SMALL
    data, algos = load(args.results)
    steps = data["steps"]
    bids = data["critic_bids"]

    n = len(algos)
    fig = plt.figure(figsize=(5.4 * n, 10))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45, 1.0], hspace=0.3)
    top = outer[0].subgridspec(1, n, wspace=0.1)
    bottom = outer[1].subgridspec(1, 2, wspace=0.24)

    vmax = max(np.abs(data[f"critic_grad/{a}"][args.seed]).max() for a in algos)

    heat_axes = []
    for col, algo in enumerate(algos):
        grad = data[f"critic_grad/{algo}"][args.seed]
        actor_bids = data[f"greedy/{algo}"][args.seed]

        ax = fig.add_subplot(top[col])
        heat_axes.append(ax)
        im = ax.pcolormesh(
            bids, steps, grad,
            cmap=DIVERGING,
            norm=SymLogNorm(linthresh=1e-4, vmin=-vmax, vmax=vmax, base=10),
            shading="nearest", rasterized=True,
        )
        for x in (p.dec_threshold, p.eom_price):
            ax.axvline(x, color=INK, lw=0.9, alpha=0.45, ls="--")
        # halo so the trajectory stays readable over both colour poles
        ax.plot(actor_bids, steps, lw=3.2, color="white", solid_capstyle="round")
        ax.plot(actor_bids, steps, lw=1.6, color=INK, solid_capstyle="round")

        ax.set_title(algo, loc="left", fontsize=11, color=COLORS.get(algo, INK))
        ax.set_xlabel("bid price (EUR/MWh)")
        ax.set_xlim(bids[0], bids[-1])
        if col == 0:
            ax.set_ylabel("environment steps")
            ax.annotate(
                "actor",
                xy=(actor_bids[len(actor_bids) // 2], steps[len(steps) // 2]),
                xytext=(-44, 0), textcoords="offset points", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1),
            )
        else:
            ax.tick_params(labelleft=False)

    cbar = fig.colorbar(im, ax=heat_axes, fraction=0.02, pad=0.012)
    cbar.set_label("dQ/d(bid)   (symlog, autograd)", fontsize=9, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    # --- bottom left: does anything actually reach the parameters? ------------
    ax_mag = fig.add_subplot(bottom[0])
    ax_mag.axhspan(ADAM_EPS * 100, 1e2, color="#1baf7a", alpha=0.07, lw=0)
    ax_mag.axhspan(FLOOR, ADAM_EPS, color="#e34948", alpha=0.07, lw=0)
    ax_mag.axhline(ADAM_EPS, ls="--", lw=1.3, color=INK)
    ax_mag.annotate(
        "Adam eps = 1e-8", xy=(steps[-1], ADAM_EPS), xytext=(-4, 5),
        textcoords="offset points", ha="right", fontsize=8.5, color=INK,
    )
    ax_mag.annotate(
        "above here Adam still delivers ~lr-sized steps,\n"
        "however small the gradient is",
        xy=(steps[1], 1e-2), fontsize=8.5, color="#137f59",
    )

    # TD3 and DDPG both die at exactly the same step and would draw on top of
    # each other -- dash the later series so both stay visible.
    styles = ["-", "--", "-.", ":"]
    for i, algo in enumerate(algos):
        a = data[f"greedy/{algo}"][args.seed] / p.max_bid_price
        g = at_actor(bids, data[f"critic_grad/{algo}"][args.seed],
                     data[f"greedy/{algo}"][args.seed])
        product = np.abs(g) * headroom(a, args.activation)
        ax_mag.semilogy(
            steps, np.maximum(product, FLOOR), lw=2, ls=styles[i % len(styles)],
            color=COLORS.get(algo, MUTED), label=algo,
        )
        dead = product == 0.0
        if dead.any():
            # stagger the dead-markers so overlapping series stay countable
            ax_mag.plot(
                steps[dead], np.full(dead.sum(), FLOOR * 10 ** (0.55 * i)), "v",
                ms=5, color=COLORS.get(algo, MUTED), clip_on=False,
            )
    ax_mag.set_ylim(FLOOR, 1e0)
    factor = "1 - a²" if args.activation == "tanh" else "(1 - |a|)²"
    ax_mag.set_title(
        f"|dQ/da x {factor}| -- the gradient that reaches the actor's weights",
        loc="left", fontsize=10.5, color=INK,
    )
    ax_mag.set_xlabel("environment steps")
    ax_mag.set_ylabel("chain-rule product")
    ax_mag.legend(frameon=False, fontsize=9, loc="center right")
    ax_mag.annotate(
        "▼ = exactly 0: tanh saturated to 1.0 in float32, so 1 - a² is\n"
        "identically zero. Adam gets 0/(0+eps) -- no eps can help.",
        xy=(steps[0], FLOOR), xytext=(2, 26), textcoords="offset points",
        fontsize=8, color="#b03a39",
    )

    # --- bottom right: is the direction consistent? ---------------------------
    ax_dir = fig.add_subplot(bottom[1])
    for algo in algos:
        g = at_actor(bids, data[f"critic_grad/{algo}"][args.seed],
                     data[f"greedy/{algo}"][args.seed])
        ax_dir.plot(steps, g, lw=2, color=COLORS.get(algo, MUTED), label=algo)
    ax_dir.axhline(0.0, lw=1.2, color=INK, zorder=0)
    ax_dir.set_yscale("symlog", linthresh=1e-6)
    ax_dir.set_title(
        "dQ/d(bid) at the actor's own action (sign = which way it is pushed)",
        loc="left", fontsize=10.5, color=INK,
    )
    ax_dir.set_xlabel("environment steps")
    ax_dir.set_ylabel("dQ/d(bid) at a = π(s)")
    ax_dir.legend(frameon=False, fontsize=9, loc="best")

    for ax in (*heat_axes, ax_mag, ax_dir):
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in (ax_mag, ax_dir):
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)

    fig.suptitle(
        "The actor stops moving before the gradient runs out",
        x=0.006, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 0.963,
        "Adam rescales a consistently-signed gradient to ~lr-sized steps, so 1e-5 is not small. "
        "The tanh output saturates instead, and that factor multiplies the update to exactly zero.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
