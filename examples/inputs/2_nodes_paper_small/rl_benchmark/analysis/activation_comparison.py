# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
tanh vs softsign actor output, on identical runs.

The single-line change ``net.mu[-1] = nn.Softsign()`` is the whole difference.
Everything else -- seeds, hyperparameters, warmup, noise schedule, the critic
architecture -- is identical between the two runs plotted here.

What the figure shows: **both** actors run away to the +100 action ceiling the
instant the warmup ends. They are not distinguished by where they go, only by
whether they can come back. In float32 ``tanh`` pins to exactly 1.0 (headroom
``1 - a² = 0``) and the actor is frozen for the remaining 8800 steps; ``softsign``
retains ~1e-2 of headroom at the same bid and climbs back into the profitable
band within about 1000 steps.

Usage::

    python run_benchmark.py --algos TD3 DDPG --critic-grid 201 --results <o>/critic_evolution.npz
    python run_benchmark.py --algos TD3 DDPG --critic-grid 201 --actor-activation softsign \\
        --results <o>/softsign.npz
    python activation_comparison.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, resolve  # noqa: E402  (also sets sys.path)
from incdec_reward import PAPER_SMALL, reward_from_bid  # noqa: E402
from run_benchmark import COLORS, INK, MUTED  # noqa: E402

FLOOR = 1e-13


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tanh", type=Path, default=resolve("critic_evolution.npz"))
    parser.add_argument("--softsign", type=Path, default=resolve("softsign.npz"))
    parser.add_argument("--algos", nargs="+", default=["TD3", "DDPG"])
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "activation_comparison.png"
    )
    args = parser.parse_args()

    p = PAPER_SMALL
    runs = {"tanh": np.load(args.tanh), "softsign": np.load(args.softsign)}
    steps = runs["tanh"]["steps"]

    fig, (ax_bid, ax_head) = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- left: where the actor actually bids -------------------------------
    ax_bid.axhspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.10, lw=0)
    ax_bid.axhline(p.optimal_bid, ls="--", lw=1.2, color=INK, zorder=1)
    ax_bid.annotate(
        f"optimum {p.optimal_bid:.0f}", xy=(steps[-1], p.optimal_bid),
        xytext=(-4, -12), textcoords="offset points", ha="right",
        fontsize=8.5, color=INK,
    )
    ax_bid.annotate(
        "dec'd band", xy=(steps[-1], p.eom_price), xytext=(-4, 4),
        textcoords="offset points", ha="right", fontsize=8.5, color="#137f59",
    )

    for name, data in runs.items():
        dashed = name == "tanh"
        for algo in args.algos:
            key = f"greedy/{algo}"
            if key not in data.files:
                continue
            ax_bid.plot(
                steps, data[key][0], lw=2.1, ls="--" if dashed else "-",
                color=COLORS.get(algo, MUTED), alpha=0.95,
                label=f"{algo} · {name}",
            )
    ax_bid.set_title(
        "both run away to the ceiling; only softsign comes back",
        loc="left", fontsize=11, color=INK,
    )
    ax_bid.set_xlabel("environment steps")
    ax_bid.set_ylabel("greedy bid (EUR/MWh)")
    ax_bid.set_ylim(-p.max_bid_price, p.max_bid_price)
    ax_bid.set_xlim(steps[0], steps[-1])
    ax_bid.legend(frameon=False, fontsize=9, loc="lower right", ncols=2)

    # --- right: the headroom that decides it --------------------------------
    for name, data in runs.items():
        dashed = name == "tanh"
        for algo in args.algos:
            key = f"greedy/{algo}"
            if key not in data.files:
                continue
            a = data[key][0] / p.max_bid_price
            head = np.maximum(1 - a**2, 0.0)
            ax_head.semilogy(
                steps, np.maximum(head, FLOOR), lw=2.1,
                ls="--" if dashed else "-",
                color=COLORS.get(algo, MUTED), label=f"{algo} · {name}",
            )
            dead = head == 0.0
            if dead.any():
                ax_head.plot(
                    steps[dead], np.full(dead.sum(), FLOOR), "v", ms=5,
                    color=COLORS.get(algo, MUTED), clip_on=False,
                )
    ax_head.set_ylim(FLOOR, 2)
    ax_head.set_title(
        "tanh headroom 1 - a²  (▼ = exactly 0, gradient dead)",
        loc="left", fontsize=11, color=INK,
    )
    ax_head.set_xlabel("environment steps")
    ax_head.set_ylabel("1 - a²")
    ax_head.set_xlim(steps[0], steps[-1])
    ax_head.legend(frameon=False, fontsize=9, loc="lower left", ncols=2)
    ax_head.annotate(
        "softsign keeps ~1e-2 of headroom at the same bid,\n"
        "because it approaches ±1 polynomially, not exponentially",
        xy=(steps[len(steps) // 3], 3e-2), fontsize=8.5, color="#137f59",
    )

    for ax in (ax_bid, ax_head):
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    # final performance, straight from the data
    caption = "   ".join(
        f"{algo} {name}: {data[f'greedy/{algo}'][:, -1].mean():.2f} EUR "
        f"({reward_from_bid(data[f'greedy/{algo}'][:, -1], p).mean():+.3f})"
        for name, data in runs.items()
        for algo in args.algos
        if f"greedy/{algo}" in data.files
    )

    fig.suptitle(
        "One line: nn.Tanh() -> nn.Softsign() on the actor output",
        x=0.006, y=1.01, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.text(0.006, 0.963, caption, fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
