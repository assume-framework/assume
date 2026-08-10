# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Draw critic evolution and final twin-critic landscape from one ASSUME run.

The input is a per-seed ``.npz`` produced by ``assume_config_sweep.py``. No
checkpoint or retraining is required because both critics and both spatial
autograd fields were recorded at every training block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR  # noqa: E402
from incdec_reward import PAPER_SMALL, reward_from_bid  # noqa: E402
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--observation", type=int, default=0,
        help="fixed observation row to draw in the temporal fields",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR / "assume_run_diagnostics.png")
    args = parser.parse_args()

    data = np.load(args.results, allow_pickle=False)
    required = {
        "steps", "critic_bids", "critic_q/MATD3", "critic_q2/MATD3",
        "critic_grad/MATD3", "critic_grad2/MATD3", "greedy/MATD3",
    }
    missing = required - set(data.files)
    if missing:
        raise SystemExit(f"{args.results} is missing {sorted(missing)}")

    steps = data["steps"]
    bids = data["critic_bids"]
    q1 = data["critic_q/MATD3"]
    q2 = data["critic_q2/MATD3"]
    g1 = data["critic_grad/MATD3"]
    g2 = data["critic_grad2/MATD3"]
    actor = data["greedy/MATD3"]
    obs = min(max(args.observation, 0), q1.shape[0] - 1)
    config = json.loads(str(data["config_json"])) if "config_json" in data.files else {}
    label = str(data["label"]) if "label" in data.files else args.results.stem
    seed = int(data["seed"]) if "seed" in data.files else -1

    fig = plt.figure(figsize=(14, 10.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.30, wspace=0.19)
    vmax = max(np.abs(g1[obs]).max(), np.abs(g2[obs]).max(), 1e-8)
    norm = SymLogNorm(linthresh=1e-4, vmin=-vmax, vmax=vmax, base=10)
    heat_axes = []
    for col, (grad, title) in enumerate(((g1, "Q1 gradient field (actor objective)"),
                                         (g2, "Q2 gradient field"))):
        ax = fig.add_subplot(grid[0, col])
        heat_axes.append(ax)
        mesh = ax.pcolormesh(
            bids, steps, grad[obs], cmap=DIVERGING, norm=norm,
            shading="nearest", rasterized=True,
        )
        for x in (PAPER_SMALL.dec_threshold, PAPER_SMALL.eom_price):
            ax.axvline(x, color=INK, lw=0.9, alpha=0.5, ls="--")
        ax.plot(actor[obs], steps, lw=3.2, color="white", solid_capstyle="round")
        ax.plot(actor[obs], steps, lw=1.5, color=INK, solid_capstyle="round")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("bid price (EUR/MWh)")
        ax.set_xlim(bids[0], bids[-1])
        if col == 0:
            ax.set_ylabel("critic gradient steps")
        else:
            ax.tick_params(labelleft=False)
    cbar = fig.colorbar(mesh, ax=heat_axes, fraction=0.025, pad=0.015)
    cbar.set_label("dQ/d(bid), symlog autograd", color=MUTED)

    # Final landscape: faint curves retain the spatial variation across all
    # fixed observations; the selected observation is highlighted.
    ax_land = fig.add_subplot(grid[1, 0])
    reward = reward_from_bid(bids, PAPER_SMALL)
    ax_land_r = ax_land.twinx()
    reward_line, = ax_land_r.plot(
        bids, reward, color="#1baf7a", lw=1.7, alpha=0.75, label="true reward"
    )
    legend_lines = []
    for values, color, name in ((q1[:, -1], "#2a78d6", "Q1"),
                                (q2[:, -1], "#eb6834", "Q2")):
        for row in values:
            ax_land.plot(bids, row, color=color, lw=0.8, alpha=0.16)
        selected, = ax_land.plot(bids, values[obs], color=color, lw=2.0, label=name)
        legend_lines.append(selected)
    actor_last = actor[obs, -1]
    actor_line = ax_land.axvline(
        actor_last, color=INK, lw=1.2, ls=":", label="actor"
    )
    ax_land.set_title("final twin-critic landscape", loc="left", fontsize=10.5)
    ax_land.set_xlabel("bid price (EUR/MWh)")
    ax_land.set_ylabel("estimated Q")
    ax_land_r.set_ylabel("true immediate reward", color="#137f59", labelpad=10)
    legend_lines.extend((actor_line, reward_line))
    ax_land.legend(
        legend_lines,
        [line.get_label() for line in legend_lines],
        frameon=False,
        fontsize=8.5,
        loc="best",
    )

    ax_path = fig.add_subplot(grid[1, 1])
    for row in actor:
        ax_path.plot(steps, row, color="#2a78d6", lw=0.9, alpha=0.24)
    ax_path.plot(steps, actor[obs], color="#2a78d6", lw=2.0,
                 label=f"actor, observation {obs}")
    argmax1 = bids[q1.argmax(axis=2)]
    ax_path.plot(steps, argmax1[obs], color="#eb6834", lw=1.7,
                 label="argmax Q1")
    ax_path.axhspan(PAPER_SMALL.dec_threshold, PAPER_SMALL.eom_price,
                    color="#1baf7a", alpha=0.12, lw=0)
    ax_path.axhline(PAPER_SMALL.optimal_bid, color=INK, lw=1, ls="--")
    ax_path.set_ylim(-100, 100)
    ax_path.set_title("actor and critic-preferred bid", loc="left", fontsize=10.5)
    ax_path.set_xlabel("critic gradient steps")
    ax_path.set_ylabel("bid price (EUR/MWh)")
    ax_path.legend(frameon=False, fontsize=8.5)

    for ax in (*heat_axes, ax_land, ax_path):
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in (ax_land, ax_path):
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)

    fig.suptitle(
        f"ASSUME MATD3 critic diagnostics — {label}, seed {seed}",
        x=0.006, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 0.965,
        f"observation {obs + 1}/{q1.shape[0]} | lr {config.get('learning_rate', '?')} | "
        f"gradient steps/block {config.get('gradient_steps', '?')} | "
        f"batch {config.get('batch_size', '?')} | policy delay {config.get('policy_delay', '?')}",
        fontsize=9, color=MUTED,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
