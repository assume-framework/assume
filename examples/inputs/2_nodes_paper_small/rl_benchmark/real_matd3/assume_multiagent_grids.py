# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 13's three per-seed views: nothing pooled, every trial drawn on its own.

``assume_multiagent_film.py`` pools the seeds with the probed observations, which
is the right summary but hides the thing run 13 turned on -- that ``act-all-x2``
at 25 episodes looked like three different outcomes and was in fact three runs
caught at different points of one descent. These three figures show every trial.

Figures
-------
``critic``   condition x seed grid of ``dQ1/d(bid)`` at ``diesel_0``. Deliberately
             stripped of per-cell chrome: one shared x axis, one shared **step**
             axis so a 1200-update run fills only the lower part of the panel and
             the budgets are visually comparable, one shared colour bar. Each
             panel is normalised by its own 98th percentile |grad|, because the
             action-scale conditions' recorded gradient carries the factor S --
             so the bar reads as *relative* strength and sign, never magnitude
             across panels.
``bids``     the same grid for every agent's greedy bid, with the usual axes and
             one shared unit legend.
``summary``  run 13's summary row at three seeds: ``diesel_0``'s bid, the
             observation disagreement about ``argmax Q1``, and the fleet reward.
             Thin lines are seeds, thick lines their median.

Usage::

    python real_matd3/assume_multiagent_grids.py
    python real_matd3/assume_multiagent_grids.py --only critic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR  # noqa: E402  (also sets sys.path)
from assume_multiagent_actshare import (  # noqa: E402
    CONDITIONS,
    act_share_from_sd,
    result_path,
)
from assume_multiagent_film import CONDITION_COLOR, FOCUS, argmax_spread  # noqa: E402
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

#: rows of every grid, ordered by budget then own act_share -- the order the
#: ladder is read in
ORDER = [
    "baseline-25", "act-all-x2", "act-all-x15",
    "baseline", "act-all-x2-50", "act-own-x15",
]
SEEDS = [42, 1, 2]


def load(out_dir: Path, name: str, seed: int) -> dict | None:
    path = result_path(out_dir, name, seed)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    units = [str(u) for u in d["unit_ids"]]
    i = units.index(FOCUS)
    scale, mode = CONDITIONS[name]["scale"], CONDITIONS[name]["mode"]
    return {
        "name": name, "seed": seed, "units": units,
        "steps": d["steps"], "bids": d["critic_bids"],
        "grad_focus": d["critic_grad/MATD3"][i],
        "q1": d["critic_q/MATD3"],
        "greedy": d["greedy/MATD3"],
        "greedy_focus": d["greedy/MATD3"][i],
        "rewards": d["rewards"],
        "share": act_share_from_sd(
            d["buffer_sd_obs"], d["buffer_sd_act"], int(d["unique_obs_dim"]),
            scale, mode).mean(),
    }


def grid(out_dir: Path, conditions: list[str], seeds: list[int]):
    return {(c, s): load(out_dir, c, s) for c in conditions for s in seeds}


def strip(ax, keep_left: bool, keep_bottom: bool) -> None:
    """Remove every piece of chrome a shared-axis grid does not need."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7, length=2.5, pad=1.5)
    if not keep_left:
        ax.tick_params(labelleft=False)
    if not keep_bottom:
        ax.tick_params(labelbottom=False)


# ------------------------------------------------------------------- critic


def plot_critic(runs: dict, conditions: list[str], seeds: list[int], out: Path) -> None:
    live = [c for c in conditions if any(runs[(c, s)] for s in seeds)]
    top = max(int(r["steps"][-1]) for r in runs.values() if r)

    fig, axes = plt.subplots(
        len(live), len(seeds), figsize=(2.55 * len(seeds) + 1.4, 1.72 * len(live) + 1.1),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(hspace=0.10, wspace=0.06, right=0.86)
    mesh = None

    for row, name in enumerate(live):
        for col, seed in enumerate(seeds):
            ax = axes[row][col]
            run = runs[(name, seed)]
            if run is None:
                ax.set_axis_off()
                continue
            field = np.median(run["grad_focus"], axis=0)
            # normalised per panel: the action-scale conditions' gradient carries
            # the factor S, so only sign and relative strength are comparable
            scale = np.percentile(np.abs(field), 98) or 1.0
            mesh = ax.pcolormesh(
                run["bids"], run["steps"], field / scale,
                cmap=DIVERGING, shading="nearest",
                norm=SymLogNorm(linthresh=0.02, vmin=-1, vmax=1),
            )
            ax.plot(np.median(run["greedy_focus"], axis=0), run["steps"],
                    lw=1.1, color=INK, alpha=0.8)
            ax.set_ylim(0, top)
            strip(ax, keep_left=col == 0, keep_bottom=row == len(live) - 1)
            if row == 0:
                ax.set_title(f"seed {seed}", fontsize=9, color=INK, pad=4)
            if col == 0:
                r0 = next(runs[(name, s)] for s in seeds if runs[(name, s)])
                ax.set_ylabel(f"{name}\nact_share {r0['share']:.3f}",
                              fontsize=8, color=INK, labelpad=4)

    # interior ticks only: with shared axes the edge labels of neighbouring
    # panels sit on top of each other
    for col in range(len(seeds)):
        axes[-1][col].set_xticks([-50, 0, 50])
        axes[-1][col].set_xlabel("bid [EUR/MWh]", fontsize=8.5, color=MUTED)

    cax = fig.add_axes([0.885, 0.13, 0.018, 0.72])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("dQ1/d(bid), normalised within each panel\n"
                   "blue: pulls the bid down", fontsize=7.5, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)

    fig.suptitle(f"Run 13 -- critic evolution at {FOCUS}, every trial",
                 x=0.005, y=0.995, ha="left", fontsize=12.5, fontweight="bold",
                 color=INK)
    fig.text(0.005, 0.963,
             f"shared step axis (0-{top}), so the 1200-update rows fill only the "
             f"lower part. Black: {FOCUS}'s own bid.",
             ha="left", fontsize=7.5, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")


# --------------------------------------------------------------------- bids


def plot_bids(runs: dict, conditions: list[str], seeds: list[int], out: Path) -> None:
    live = [c for c in conditions if any(runs[(c, s)] for s in seeds)]
    top = max(int(r["steps"][-1]) for r in runs.values() if r)
    cmap = plt.get_cmap("viridis")
    handles = None

    fig, axes = plt.subplots(
        len(live), len(seeds), figsize=(3.5 * len(seeds), 2.3 * len(live)),
        sharex=True, sharey=True, squeeze=False,
    )
    fig.subplots_adjust(hspace=0.42, wspace=0.13, bottom=0.10)

    for row, name in enumerate(live):
        for col, seed in enumerate(seeds):
            ax = axes[row][col]
            run = runs[(name, seed)]
            if run is None:
                ax.set_axis_off()
                continue
            greedy = np.median(run["greedy"], axis=1)
            for i, unit in enumerate(run["units"]):
                ax.plot(run["steps"], greedy[i], lw=1.0,
                        color=cmap(i / max(1, len(run["units"]) - 1)),
                        alpha=0.85, label=None if unit == FOCUS else unit)
            ax.plot(run["steps"], greedy[run["units"].index(FOCUS)],
                    lw=2.1, color=INK, alpha=0.9, zorder=5, label=FOCUS)
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
            ax.axhline(0, color=MUTED, lw=0.7, ls=":")
            ax.set_ylim(-105, 105)
            ax.set_xlim(0, top)
            ax.set_title(f"{name}  seed {seed}", loc="left", fontsize=9, color=INK)
            ax.grid(True, color=MUTED, alpha=0.2, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            if col == 0:
                ax.set_ylabel("bid [EUR/MWh]", fontsize=8.5, color=MUTED)
            if row == len(live) - 1:
                ax.set_xlabel("critic gradient steps", fontsize=8.5, color=MUTED)

    fig.legend(handles, labels, loc="lower center", ncol=12, frameon=False,
               fontsize=8, labelcolor=MUTED, bbox_to_anchor=(0.5, -0.012))
    fig.suptitle(f"Run 13 -- every agent's greedy bid, every trial ({FOCUS} in black)",
                 x=0.005, y=0.997, ha="left", fontsize=12.5, fontweight="bold",
                 color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")


# ------------------------------------------------------------------ summary


# argmax_spread is imported from assume_multiagent_film at the top of this
# module, so the per-seed grid and the pooled view cannot drift apart. Its
# definition lives in analysis/critic_coherence.py.


def plot_summary(runs: dict, conditions: list[str], seeds: list[int], out: Path) -> None:
    live = [c for c in conditions if any(runs[(c, s)] for s in seeds)]
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.4))

    series = {
        0: lambda r: np.median(r["greedy_focus"], axis=0),
        1: argmax_spread,
        2: lambda r: r["rewards"].sum(axis=1),
    }
    for name in live:
        colour = CONDITION_COLOR.get(name, MUTED)
        present = [runs[(name, s)] for s in seeds if runs[(name, s)]]
        for k, ax in enumerate(axes):
            stack = [series[k](r) for r in present]
            for y, r in zip(stack, present):
                ax.plot(r["steps"], y, lw=0.8, color=colour, alpha=0.35)
            n = min(len(y) for y in stack)
            ax.plot(present[0]["steps"][:n],
                    np.median([y[:n] for y in stack], axis=0),
                    lw=2.2, color=colour, label=f"{name} (n={len(present)})")

    axes[0].axhspan(30, 49, color=MUTED, alpha=0.16, zorder=0)
    axes[0].annotate("single-agent band [30, 49]", xy=(0.02, 0.06),
                     xycoords="axes fraction", fontsize=7.5, color=MUTED)
    axes[0].set_ylim(-105, 105)
    axes[0].set_title(f"{FOCUS}: greedy bid", loc="left", fontsize=10, color=INK)
    axes[0].set_ylabel("bid [EUR/MWh]")
    axes[1].set_title("argmax Q1 disagreement across observations\n"
                      "(run 10's incoherence statistic, mean over agents)",
                      loc="left", fontsize=10, color=INK)
    axes[1].set_ylabel("EUR/MWh")
    axes[2].set_title("fleet reward: sum over agents of the mean\n"
                      "stored reward over the most recent episode",
                      loc="left", fontsize=10, color=INK)

    for ax in axes:
        ax.set_xlabel("critic gradient steps")
        ax.legend(fontsize=7.5, frameon=False, labelcolor=MUTED)
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle("Run 13 -- summary at three seeds (thin: seeds, thick: median)",
                 x=0.005, y=1.06, ha="left", fontsize=12.5, fontweight="bold",
                 color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=ORDER)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--only", choices=["critic", "bids", "summary"], default=None)
    parser.add_argument(
        "--data-dir", type=Path,
        default=OUT_DIR / "runs" / "data" / "13-multiagent-actshare")
    parser.add_argument("--img-dir", type=Path, default=OUT_DIR / "runs" / "img")
    args = parser.parse_args()

    runs = grid(args.data_dir, args.conditions, args.seeds)
    if not any(runs.values()):
        raise SystemExit(f"no results under {args.data_dir}")

    if args.only in (None, "critic"):
        plot_critic(runs, args.conditions, args.seeds,
                    args.img_dir / "13-multiagent-critic-grid.png")
    if args.only in (None, "bids"):
        plot_bids(runs, args.conditions, args.seeds,
                  args.img_dir / "13-multiagent-bids-grid.png")
    if args.only in (None, "summary"):
        plot_summary(runs, args.conditions, args.seeds,
                     args.img_dir / "13-multiagent-summary.png")


if __name__ == "__main__":
    main()
