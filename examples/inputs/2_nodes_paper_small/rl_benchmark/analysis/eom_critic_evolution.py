# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Draw the plain-EOM critic films recorded by ``real_matd3/eom_critic_film.py``.

``critic_evolution.py`` is the single-agent, single-bid-axis version of this and
carries the argument about why a *small* gradient is not the problem (Adam is
scale-invariant; a gradient of exactly zero is). That argument is unchanged here,
so it is not repeated -- what changes is the shape of the data:

* several learning units per run, each with its own critic, so the heatmap
  becomes a **row of panels, one per learning unit**;
* the bid structure varies by case. The two-bid cases (``02a``-``02c``,
  ``EnergyLearningStrategy``) have ``act_dim = 2``, hence ``act_dim + 1`` sweeps
  per agent and **two** actor trajectories -- ``min`` of its actions is the
  inflexible price and ``max`` the flexible one, which is how ``calculate_bids``
  assigns them. The single-bid cases (``sb02a``-``sb02c``,
  ``EnergyLearningSingleBidStrategy``) have ``act_dim = 1``, one sweep and one
  trajectory. ``bid_series`` is the one place that knows the difference;
* there is no closed-form landscape to draw reference lines against. On the
  inc-dec scenario the figures mark the dec threshold and the EOM price; the
  only comparable line here is each unit's **marginal cost**, which is where a
  truthful bidder would sit, so that is what is drawn.

Two figures per invocation:

``film``     one PNG per (case, seed): a panel per learning unit showing
             ``dQ1/d(bid)`` over (bid, critic updates) with the actor's bid
             trajectory on top, plus a greedy-bid axis and a buffer-reward
             axis underneath.
``summary``  one PNG for the set: final bid, ``act_share`` and mean buffer
             reward across the cases drawn, i.e. against the number of learning
             agents. Thin lines are seeds, thick lines their median. The x axis
             follows ``CASES`` order, so a run of all six puts the two-bid
             ladder and its single-bid twins side by side.

Usage::

    python analysis/eom_critic_evolution.py                    # sb02a-c
    python analysis/eom_critic_evolution.py --cases 02a 02b 02c
    python analysis/eom_critic_evolution.py --sweep a0 --only film
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR  # noqa: E402  (also sets sys.path)
from eom_critic_film import (  # noqa: E402
    CASES,
    DEFAULT_CASES,
    SEEDS,
    act_share_from_sd,
    result_path,
)
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

def marginal_costs(d) -> np.ndarray:
    """Each unit's marginal cost in EUR/MWh, from its probed observations.

    ``EnergyLearningStrategy``'s observation puts total capacity and marginal
    cost in the last two slots, the latter scaled by ``max_bid_price`` (100 in
    all three study cases). Reading it out of the recording rather than off the
    CSV keeps the line correct when the fuel price series moves within the
    horizon, and needs no per-scenario table: it is a median over the probed
    observations, so a moving cost shows as one representative line.
    """
    obs = d["observations"]  # (n_obs, n_agents, obs_dim)
    return np.median(obs[:, :, -1], axis=0) * 100.0


def bid_series(run: dict, i: int) -> list[tuple[np.ndarray, str]]:
    """Agent *i*'s greedy bid price(s) over time, with the line style to draw.

    ``EnergyLearningStrategy`` assigns its two actions by ``min``/``max`` in
    ``calculate_bids``, so the low one is the inflexible block's price and the
    high one the flexible block's — two lines, solid and dashed. At ``act_dim``
    1 (``EnergyLearningSingleBidStrategy``) the two collapse onto the single
    action, and drawing both would put a dashed line exactly on top of a solid
    one; one line is returned instead.
    """
    greedy = run["greedy"][i]                       # (act_dim, n_obs, frames)
    low = np.median(greedy.min(axis=0), axis=0)
    if greedy.shape[0] == 1:
        return [(low, "-")]
    return [(low, "-"), (np.median(greedy.max(axis=0), axis=0), "--")]


def strip(ax, keep_left: bool = True, keep_bottom: bool = True) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if not keep_left:
        ax.tick_params(labelleft=False)
    if not keep_bottom:
        ax.tick_params(labelbottom=False)
    ax.set_axisbelow(True)


def load(out_dir: Path, case: str, seed: int):
    path = result_path(out_dir, case, seed)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    sweeps = [str(s) for s in d["sweeps"]]
    return {
        "case": case,
        "seed": seed,
        "units": [str(u) for u in d["unit_ids"]],
        "sweeps": sweeps,
        "steps": d["steps"],
        "bids": d["critic_bids"],
        "q1": d["critic_q/MATD3"],       # (agents, sweeps, obs, frames, grid)
        "grad": d["critic_grad/MATD3"],  # same
        "greedy": d["greedy/MATD3"],     # (agents, act_dim, obs, frames), EUR/MWh
        "rewards": d["rewards"],         # (frames, agents)
        "mc": marginal_costs(d),
        "share": act_share_from_sd(
            d["buffer_sd_obs"], d["buffer_sd_act"], int(d["unique_obs_dim"])
        ),
    }


# --------------------------------------------------------------------- film


def plot_film(run: dict, sweep: str, out: Path) -> None:
    s = run["sweeps"].index(sweep)
    units, steps, bids = run["units"], run["steps"], run["bids"]
    n = len(units)

    # one panel per unit, wrapped so ten units stay readable
    cols = min(n, 5)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.1 * cols + 1.2, 3.0 * rows + 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0 * rows, 4.4], hspace=0.32)
    top = outer[0].subgridspec(rows, cols, wspace=0.10, hspace=0.28)
    bottom = outer[1].subgridspec(1, 2, wspace=0.22)

    # the gradient is compared across panels of ONE run, where no action-scale
    # patch is installed, so a shared scale is honest here (unlike run 13's
    # grids, which normalise per panel because the scale factor S differs)
    field = np.median(run["grad"][:, s], axis=1)  # (agents, frames, grid)
    vmax = float(np.percentile(np.abs(field), 99.5)) or 1e-6

    axes = []
    for i, unit in enumerate(units):
        ax = fig.add_subplot(top[i // cols, i % cols])
        axes.append(ax)
        im = ax.pcolormesh(
            bids, steps, field[i],
            cmap=DIVERGING,
            norm=SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax, base=10),
            shading="nearest", rasterized=True,
        )
        ax.axvline(run["mc"][i], color=INK, lw=0.9, alpha=0.45, ls="--")
        for series, style in bid_series(run, i):
            ax.plot(series, steps, lw=3.0, color="white", solid_capstyle="round")
            ax.plot(series, steps, lw=1.5, color=INK, ls=style, solid_capstyle="round")
        ax.set_title(f"{unit}   act_share {run['share'][i]:.3f}",
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xlim(bids[0], bids[-1])
        strip(ax, keep_left=i % cols == 0, keep_bottom=i // cols == rows - 1)
        if i % cols == 0:
            ax.set_ylabel("critic updates", fontsize=9, color=MUTED)
        if i // cols == rows - 1:
            ax.set_xlabel("bid price (EUR/MWh)", fontsize=9, color=MUTED)

    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.012)
    cbar.set_label(f"dQ1/d(bid), sweep '{sweep}'   (symlog, autograd)",
                   fontsize=9, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    ax_bid = fig.add_subplot(bottom[0])
    ax_rew = fig.add_subplot(bottom[1])
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, n))
    single_bid = run["greedy"].shape[1] == 1
    for i, unit in enumerate(units):
        series = bid_series(run, i)
        ax_bid.plot(steps, series[0][0], lw=1.6, color=colours[i], label=unit)
        for extra, _ in series[1:]:
            ax_bid.plot(steps, extra, lw=1.0, color=colours[i], alpha=0.5, ls="--")
        ax_rew.plot(steps, run["rewards"][:, i], lw=1.4, color=colours[i])
    ax_bid.axhline(float(np.median(run["mc"])), lw=1.1, color=INK, ls="--", alpha=0.5)
    ax_bid.annotate("marginal cost", xy=(steps[0], float(np.median(run["mc"]))),
                    xytext=(3, 4), textcoords="offset points",
                    fontsize=8, color=MUTED)
    ax_bid.set_title(
        "greedy bid, one price per unit" if single_bid
        else "greedy bid: inflexible (solid) and flexible (dashed)",
        loc="left", fontsize=10, color=INK,
    )
    ax_bid.set_xlabel("critic updates", fontsize=9, color=MUTED)
    ax_bid.set_ylabel("EUR/MWh", fontsize=9, color=MUTED)
    ax_bid.legend(frameon=False, fontsize=8, ncol=1 + n // 4, loc="lower left")

    ax_rew.axhline(0.0, lw=1.1, color=INK, alpha=0.5)
    ax_rew.set_title("mean stored reward since the previous frame",
                     loc="left", fontsize=10, color=INK)
    ax_rew.set_xlabel("critic updates", fontsize=9, color=MUTED)
    for ax in (ax_bid, ax_rew):
        strip(ax)
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)

    case = CASES[run["case"]]
    fig.suptitle(
        f"{run['case']} ({case['scenario']}/{case['study_case']}) seed "
        f"{run['seed']} -- {n} learning unit(s) on a plain EOM, "
        f"{'single' if single_bid else 'two'}-bid",
        x=0.006, y=0.998, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ------------------------------------------------------------------- summary


def plot_summary(runs: list[dict], out: Path) -> None:
    cases = [c for c in CASES if any(r["case"] == c for r in runs)]
    x = np.arange(len(cases))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    titles = (
        "final bid, median over learning units",
        "act_share, mean over learning units",
        "mean buffer reward at the last frame",
    )

    def value(run: dict, which: str) -> float:
        if which == "bid":
            # the low action, i.e. the inflexible price at act_dim 2 and the
            # single price at act_dim 1
            final = np.stack([bid_series(run, i)[0][0][-1]
                              for i in range(len(run["units"]))])
            return float(np.median(final))
        if which == "share":
            return float(run["share"].mean())
        return float(run["rewards"][-1].mean())

    for ax, title, which in zip(axes, titles, ("bid", "share", "reward")):
        per_seed = {seed: [] for seed in SEEDS}
        medians = []
        for case in cases:
            vals = []
            for seed in SEEDS:
                r = next((r for r in runs
                          if r["case"] == case and r["seed"] == seed), None)
                per_seed[seed].append(np.nan if r is None else value(r, which))
                if r is not None:
                    vals.append(value(r, which))
            medians.append(np.median(vals) if vals else np.nan)
        for seed, series in per_seed.items():
            ax.plot(x, series, lw=1.0, marker="o", ms=3.5, alpha=0.45,
                    color=MUTED, label=f"seed {seed}" if ax is axes[0] else None)
        ax.plot(x, medians, lw=2.4, marker="o", ms=5, color=INK, label="median"
                if ax is axes[0] else None)
        ax.set_xticks(x)
        # the x axis is CASES' own order, so the two-bid trio and its single-bid
        # twins sit as two consecutive ladders rather than interleaved
        ax.set_xticklabels(
            [f"{c}\n{len(next(r for r in runs if r['case'] == c)['units'])} learners"
             for c in cases]
        )
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        strip(ax)
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
    axes[0].legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Plain EOM: what changes as the number of learning agents grows "
        "(sb* = single-bid)",
        x=0.006, y=1.02, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES,
                        choices=list(CASES),
                        help="which of eom_critic_film.CASES to draw; defaults "
                             "to the single-bid trio")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--sweep", default="diag",
                        help="which recorded action sweep to draw: 'diag' moves "
                             "the unit's whole bid, 'a0'/'a1' one component")
    parser.add_argument("--only", choices=("film", "summary"), default=None)
    parser.add_argument(
        "--data-dir", type=Path,
        default=OUT_DIR / "runs" / "data" / "14-eom-critic-evolution",
    )
    parser.add_argument("--img-dir", type=Path, default=OUT_DIR / "runs" / "img")
    args = parser.parse_args()

    args.img_dir.mkdir(parents=True, exist_ok=True)
    runs = [r for r in (load(args.data_dir, s, seed)
                        for s in args.cases for seed in args.seeds) if r]
    if not runs:
        raise SystemExit(f"no recorded runs under {args.data_dir}")

    if args.only in (None, "film"):
        for run in runs:
            if args.sweep not in run["sweeps"]:
                print(f"  {run['case']} seed {run['seed']}: no sweep "
                      f"'{args.sweep}' (has {run['sweeps']}) -- skipped")
                continue
            plot_film(
                run, args.sweep,
                args.img_dir / f"14-eom-{run['case']}-seed{run['seed']}.png",
            )
    if args.only in (None, "summary"):
        plot_summary(runs, args.img_dir / "14-eom-summary.png")


if __name__ == "__main__":
    main()
