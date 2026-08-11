# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 13's figure -- what raising ``act_share`` does when eleven agents share the
market.

Run 12 answered the single-agent question: the action is 1 of 75 critic inputs,
and raising its share of the input variation is what makes the critic learn the
bid at all. ``HANDOFF.md`` then predicted the multi-agent case should be *worse*,
because a centralised critic adds an observation block and an action dimension per
agent while agent *i*'s own action stays one scalar. It is: the measured own-action
share in ``inc_dec_learning`` is **0.016**, about half the single-agent 0.030.

This draws the two runs side by side.

Panels
------
row 1   ``dQ1/d(bid)`` over (bid, critic updates) for ``diesel_0`` -- the unit the
        rest of this archive is about -- as the median over the 6 probed
        observations, with that agent's own bid trajectory on top. **Each facet
        carries its own robust colour scale**: the scaled condition fits
        ``Q(s, .., S*a_i, ..)``, so its recorded gradient carries the factor S and
        magnitudes are not comparable across facets. Sign and structure are.
row 2   every agent's bid over training, coloured by marginal cost. This is the
        fleet-level version of "does the actor leave the ceiling".
row 3   left: how much the six probed observations disagree about the preferred
        bid, averaged over agents -- run 10's incoherence statistic, which run 12
        drove from 18.1 to 1.5 in the single-agent case. right: mean stored reward
        per agent over the most recent episode, summed over the fleet.

The closed-form ``incdec_reward`` landscape is **not** drawn: it was derived with
the rest of the fleet bidding naively, and here all eleven units learn.

Usage::

    python real_matd3/assume_multiagent_actshare.py      # produces the data
    python real_matd3/assume_multiagent_film.py
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
    DEFAULT_CONDITIONS,
    act_share_from_sd,
    action_block_share,
    result_path,
)
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

#: the unit the single-agent archive follows; row 1 is drawn for it
FOCUS = "diesel_0"
CONDITION_COLOR = {
    "baseline": "#0b0b0b",
    "baseline-25": "#8a8985",
    "act-own-x15": "#2a78d6",
    "act-all-x2": "#1baf7a",
    "act-all-x2-50": "#0f7d57",
    "act-all-x15": "#eb6834",
}


def load(out_dir: Path, name: str, seeds: list[int]) -> dict | None:
    """Pool a condition's seeds along the observation axis.

    Run 12's film does the same: a probed observation and a random seed are both
    just repeated draws of the condition, so ``critic_q`` becomes
    ``(agents, seed * obs, frames, grid)`` and every median below is over both.
    """
    q1, grad, greedy, rewards, shares, blocks = [], [], [], [], [], []
    units = steps = bids = None
    for seed in seeds:
        path = result_path(out_dir, name, seed)
        if not path.exists():
            continue
        d = np.load(path, allow_pickle=False)
        scale, mode = CONDITIONS[name]["scale"], CONDITIONS[name]["mode"]
        args = (d["buffer_sd_obs"], d["buffer_sd_act"], int(d["unique_obs_dim"]),
                scale, mode)
        shares.append(act_share_from_sd(*args))
        blocks.append(action_block_share(*args))
        q1.append(d["critic_q/MATD3"])
        grad.append(d["critic_grad/MATD3"])
        greedy.append(d["greedy/MATD3"])
        rewards.append(d["rewards"])
        units = [str(u) for u in d["unit_ids"]]
        steps, bids = d["steps"], d["critic_bids"]
    if not q1:
        return None
    return {
        "seeds": len(q1),
        "units": units,
        "steps": steps,
        "bids": bids,
        "q1": np.concatenate(q1, axis=1),
        "grad": np.concatenate(grad, axis=1),
        "greedy": np.concatenate(greedy, axis=1),
        "rewards": np.mean(rewards, axis=0),
        "share": np.mean(shares, axis=0),
        "block": np.mean(blocks, axis=0),
    }


def argmax_spread(run: dict) -> np.ndarray:
    """Mean pairwise disagreement of ``argmax Q1`` over probed observations.

    Defined once in ``analysis/critic_coherence.py`` so runs 10-13 are on one
    footing; see that module for why it used to differ between them.
    """
    from critic_coherence import argmax_disagreement, peak_bids

    peak = peak_bids(run["bids"], run["q1"], axis=3)   # (agents, obs, frames)
    return argmax_disagreement(peak, axis=1).mean(axis=0)


def plot(runs: dict[str, dict], out: Path) -> None:
    names = [n for n in runs if runs[n] is not None]
    n = len(names)
    # the shortest run's budget: everything below it is a matched comparison,
    # everything above is the longer runs continuing alone
    common = min(int(runs[name]["steps"][-1]) for name in names)
    fig = plt.figure(figsize=(4.9 * n, 12.0))
    gs = fig.add_gridspec(3, n, height_ratios=[1.25, 1.0, 0.85], hspace=0.46, wspace=0.22)

    # ---- row 1: the critic field at diesel_0 ----------------------------
    for col, name in enumerate(names):
        run = runs[name]
        ax = fig.add_subplot(gs[0, col])
        i = run["units"].index(FOCUS)
        field = np.median(run["grad"][i], axis=0)           # (frames, grid)
        steps = run["steps"]
        scale = np.percentile(np.abs(field), 98) or 1.0
        mesh = ax.pcolormesh(
            run["bids"], steps, field,
            cmap=DIVERGING, shading="nearest",
            norm=SymLogNorm(linthresh=scale / 50, vmin=-scale, vmax=scale),
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label("blue: pulls the bid down", fontsize=8, color=MUTED)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        ax.plot(np.median(run["greedy"][i], axis=0), steps,
                lw=1.4, color=INK, alpha=0.75)
        if steps[-1] > common:
            ax.axhline(common, color=INK, lw=1.0, ls="--", alpha=0.5)
        ax.set_title(
            f"{name}   own act_share {run['share'].mean():.3f}, "
            f"action block {run['block'].mean():.3f}\n"
            f"dQ1/d(bid) at {FOCUS}, median over {run['seeds']} seeds "
            f"x 6 observations",
            loc="left", fontsize=9.5, color=INK,
        )
        ax.set_xlabel("bid [EUR/MWh]")
        if col == 0:
            ax.set_ylabel("critic gradient steps")

    # ---- row 2: every agent's bid ---------------------------------------
    for col, name in enumerate(names):
        run = runs[name]
        ax = fig.add_subplot(gs[1, col])
        greedy = np.median(run["greedy"], axis=1)           # (agents, frames)
        cmap = plt.get_cmap("viridis")
        for i, unit in enumerate(run["units"]):
            colour = cmap(i / max(1, len(run["units"]) - 1))
            ax.plot(run["steps"], greedy[i], lw=1.3, color=colour, alpha=0.85,
                    label=unit if col == 0 else None)
        ax.plot(run["steps"], greedy[run["units"].index(FOCUS)],
                lw=2.4, color=INK, alpha=0.9, zorder=5)
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        if run["steps"][-1] > common:
            ax.axvline(common, color=INK, lw=1.0, ls="--", alpha=0.5)
        ax.set_ylim(-105, 105)
        ax.set_title(f"every agent's greedy bid ({FOCUS} in black)",
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("critic gradient steps")
        if col == 0:
            ax.set_ylabel("bid [EUR/MWh]")
            ax.legend(fontsize=7, ncol=2, frameon=False, labelcolor=MUTED, loc="lower right")

    # ---- row 3: the three conditions compared on one axis each -----------
    sub = gs[2, :].subgridspec(1, 3, wspace=0.28)
    ax_d = fig.add_subplot(sub[0, 0])
    ax_s = fig.add_subplot(sub[0, 1])
    ax_r = fig.add_subplot(sub[0, 2])
    for name in names:
        run = runs[name]
        colour = CONDITION_COLOR.get(name, MUTED)
        i = run["units"].index(FOCUS)
        ax_d.plot(run["steps"], np.median(run["greedy"][i], axis=0),
                  lw=1.9, color=colour, label=name)
        ax_s.plot(run["steps"], argmax_spread(run), lw=1.9, color=colour, label=name)
        ax_r.plot(run["steps"], run["rewards"].sum(axis=1), lw=1.9, color=colour,
                  label=name)
    # the single-agent profitable band, for orientation only -- the multi-agent
    # landscape is not the one it was measured on
    ax_d.axhspan(30, 49, color=MUTED, alpha=0.16, zorder=0)
    ax_d.annotate("single-agent band [30, 49]", xy=(0.02, 0.06),
                  xycoords="axes fraction", fontsize=7.5, color=MUTED)
    ax_d.set_ylim(-105, 105)
    ax_d.set_title(f"{FOCUS}: greedy bid", loc="left", fontsize=9.5, color=INK)
    ax_d.set_ylabel("bid [EUR/MWh]")
    ax_s.set_title("argmax Q1 disagreement across observations\n"
                   "(run 10's incoherence statistic, mean over agents)",
                   loc="left", fontsize=9.5, color=INK)
    ax_s.set_ylabel("EUR/MWh")
    ax_r.set_title("fleet reward: sum over agents of the mean\n"
                   "stored reward over the most recent episode",
                   loc="left", fontsize=9.5, color=INK)
    for ax in (ax_d, ax_s, ax_r):
        ax.set_xlabel("critic gradient steps")
        ax.axvline(common, color=INK, lw=1.0, ls="--", alpha=0.4, zorder=0)
        ax.legend(fontsize=7.5, frameon=False, labelcolor=MUTED)

    for ax in fig.axes:
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(
        "Run 13 -- eleven learning agents: the own-action share halves, and what "
        "raising it back does",
        x=0.006, y=1.045, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 1.012,
        f"inc_dec_learning, true reward, seeds pooled with observations. "
        f"Dashed line: {common} critic "
        f"updates, where the 25-episode runs stop. Row 1 facets carry their own "
        f"colour scale -- the scaled conditions' recorded dQ/d(bid) carries the "
        f"factor S, so signs compare across facets and magnitudes do not.",
        ha="left", fontsize=8.5, color=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument(
        "--data-dir", type=Path,
        default=OUT_DIR / "runs" / "data" / "13-multiagent-actshare",
    )
    parser.add_argument(
        "--out", type=Path,
        default=OUT_DIR / "runs" / "img" / "13-multiagent-actshare.png",
    )
    args = parser.parse_args()

    runs = {n: load(args.data_dir, n, args.seeds) for n in args.conditions}
    missing = [n for n, r in runs.items() if r is None]
    if missing:
        print(f"  no results for {missing}")
    if all(r is None for r in runs.values()):
        raise SystemExit("nothing to draw")
    plot({n: r for n, r in runs.items() if r is not None}, args.out)


if __name__ == "__main__":
    main()
