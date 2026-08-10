# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 12's figure -- the descent window ASSUME's critic never had before.

Run 06 measured a window on the SB3 surrogate: the plateau slope flips toward the
band at step ~1400, an unbroken descent path exists for ~600 steps, and then the
plateau is learned as genuinely flat and the path fragments into noise. On the
real learner with the true reward that window was never observed at all -- run 09
filmed 2560 critic updates of incoherent field, and run 11 found no configuration
that produced one.

Run 12 raises the action's share of the critic's input, either by cutting
``foresight`` or by scaling the critic's action input, and this draws what that
does to the field. The questions the figure answers, in order:

* does a coherent leftward pull appear at all, and at which update;
* does it *close* the way run 06's did, or stay open;
* how many updates the actor needs to cross from the ceiling into the band.

Panels
------
top     ``dQ1/d(bid)`` over (bid, critic updates), one facet per condition, the
        median over 3 seeds x 6 probed observations, with the per-seed actor
        trajectories drawn on top. **Each facet is normalised by its own robust
        scale**: the action-scale conditions fit ``Q(s, S*a)``, so their recorded
        gradient carries the factor ``S`` and raw magnitudes are not comparable
        across rows. Sign and spatial structure are.
bottom  left: the share of the ``[50, 100]`` plateau pulling toward the band --
        1.0 is an unbroken descent path, and this is the window opening and
        closing. right: the actor's bid, i.e. how long the crossing took.

Usage::

    python real_matd3/assume_actshare_sweep.py --workers 5    # produces the data
    python real_matd3/assume_actshare_film.py
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
from assume_actshare_sweep import (  # noqa: E402
    BAND,
    CONDITIONS,
    SEEDS,
    UNIQUE_OBS_DIM,
    act_share,
    result_path,
)
from incdec_reward import PAPER_SMALL  # noqa: E402
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

#: One colour per condition, ordered by act_share: grey for the failing baseline,
#: then the house blue->green ramp as the action gains weight.
CONDITION_COLOR = {
    "baseline": "#8a8985",
    "foresight-6": "#9dc2ec",
    "foresight-3": "#2a78d6",
    "act-x10": "#eb6834",
    "act-x30": "#1baf7a",
}
PLATEAU = (50.0, 100.0)


def descent_stop(bids: np.ndarray, grad: np.ndarray, start: float) -> float:
    """Walk left from ``start`` while ``dQ/d(bid) < 0``; report where it stops.

    Same definition as ``analysis/descent_window.py``, so run 12's window numbers
    are on the same footing as run 06's.
    """
    k = int(np.argmin(np.abs(bids - start)))
    while k > 0 and grad[k] < 0:
        k -= 1
    return float(bids[k])


def load(out_dir: Path, name: str, seeds: list[int]):
    """Stack a condition's seeds. Returns (steps, bids, grad, greedy).

    ``grad`` is ``(seed * obs, frames, grid)`` and ``greedy`` ``(seed * obs, frames)``
    -- the probed observations are pooled with the seeds, since both are just
    repeated draws of the same condition.
    """
    grads, greedys, steps, bids = [], [], None, None
    for seed in seeds:
        path = result_path(out_dir, name, seed)
        if not path.exists():
            continue
        d = np.load(path)
        grads.append(d["critic_grad/MATD3"])
        greedys.append(d["greedy/MATD3"])
        steps, bids = d["steps"], d["critic_bids"]
    if not grads:
        raise SystemExit(f"no results for {name} in {out_dir}")
    return steps, bids, np.concatenate(grads), np.concatenate(greedys)


def window_stats(bids: np.ndarray, grad: np.ndarray, greedy: np.ndarray) -> dict:
    """Per-frame field descriptions plus the actor's crossing, for one trace."""
    plateau = (bids >= PLATEAU[0]) & (bids <= PLATEAU[1])
    left = (grad[:, plateau] < 0).mean(axis=1)
    clean = np.array(
        [descent_stop(bids, row, PAPER_SMALL.max_bid_price) <= PAPER_SMALL.eom_price
         for row in grad]
    )
    in_band = (greedy >= BAND[0]) & (greedy <= BAND[1])
    return {"left": left, "clean": clean, "in_band": in_band}


#: Tolerance around the band for "has settled". Wide enough that one noisy frame
#: at the rim does not reopen an otherwise finished trajectory.
SETTLE_PAD = 3.0


def settle_frame(steps: np.ndarray, greedy: np.ndarray) -> float:
    """First update from which the actor stays in the band for the rest of the run.

    ``in_band`` alone is not the crossing: at ``foresight-3`` the randomly
    initialised actor already emits ~40, so its first in-band frame is frame 1 --
    it then climbs to the ceiling with the incomplete critic (phase 1) and has to
    come back. This measures the arrival that sticks.
    """
    inside = (greedy >= BAND[0] - SETTLE_PAD) & (greedy <= BAND[1] + SETTLE_PAD)
    # scan from the end: the settle point is where the final unbroken run begins
    if not inside[-1]:
        return np.nan
    k = len(inside) - 1
    while k > 0 and inside[k - 1]:
        k -= 1
    return float(steps[k])


def report(out_dir: Path, conditions: list[str], seeds: list[int]) -> dict:
    print("\nrun 12 -- the descent window, measured as in run 06")
    print("'clean path' = an unbroken dQ/d(bid) < 0 walk from bid 100 reaches the band")
    print("updates are critic gradient steps; 800 is the 40-episode budget\n")
    header = (f"{'condition':<13}{'act_share':>10}{'starts':>8}{'peak bid':>10}"
              f"{'first clean':>12}{'clean share':>12}{'open at end':>12}"
              f"{'pulled left':>12}{'settles':>9}{'settles at':>12}")
    print(header)
    print("-" * len(header))

    traces = {}
    for name in conditions:
        steps, bids, grad, greedy = load(out_dir, name, seeds)
        stats = [window_stats(bids, grad[i], greedy[i]) for i in range(len(grad))]
        traces[name] = (steps, bids, grad, greedy, stats)

        first, share = [], []
        for st in stats:
            c = st["clean"]
            first.append(steps[np.argmax(c)] if c.any() else np.nan)
            share.append(c.mean())
        first = np.array(first)
        settle = np.array([settle_frame(steps, row) for row in greedy])
        # "open at end" = an unbroken path still exists at the final frame
        still = float(np.mean([st["clean"][-1] for st in stats]))
        # the field at the actor's own action, which is what actually moves it
        pulled = float(np.mean([
            np.interp(greedy[i, -1], bids, grad[i, -1]) < 0 for i in range(len(grad))
        ]))
        k = int(CONDITIONS[name]["foresight"])
        print(
            f"{name:<13}{act_share(k, float(CONDITIONS[name]['action_scale'])):>10.3f}"
            f"{np.median(greedy[:, 0]):>8.1f}{np.median(greedy.max(axis=1)):>10.1f}"
            f"{np.nanmedian(first) if np.isfinite(first).any() else np.nan:>12.0f}"
            f"{np.mean(share):>12.0%}{still:>12.0%}{pulled:>12.0%}"
            f"{f'{np.isfinite(settle).sum()}/{len(settle)}':>9}"
            f"{np.nanmedian(settle) if np.isfinite(settle).any() else np.nan:>12.0f}"
        )
    print("\n'settles' counts traces ending inside the band +-3 EUR and 'settles at' is")
    print("the update from which they never leave again -- not the first in-band frame,")
    print("which at foresight-3 is frame 1 because the initial actor already emits ~40.")
    print("'first clean' walks left from bid 100 as run 06 does, so it post-dates the")
    print("crossing whenever the actor never sat at the ceiling. 'pulled left' is the")
    print("sign of dQ1/d(bid) at the actor's own action on the final frame: at a")
    print("converged optimum it should be a coin flip, not 100%.")
    return traces


def plot(traces: dict, conditions: list[str], out: Path) -> None:
    p = PAPER_SMALL
    n = len(conditions)
    fig = plt.figure(figsize=(3.5 * n, 9.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.28)
    top = outer[0].subgridspec(1, n, wspace=0.13)
    bottom = outer[1].subgridspec(1, 2, wspace=0.2)

    heat = []
    for col, name in enumerate(conditions):
        steps, bids, grad, greedy, _ = traces[name]
        field = np.median(grad, axis=0)
        # each condition on its own robust scale -- see the module docstring
        scale = np.percentile(np.abs(field), 99.5) or 1.0
        k = int(CONDITIONS[name]["foresight"])
        share = act_share(k, float(CONDITIONS[name]["action_scale"]))

        ax = fig.add_subplot(top[col])
        heat.append(ax)
        im = ax.pcolormesh(
            bids, steps, field / scale,
            cmap=DIVERGING,
            norm=SymLogNorm(linthresh=1e-3, vmin=-1.0, vmax=1.0, base=10),
            shading="nearest", rasterized=True,
        )
        for x in (p.dec_threshold, p.eom_price):
            ax.axvline(x, color=INK, lw=0.9, alpha=0.45, ls="--")
        for row in greedy:
            ax.plot(row, steps, lw=2.6, color="white", alpha=0.7,
                    solid_capstyle="round")
        for row in greedy:
            ax.plot(row, steps, lw=0.9, color=INK, alpha=0.55,
                    solid_capstyle="round")
        ax.set_title(f"{name}\nact_share {share:.3f}", loc="left", fontsize=10,
                     color=CONDITION_COLOR.get(name, INK))
        ax.set_xlabel("bid price (EUR/MWh)")
        ax.set_xlim(bids[0], bids[-1])
        ax.set_ylim(steps[0], steps[-1])
        # inset ticks: at +-100 the neighbouring facets' labels collide
        ax.set_xticks([-50, 0, 50])
        if col == 0:
            ax.set_ylabel("critic gradient steps")
        else:
            ax.tick_params(labelleft=False)

    cbar = fig.colorbar(im, ax=heat, fraction=0.02, pad=0.012)
    cbar.set_label("dQ1/d(bid), each facet on its own scale (symlog)",
                   fontsize=9, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    # --- bottom left: the window -------------------------------------------
    ax_w = fig.add_subplot(bottom[0])
    for name in conditions:
        steps, _, _, _, stats = traces[name]
        left = np.median([st["left"] for st in stats], axis=0)
        ax_w.plot(steps, left, lw=2.1, color=CONDITION_COLOR.get(name, MUTED),
                  label=name)
    ax_w.axhline(1.0, ls=":", lw=1.2, color=MUTED, zorder=0)
    ax_w.axhline(0.5, ls="--", lw=1.0, color=MUTED, zorder=0)
    ax_w.annotate("coin flip", xy=(1.0, 0.5), xycoords=("axes fraction", "data"),
                  xytext=(-4, 4), textcoords="offset points", ha="right",
                  fontsize=8, color=MUTED)
    ax_w.set_ylim(-0.03, 1.08)
    ax_w.set_title(
        "share of the plateau [50, 100] pulling toward the band "
        "(1.0 = unbroken descent path)",
        loc="left", fontsize=10, color=INK,
    )
    ax_w.set_xlabel("critic gradient steps")
    ax_w.set_ylabel("share with dQ1/d(bid) < 0")
    ax_w.legend(frameon=False, fontsize=8.5, loc="lower right", ncols=2)

    # --- bottom right: the crossing ----------------------------------------
    ax_a = fig.add_subplot(bottom[1])
    ax_a.axhspan(*BAND, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    ax_a.annotate("profitable band", xy=(1.0, BAND[1]),
                  xycoords=("axes fraction", "data"), xytext=(-4, 4),
                  textcoords="offset points", ha="right", fontsize=8, color="#137f59")
    for name in conditions:
        steps, _, _, greedy, _ = traces[name]
        for row in greedy:
            ax_a.plot(steps, row, lw=0.7, alpha=0.18,
                      color=CONDITION_COLOR.get(name, MUTED))
        ax_a.plot(steps, np.median(greedy, axis=0), lw=2.1,
                  color=CONDITION_COLOR.get(name, MUTED), label=name)
    ax_a.set_title("the actor's greedy bid (thin = every seed x observation)",
                   loc="left", fontsize=10, color=INK)
    ax_a.set_xlabel("critic gradient steps")
    ax_a.set_ylabel("bid price (EUR/MWh)")
    ax_a.legend(frameon=False, fontsize=8.5, loc="lower left")

    for ax in (ax_w, ax_a):
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        ax.set_xlim(0, max(traces[c][0][-1] for c in conditions))
    for ax in (*heat, ax_w, ax_a):
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(
        "Raising the action's share of the critic's input creates the descent "
        "window ASSUME never had",
        x=0.006, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 0.966,
        "True reward, no shaping, 40 episodes, 3 seeds. The baseline's field never "
        "develops a coherent pull and no actor settles; at act_share >= 0.19 every "
        "actor settles in the band, and from 0.23 the descent path is there from "
        "~20 updates instead of appearing transiently.",
        fontsize=9, color=MUTED, ha="left",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}")


def plot_dose(traces: dict, conditions: list[str], out: Path) -> None:
    """The claim itself: outcome against act_share, for both levers at once.

    The two levers are mechanically unrelated -- one removes observation
    dimensions, the other rescales the action input -- so if they fall on one
    curve, act_share is the variable and not a coincidence of either.
    """
    from incdec_reward import reward_from_bid

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    marker = {"foresight": "o", "action": "^"}

    for name in conditions:
        steps, _, _, greedy, _ = traces[name]
        k = int(CONDITIONS[name]["foresight"])
        scale = float(CONDITIONS[name]["action_scale"])
        share = act_share(k, scale)
        lever = "foresight" if k != 24 else "action"
        if name == "baseline":
            lever = "foresight"  # the shared origin of both ladders
        colour = CONDITION_COLOR.get(name, MUTED)

        final_bid = greedy[:, -1]
        final_rew = np.array([reward_from_bid(b, PAPER_SMALL) for b in final_bid])
        for ax, y in ((axes[0], final_rew), (axes[1], final_bid)):
            ax.scatter(np.full(len(y), share), y, s=16, alpha=0.35, color=colour,
                       marker=marker[lever], lw=0)
            ax.scatter([share], [np.median(y)], s=110, color=colour,
                       marker=marker[lever], edgecolor="white", lw=1.3, zorder=3,
                       label=f"{name} ({'obs dims' if lever == 'foresight' else 'action scale'})")

    axes[0].axhline(PAPER_SMALL.optimal_reward, ls=":", lw=1.2, color=MUTED)
    axes[0].annotate("deterministic optimum +0.190", xy=(0.0, PAPER_SMALL.optimal_reward),
                     xycoords=("axes fraction", "data"), xytext=(4, -11),
                     textcoords="offset points", fontsize=8, color=MUTED)
    axes[0].axhline(0.15, ls="--", lw=1.0, color="#137f59")
    axes[0].annotate("'solved' bar", xy=(0.0, 0.15), xycoords=("axes fraction", "data"),
                     xytext=(4, -11), textcoords="offset points", fontsize=8,
                     color="#137f59")
    axes[0].set_ylabel("final true reward")
    axes[0].set_title("outcome against the action's share of critic input",
                      loc="left", fontsize=10.5, color=INK)

    axes[1].axhspan(*BAND, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
    axes[1].set_ylabel("final bid (EUR/MWh)")
    axes[1].set_title("both levers land on one curve", loc="left", fontsize=10.5,
                      color=INK)

    shares = sorted(
        act_share(int(CONDITIONS[n]["foresight"]), float(CONDITIONS[n]["action_scale"]))
        for n in conditions
    )
    for ax in axes:
        # linear, with a tick only where a condition actually sits: a log axis
        # puts five points inside one decade and its minor labels collide
        ax.set_xticks(shares)
        # rotated: 0.191 and 0.234 sit close enough to collide when horizontal
        ax.set_xticklabels([f"{s:.3f}" for s in shares], fontsize=8,
                           rotation=30, ha="right")
        ax.set_xlim(shares[0] - 0.04, shares[-1] + 0.04)
        ax.set_xlabel("act_share  =  sd(a) / (sd(a) + sum_j sd(obs_j))")
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")

    fig.suptitle(
        "One variable orders the live runs, whichever side of the ratio is moved",
        x=0.006, y=1.10, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 1.035,
        "Circles cut observation dimensions (foresight); triangles scale the "
        "critic's action input. Faint marks are the 6 probed observations of each "
        "of 3 seeds; solid marks are medians. True reward, 40 episodes.",
        fontsize=8.5, color=MUTED, ha="left",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--data-dir", type=Path, default=OUT_DIR / "runs" / "data" / "12-actshare"
    )
    parser.add_argument(
        "--out", type=Path,
        default=OUT_DIR / "runs" / "img" / "12-actshare-descent-window.png",
    )
    parser.add_argument(
        "--dose-out", type=Path,
        default=OUT_DIR / "runs" / "12-actshare-dose-response.png",
    )
    args = parser.parse_args()

    traces = report(args.data_dir, args.conditions, args.seeds)
    plot(traces, args.conditions, args.out)
    plot_dose(traces, args.conditions, args.dose_out)


if __name__ == "__main__":
    main()
