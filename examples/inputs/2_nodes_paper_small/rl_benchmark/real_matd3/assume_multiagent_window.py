# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 13 -- was ``act-all-x2``'s descent window still opening when the budget ran out?

The run 13 figure shows ``act-all-x2``'s critic field at ``diesel_0`` as almost
entirely positive, with a blue wedge appearing in the top-right corner only near
the end of its 1200 updates. That looks like a window that has just opened, which
would mean the condition is under-trained rather than unrescued -- exactly what
run 12 found for ``foresight-3`` (``pulled left`` 100 % at the final frame, bid
still descending at 40.4 when the 800-update budget expired).

This tests that reading rather than eyeballing it, with **run 06's definitions**
imported from ``assume_actshare_film`` so the numbers sit on the same footing as
runs 06 and 12:

``left``          share of the ``[50, 100]`` plateau with ``dQ1/d(bid) < 0``, i.e.
                  pulling the bid down. 1.0 is an unbroken leftward pull.
``descent stop``  how far left an unbroken ``dQ1/d(bid) < 0`` walk from +100 gets
                  before the sign flips. Lower is a deeper path; reaching 49 is
                  run 06's "clean path".
``pulled left``   the sign at the actor's *own* bid -- run 12 reads a coin flip as
                  converged and 100 % as still moving.

A window that is opening shows `left` and the depth of the descent both still
**trending** at the final frame. A window that has closed, or never opened, does
not. The trend over the final quarter of frames is printed with each panel.

Caveat carried from run 12: the recorded ``dQ/d(bid)`` carries the action-scale
factor S, so **signs are comparable across conditions and magnitudes are not**.
Every statistic here is a sign statistic for that reason.

Usage::

    python real_matd3/assume_multiagent_window.py
    python real_matd3/assume_multiagent_window.py --focus gas_0
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
from assume_actshare_film import descent_stop  # noqa: E402  run 06's definition
from assume_multiagent_actshare import (  # noqa: E402
    CONDITIONS,
    act_share_from_sd,
    result_path,
)
from assume_multiagent_film import CONDITION_COLOR, FOCUS  # noqa: E402
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

#: the zero-profit plateau and the profitable band of the *single-agent*
#: landscape. Here they are orientation only -- with eleven learners the reward
#: is no longer the closed-form one -- but keeping run 06's numbers makes the
#: window statistics directly comparable.
PLATEAU = (50.0, 100.0)
BAND = (30.0, 49.0)
CEILING = 100.0

#: the condition this figure is about; the others are drawn as reference
SUBJECT = "act-all-x2"
REFERENCE = ["baseline", "act-all-x15", "act-own-x15"]


def load(out_dir: Path, name: str, seed: int, focus: str) -> dict | None:
    path = result_path(out_dir, name, seed)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    units = [str(u) for u in d["unit_ids"]]
    if focus not in units:
        raise SystemExit(f"{focus} not among {units}")
    i = units.index(focus)
    scale, mode = CONDITIONS[name]["scale"], CONDITIONS[name]["mode"]
    return {
        "name": name,
        "seed": seed,
        "steps": d["steps"],
        "bids": d["critic_bids"],
        "grad": d["critic_grad/MATD3"][i],     # (obs, frames, grid)
        "greedy": d["greedy/MATD3"][i],        # (obs, frames)
        "share": act_share_from_sd(
            d["buffer_sd_obs"], d["buffer_sd_act"], int(d["unique_obs_dim"]),
            scale, mode,
        ).mean(),
    }


def window_series(run: dict) -> dict:
    """Run 06's three statistics per frame, pooled over the probed observations."""
    bids, grad, greedy = run["bids"], run["grad"], run["greedy"]
    plateau = (bids >= PLATEAU[0]) & (bids <= PLATEAU[1])

    left = (grad[:, :, plateau] < 0).mean(axis=(0, 2))
    stop = np.array([
        [descent_stop(bids, grad[o, f], CEILING) for f in range(grad.shape[1])]
        for o in range(grad.shape[0])
    ]).mean(axis=0)

    # sign at the actor's own bid, per observation and frame
    idx = np.abs(bids[None, None, :] - greedy[:, :, None]).argmin(axis=2)
    at_actor = np.take_along_axis(grad, idx[:, :, None], axis=2)[:, :, 0]
    pulled = (at_actor < 0).mean(axis=0)
    return {"left": left, "stop": stop, "pulled": pulled,
            "bid": np.median(greedy, axis=0)}


def trend(steps: np.ndarray, y: np.ndarray, tail: float = 0.25) -> float:
    """Least-squares slope over the final ``tail`` of the run, per 1000 updates."""
    k = max(2, int(len(y) * tail))
    x = steps[-k:].astype(float)
    return float(np.polyfit(x, y[-k:], 1)[0] * 1000)


def plot(subject: list[dict], reference: list[dict], focus: str, out: Path) -> None:
    n = len(subject)
    fig = plt.figure(figsize=(5.6 * max(n, 2), 9.6))
    gs = fig.add_gridspec(2, max(n, 2), height_ratios=[1.15, 1.0],
                          hspace=0.34, wspace=0.24)

    # ---- row 1: the subject's field, one facet per seed --------------------
    for col, run in enumerate(subject):
        ax = fig.add_subplot(gs[0, col])
        field = np.median(run["grad"], axis=0)
        scale = np.percentile(np.abs(field), 98) or 1.0
        mesh = ax.pcolormesh(
            run["bids"], run["steps"], field, cmap=DIVERGING, shading="nearest",
            norm=SymLogNorm(linthresh=scale / 50, vmin=-scale, vmax=scale),
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02, fraction=0.05)
        cbar.set_label("blue: pulls the bid down", fontsize=8, color=MUTED)
        cbar.ax.tick_params(colors=MUTED, labelsize=7)
        ax.plot(np.median(run["greedy"], axis=0), run["steps"],
                lw=1.5, color=INK, alpha=0.8)
        ax.axvspan(*BAND, color=INK, alpha=0.10, zorder=0)
        ax.set_title(f"{run['name']}  seed {run['seed']}   "
                     f"own act_share {run['share']:.3f}\n"
                     f"dQ1/d(bid) at {focus}, median over 6 observations",
                     loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("bid [EUR/MWh]")
        if col == 0:
            ax.set_ylabel("critic gradient steps")

    # ---- row 2: the window statistics, subject against reference -----------
    sub = gs[1, :].subgridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(sub[0, k]) for k in range(3)]
    keys = ("left", "stop", "pulled")
    titles = (
        f"share of the {PLATEAU[0]:.0f}-{PLATEAU[1]:.0f} plateau pulling down\n"
        "1.0 = unbroken leftward pull (run 06's window)",
        "descent stop: how far left an unbroken\n"
        "dQ1/d(bid) < 0 walk from +100 reaches",
        "pulled left at the actor's own bid\n"
        "0.5 = converged, 1.0 = still descending",
    )

    for run in reference + subject:
        s = window_series(run)
        is_subject = run in subject
        colour = CONDITION_COLOR.get(run["name"], MUTED)
        for ax, key in zip(axes, keys):
            ax.plot(run["steps"], s[key],
                    lw=2.2 if is_subject else 1.2,
                    alpha=0.95 if is_subject else 0.45,
                    ls="-" if run["seed"] == 42 else "--",
                    color=colour,
                    label=f"{run['name']} s{run['seed']}")

    for ax, title in zip(axes, titles):
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("critic gradient steps")
        ax.legend(fontsize=6.8, frameon=False, labelcolor=MUTED, ncol=2)
    axes[1].axhline(BAND[1], ls=":", lw=1.1, color=MUTED)
    axes[1].annotate("band rim 49", xy=(0.02, 0.06), xycoords="axes fraction",
                     fontsize=7.5, color=MUTED)
    axes[1].set_ylabel("bid [EUR/MWh]")
    axes[2].axhline(0.5, ls="--", lw=1.0, color=MUTED, zorder=0)

    for ax in fig.axes:
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(
        f"Run 13 -- is {SUBJECT}'s descent window still opening at the end of its "
        f"budget?",
        x=0.006, y=1.03, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.text(0.006, 1.003,
             f"{focus}, run 06's definitions. Solid: seed 42, dashed: seed 1. "
             f"Bold: {SUBJECT}. Recorded dQ/d(bid) carries the action scale S, so "
             f"every statistic here is a sign statistic.",
             ha="left", fontsize=8.5, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    print(f"  wrote {out}")


def report(runs: list[dict]) -> None:
    print(f"\n  window statistics at the final frame, and the trend over the last "
          f"quarter of each run\n")
    header = (f"  {'condition':<14} {'seed':>4} {'updates':>8} {'left':>7} "
              f"{'d left':>8} {'stop':>7} {'d stop':>8} {'pulled':>7} {'bid':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for run in runs:
        s = window_series(run)
        print(f"  {run['name']:<14} {run['seed']:>4} {int(run['steps'][-1]):>8} "
              f"{s['left'][-1]:7.2f} {trend(run['steps'], s['left']):+8.3f} "
              f"{s['stop'][-1]:7.1f} {trend(run['steps'], s['stop']):+8.1f} "
              f"{s['pulled'][-1]:7.2f} {s['bid'][-1]:7.1f}")
    print("\n  d left / d stop are least-squares slopes per 1000 updates over the "
          "final quarter.\n  A window still opening has d left > 0 and d stop < 0 "
          "at the end.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default=SUBJECT)
    parser.add_argument("--reference", nargs="*", default=REFERENCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2])
    parser.add_argument("--focus", default=FOCUS)
    parser.add_argument(
        "--data-dir", type=Path,
        default=OUT_DIR / "runs" / "data" / "13-multiagent-actshare",
    )
    parser.add_argument(
        "--out", type=Path,
        default=OUT_DIR / "runs" / "img" / "13-multiagent-window.png",
    )
    args = parser.parse_args()

    subject = [r for s in args.seeds
               if (r := load(args.data_dir, args.subject, s, args.focus))]
    reference = [r for name in args.reference for s in args.seeds
                 if (r := load(args.data_dir, name, s, args.focus))]
    if not subject:
        raise SystemExit(f"no results for {args.subject}")
    report(subject + reference)
    plot(subject, reference, args.focus, args.out)


if __name__ == "__main__":
    main()
