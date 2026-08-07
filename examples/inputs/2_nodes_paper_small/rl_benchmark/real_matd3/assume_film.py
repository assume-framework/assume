# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
The two ASSUME films side by side: shaped reward vs true reward.

``assume_training_probe.py`` records ASSUME's own critic over a live learning run.
This draws a pair of them. The reward shaping at
``learning_strategies.py:1583-1589`` is unconditional, so the two runs differ only
in whether those seven lines are commented out -- and nothing in the config
records which, hence the ``label`` stored in each npz.

What the panels answer
----------------------
1. **Where does the critic point, and where does the actor stand?** The heatmap is
   ``dQ1/d(bid)`` over (bid, cumulative critic updates), with the actor's own bid
   drawn on top. Blue pulls toward the band, orange pulls toward the ceiling.
2. **Does the critic ever prefer the band?** ``argmax Q1`` per update, against the
   30-49 EUR band. This is the "did it converge" question, answered per frame
   rather than once at the end.
3. **What is the actor doing?** Its bid over training, for every probed
   observation.

Usage::

    python real_matd3/assume_film.py
    python real_matd3/assume_film.py --runs shaped=<a>.npz unshaped=<b>.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, resolve  # noqa: E402  (also sets sys.path)
from critic_evolution import DIVERGING  # noqa: E402
from incdec_reward import PAPER_SMALL  # noqa: E402
from run_benchmark import INK, MUTED  # noqa: E402

ALGO = "MATD3"
#: One hue per film, assigned in fixed order so a run keeps its colour.
HUES = ("#2a78d6", "#eb6834")


def load(pairs: list[tuple[str, Path]]) -> dict[str, dict]:
    runs = {}
    for label, path in pairs:
        d = np.load(path)
        stored = str(d["label"]) if "label" in d.files else ""
        runs[label] = {
            "steps": d["steps"],
            "bids": d["critic_bids"],
            "q": d[f"critic_q/{ALGO}"],
            "grad": d[f"critic_grad/{ALGO}"],
            "actor": d[f"greedy/{ALGO}"],
            "buffer_fill": d["buffer_fill"],
            "stored_label": stored,
        }
        if stored and stored != label:
            print(f"  note: {path.name} is labelled '{stored}', drawn as '{label}'")
    return runs


def report(runs: dict[str, dict]) -> None:
    p = PAPER_SMALL
    print(f"\n  band {p.dec_threshold:.0f}-{p.eom_price:.0f} EUR/MWh\n")
    print(
        f"  {'film':<11}{'updates':>9}{'buffer':>8}{'argmax first':>14}"
        f"{'argmax last':>13}{'in band':>9}{'actor first':>13}{'actor last':>12}"
    )
    print("  " + "-" * 89)
    for label, r in runs.items():
        argmax = r["bids"][r["q"].argmax(axis=2)]  # (obs, frames)
        in_band = np.mean(
            (argmax[:, -1] >= p.dec_threshold) & (argmax[:, -1] <= p.eom_price)
        )
        print(
            f"  {label:<11}{r['steps'][-1]:>9}{r['buffer_fill'][-1]:>8}"
            f"{np.median(argmax[:, 0]):>14.1f}{np.median(argmax[:, -1]):>13.1f}"
            f"{in_band:>8.0%}{np.median(r['actor'][:, 0]):>13.1f}"
            f"{np.median(r['actor'][:, -1]):>12.1f}"
        )
    print()


def plot(runs: dict[str, dict], out: Path) -> None:
    p = PAPER_SMALL
    labels = list(runs)
    n = len(labels)

    fig = plt.figure(figsize=(6.6 * n, 10.2))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.30)
    top = outer[0].subgridspec(1, n, wspace=0.08)
    bottom = outer[1].subgridspec(1, 2, wspace=0.22)

    vmax = max(np.abs(r["grad"]).max() for r in runs.values())

    # --- row 1: the field, one heatmap per film ------------------------------
    heat = []
    for col, label in enumerate(labels):
        r = runs[label]
        ax = fig.add_subplot(top[col])
        heat.append(ax)
        im = ax.pcolormesh(
            r["bids"], r["steps"], r["grad"][0],
            cmap=DIVERGING,
            norm=SymLogNorm(linthresh=1e-4, vmin=-vmax, vmax=vmax, base=10),
            shading="nearest", rasterized=True,
        )
        for x in (p.dec_threshold, p.eom_price):
            ax.axvline(x, color=INK, lw=0.9, alpha=0.4, ls="--")
        for row in r["actor"]:
            ax.plot(row, r["steps"], lw=2.6, color="white", solid_capstyle="round")
        for row in r["actor"]:
            ax.plot(row, r["steps"], lw=1.2, color=INK, alpha=0.8,
                    solid_capstyle="round")
        ax.set_title(f"{label} reward", loc="left", fontsize=11.5, color=HUES[col])
        ax.set_xlabel("bid price (EUR/MWh)")
        ax.set_xlim(r["bids"][0], r["bids"][-1])
        ax.set_ylim(r["steps"][0], r["steps"][-1])
        if col == 0:
            ax.set_ylabel("critic gradient steps")
            ax.annotate(
                "the actor, one line\nper probed observation",
                xy=(-95, r["steps"][-1] * 0.62), fontsize=8.5, color=INK,
            )
        else:
            ax.tick_params(labelleft=False)

    cbar = fig.colorbar(im, ax=heat, fraction=0.02, pad=0.012)
    cbar.set_label("dQ1/d(bid)   (symlog, autograd)", fontsize=9, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    # --- row 2 left: does the critic ever prefer the band? -------------------
    ax_arg = fig.add_subplot(bottom[0])
    ax_arg.axhspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.12, lw=0)
    ax_arg.axhline(p.optimal_bid, ls="--", lw=1.1, color=INK, zorder=2)
    for col, label in enumerate(labels):
        r = runs[label]
        argmax = r["bids"][r["q"].argmax(axis=2)]
        ax_arg.plot(r["steps"], np.median(argmax, axis=0), lw=2.1,
                    color=HUES[col], label=label)
        ax_arg.fill_between(r["steps"], argmax.min(axis=0), argmax.max(axis=0),
                            color=HUES[col], alpha=0.15, lw=0)
    ax_arg.set_title(
        "argmax Q1 — the bid the critic prefers",
        loc="left", fontsize=10.5, color=INK,
    )
    ax_arg.set_xlabel("critic gradient steps  (median, min–max over observations)")
    ax_arg.set_ylabel("bid price (EUR/MWh)")
    ax_arg.set_ylim(-p.max_bid_price, p.max_bid_price)
    ax_arg.legend(frameon=False, fontsize=9, loc="lower right")

    # --- row 2 right: and where does the actor actually bid? -----------------
    ax_act = fig.add_subplot(bottom[1])
    ax_act.axhspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.12, lw=0)
    ax_act.axhline(p.optimal_bid, ls="--", lw=1.1, color=INK, zorder=2)
    for col, label in enumerate(labels):
        r = runs[label]
        ax_act.plot(r["steps"], np.median(r["actor"], axis=0), lw=2.1,
                    color=HUES[col], label=label)
        ax_act.fill_between(r["steps"], r["actor"].min(axis=0), r["actor"].max(axis=0),
                            color=HUES[col], alpha=0.15, lw=0)
    ax_act.set_title(
        "the actor's own bid",
        loc="left", fontsize=10.5, color=INK,
    )
    ax_act.set_xlabel("critic gradient steps")
    ax_act.set_ylabel("bid price (EUR/MWh)")
    ax_act.set_ylim(-p.max_bid_price, p.max_bid_price)
    ax_act.legend(frameon=False, fontsize=9, loc="lower right")
    ax_act.annotate(
        f"band {p.dec_threshold:.0f}–{p.eom_price:.0f}",
        xy=(0.02, p.eom_price + 4), xycoords=("axes fraction", "data"),
        fontsize=8.5, color="#137f59",
    )

    for ax in (*heat, ax_arg, ax_act):
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)
    for ax in (ax_arg, ax_act):
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)

    fig.suptitle(
        "ASSUME's own MATD3, filmed over a 40-episode learning run",
        x=0.006, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 0.962,
        "Q1 is the surface matd3.py has the actor ascend. The two runs differ only in "
        "whether the reward shaping at learning_strategies.py:1583-1589 is active.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", default=None, metavar="LABEL=PATH",
        help="films to draw; defaults to the shaped/unshaped pair",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "assume_film.png"
    )
    args = parser.parse_args()

    pairs = (
        [(lbl, Path(path)) for lbl, path in (r.split("=", 1) for r in args.runs)]
        if args.runs
        # deliberately the *clean* unshaped run: assume_probe_unshaped.npz was
        # labelled unshaped but preloads a buffer of shaped rewards, and those
        # transitions change the outcome completely
        else [("shaped", resolve("assume_probe_shaped.npz")),
              ("unshaped", resolve("assume_probe_unshaped_clean.npz"))]
    )
    runs = load(pairs)
    report(runs)
    plot(runs, args.out)


if __name__ == "__main__":
    main()
