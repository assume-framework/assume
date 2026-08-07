# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
When was the descent possible? -- the escape is a transient, not a standing option.

``runs/README.md`` §7 argues that the fragmented gradient field is *not* what
binds, on the evidence that the softsign actor crossed the plateau while scoring
worse than tanh on every path-consistency metric. That comparison is taken over
**every post-warmup probe of the whole run**, and this script shows why that hides
the mechanism:

* Right after warmup the critic has learned only the coarse shape -- low bids lose,
  high bids do not -- so ``Q`` rises monotonically with the bid and *both* actors
  run to the ceiling.
* A few hundred steps later the critic discovers the band's spike, and the slope
  over the whole ``[49, 100]`` plateau flips to a single coherent leftward pull.
  For a few hundred steps there is an unbroken descent path from the ceiling into
  the band.
* Then the critic finishes learning that the plateau really is flat. The coherent
  pull collapses into sign-alternating noise, and no descent path exists any more.

So the path metrics are dominated by the third phase -- which is *after* the
outcome was decided, and for tanh is partly a consequence of the actor having
parked at the ceiling and filled the buffer there. Restricted to the interval in
which the softsign actor actually crossed, the field is clean for both
activations. Saturation, not fragmentation, is still the discriminator -- but the
crossing was only ever available in a window.

Needs two runs recorded with ``--critic-grid``, identical but for the activation::

    python run_benchmark.py --algos TD3 DDPG --seeds 3 --timesteps 4000 \\
        --eval-every 25 --critic-grid 401 --actor-activation softsign \\
        --results <o>/window_softsign.npz
    python run_benchmark.py ... --actor-activation tanh --results <o>/window_tanh.npz
    python descent_window.py
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
from run_benchmark import COLORS, INK, MUTED  # noqa: E402

#: Bid above which the actor counts as "at the ceiling", and below which it counts
#: as having escaped. The gap is wide so a single noisy probe cannot open or close
#: an escape interval.
CEILING = 90.0
ESCAPED = 60.0

def descent_stop(bids: np.ndarray, grad: np.ndarray, start: float) -> float:
    """Walk left from ``start`` while ``dQ/da < 0`` and report where it stops.

    ``dQ/da < 0`` is what pulls a gradient-ascent actor toward lower bids, so this
    is the furthest the actor could get by following the field it can see, without
    ever crossing a stationary point.
    """
    k = int(np.argmin(np.abs(bids - start)))
    while k > 0 and grad[k] < 0:
        k -= 1
    return float(bids[k])


def probe_stats(bids: np.ndarray, grad: np.ndarray) -> dict[str, np.ndarray]:
    """Per-probe descriptions of the field between the ceiling and the band.

    ``grad`` is ``(probes, grid)``. All three quantities are properties of the
    critic alone -- none of them looks at where the actor happens to be, so they
    can be compared across activations on equal terms.
    """
    p = PAPER_SMALL
    plateau = bids >= p.eom_price + 1.0
    span = (bids >= p.dec_threshold) & (bids <= p.max_bid_price)

    reversals = np.empty(len(grad))
    stop = np.empty(len(grad))
    for i, row in enumerate(grad):
        sign = np.sign(row[span])
        sign = sign[sign != 0]
        reversals[i] = np.sum(np.diff(sign) != 0) if len(sign) > 1 else 0
        stop[i] = descent_stop(bids, row, p.max_bid_price)

    return {
        #: signed mean slope over the plateau -- negative pulls toward the band
        "pull": grad[:, plateau].mean(axis=1),
        #: share of the plateau that pulls left at all
        "left_frac": (grad[:, plateau] < 0).mean(axis=1),
        "reversals": reversals,
        "stop": stop,
        #: an unbroken descent from the ceiling reaches the band
        "clean": stop <= p.eom_price,
    }


def escape_interval(steps: np.ndarray, greedy: np.ndarray) -> tuple[float, float]:
    """``(start, end)`` steps over which the actor crossed the plateau.

    Start is the last probe at which it still sat above ``CEILING``, end the first
    probe at which it is below ``ESCAPED``. ``(nan, nan)`` if it never left.
    """
    below = np.flatnonzero(greedy < ESCAPED)
    above = np.flatnonzero(greedy > CEILING)
    if not len(above) or not len(below):
        return (np.nan, np.nan)
    after = below[below > above[0]]
    if not len(after):
        return (np.nan, np.nan)
    end = after[0]
    start = above[above < end][-1]
    return (float(steps[start]), float(steps[end]))


#: Phase names in the order they happen, so the table reads as a sequence.
PHASES = ("all post-warmup", "warmup -> flip", "flip -> crossed", "after crossing")


def phase_table(
    steps: np.ndarray,
    stats: dict[str, np.ndarray],
    window: tuple[float, float],
    warmup: int,
) -> dict[str, dict[str, float]]:
    """The path metrics split by phase instead of pooled over the run.

    ``window`` is ``(flip, crossed)``: the probe at which the plateau slope first
    turns toward the band, and the probe at which the actor has finished crossing.
    Pooling over the run -- the first row -- is what ``runs/README.md`` §7 does,
    and it averages phase 2 into phase 3.
    """
    lo, hi = window
    phases = {
        "all post-warmup": steps > warmup,
        "warmup -> flip": (steps > warmup) & (steps < lo),
        "flip -> crossed": (steps >= lo) & (steps <= hi),
        "after crossing": steps > hi,
    }
    return {
        name: {
            "n": int(mask.sum()),
            "clean": float(stats["clean"][mask].mean()) if mask.any() else np.nan,
            "reversals": float(np.median(stats["reversals"][mask])) if mask.any() else np.nan,
            "pull": float(np.median(stats["pull"][mask])) if mask.any() else np.nan,
            "left_frac": float(np.median(stats["left_frac"][mask])) if mask.any() else np.nan,
        }
        for name, mask in phases.items()
    }


def report(runs: dict[str, np.lib.npyio.NpzFile], algo: str, warmup: int) -> dict:
    """Print the per-seed timings and the phase table; return what the figure needs.

    The **first** run is the reference: runs whose actor never leaves the ceiling
    have no crossing of their own, so they borrow the reference's per-seed
    crossing to be scored over the same interval.
    """
    out: dict = {}
    reference = next(iter(runs))
    seeds = len(runs[reference][f"greedy/{algo}"])
    steps = runs[reference]["steps"]
    bids = runs[reference]["critic_bids"]

    print(f"\n  {algo}: when the field turns, and when the actor moves\n")
    print(
        f"  {'activation':<11}{'seed':>5}{'first clean path':>19}{'pull flips':>13}"
        f"{'actor escapes':>16}{'final bid':>12}"
    )
    print("  " + "-" * 76)

    for name, data in runs.items():
        grad = data[f"critic_grad/{algo}"]
        greedy = data[f"greedy/{algo}"]
        per_seed = []
        for s in range(seeds):
            st = probe_stats(bids, grad[s])
            post = steps > warmup
            clean_post = st["clean"] & post
            flip_post = (st["pull"] < 0) & post
            first_clean = steps[np.argmax(clean_post)] if clean_post.any() else np.nan
            first_flip = steps[np.argmax(flip_post)] if flip_post.any() else np.nan
            # a run whose actor never escapes has no crossing of its own, so it
            # borrows the reference run's paired seed and is scored over the same
            # interval -- same seed, same data, differing only in what is varied
            own = escape_interval(steps, greedy[s])
            crossed = own[1] if np.isfinite(own[1]) else escape_interval(
                steps, runs[reference][f"greedy/{algo}"][s]
            )[1]
            window = (first_flip, crossed)
            per_seed.append({"stats": st, "window": window, "greedy": greedy[s]})
            esc = "never" if not np.isfinite(own[0]) else f"{own[0]:.0f}-{own[1]:.0f}"
            print(
                f"  {name:<11}{s:>5}{first_clean:>19.0f}{first_flip:>13.0f}"
                f"{esc:>16}{greedy[s, -1]:>12.2f}"
            )
        out[name] = per_seed

    print("\n  path metrics by phase (median over seeds)\n")
    print(
        f"  {'activation':<11}{'phase':<20}{'probes':>8}{'clean path':>12}"
        f"{'reversals':>11}{'left share':>12}{'mean dQ/da':>13}"
    )
    print("  " + "-" * 87)
    for name, per_seed in out.items():
        for phase in PHASES:
            rows = [
                phase_table(steps, d["stats"], d["window"], warmup)[phase]
                for d in per_seed
            ]
            agg = {k: np.nanmedian([r[k] for r in rows]) for k in rows[0]}
            print(
                f"  {name:<11}{phase:<20}{agg['n']:>8.0f}{agg['clean']:>11.0%}"
                f"{agg['reversals']:>11.0f}{agg['left_frac']:>11.0%}"
                f"{agg['pull']:>13.1e}"
            )
        print()
    return out


def plot(
    runs: dict[str, np.lib.npyio.NpzFile],
    stats: dict,
    algo: str,
    seed: int,
    warmup: int,
    out: Path,
) -> None:
    p = PAPER_SMALL
    reference = next(iter(runs))
    steps = runs[reference]["steps"]
    bids = runs[reference]["critic_bids"]
    window = stats[reference][seed]["window"]

    fig = plt.figure(figsize=(15, 9.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 1.0], hspace=0.30)
    top = outer[0].subgridspec(1, 2, wspace=0.08)
    bottom = outer[1].subgridspec(1, 2, wspace=0.22)

    vmax = max(np.abs(d[f"critic_grad/{algo}"][seed]).max() for d in runs.values())

    # --- row 1: the field itself, with the actor drawn on top ----------------
    heat = []
    for col, (name, data) in enumerate(runs.items()):
        ax = fig.add_subplot(top[col])
        heat.append(ax)
        im = ax.pcolormesh(
            bids, steps, data[f"critic_grad/{algo}"][seed],
            cmap=DIVERGING,
            norm=SymLogNorm(linthresh=1e-5, vmin=-vmax, vmax=vmax, base=10),
            shading="nearest", rasterized=True,
        )
        for x in (p.dec_threshold, p.eom_price):
            ax.axvline(x, color=INK, lw=0.9, alpha=0.4, ls="--")
        greedy = data[f"greedy/{algo}"][seed]
        ax.plot(greedy, steps, lw=3.4, color="white", solid_capstyle="round")
        ax.plot(greedy, steps, lw=1.7, color=INK, solid_capstyle="round")
        if np.isfinite(window[0]):
            for y in window:
                ax.axhline(y, color="#137f59", lw=1.4, ls=(0, (4, 3)))
        ax.set_title(f"{algo} · {name}", loc="left", fontsize=11,
                     color=COLORS.get(algo, INK))
        ax.set_xlabel("bid price (EUR/MWh)")
        ax.set_xlim(bids[0], bids[-1])
        ax.set_ylim(steps[0], steps[-1])
        if col == 0:
            ax.set_ylabel("environment steps")
            ax.annotate(
                "the softsign actor\ncrosses here",
                xy=(-95, window[1]), xytext=(0, 8), textcoords="offset points",
                fontsize=8.5, color="#137f59",
            )
        else:
            ax.tick_params(labelleft=False)
            ax.annotate(
                "same field, same steps —\nbut a = 1.0 exactly, so no step is taken",
                xy=(-95, window[1]), xytext=(0, 8), textcoords="offset points",
                fontsize=8.5, color="#b03a39",
            )

    cbar = fig.colorbar(im, ax=heat, fraction=0.02, pad=0.012)
    cbar.set_label("dQ/d(bid)   (symlog, autograd)", fontsize=9, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    # --- row 2 left: the coherent pull, and its decay -------------------------
    ax_pull = fig.add_subplot(bottom[0])
    ax_pull.axhline(0.0, lw=1.2, color=INK, zorder=0)
    for name, per_seed in stats.items():
        for s, d in enumerate(per_seed):
            ax_pull.plot(
                steps, d["stats"]["pull"], lw=1.9 if s == seed else 1.0,
                alpha=1.0 if s == seed else 0.25,
                ls="--" if name != reference else "-",
                color=COLORS.get(algo, MUTED),
                label=f"{algo} · {name}" if s == seed else None,
            )
    ax_pull.set_yscale("symlog", linthresh=1e-6)
    ax_pull.set_title(
        "mean dQ/d(bid) over the plateau [50, 100] — below zero pulls toward the band",
        loc="left", fontsize=10.5, color=INK,
    )
    ax_pull.set_xlabel("environment steps")
    ax_pull.set_ylabel("mean dQ/d(bid)")
    ax_pull.legend(frameon=False, fontsize=9, loc="upper right")

    # --- row 2 right: how much of the plateau agrees --------------------------
    ax_left = fig.add_subplot(bottom[1])
    for name, per_seed in stats.items():
        for s, d in enumerate(per_seed):
            ax_left.plot(
                steps, d["stats"]["left_frac"], lw=1.9 if s == seed else 1.0,
                alpha=1.0 if s == seed else 0.25,
                ls="--" if name != reference else "-",
                color=COLORS.get(algo, MUTED),
                label=f"{algo} · {name}" if s == seed else None,
            )
    ax_left.axhline(1.0, ls=":", lw=1.2, color=MUTED, zorder=0)
    ax_left.set_ylim(-0.03, 1.08)
    ax_left.set_title(
        "share of the plateau pulling toward the band (1.0 = an unbroken path)",
        loc="left", fontsize=10.5, color=INK,
    )
    ax_left.set_xlabel("environment steps")
    ax_left.set_ylabel("share of [50, 100] with dQ/d(bid) < 0")
    ax_left.legend(frameon=False, fontsize=9, loc="center left")

    for ax in (ax_pull, ax_left):
        if np.isfinite(window[0]):
            ax.axvspan(*window, color="#1baf7a", alpha=0.12, lw=0, zorder=0)
        ax.axvline(warmup, ls=":", lw=1.2, color=INK, zorder=0)
        ax.set_xlim(steps[0], steps[-1])
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
    ax_left.annotate(
        "the escape window", xy=(window[1], 1.02), xytext=(6, 0),
        textcoords="offset points", fontsize=8.5, color="#137f59",
    )

    for ax in (*heat, ax_pull, ax_left):
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    fig.suptitle(
        "The descent path exists for a few hundred steps, then closes",
        x=0.006, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.006, 0.962,
        "Both activations see the same clean field during the window; only softsign "
        "can take a step while it is open. Pooling the path metrics over the whole "
        "run averages the window away.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        default=None,
        metavar="LABEL=PATH",
        help="runs to compare, first one is the reference whose crossing defines "
        "the window. Defaults to the softsign/tanh pair of run 06",
    )
    parser.add_argument("--algo", default="TD3")
    parser.add_argument("--seed", type=int, default=0, help="which seed the figure draws")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "descent_window.png")
    args = parser.parse_args()

    pairs = (
        [tuple(r.split("=", 1)) for r in args.runs]
        if args.runs
        else [("softsign", str(resolve("window_softsign.npz"))),
              ("tanh", str(resolve("window_tanh.npz")))]
    )
    runs = {label: np.load(path) for label, path in pairs}
    warmup = int(runs[next(iter(runs))]["cfg/warmup"])

    stats = report(runs, args.algo, warmup)
    plot(runs, stats, args.algo, args.seed, warmup, args.out)


if __name__ == "__main__":
    main()
