# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
One figure for the live architecture sweep (run 18a), panel per architecture.

The live counterpart of ``analysis/offline_arch_film.py``, drawn the same way
and ordered the same way so the two can be laid side by side: ``dQ1/d(bid)``
over (training frame, bid) as a heatmap, **blue is a gradient pushing the bid
down**, the argmax trace on top, panels sorted by parameter count.

What changes between the two figures is everything the offline harness took
out. There ``gamma = 0``, the buffer was frozen and no actor moved the action
distribution; here the bootstrap, the growing buffer and the actor are all
back. So a variant that opened a descent path offline and does not here has
been beaten by the *loop*, not by the regression problem -- which is the one
distinction run 17 could not make and this figure exists to make.

The landscape, for reading the panels (``surrogate/incdec_reward.py``):

* **bid > 49** -- not dispatched, reward 0, flat
* **30 <= bid <= 49** -- dispatched then dec'd, reward ``(49 - bid) / 100``.
  This is the band, shaded in every panel.
* **bid < 30** -- dispatched, not dec'd, reward -0.170, flat

So the correct field is **negative slope inside the band** and the optimum a
stochastic policy should aim at is **32.31**, not 30 -- the cliff one tick below
30 costs 0.36. Both lines are drawn.

The x axis is training frames, not offline update steps: the study case runs
24 h at ``train_freq 12h`` with a preloaded buffer, so every episode leaves two
frames and the axis is ``2 x episodes`` long.

Usage::

    python analysis/live_arch_film.py
    python analysis/live_arch_film.py --archs baseline split sbn-d2-8M
    python analysis/live_arch_film.py --round hpo
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "real_matd3"))

from _layout import OUT_DIR  # noqa: E402  (also sets sys.path)
from assume_arch_sweep import ARCHS, SHAPE, result_path  # noqa: E402
from critic_architectures import param_count  # noqa: E402
from hpo_grid import resolve as resolve_cells  # noqa: E402
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

#: the reward band and the constrained optimum, from surrogate/incdec_reward.py
BAND = (30.0, 49.0)
OPTIMUM = 32.31

DEFAULT_DATA = OUT_DIR / "runs" / "data" / "18-live-arch"


def strip(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8, length=3)


def load(data_dir: Path, round_: str, name: str, seed: int) -> dict | None:
    path = result_path(data_dir, round_, name, seed)
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    # (n_obs, frames, grid) -> mean over the probed observations. Their spread
    # is run 10's coherence question and is not what this figure is about.
    return {
        "steps": d["steps"],
        "bids": d["critic_bids"],
        "grad": d["critic_grad/MATD3"].mean(axis=0),
        "q1": d["critic_q/MATD3"].mean(axis=0),
    }


def plot(panels: list[tuple[str, dict]], round_: str, seed: int,
         out: Path, ncols: int) -> None:
    grad = np.stack([p["grad"] for _, p in panels])
    # one symmetric colour scale across every panel, or the architectures are
    # not comparable by eye -- which is the entire point of the figure
    scale = float(np.percentile(np.abs(grad), 99.5)) or 1e-6
    norm = SymLogNorm(linthresh=scale / 50, vmin=-scale, vmax=scale, base=10)

    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows),
                             squeeze=False, sharex=True, sharey=True,
                             layout="constrained")
    mesh = None
    for ax, (title, p) in zip(axes.ravel(), panels):
        mesh = ax.pcolormesh(p["steps"], p["bids"], p["grad"].T, cmap=DIVERGING,
                             norm=norm, shading="nearest", rasterized=True)
        ax.axhspan(*BAND, color=INK, alpha=0.10, lw=0)
        ax.axhline(OPTIMUM, color=INK, lw=0.9, ls="--", alpha=0.75)
        ax.plot(p["steps"], p["bids"][np.argmax(p["q1"], axis=1)],
                color=INK, lw=1.4)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_ylim(-100, 100)
        strip(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("bid (EUR/MWh)")

    # sharex hides tick labels on every row but the last, and the last row is
    # usually partial -- which would leave whole columns with no x axis at all
    for col in range(ncols):
        visible = [r for r in range(nrows) if axes[r][col].get_visible()]
        if visible:
            axes[visible[-1]][col].tick_params(labelbottom=True)
            axes[visible[-1]][col].set_xlabel("training frame")

    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes, fraction=0.018, pad=0.012)
        cb.set_label("dQ1/d(bid)   blue = push the bid down", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Live MATD3 on inc-dec: does any "
        + ("architecture" if round_ == "arch" else "setting")
        + " open a descent path into the reward band?",
        x=0.004, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.supxlabel(
        f"shaded = reward band [30, 49]    dashed = constrained optimum "
        f"{OPTIMUM}    solid = argmax Q1        "
        f"seed {seed}, each panel averaged over the probed observations"
        + ("    panels ordered by parameter count" if round_ == "arch" else ""),
        x=0.004, ha="left", fontsize=9.5, color=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", default="arch", choices=["arch", "hpo"])
    p.add_argument("--archs", nargs="+", default=None,
                   help="restrict the 'arch' round to these architectures")
    p.add_argument("--cells", nargs="+", default=["all"],
                   help="which cells the 'hpo' round draws; a cell name, an "
                        "axis group or 'all'. See real_matd3/hpo_grid.py")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    names = (args.archs or ARCHS) if args.round == "arch" else resolve_cells(args.cells)
    found = [(n, d) for n, d in ((n, load(args.data_dir, args.round, n, args.seed))
                                 for n in names) if d is not None]
    if not found:
        raise SystemExit(
            f"no films for round {args.round!r} seed {args.seed} under "
            f"{args.data_dir}\nrecord them with:  bash cluster/critic_arch.sh"
        )
    if len(found) < len(names):
        missing = sorted(set(names) - {n for n, _ in found})
        print(f"  no film for {len(missing)} cell(s): {', '.join(missing)}")

    if args.round == "arch":
        # by capacity, so the two families' matched rungs sit side by side and
        # the ladder reads off the rows -- the same order run 17's figure uses
        sized = []
        for name, data in found:
            try:
                n_params = param_count(name, **SHAPE)
            except Exception:
                n_params = 0
            sized.append((n_params, f"{name}   ({n_params:,} params)", data))
        sized.sort(key=lambda row: row[0])
        panels = [(title, data) for _, title, data in sized]
    else:
        panels = [(name, data) for name, data in found]

    out = args.out or (OUT_DIR / "runs" / "img" / "18"
                       / f"18-live-{args.round}-grad.png")
    plot(panels, args.round, args.seed, out, args.ncols)


if __name__ == "__main__":
    main()
