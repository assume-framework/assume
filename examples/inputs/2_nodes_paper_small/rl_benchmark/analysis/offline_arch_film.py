# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
One figure for the whole offline architecture screen (workstream B, run 17).

Draws ``dQ1/d(bid)`` over (critic updates, bid) as a heatmap, one panel per
critic architecture, from the film ``assume_offline_critic.py --film`` writes.
Same reading as the live critic films in ``eom_critic_evolution.py``: **blue is
a gradient pushing the bid down, red one pushing it up**, and what matters is
whether a descent path ever opens between the ceiling and the reward band.

Why a gradient and not ``Q`` itself: the actor climbs the gradient, so a flat
region is a region the actor cannot cross however right the critic's peak is.
``Q`` is drawn too, as the argmax trace on top -- the thin line is where the
critic would send a greedy bid at that moment.

The landscape, for reading the panels (``surrogate/incdec_reward.py``):

* **bid > 49** -- not dispatched, reward 0, flat
* **30 <= bid <= 49** -- dispatched then dec'd, reward ``(49 - bid) / 100``,
  peaking at bid 30. This is the band, shaded in every panel.
* **bid < 30** -- dispatched, not dec'd, reward -0.170, flat

So the correct field is **negative slope inside the band** (down toward 30) and
the optimum a stochastic policy should aim at is **32.31**, not 30 -- the cliff
one tick below 30 costs 0.36. Both lines are drawn.

The failure this is a picture of: every architecture in the archive before this
one parks its argmax at exactly **100.0**, the far ceiling, and never develops
any slope inside the band at all.

Usage::

    python analysis/offline_arch_film.py
    python analysis/offline_arch_film.py --arch baseline simba simba-nornorm
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
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

#: the reward band and the constrained optimum, from surrogate/incdec_reward.py
BAND = (30.0, 49.0)
OPTIMUM = 32.31

DEFAULT_FILM = OUT_DIR / "runs" / "data" / "17-offline-arch" / "films.npz"


def strip(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8, length=3)


def plot(film: Path, out: Path, which: list[str] | None, ncols: int,
         order: str = "params") -> None:
    d = np.load(film, allow_pickle=False)
    names = [str(a) for a in d["arch"]]
    keep = [i for i, n in enumerate(names) if which is None or n in which]
    if not keep:
        raise SystemExit(f"none of {which} in the film ({names})")

    bids, params = d["bids"], d["params"]
    if order == "params":
        # Sorting by capacity is what makes the ladder legible: the matched
        # widths land side by side, so the reading is "at this size, does the
        # trunk open a path where the MLP does not" rather than a hunt across
        # the grid. Stable, so variants at one width keep registry order.
        keep.sort(key=lambda i: int(params[i]))
    # (arch, frames, probe, bids) -> mean over the probed observations. The
    # spread across probes is the run 10 coherence question and is not what
    # this figure is about; averaging keeps one panel per architecture.
    grad = d["grad"][:, :, :, :].mean(axis=2)
    q = d["q"].mean(axis=2)
    steps = d["steps"]

    # one symmetric colour scale across every panel, or the architectures are
    # not comparable by eye -- which is the entire point of the figure
    scale = float(np.percentile(np.abs(grad[keep]), 99.5)) or 1e-6
    norm = SymLogNorm(linthresh=scale / 50, vmin=-scale, vmax=scale, base=10)

    nrows = int(np.ceil(len(keep) / ncols))
    # constrained layout, because the figure carries a two-line suptitle, a
    # per-panel title and a shared colorbar; with tight_layout the title lands
    # on top of the first row's panel titles
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows),
                             squeeze=False, sharex=True, sharey=True,
                             layout="constrained")
    mesh = None
    for ax, i in zip(axes.ravel(), keep):
        mesh = ax.pcolormesh(steps[i], bids, grad[i].T, cmap=DIVERGING,
                             norm=norm, shading="nearest", rasterized=True)
        ax.axhspan(*BAND, color=INK, alpha=0.10, lw=0)
        ax.axhline(OPTIMUM, color=INK, lw=0.9, ls="--", alpha=0.75)
        # where a greedy actor would bid at each moment
        ax.plot(steps[i], bids[np.argmax(q[i], axis=1)], color=INK, lw=1.4)
        ax.set_title(f"{names[i]}   ({params[i]:,} params)", loc="left",
                     fontsize=9.5, color=INK)
        ax.set_ylim(-100, 100)
        strip(ax)
    for ax in axes.ravel()[len(keep):]:
        ax.set_visible(False)
    for row in axes:
        row[0].set_ylabel("bid (EUR/MWh)")

    # sharex hides tick labels on every row but the last, and the last row is
    # usually a partial one -- which would leave whole columns with no x axis
    # at all. Label the bottom-most *visible* panel of each column instead.
    for col in range(ncols):
        visible = [r for r in range(nrows) if axes[r][col].get_visible()]
        if not visible:
            continue
        ax = axes[visible[-1]][col]
        ax.tick_params(labelbottom=True)
        ax.set_xlabel("critic updates")

    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes, fraction=0.018, pad=0.012)
        cb.set_label("dQ1/d(bid)   blue = push the bid down", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Offline critic screen: does any architecture open a descent path "
        "into the reward band?",
        x=0.004, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    # the key goes at the bottom as a caption: placed above the title it reads
    # before the thing it is explaining, and constrained layout puts the
    # suptitle flush against the top edge anyway
    fig.supxlabel(
        f"shaded = reward band [30, 49]    dashed = constrained optimum "
        f"{OPTIMUM}    solid = argmax Q1        "
        f"gamma = 0, seed {int(d['seed'])}, each panel averaged over the "
        f"{d['q'].shape[2]} probed observations"
        + ("    panels ordered by parameter count" if order == "params" else ""),
        x=0.004, ha="left", fontsize=9.5, color=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--film", type=Path, default=DEFAULT_FILM)
    p.add_argument("--arch", nargs="+", default=None,
                   help="restrict to these architectures, in the film's order")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--order", choices=["params", "registry"], default="params",
                   help="panel order: ascending parameter count (default, so "
                        "the matched widths sit together) or the film's own")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not args.film.exists():
        raise SystemExit(
            f"no film at {args.film}\n"
            "record one with:  python real_matd3/assume_offline_critic.py "
            "--round arch --film 80"
        )
    out = args.out or (OUT_DIR / "runs" / "img" / "17"
                       / "17-offline-arch-grad.png")
    plot(args.film, out, args.arch, args.ncols, args.order)


if __name__ == "__main__":
    main()
