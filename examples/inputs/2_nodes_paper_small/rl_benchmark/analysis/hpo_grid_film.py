# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
One figure for a whole hyperparameter sweep: critic gradient evolution per cell.

Draws ``dQ1/d(bid)`` over (critic updates, bid) as a heatmap, **one panel per
hyperparameter cell**, from the films ``eom_critic_film.py --hp <cell>`` writes.
Same reading as every other critic film in this benchmark: **blue is a gradient
pushing the bid down, red one pushing it up**, and what matters is whether a
descent path ever opens toward the equilibrium.

Read one regime at a time
-------------------------
``--regime`` picks which probed observations enter the panels, and on the EOM
cases it is the whole point. Run 14b found the critic fits the regime it sees
often and leaves the rare one as noise, so an aggregate panel mostly reports
the common regime and a setting that helps only there looks like a win it is
not. On ``p1`` the two are:

* **bertrand** -- at least one learner is undispatched and undercuts anything
  above cost, so the Nash equilibrium is **everyone at marginal cost**;
* **pivotal** -- every learner runs and one only partly, so it cannot be
  replaced by a peer and the equilibrium is the **backup's marginal cost**.

``--regime each`` writes one figure per regime, which is the intended use.

The equilibrium each panel should be converging to therefore *differs by
regime*, and is drawn as the dashed line -- read from the case's own merit
order, not hardcoded, so it follows the scenario.

Usage::

    python analysis/hpo_grid_film.py --data-dir .../18c-hpo-eom --regime each
    python analysis/hpo_grid_film.py --cells lr --regime pivotal
    python analysis/hpo_grid_film.py --cells lr --unit pp_6
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
from eom_critic_evolution import (  # noqa: E402
    load,
    marginal_costs,
    pick_units,
    subset,
)
from eom_critic_film import CASES, REGIMES  # noqa: E402
from hpo_grid import describe as describe_cell  # noqa: E402
from hpo_grid import resolve as resolve_cells  # noqa: E402
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

DEFAULT_DATA = OUT_DIR / "runs" / "data" / "18c-hpo-eom"


def strip(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8, length=3)


def panel_data(run: dict, sweep: str, regime: str | None,
               units: list[str] | None) -> dict | None:
    """One cell's mean gradient and Q field, restricted to a regime and units."""
    if sweep not in run["sweeps"]:
        return None
    view = subset(run, regime)
    if units:
        view = pick_units(view, units)
    if not view or not len(view["regime"]):
        return None
    s = view["sweeps"].index(sweep)
    # (agents, obs, frames, grid) -> mean over agents and probed observations.
    # The spread across those is a different question (run 10's coherence); one
    # panel per cell is what makes the sweep comparable at a glance.
    return {
        "steps": view["steps"],
        "bids": view["bids"],
        "grad": view["grad"][:, s].mean(axis=(0, 1)),
        "q1": view["q1"][:, s].mean(axis=(0, 1)),
        "mc": view["mc"],
    }


def plot(panels: list[tuple[str, dict]], case: str, regime: str | None,
         seed: int, out: Path, ncols: int) -> None:
    grad = np.stack([p["grad"] for _, p in panels])
    scale = float(np.percentile(np.abs(grad), 99.5)) or 1e-6
    norm = SymLogNorm(linthresh=scale / 50, vmin=-scale, vmax=scale, base=10)

    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows),
                             squeeze=False, sharex=True, sharey=True,
                             layout="constrained")
    mesh = None
    for ax, (name, p) in zip(axes.ravel(), panels):
        mesh = ax.pcolormesh(p["steps"], p["bids"], p["grad"].T, cmap=DIVERGING,
                             norm=norm, shading="nearest", rasterized=True)
        # each learning unit's own marginal cost, read out of the recorded
        # observations rather than off a per-scenario table (see
        # eom_critic_evolution.marginal_costs). In the bertrand regime the Nash
        # equilibrium IS this line; in the pivotal one it is the backup's, i.e.
        # above every line drawn.
        for level in np.unique(np.round(np.asarray(p["mc"], dtype=float), 2)):
            ax.axhline(float(level), color=INK, lw=0.9, ls="--", alpha=0.6)
        ax.plot(p["steps"], p["bids"][np.argmax(p["q1"], axis=1)],
                color=INK, lw=1.4)
        ax.set_title(f"{name}", loc="left", fontsize=9.5, color=INK)
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
            axes[visible[-1]][col].set_xlabel("critic updates")

    if mesh is not None:
        cb = fig.colorbar(mesh, ax=axes, fraction=0.018, pad=0.012)
        cb.set_label("dQ1/d(bid)   blue = push the bid down", fontsize=9)
        cb.ax.tick_params(labelsize=8)

    what = REGIMES.get(regime or "", "")
    fig.suptitle(
        f"Hyperparameter sweep on {case}"
        + (f", {regime} regime (NE: {what})" if regime not in (None, "all") else ""),
        x=0.004, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.supxlabel(
        f"dashed = marginal costs of the merit order    solid = argmax Q1    "
        f"seed {seed}, averaged over the learning units and the probed "
        f"observations of this regime",
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
    p.add_argument("--case", default="p1", choices=list(CASES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cells", nargs="+", default=["all"],
                   help="cell names, an axis group (centre, lr, batch, delay, "
                        "wd) or 'all'. See real_matd3/hpo_grid.py")
    p.add_argument("--sweep", default="diag")
    p.add_argument("--regime", default="each", choices=[*REGIMES, "all", "each"],
                   help="'each' writes one figure per regime present, which is "
                        "the intended use on the EOM cases -- an aggregate "
                        "panel mostly reports whichever regime is common")
    p.add_argument("--unit", nargs="+", default=None,
                   help="restrict to these learning units (default: all)")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--img-dir", type=Path, default=None)
    args = p.parse_args()

    cells = resolve_cells(args.cells)
    runs = {c: load(args.data_dir, args.case, args.seed, c) for c in cells}
    have = {c: r for c, r in runs.items() if r is not None}
    if not have:
        raise SystemExit(
            f"no films for {args.case} seed {args.seed} under {args.data_dir}\n"
            f"record them with:  bash cluster/hpo_eom.sh"
        )
    missing = [c for c in cells if c not in have]
    if missing:
        print(f"  no film for {len(missing)} cell(s): {', '.join(missing)}")

    img = args.img_dir or (OUT_DIR / "runs" / "img" / "18c")
    regimes = (
        list(dict.fromkeys(np.concatenate([r["regime"] for r in have.values()])))
        if args.regime == "each" else [args.regime]
    )
    for regime in regimes:
        panels = []
        for name, run in have.items():
            data = panel_data(run, args.sweep, regime, args.unit)
            if data is None:
                continue
            panels.append((f"{name}   {describe_cell(name)}", data))
        if not panels:
            print(f"  regime {regime!r}: no cell has probes there -- skipped")
            continue
        tag = "" if regime in (None, "all") else f"-{regime}"
        plot(panels, args.case, regime, args.seed,
             img / f"18c-hpo-{args.case}{tag}.png", args.ncols)


if __name__ == "__main__":
    main()
