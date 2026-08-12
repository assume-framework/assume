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

Four figures, selectable with ``--only``:

``film``            one PNG per (case, seed): a panel per learning unit showing
                    ``dQ1/d(bid)`` over (bid, critic updates) with the actor's
                    bid trajectory on top, plus a greedy-bid axis and a
                    buffer-reward axis underneath.
``summary``         one PNG for the set: final bid, ``act_share`` and mean
                    buffer reward across the cases drawn. Thin lines are seeds,
                    thick lines their median. The x axis follows ``CASES``
                    order, so a run of all six puts the two-bid ladder and its
                    single-bid twins side by side.
``regime-heatmap``  the critic's **field** in each demand regime side by side on
                    one shared colour scale, plus their difference.
``separation``      the same comparison reduced to two lines per regime — where
                    the actor bids and where ``argmax Q1`` sits — against both
                    equilibrium prices.

The last two need a run recorded with ``eom_critic_film.py --obs-regimes``,
which stratifies the probed observations by demand regime. Without it every
observation is labelled ``"any"`` and both are skipped. Together they separate
the two ways a policy can fail to price the pivotal hours: ``separation`` shows
whether the actor ends up somewhere different, ``regime-heatmap`` whether the
critic's landscape differs at all — a peak can move while the field stays flat,
and a flat field is a gradient the actor cannot climb however right the peak is.

Usage::

    python analysis/eom_critic_evolution.py                    # sb02a-c
    python analysis/eom_critic_evolution.py --cases 02a 02b 02c
    python analysis/eom_critic_evolution.py --sweep a0 --only film

    # per-equilibrium views, after an --obs-regimes run
    python analysis/eom_critic_evolution.py --only regime-heatmap
    python analysis/eom_critic_evolution.py --regime each --only film
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR  # noqa: E402  (also sets sys.path)
from eom_critic_film import (  # noqa: E402
    CASES,
    DEFAULT_CASES,
    REGIMES,
    SEEDS,
    act_share_from_sd,
    merit_order,
    result_path,
)
from run_benchmark import DIVERGING, INK, MUTED  # noqa: E402

#: shared with analysis/eom_exploitability.py so a regime is the same colour
#: wherever it is drawn
REGIME_COLOR = {"idle": "#b8b6b0", "bertrand": "#2f7fb8",
                "pivotal": "#d1642f", "backup": "#6b4f8a", "any": INK}

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


def frame_schedule(case: str, n_frames: int, every: int = 4) -> dict | None:
    """Where in the simulated horizon each frame sits.

    ⚠️ **The x axis of these figures is not a time axis, and it is aliased.**
    Every episode replays the same calendar month, so a frame carries two
    unrelated coordinates at once: how far training has got, and where in the
    month the snapshot was taken. ``learning_role.sync_train_freq_with_
    simulation_horizon`` snaps ``train_freq`` so it divides the horizon evenly —
    ``100h`` over 30 days becomes **90h**, over 31 days **93h**, eight blocks
    either way — and the recorder snapshots every ``every``-th block. At
    ``every = 4`` that is **two frames per episode**, so the frame index samples
    exactly two phases of the month forever: even frames end mid-month, odd
    frames end at the month's end. Any period-2 wobble in a trace is the
    calendar, not the learner.

    Returns the per-frame episode, block, snapshot time and the window of
    simulated time the frame's reward average covers, or ``None`` for a case
    whose horizon is not known here.
    """
    spec = {"sb02a": ("2019-03-01", "2019-03-31"), "02a": ("2019-03-01", "2019-03-31")}
    start_s, end_s = spec.get(case, ("2019-03-01", "2019-04-01"))
    start, end = pd.Timestamp(start_s), pd.Timestamp(end_s)

    total = end - start
    quotient, remainder = divmod(total, pd.Timedelta("100h"))
    n_blocks = int(quotient) + (1 if remainder else 0)
    train_freq = pd.Timedelta(hours=int((total / n_blocks).total_seconds() // 3600))

    rows = []
    for f in range(n_frames):
        done = (f + 1) * every                      # blocks completed by this frame
        episode, block = divmod(done - 1, n_blocks)
        t = start + (block + 1) * train_freq
        w0 = start + max(0, block + 1 - every) * train_freq
        rows.append({"episode": episode, "block": block + 1, "time": t, "window": (w0, t)})
    return {"start": start, "end": end, "train_freq": train_freq,
            "blocks_per_episode": n_blocks, "frames_per_episode": n_blocks // every,
            "rows": rows}


def demand_series(case: str):
    """Hourly demand of the scenario this case runs, or ``None`` if unreadable."""
    scenario = CASES[case]["scenario"]
    path = (Path(__file__).resolve().parents[4] / "inputs" / scenario / "demand_df.csv")
    if not path.is_file():
        return None
    s = pd.read_csv(path, index_col=0, parse_dates=True)["demand_EOM"]
    return s.resample("1h").mean()


def subset(run: dict, regime: str | None) -> dict:
    """The run restricted to the observations probed in one demand regime.

    The observation axis is shared by every recorded array, so restricting it is
    a slice — which is what makes "the critic film in the pivotal hours" a view
    of an existing recording rather than a second run. ``None`` or ``"all"``
    returns the run unchanged.
    """
    if regime in (None, "all"):
        return run
    m = run["regime"] == regime
    if not m.any():
        return {}
    out = dict(run)
    out["regime"] = run["regime"][m]
    out["q1"] = run["q1"][:, :, m]          # (agents, sweeps, obs, frames, grid)
    out["grad"] = run["grad"][:, :, m]
    out["greedy"] = run["greedy"][:, :, m]  # (agents, act_dim, obs, frames)
    out["regime_label"] = regime
    return out


def pick_units(run: dict, names: list[str] | None) -> dict:
    """The run restricted to some of its learning units.

    The companion of :func:`subset`: that one slices the observation axis (and
    so the demand regime), this one slices the agent axis. Every recorded array
    keeps both un-aggregated — ``critic_grad`` is ``(agent, sweep, observation,
    frame, bid)`` on disk — so any (unit × seed × regime) view is a slice, never
    a re-run. ``None`` returns the run unchanged.
    """
    if not names:
        return run
    idx = [i for i, u in enumerate(run["units"]) if u in set(names)]
    if not idx:
        return {}
    out = dict(run)
    out["units"] = [run["units"][i] for i in idx]
    out["q1"] = run["q1"][idx]
    out["grad"] = run["grad"][idx]
    out["greedy"] = run["greedy"][idx]
    out["rewards"] = run["rewards"][:, idx]
    out["mc"] = run["mc"][idx]
    out["share"] = run["share"][idx]
    return out


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
    n_obs = d["observations"].shape[0]
    # runs recorded without --obs-regimes have no label at all
    regime = ([str(r) for r in d["obs_regime"]] if "obs_regime" in d.files
              else ["any"] * n_obs)
    return {
        "case": case,
        "regime": np.array(regime),
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

    # One panel per unit, wrapped so ten units stay readable. The grid keeps at
    # least three columns even for a single learner: the bottom row is two axes
    # side by side and needs the width, and sizing the figure off `n` alone made
    # the sb02a films collide (heatmap squeezed against the colorbar, the two
    # bottom titles overlapping).
    cols = min(n, 5)
    grid_cols = max(cols, 3)
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.1 * grid_cols + 1.6, 3.0 * rows + 5.0))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.0 * rows, 4.4], hspace=0.32)
    top = outer[0].subgridspec(rows, grid_cols, wspace=0.10, hspace=0.28)
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

    # the bar is a fraction of the axes it is attached to, so it has to shrink
    # as the panel count does or it swallows a one-panel figure
    cbar = fig.colorbar(im, ax=axes, fraction=0.045 / grid_cols, pad=0.012)
    cbar.set_label(f"dQ1/d(bid), sweep '{sweep}'   (symlog, autograd)",
                   fontsize=9, color=MUTED)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)

    ax_bid = fig.add_subplot(bottom[0])
    ax_rew = fig.add_subplot(bottom[1])
    colours = plt.cm.viridis(np.linspace(0.05, 0.9, n))
    single_bid = run["greedy"].shape[1] == 1

    sched = frame_schedule(run["case"], len(steps))
    phase = phases = None
    if sched:
        phases = sched["frames_per_episode"]
        phase = np.array([r["block"] for r in sched["rows"]])
        phase = np.unique(phase, return_inverse=True)[1]
    for i, unit in enumerate(units):
        series = bid_series(run, i)
        ax_bid.plot(steps, series[0][0], lw=1.6, color=colours[i], label=unit)
        for extra, _ in series[1:]:
            ax_bid.plot(steps, extra, lw=1.0, color=colours[i], alpha=0.5, ls="--")
        # Split the reward by horizon phase. The frame index samples the month
        # at `frames_per_episode` fixed positions and nothing else, so an
        # unsplit trace shows a sawtooth that is the calendar rather than the
        # learner -- see frame_schedule().
        if phase is None:
            ax_rew.plot(steps, run["rewards"][:, i], lw=1.4, color=colours[i])
        else:
            for p, style in zip(range(phases), ("-", "--", ":", "-.")):
                m = phase == p
                ax_rew.plot(steps[m], run["rewards"][m, i], lw=1.4, ls=style,
                            color=colours[i])
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

    if sched:
        # Name the phases and say what the demand was in each. Without this the
        # reward panel's sawtooth reads as instability; it is the month.
        dem = demand_series(run["case"])
        parts = []
        for p, style in zip(range(phases), ("solid", "dashed", "dotted", "dashdot")):
            first = next(r for r in sched["rows"] if r["block"]
                         == sorted({r["block"] for r in sched["rows"]})[p])
            w0, w1 = first["window"]
            label = f"{style} {w0:%m-%d %H:%M}→{w1:%m-%d %H:%M}"
            if dem is not None:
                label += f" ({dem.loc[w0:w1 - pd.Timedelta('1h')].mean():.0f} MW)"
            parts.append(label)
        ax_rew.set_title(
            "mean stored reward, split by the horizon window it averages\n"
            + ";  ".join(parts),
            loc="left", fontsize=9, color=INK,
        )
        # the honest training-progress axis: every episode replays the calendar
        top_ax = ax_rew.secondary_xaxis(
            "top",
            functions=(lambda s: s / (sched["blocks_per_episode"] * 10),
                       lambda e: e * sched["blocks_per_episode"] * 10),
        )
        top_ax.set_xlabel("training episodes (each replays the same month)",
                          fontsize=8.5, color=MUTED)
        top_ax.tick_params(colors=MUTED, labelsize=8)
    for ax in (ax_bid, ax_rew):
        strip(ax)
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)

    case = CASES[run["case"]]
    reg = run.get("regime_label")
    fig.suptitle(
        f"{run['case']} ({case['scenario']}/{case['study_case']}) seed "
        f"{run['seed']} -- {n} learning unit(s) on a plain EOM, "
        f"{'single' if single_bid else 'two'}-bid"
        + (f"  |  {reg} hours only, NE = {REGIMES[reg]}" if reg else ""),
        x=0.006, y=0.998, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# -------------------------------------------------- regime heatmap compare


def plot_regime_heatmaps(run: dict, sweep: str, out: Path) -> None:
    """The critic's field in each demand regime, side by side, plus the difference.

    ``plot_separation`` reduces the critic to one number per frame — where the
    peak of ``Q1`` sits. That answers "does the actor end up somewhere else"
    but not "does the critic *know* the hours are different", which is the
    question that separates a critic failure from an actor failure: the peak can
    move while the field stays flat, and a flat field is a gradient the actor
    cannot climb no matter how right the peak is.

    So: one panel per regime showing ``dQ1/d(bid)`` over (bid, critic updates),
    medianed over the learning units and over that regime's probed
    observations, on **one shared colour scale** so the panels are comparable;
    then a **difference panel**, the last regime minus the first, on its own
    scale. Marginal cost and the backup's marginal cost are marked, because
    those are the two equilibrium prices the panels are being asked to
    distinguish.
    """
    regimes = [g for g in REGIMES if g in set(run["regime"])]
    if len(regimes) < 2:
        print(f"  {run['case']} seed {run['seed']}: only regime(s) {regimes} "
              "probed -- nothing to compare")
        return

    s = run["sweeps"].index(sweep)
    steps, bids = run["steps"], run["bids"]
    spec = merit_order(run["case"])

    fields = {}
    for regime in regimes:
        v = subset(run, regime)
        # median over agents and over that regime's observations
        fields[regime] = np.median(v["grad"][:, s], axis=(0, 1))    # (frames, grid)

    n = len(regimes)
    # constrained layout, because the shared colour bar sits *between* the
    # regime panels and the difference panel and its label runs into the latter
    # under the default one
    fig, axes = plt.subplots(1, n + 1, figsize=(4.3 * (n + 1), 4.8),
                             squeeze=False, sharey=True, layout="constrained")
    axes = axes[0]

    vmax = float(np.percentile(np.abs(np.stack(list(fields.values()))), 99.5)) or 1e-6
    for ax, regime in zip(axes, regimes):
        im = ax.pcolormesh(
            bids, steps, fields[regime], cmap=DIVERGING,
            norm=SymLogNorm(linthresh=vmax * 1e-3, vmin=-vmax, vmax=vmax, base=10),
            shading="nearest", rasterized=True)
        v = subset(run, regime)
        traj = np.median(v["greedy"].min(axis=1), axis=(0, 1))
        ax.plot(traj, steps, lw=3.0, color="white", solid_capstyle="round")
        ax.plot(traj, steps, lw=1.5, color=INK, solid_capstyle="round")
        ax.set_title(f"{regime} — NE {REGIMES[regime]}", loc="left",
                     fontsize=10, color=REGIME_COLOR[regime])
        ax.set_xlabel("bid price (EUR/MWh)", fontsize=9, color=MUTED)
    fig.colorbar(im, ax=list(axes[:n]), fraction=0.03, pad=0.012).set_label(
        f"dQ1/d(bid), sweep '{sweep}'   (symlog, shared scale)",
        fontsize=8.5, color=MUTED)

    # the difference: what the critic does with the extra information
    a, b = regimes[0], regimes[-1]
    diff = fields[b] - fields[a]
    dmax = float(np.percentile(np.abs(diff), 99.5)) or 1e-6
    ax = axes[n]
    im2 = ax.pcolormesh(
        bids, steps, diff, cmap=DIVERGING,
        norm=SymLogNorm(linthresh=dmax * 1e-3, vmin=-dmax, vmax=dmax, base=10),
        shading="nearest", rasterized=True)
    ax.set_title(f"difference: {b} − {a}", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("bid price (EUR/MWh)", fontsize=9, color=MUTED)
    fig.colorbar(im2, ax=ax, fraction=0.06, pad=0.012).set_label(
        "Δ dQ1/d(bid)  (own scale)", fontsize=8.5, color=MUTED)

    for ax in axes:
        for level, style in ((spec["mc"], "--"), (spec["backup_mc"], ":")):
            if bids[0] <= level <= bids[-1]:
                ax.axvline(level, lw=1.0, ls=style, color=INK, alpha=0.5)
        ax.set_xlim(bids[0], bids[-1])
        strip(ax)
    axes[0].set_ylabel("critic updates", fontsize=9, color=MUTED)

    fig.suptitle(
        f"{run['case']} seed {run['seed']} — does the critic see the two "
        f"equilibria differently?   (dashed = marginal cost {spec['mc']:.1f}, "
        f"dotted = backup {spec['backup_mc']:.1f})",
        x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK,
    )
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ----------------------------------------------------------- NE separation


def plot_separation(runs: list[dict], out: Path) -> None:
    """Does one actor/critic hold both equilibria at once?

    The stage game's Nash equilibrium switches with demand: at ``bertrand``
    demand a learner is undispatched and undercuts, so NE is **marginal cost**;
    at ``pivotal`` demand every learner runs and one is only partly dispatched,
    so NE is the **backup's marginal cost**. Both are drawn as reference lines.

    Two rows: what the **actor** does (greedy bid, median over units) and what
    the **critic** wants (``argmax Q1`` over the bid grid, same median). If a
    run separates the states, the two regime lines sit at different heights,
    each near its own reference. If it does not, they lie on top of each other —
    a single price for a game with two answers.
    """
    runs = [r for r in runs if len(set(r["regime"])) > 1]
    if not runs:
        print("  no run carries more than one probed regime -- "
              "re-record with --obs-regimes (see eom_critic_film.py)")
        return

    cases = list(dict.fromkeys(r["case"] for r in runs))
    fig, axes = plt.subplots(2, len(cases), figsize=(4.9 * len(cases), 7.2),
                             squeeze=False, sharex="col")
    for col, case in enumerate(cases):
        spec = merit_order(case)
        for row, (key, name) in enumerate(((None, "actor: greedy bid"),
                                           (None, "critic: argmax Q1"))):
            ax = axes[row][col]
            for r in [x for x in runs if x["case"] == case]:
                for regime in [g for g in REGIMES if g in set(r["regime"])]:
                    s = subset(r, regime)
                    if not s:
                        continue
                    if row == 0:
                        y = np.median(s["greedy"].min(axis=1), axis=(0, 1))
                    else:
                        sw = s["sweeps"].index("diag")
                        peak = s["bids"][np.argmax(s["q1"][:, sw], axis=-1)]
                        y = np.median(peak, axis=(0, 1))
                    ax.plot(s["steps"], y, lw=1.6, color=REGIME_COLOR[regime],
                            alpha=0.85,
                            label=f"{regime} → NE {REGIMES[regime]}"
                            if r is runs[0] or True else None)
            for level, text, style in ((spec["mc"], "marginal cost", "--"),
                                       (spec["backup_mc"], "backup mc", ":")):
                ax.axhline(level, lw=1.1, ls=style, color=INK, alpha=0.55)
                ax.annotate(text, xy=(0, level), xytext=(3, 3),
                            textcoords="offset points", fontsize=7.5, color=MUTED)
            ax.set_title(f"{case} — {name}", loc="left", fontsize=10.5, color=INK)
            strip(ax)
            ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
            if col == 0:
                ax.set_ylabel("EUR/MWh", fontsize=9, color=MUTED)
            if row == 1:
                ax.set_xlabel("critic updates", fontsize=9, color=MUTED)
        # one legend per column, deduplicated
        h, lab = axes[0][col].get_legend_handles_labels()
        seen = dict(zip(lab, h))
        axes[0][col].legend(seen.values(), seen.keys(), frameon=False, fontsize=8)

    fig.suptitle(
        "Can one policy hold both equilibria? — probes split by demand regime",
        x=0.006, y=1.0, ha="left", fontsize=13, fontweight="bold", color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
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
    parser.add_argument(
        "--only",
        choices=("film", "summary", "separation", "regime-heatmap"),
        default=None)
    parser.add_argument(
        "--regime", default=None, choices=[*REGIMES, "all", "each"],
        help="draw the film from the observations probed in one demand regime "
             "only; 'each' writes one film per regime present. Needs a run "
             "recorded with eom_critic_film.py --obs-regimes")
    parser.add_argument(
        "--unit", nargs="+", default=None,
        help="restrict the film to these learning units, or 'each' for one file "
             "per unit. Combines with --regime, so '--unit each --regime each' "
             "is one film per (unit x regime) for every seed drawn")
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
            regimes = (list(dict.fromkeys(run["regime"])) if args.regime == "each"
                       else [args.regime])
            units = (run["units"] if args.unit == ["each"]
                     else [args.unit])          # None, or one explicit group
            for regime in regimes:
                for group in units:
                    names = [group] if isinstance(group, str) else group
                    view = pick_units(subset(run, regime), names)
                    if not view:
                        print(f"  {run['case']} seed {run['seed']}: nothing "
                              f"matches regime={regime!r} unit={names} -- skipped")
                        continue
                    tag = "".join([
                        f"-{regime}" if regime not in (None, "all") else "",
                        f"-{'_'.join(names)}" if names else "",
                    ])
                    plot_film(
                        view, args.sweep,
                        args.img_dir
                        / f"14-eom-{run['case']}-seed{run['seed']}{tag}.png",
                    )
    if args.only in (None, "regime-heatmap"):
        for run in runs:
            if len(set(run["regime"])) > 1 and args.sweep in run["sweeps"]:
                plot_regime_heatmaps(
                    run, args.sweep,
                    args.img_dir / f"14-eom-regimes-{run['case']}-seed{run['seed']}.png")
    if args.only in (None, "summary"):
        plot_summary(runs, args.img_dir / "14-eom-summary.png")
    if args.only in (None, "separation"):
        plot_separation(runs, args.img_dir / "14-eom-ne-separation.png")


if __name__ == "__main__":
    main()
