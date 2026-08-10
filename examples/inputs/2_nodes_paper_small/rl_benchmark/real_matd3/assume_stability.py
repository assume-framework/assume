# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Are the two ASSUME films reproducible, or were they two draws?

Run 09 filmed one shaped and one unshaped MATD3 run and reported a clean split:
with shaping ``argmax Q1`` falls 100.0 -> 48.8 and the actor follows to 49.0;
without it, both stay at the ceiling for all 2560 critic updates. That is **one
seed each**, and run 08 is the standing warning against reading a single run on
this landscape as a measurement -- in the SB3 analogue, changing the BLAS thread
count alone moved a seed from +31.60 to -60.49.

This runs the same film several times per condition and reports the spread.

How the seed is varied
----------------------
ASSUME has a real seed knob and this uses it. ``loader_csv.py:555`` calls
``set_random_seed(config.get("seed", 42), learning_mode=...)`` once, while the
scenario is being read; nothing re-seeds afterwards, and ``setup_world()`` never
looks at ``config["seed"]`` again. ``assume_training_probe.py --seed`` re-applies
that same call after the scenario is loaded and before ``run_learning()`` builds
the networks.

Re-applying it *after* the load rather than editing the tracked ``config.yaml``
narrows what the seed touches, and the narrowing is deliberate: the scenario's
CSV tables and its forecaster are deterministic reads with no RNG draws
(``common/forecaster.py`` and ``forecast_algorithms.py`` contain none), so the
only things left downstream are the ones a stability sweep is about -- actor and
critic initialisation, the exploration noise, and the replay-buffer batch draws.
Verified: two runs at ``--seed 1`` are bit-identical in every recorded array,
``--seed 2`` is not.

Why the two conditions cannot run together
------------------------------------------
The shaping at ``learning_strategies.py:1583-1589`` is a source edit, not a config
flag, so a process can only ever be in one condition. This script therefore runs
**one condition per invocation** and refuses to start if the source does not match
the condition asked for::

    # repo as committed -- shaping commented out
    python real_matd3/assume_stability.py --condition unshaped --seeds 42 1 2 3 4 5

    # now uncomment learning_strategies.py:1583-1589
    python real_matd3/assume_stability.py --condition shaped   --seeds 42 1 2 3 4 5
    # and comment it back out

    python real_matd3/assume_stability.py --report

Seed 42 is ASSUME's own default, so that row is the closest thing to a rerun of
run 09's films.

Each run is a separate process at ``torch.set_num_threads(1)`` with its own
database, save folder and working directory -- ``run_learning`` drops
``tensorboard/`` and ``assume.log`` into the cwd, and rmtree's its save path
before starting. One thread rather than the default 14 because run 08 showed the
accumulation order of a matrix product is enough to change the outcome here; the
sweep's runs at least have to be comparable to each other.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, RUNS, SCENARIO  # noqa: E402  (also sets sys.path)
from incdec_reward import PAPER_SMALL, reward_from_bid  # noqa: E402
from run_benchmark import INK, MUTED  # noqa: E402

ALGO = "MATD3"
REPO = SCENARIO.parents[2]
PROBE = Path(__file__).resolve().parent / "assume_training_probe.py"

#: The source file whose 1583-1589 block *is* the condition.
SHAPING_SOURCE = REPO / "assume" / "strategies" / "learning_strategies.py"
#: One line of the shaped branch, matched at the start of a line so a commented
#: copy does not count. Deliberately a line from the middle of the block rather
#: than the ``if``, which is too common a shape to be distinctive.
SHAPING_LINE = re.compile(r"^\s*reward = \(price - marginal_cost\) / 100\s*$", re.M)

#: Each condition's buffer. This is not cosmetic: the ``g0`` study case preloads
#: ``single_10ep_gradient.npz``, whose 280 stored rewards are *shaped*, so an
#: unshaped run left on the default still trains on a third shaped data and
#: appears to flip. Run 09 got this wrong once and it reversed the conclusion.
BUFFERS = {
    "shaped": None,  # the study case's own, shaped, buffer
    "unshaped": "buffers/single_10ep_standard.npz",
}
#: Fixed hue per condition, same assignment as ``assume_film.py`` so a reader
#: moving between the two figures keeps the same colour for the same run.
HUES = {"shaped": "#2a78d6", "unshaped": "#eb6834"}

#: ``argmax Q1`` at or below this counts as "the critic has left the ceiling".
#: 90 rather than something inside the band, because the question this sweep
#: answers is whether the critic ever develops *any* preference below +100 --
#: where it stops afterwards is run 09's question, not this one.
CEILING = 90.0


def shaping_active() -> bool:
    """Whether the reward shaping is uncommented in the working tree."""
    return SHAPING_LINE.search(SHAPING_SOURCE.read_text(encoding="utf-8")) is not None


def out_path(root: Path, condition: str, seed: int) -> Path:
    return root / f"assume_stab_{condition}_seed{seed}.npz"


# --------------------------------------------------------------------------- #
# running


def launch(condition: str, seed: int, args) -> tuple[str, int, int, float, Path]:
    """One probe process. Returns (condition, seed, returncode, seconds, log)."""
    tag = f"{condition}_seed{seed}"
    scratch = args.out_dir / "scratch" / tag
    scratch.mkdir(parents=True, exist_ok=True)

    # the probe refuses to start on a non-empty save folder, and rightly so --
    # run_learning would rmtree it. Clearing it here is what makes a rerun of a
    # single seed a one-liner.
    relative = Path("learned_strategies") / f"probe_stab_{tag}"
    shutil.rmtree(SCENARIO / relative, ignore_errors=True)

    cmd = [
        sys.executable, str(PROBE),
        "--study-case", args.study_case,
        "--episodes", str(args.episodes),
        "--n-obs", str(args.n_obs),
        "--grid", str(args.grid),
        "--seed", str(seed),
        "--threads", "1",
        "--label", condition,
        "--db-uri", f"sqlite:///{scratch / 'probe.db'}",
        "--save-path", str(relative),
        "--out", str(out_path(args.out_dir, condition, seed)),
    ]
    if BUFFERS[condition] is not None:
        cmd += ["--load-buffer", BUFFERS[condition]]

    log = scratch / "run.log"
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        # cwd is the scratch folder so run_learning's tensorboard/ and assume.log
        # land there rather than in whichever folder the sweep was started from,
        # and so concurrent runs do not share a tensorboard directory (it gets
        # rmtree'd at the start of every run, and the simulation_id is the same
        # for all of them)
        proc = subprocess.run(cmd, cwd=scratch, stdout=fh, stderr=subprocess.STDOUT)
    return condition, seed, proc.returncode, time.perf_counter() - t0, log


def train(condition: str, seeds: list[int], args) -> None:
    active = shaping_active()
    if active != (condition == "shaped"):
        state = "uncommented" if active else "commented out"
        raise SystemExit(
            f"--condition {condition} but the shaping at {SHAPING_SOURCE.name}"
            f":1583-1589 is {state}.\n"
            "The shaping is a source edit, not a config flag, so the condition has "
            "to be set by hand:\n"
            "  shaped   -> uncomment those seven lines\n"
            "  unshaped -> leave them commented (the repo's committed state)\n"
            "Whatever you do, leave them commented out when you are finished."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {condition}: {len(seeds)} seeds on {args.workers} workers, "
          f"{args.episodes} episodes each\n")

    t0 = time.perf_counter()
    failures = []
    # threads, not processes: each job is a blocking subprocess.run, so the pool
    # only has to wait on them
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(launch, condition, s, args) for s in seeds]
        for i, fut in enumerate(jobs, 1):
            cond, seed, code, secs, log = fut.result()
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"  [{i}/{len(seeds)}] {cond} seed {seed:<3} {secs / 60:5.1f} min  "
                  f"{status}", flush=True)
            if code != 0:
                failures.append((seed, log))

    print(f"\n  {(time.perf_counter() - t0) / 60:.0f} min wall")
    for seed, log in failures:
        print(f"  seed {seed} failed -- see {log}")


# --------------------------------------------------------------------------- #
# reading and summarising


def find_runs(out_dir: Path) -> list[Path]:
    """Fresh files in ``out_dir`` win; the archive fills in the rest.

    Same rule as ``_layout.resolve()``, applied to a glob rather than one name,
    so a single re-run seed shadows its archived copy without the caller having
    to say so.
    """
    found: dict[str, Path] = {}
    for path in sorted((RUNS / "data").glob("*/assume_stab_*.npz")):
        found[path.name] = path
    for path in sorted(out_dir.glob("assume_stab_*.npz")):
        found[path.name] = path
    return sorted(found.values(), key=lambda p: p.name)


def load(paths: list[Path]) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for path in paths:
        d = np.load(path)
        condition = str(d["label"])
        seed = int(d["seed"])
        if not condition:
            raise SystemExit(f"{path.name} carries no label; which condition is it?")
        runs.setdefault(condition, {})[seed] = {
            "steps": d["steps"],
            # (obs, frames) -- the bid the critic prefers, per probed observation
            "argmax": d["critic_bids"][d[f"critic_q/{ALGO}"].argmax(axis=2)],
            "actor": d[f"greedy/{ALGO}"],
            "path": path,
        }
    return {c: dict(sorted(runs[c].items())) for c in sorted(runs, reverse=True)}


def metrics(run: dict) -> dict[str, float]:
    """Per-seed summary of one film.

    The headline quantities are medians over the probed observations, matching
    ``assume_film.py`` -- the critic is swept from six real observations and the
    question is what it prefers *typically*, not at one arbitrary hour.

    ``argmax_range`` is there because that median turned out to be the wrong
    summary on its own. A critic that has actually learned a preference gives
    nearly the same ``argmax Q1`` whichever observation it is asked from; the
    unshaped runs give six answers spread over most of the bid axis, so their
    median moves a long way on very little. Read ``argmax_last`` and
    ``argmax_range`` together or not at all.
    """
    argmax = np.median(run["argmax"], axis=0)  # (frames,)
    actor = np.median(run["actor"], axis=0)
    below = np.flatnonzero(argmax <= CEILING)
    last = run["argmax"][:, -1]  # (obs,)
    return {
        "argmax_last": argmax[-1],
        "argmax_range": last.max() - last.min(),
        "argmax_min": argmax.min(),
        "left_ceiling": float(len(below) > 0),
        "t_flip": float(run["steps"][below[0]]) if len(below) else np.nan,
        "actor_last": actor[-1],
        "actor_reward": float(reward_from_bid(actor[-1], PAPER_SMALL)),
    }


def summarize(runs: dict[str, dict[int, dict]]) -> None:
    p = PAPER_SMALL
    print(
        f"\n  band {p.dec_threshold:.0f}-{p.eom_price:.0f} EUR/MWh, optimum "
        f"{p.optimal_bid:.0f} -> {p.optimal_reward:+.3f}   |   "
        f"'left ceiling' = median argmax Q1 reached <= {CEILING:.0f} at some frame\n"
    )
    print(f"  {'condition':<11}{'seeds':>6}{'argmax Q1 last':>23}"
          f"{'spread over obs':>17}{'actor last':>22}{'true reward':>17}"
          f"{'left ceiling':>14}")
    print("  " + "-" * 110)
    for condition, seeds in runs.items():
        m = {k: np.array([metrics(r)[k] for r in seeds.values()])
             for k in metrics(next(iter(seeds.values())))}
        n = len(seeds)
        left = f"{int(m['left_ceiling'].sum())}/{n}"
        print(
            f"  {condition:<11}{n:>6}"
            f"{m['argmax_last'].mean():>14.1f} +-{m['argmax_last'].std():<6.1f}"
            f"{m['argmax_range'].mean():>17.1f}"
            f"{m['actor_last'].mean():>13.1f} +-{m['actor_last'].std():<6.1f}"
            f"{m['actor_reward'].mean():>+9.3f} +-{m['actor_reward'].std():<5.3f}"
            f"{left:>14}"
        )

    print(f"\n  {'condition':<11}{'seed':>6}{'argmax Q1 last':>16}"
          f"{'spread over obs':>17}{'argmax Q1 min':>15}{'actor last':>12}"
          f"{'true reward':>13}{'t_flip':>9}")
    print("  " + "-" * 99)
    for condition, seeds in runs.items():
        for seed, run in seeds.items():
            m = metrics(run)
            flip = "never" if np.isnan(m["t_flip"]) else f"{m['t_flip']:.0f}"
            print(
                f"  {condition:<11}{seed:>6}{m['argmax_last']:>16.1f}"
                f"{m['argmax_range']:>17.1f}{m['argmax_min']:>15.1f}"
                f"{m['actor_last']:>12.1f}{m['actor_reward']:>+13.3f}{flip:>9}"
            )
    print()


# --------------------------------------------------------------------------- #
# figure


def plot(runs: dict[str, dict[int, dict]], out: Path) -> None:
    """Trajectories on top, end states below.

    Nothing is averaged and nothing is band-shaded across seeds: run 08's lesson
    is that the failure mode here is bimodal, and a mean of "48.8 and 100.0"
    describes no run that ever happened. Every line and every dot is one seed.
    """
    import matplotlib.pyplot as plt

    p = PAPER_SMALL
    conditions = list(runs)
    #: Shared by all four panels. The reachable bid range is +-100, but nothing
    #: in these runs goes below ~16, and the distinction the figure has to carry
    #: is 49 against 100 -- half an axis of empty plateau would cost the
    #: resolution that separates them.
    SPAN = (10, 105)

    fig = plt.figure(figsize=(13.2, 8.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.42, wspace=0.16)

    traj = []
    for col, (key, title, xlabel) in enumerate([
        ("argmax", "argmax Q1 — the bid the critic prefers",
         "critic gradient steps"),
        ("actor", "the actor's own bid", "critic gradient steps"),
    ]):
        ax = fig.add_subplot(grid[0, col])
        traj.append(ax)
        ax.axhspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.12, lw=0)
        ax.axhline(p.optimal_bid, ls="--", lw=1.1, color=INK, zorder=2)
        for condition in conditions:
            for i, run in enumerate(runs[condition].values()):
                ax.plot(
                    run["steps"], np.median(run[key], axis=0),
                    lw=1.5, color=HUES[condition], alpha=0.8, zorder=3,
                    label=f"{condition} ({len(runs[condition])} seeds)" if i == 0 else None,
                )
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylim(*SPAN)
        ax.set_xlim(0, max(r["steps"][-1] for s in runs.values() for r in s.values()))
        if col == 0:
            ax.set_ylabel("bid price (EUR/MWh)")
            ax.legend(frameon=False, fontsize=9, loc="lower left")
        else:
            ax.tick_params(labelleft=False)

    # --- end states, one dot per seed ---------------------------------------
    ends = []
    ys = np.arange(len(conditions))[::-1]
    rng = np.random.default_rng(0)
    for col, (key, summary, title) in enumerate([
        ("argmax", "argmax_last", "final argmax Q1 — dot per seed, faint per observation"),
        ("actor", "actor_last", "final actor bid — dot per seed, faint per observation"),
    ]):
        ax = fig.add_subplot(grid[1, col])
        ends.append(ax)
        ax.axvspan(p.dec_threshold, p.eom_price, color="#1baf7a", alpha=0.12, lw=0)
        ax.axvline(p.optimal_bid, ls="--", lw=1.1, color=INK, zorder=1)
        ax.axvline(p.max_bid_price, ls=":", lw=1.1, color=MUTED, zorder=1)
        for y, condition in zip(ys, conditions):
            # small dots first: every probed observation of every seed. Their
            # spread is the coherence of the critic, which the medians hide --
            # an unshaped run answers the same question with 30 and with 100
            # depending on which hour it is asked about
            obs = np.concatenate([r[key][:, -1] for r in runs[condition].values()])
            ax.plot(obs, y - 0.30 + rng.uniform(-0.07, 0.07, len(obs)), "o", ms=3.6,
                    color=HUES[condition], alpha=0.4, mec="none", zorder=2)
            # then the per-seed medians, which are what the table reports
            v = np.array([metrics(r)[summary] for r in runs[condition].values()])
            ax.plot(v, y + rng.uniform(-0.09, 0.09, len(v)), "o", ms=7.5,
                    color=HUES[condition], alpha=0.9, mec="white", mew=0.9, zorder=3)
            ax.plot([np.median(v)], [y], "|", ms=20, mew=2.2, color=INK, zorder=4)
            ax.annotate(
                f"{np.mean(v):.1f} ± {np.std(v):.1f}",
                xy=(0.5, y + 0.26), xycoords=("axes fraction", "data"),
                ha="center", fontsize=8.5, color=MUTED,
            )
        ax.set_yticks(ys, conditions, fontsize=10)
        ax.set_ylim(-0.75, len(conditions) - 0.3)
        ax.set_xlim(*SPAN)
        ax.set_xlabel("bid price (EUR/MWh)")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        if col == 0:
            ax.annotate("band", xy=(39.5, -0.66), ha="center", fontsize=8.5,
                        color="#137f59")
            ax.annotate("ceiling", xy=(104, -0.66), ha="right", fontsize=8.5,
                        color=MUTED)

    for ax in (*traj, *ends):
        ax.set_axisbelow(True)
        ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=9)

    counts = ", ".join(f"{c} {len(runs[c])}" for c in conditions)
    fig.suptitle(
        "Do ASSUME's two MATD3 outcomes reproduce across seeds?",
        x=0.005, y=0.985, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.005, 0.945,
        f"Seeds per condition: {counts}. ASSUME's own seed knob "
        "(loader_csv.py:555), re-applied after the scenario is loaded. Each run is "
        "a separate process at one torch thread; 40 episodes = 2560 critic updates.",
        fontsize=9, color=MUTED, ha="left",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=sorted(BUFFERS), default=None,
                        help="train this condition; omit and pass --report to read")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4, 5],
                        help="42 is ASSUME's own default, i.e. a rerun of run 09")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--n-obs", type=int, default=6)
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument("--study-case", default="inc_dec_learning_single_g0")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR / "stability",
                        help="where the per-run npz files are written and read")
    parser.add_argument("--report", action="store_true",
                        help="table and figure from whatever runs are on disk")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "assume_stability.png")
    parser.add_argument("--no-plot", action="store_true", help="table only")
    args = parser.parse_args()

    if args.condition:
        train(args.condition, args.seeds, args)
    if not (args.report or args.condition):
        raise SystemExit("nothing to do -- pass --condition to train or --report")

    paths = find_runs(args.out_dir)
    if not paths:
        raise SystemExit(f"no assume_stab_*.npz under {args.out_dir} or {RUNS / 'data'}")
    runs = load(paths)
    summarize(runs)
    if not args.no_plot:
        plot(runs, args.out)


if __name__ == "__main__":
    main()
