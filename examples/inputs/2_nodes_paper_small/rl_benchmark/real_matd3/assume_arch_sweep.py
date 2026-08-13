# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 18 -- the offline architecture screen and a hyperparameter grid, run **live**.

Two rounds, one runner, because they share everything except what varies:
run 11's ``BASELINE`` configuration, the frozen starting buffer, 40 episodes,
three seeds, and the critic film the report is read from. Only the critic
network changes in ``--round arch``; only the optimizer settings change in
``--round hpo``.

Round ``arch`` -- does run 17's offline result survive the live loop?
--------------------------------------------------------------------
The offline ``gamma = 0`` screen (``assume_offline_critic.py --round arch``)
found two things on a frozen buffer with no bootstrap, no moving target and no
actor:

* **RSNorm is disqualifying.** Six variants carrying it sat at a mean
  ``in_band`` of 0.005 with ``argmax Q1`` pinned at exactly 100.0 at *every*
  width from 143 k to 8.5 M. ``simba`` against ``simba-nornorm`` is the clean
  single-variable contrast -- identical networks, 8,483,842 parameters each --
  and is the whole difference between 100.0 and 34.1. None of them are here.
* **SimBa's residual trunk is worth about 15x the parameters.** The plain MLP
  was flat at ``in_band`` 0.10 from 105 k to 548 k and only broke through at
  8.5 M; the trunk was already at 0.27 at 143 k. Read as parameter efficiency
  at matched *outcome* -- 548 k against 8.48 M for a better argmax -- not as a
  gap at matched parameters, which understates it roughly 15-fold.

A screen is not a verdict. Offline there is no bootstrap, the buffer never
grows, and no actor moves the action distribution the critic is fitted on --
all three are back here. This round is the check.

What it runs, and why the grid is shaped that way::

    baseline                    ASSUME's CriticTD3 -- the control, and run 11's
                                cell, so it must reproduce 99.4 +- 0.1
    late                        action injected at hidden layer 2 (Lillicrap 2016)
    split                       obs and action get their own equal-width encoder
    mlp-d{2,4}-{100k,500k,2M,8M}    plain MLP, both scaling axes
    sbn-d{2,4}-{100k,500k,2M,8M}    SimBa trunk (no RSNorm), both scaling axes

**Both scaling axes.** Run 17 moved width only, at fixed depth, so "capacity"
and "width" were the same variable and neither could be blamed. Here each
family is run at two depths up the same four-rung parameter ladder, so depth
and width are separable: if the ladder's shape is the same at depth 2 and depth
4, capacity is what matters; if the deeper column is better at equal
parameters, depth is. See ``critic_architectures.sized`` -- and note the two
families' depth units differ (one SimBa block is two ``Linear`` layers), so
each family's curve is read against itself.

``split`` is the new cell. Late injection gives the action its own weight
matrix but the action still arrives raw and outnumbered; ``split`` gives it its
own hidden layer at the same width as the observation's, so from layer 2 on
neither can dominate by sheer count. That separates **equal count** from
**equal scale** -- and equal scale is what ``act_share`` moved in run 12, which
this workstream exists to replace with something from the literature.

Round ``hpo`` -- 20 cells of a coordinate sweep
-----------------------------------------------
Defined in ``hpo_grid.py`` and shared with the EOM study, so ``p1`` and inc-dec
sweep the same cells and the two tables can be laid side by side. Weight decay
is in there and is not a ``LearningConfig`` field: ``matd3.py`` constructs
``AdamW(params, lr=...)`` and nothing else, so every archived run trained at
torch's default 0.01 rather than at none. See ``optim_patches.py``.

Comparability
-------------
Every trial takes run 11's ``BASELINE`` overrides, the same 40 episodes, the
same three seeds and the same frozen starting buffer, so ``baseline`` /
``default`` here is run 11's ``baseline`` cell and should reproduce its
99.4 +- 0.1. Run 12's ``act-x30`` is the other reference point, and it is
deliberately *not* re-run: it is the invented lever this workstream replaces.

Usage::

    python real_matd3/assume_arch_sweep.py --round arch --workers 4
    python real_matd3/assume_arch_sweep.py --round hpo --cells lr
    python real_matd3/assume_arch_sweep.py --report-only --round arch hpo
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, SCENARIO  # noqa: E402  (also sets sys.path)
from assume_actshare_sweep import (  # noqa: E402
    BAND,
    BASELINE,
    BLOCKS_PER_EPISODE,
    BUFFER,
    BUFFER_RELATIVE,
    EPISODES,
    SEEDS,
    preflight,
)
from critic_architectures import (  # noqa: E402
    build as build_critic,
    describe as describe_critic,
    ladder_names,
    param_count,
)
from hpo_grid import CELLS, describe as describe_cell, resolve as resolve_cells  # noqa: E402

SELF = Path(__file__).resolve()

#: the architectures the live sweep runs, in report order. The six RSNorm
#: carriers and ``bn+late`` from run 17 are deliberately absent -- see the
#: module docstring; ``late`` stays because it is the one literature cell the
#: offline screen could not settle either way.
ARCHS: list[str] = ["baseline", "late", "split", *ladder_names()]

#: the observation width of ``inc_dec_learning_single`` at foresight 24, which
#: is also what the ladder's widths were bisected at, so the printed parameter
#: counts here are the ones ``critic_architectures`` resolved
SHAPE = dict(obs_dim=74, act_dim=1, n_agents=1, unique_obs_dim=2)


# ---------------------------------------------------------------------- child


def run_child(round_: str, name: str, rest: list[str]) -> None:
    """Install this trial's one variation, then hand over to the probe."""
    if round_ == "arch":
        if name != "baseline":
            import assume.reinforcement_learning.algorithms.matd3 as matd3

            matd3.CriticTD3 = build_critic(name)
            print(f"  critic architecture: {name} -- {describe_critic(name)}")
    else:
        from optim_patches import TORCH_DEFAULT_WEIGHT_DECAY, install_weight_decay

        wd = CELLS[name]["weight_decay"]
        # only patch when the cell actually differs. Passing 0.01 explicitly
        # would be numerically identical, but leaving AdamW untouched keeps
        # hpo/default bit-identical to arch/baseline and to run 11's cell --
        # and those three agreeing is this sweep's own consistency check.
        if wd != TORCH_DEFAULT_WEIGHT_DECAY:
            install_weight_decay(wd)
        else:
            print(f"  weight_decay {wd} is torch's AdamW default -- not patched")

    import assume_training_probe as probe

    sys.argv = ["assume_training_probe.py", *rest]
    probe.main()


# --------------------------------------------------------------------- parent


def result_path(out_dir: Path, round_: str, name: str, seed: int) -> Path:
    # the round is in the file name: 'default' and 'baseline' are the same
    # configuration run for two different questions, and collapsing them onto
    # one path would silently make each round's control the other round's
    return out_dir / f"assume_{round_}_{name}_seed{seed}.npz"


def validate_result(path: Path, name: str, seed: int, episodes: int) -> None:
    """Refuse to treat a partial archive as a finished trial.

    Same guard as run 12's, and for the same reason: the probe writes its film
    from ``finally``, so a crashed run still leaves an inspectable ``.npz`` and
    "the file exists" is not "the trial finished".
    """
    d = np.load(path, allow_pickle=False)
    missing = {"steps", "critic_bids", "critic_q/MATD3", "critic_grad/MATD3",
               "greedy/MATD3"} - set(d.files)
    if missing:
        raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
    if int(d["seed"]) != seed or str(d["label"]) != name:
        raise RuntimeError(f"{path.name} carries the wrong seed or label")
    expected = episodes * BLOCKS_PER_EPISODE
    if len(d["steps"]) != expected:
        raise RuntimeError(
            f"{path.name} has {len(d['steps'])} frames, expected {expected} "
            f"({episodes} episodes x {BLOCKS_PER_EPISODE} blocks) -- partial run"
        )


def launch(round_: str, name: str, seed: int, args) -> tuple:
    out = result_path(args.out_dir, round_, name, seed)
    if out.exists() and not args.rerun:
        validate_result(out, name, seed, args.episodes)
        return round_, name, seed, 0, 0.0, out

    tag = f"{round_}_{name}_seed{seed}"
    scratch = args.out_dir / "scratch" / tag
    scratch.mkdir(parents=True, exist_ok=True)
    relative_save = Path("learned_strategies") / f"probe_arch_{tag}"
    shutil.rmtree(SCENARIO / relative_save, ignore_errors=True)
    db = scratch / "probe.db"
    db.unlink(missing_ok=True)

    overrides = dict(BASELINE)
    if round_ == "hpo":
        overrides.update(CELLS[name]["overrides"])

    cmd = [
        sys.executable, str(SELF), "--child", round_, name, "--",
        "--study-case", args.study_case,
        "--episodes", str(args.episodes),
        "--n-obs", str(args.n_obs),
        "--grid", str(args.grid),
        "--every", "1",
        "--seed", str(seed),
        "--threads", str(args.threads),
        "--disable-tensorboard",
        "--label", name,
        "--train-freq", "12h",
        "--load-buffer", BUFFER_RELATIVE,
        "--overrides-json", json.dumps(overrides, separators=(",", ":")),
        "--db-uri", f"sqlite:///{db}",
        "--save-path", str(relative_save),
        "--out", str(out),
    ]
    log = scratch / "run.log"
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=scratch, stdout=fh, stderr=subprocess.STDOUT)
    return round_, name, seed, proc.returncode, time.perf_counter() - t0, out


def cells_for(round_: str, args) -> list[str]:
    if round_ == "arch":
        return args.archs
    return resolve_cells(args.cells)


def run(args) -> None:
    preflight()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(r, n, s) for r in args.round
            for n in cells_for(r, args) for s in args.seeds]
    print(f"\n  {len(jobs)} trials, {args.episodes} episodes, "
          f"{args.workers} workers")
    print(f"  true reward (shaping commented out), starting buffer {BUFFER.name}")
    for r in args.round:
        names = cells_for(r, args)
        print(f"  round {r}: {len(names)} cells -- {', '.join(names)}")
    print(flush=True)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, r, n, s, args) for r, n, s in jobs]
        for fut in concurrent.futures.as_completed(futures):
            round_, name, seed, rc, secs, out = fut.result()
            if rc == 0 and out.exists():
                try:
                    validate_result(out, name, seed, args.episodes)
                    status = "ok"
                except RuntimeError as exc:
                    status = f"INCOMPLETE ({exc})"
            else:
                status = f"FAILED rc={rc}"
            done += 1
            print(f"  [{done}/{len(jobs)}] {round_}/{name} seed {seed}: "
                  f"{status} ({secs / 60:.1f} min)", flush=True)


# -------------------------------------------------------------------- reading


def read_trial(path: Path) -> dict | None:
    """The four numbers a row is built from, or ``None`` if the film is absent."""
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    greedy = d["greedy/MATD3"]            # (n_obs, frames)
    q1 = d["critic_q/MATD3"]              # (n_obs, frames, grid)
    grad = d["critic_grad/MATD3"]
    bids = d["critic_bids"]
    band = (bids >= BAND[0]) & (bids <= BAND[1])
    argmax = bids[np.argmax(q1[:, -1, :], axis=1)]
    return {
        "bid": float(np.median(greedy[:, -1])),
        "argmax": float(np.median(argmax)),
        # the offline screen's headline metric, measured the same way: share of
        # probed observations whose greedy bid ends inside the reward band
        "in_band": float(np.mean([BAND[0] <= b <= BAND[1] for b in greedy[:, -1]])),
        "band_neg": float(np.mean(grad[:, -1, :][:, band] < 0)),
    }


def report(args) -> None:
    for round_ in args.round:
        names = cells_for(round_, args)
        title = ("run 18a -- critic architecture, live"
                 if round_ == "arch" else
                 "run 18b -- hyperparameter grid, live")
        print(f"\n{title}  ({args.episodes} episodes, {len(args.seeds)} seeds)")
        print("in_band and band_neg are run 17's metrics, measured the same way, so "
              "the offline\nand the live screen can be read against each other. "
              "band_neg 0.50 is a coin flip.")
        if round_ == "arch":
            print("run 11's baseline cell was a final bid of 99.4 +- 0.1; run 12's "
                  "act-x30 is the\nother reference point and is not re-run here.\n")
            first = f"{'architecture':<16}{'params':>11}"
        else:
            print("'default' is run 11's BASELINE unchanged, so bs128 and pd2 are "
                  "re-runs of it\nunder another name -- disagreement between those "
                  "three is a bug, not a result.\n")
            first = f"{'cell':<16}{'':>11}"
        header = (first + f"{'final bid':>16}{'argmax Q1':>16}"
                  f"{'in_band':>9}{'band_neg':>10}  what it changes")
        print(header)
        print("-" * len(header))

        for name in names:
            rows = [r for r in (read_trial(result_path(args.out_dir, round_, name, s))
                                for s in args.seeds) if r is not None]
            if round_ == "arch":
                try:
                    size = f"{param_count(name, **SHAPE):>11,}"
                except Exception:
                    size = f"{'?':>11}"
                what = describe_critic(name)
            else:
                size = " " * 11
                what = describe_cell(name)
            if not rows:
                print(f"{name:<16}{size}{'(no results)':>16}")
                continue
            g = lambda k: np.array([r[k] for r in rows])  # noqa: E731
            print(f"{name:<16}{size}"
                  f"{g('bid').mean():>11.1f} +-{g('bid').std():>3.1f}"
                  f"{g('argmax').mean():>11.1f} +-{g('argmax').std():>3.1f}"
                  f"{g('in_band').mean():>9.2f}{g('band_neg').mean():>10.2f}"
                  f"  {what}")

    print("\nthe measured reward lives in each trial's own rl_params table under "
          "scratch/<tag>/probe.db, and holds only the first two products of each "
          "episode (RUNS.md correction 16).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--child", nargs=2, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--round", nargs="+", default=["arch"],
                        choices=["arch", "hpo"])
    parser.add_argument("--archs", nargs="+", default=ARCHS,
                        help="which critic architectures the 'arch' round runs")
    parser.add_argument("--cells", nargs="+", default=["all"],
                        help="which hyperparameter cells the 'hpo' round runs; "
                             "a cell name, an axis group (centre, lr, batch, "
                             "delay, wd) or 'all'. See hpo_grid.py")
    parser.add_argument("--study-case", default="inc_dec_learning_single_g0")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1,
                        help="torch threads per child. Run 08 found the BLAS "
                             "thread count alone moving a surrogate seed from "
                             "+31.60 to -60.49, so this stays at 1")
    parser.add_argument("--n-obs", type=int, default=6)
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--out-dir", type=Path,
                        default=OUT_DIR / "runs" / "data" / "18-live-arch")

    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        round_, name = sys.argv[i + 1], sys.argv[i + 2]
        run_child(round_, name, sys.argv[sys.argv.index("--", i) + 1:])
        return

    args = parser.parse_args()
    if not args.report_only:
        run(args)
    report(args)


if __name__ == "__main__":
    main()
