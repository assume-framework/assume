# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 12 -- does raising the action's share of the critic's input rescue ASSUME's
MATD3 on the **true** reward?

The offline finding this tests
------------------------------
A gamma = 0 supervised fit of ASSUME's own ``CriticTD3`` on the frozen
``single_10ep_standard`` buffer reproduces the run 09/10/11 failure with no
bootstrap, no moving target and no actor: with the real 74-dim observation the
critic ends with ``argmax Q1`` at 95, a *positive* slope inside the band and
``Q(32) < Q(100)``. Shrinking the observation, scaling the action up, or simply
using fewer observation dimensions all fix it, and the outcome depends on

    act_share = sd_a / (sd_a + sum_j sd_j)      -- 0.030 as ASSUME is configured

rather than on which side of the ratio was moved. This script asks whether that
transfers to the live learner, where the actor, the bootstrap and a growing
buffer are all back in play.

Two levers, both raising act_share, neither touching the reward
--------------------------------------------------------------
``foresight``      ``obs_dim = 3 * foresight + 2`` (``base.py:952``), so a
                   shorter forecast window removes observation dimensions.
                   Applied by forcing the strategy's ``foresight`` kwarg
                   (``learning_strategies.py:1338``).
``action_scale``   the critic's *input* action is multiplied by S, i.e. it fits
                   ``Q(s, S*a)``. The actor's output, the bid mapping and the
                   environment are untouched; only the critic's input geometry
                   changes. Adam is scale-invariant, so the actor's step size is
                   unchanged even though its gradient is S times larger.

Nothing in ``assume/`` is edited: both levers are monkeypatches installed in the
child process before ``assume_training_probe.main()`` loads the scenario.

Comparability
-------------
Every trial uses run 11's ``BASELINE`` config, the same 40 episodes, the same
3 seeds and the same starting data, so ``baseline`` here is run 11's ``baseline``
cell and should reproduce its 99.4 +- 0.1.

A reduced-foresight run cannot load the 74-dim buffer, so a truncated copy is
derived from it: forecast blocks keep their **first** k entries and the price
history keeps its **last** k, because ``create_observation`` builds the first two
with ``window(..., direction="forward")`` and the third with
``direction="backward"`` (``fast_pandas.py:864-867``) -- the history block runs
oldest to newest and ends at the current hour. The transitions are otherwise the
same, in the same order.

Usage
-----
    python real_matd3/assume_actshare_sweep.py --workers 5
    python real_matd3/assume_actshare_sweep.py --report-only
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, SCENARIO  # noqa: E402

PROBE = Path(__file__).resolve().parent / "assume_training_probe.py"
SELF = Path(__file__).resolve()
REPO = SCENARIO.parents[2]
SHAPING_SOURCE = REPO / "assume" / "strategies" / "learning_strategies.py"
#: the shaping is a source edit, not a config flag; this is its uncommented form
SHAPING_LIVE = re.compile(r"^\s{8}if reward > 0:", re.MULTILINE)

BUFFER_RELATIVE = "buffers/single_10ep_standard.npz"
BUFFER = SCENARIO / "learned_strategies" / BUFFER_RELATIVE
BUFFER_SHA256 = "5f1b80b4a2cb98c1c166c35e48931e87ae24f06e92f88c46c44f768954c16a72"

FULL_FORESIGHT = 24
UNIQUE_OBS_DIM = 2
MAX_BID_PRICE = 100.0
BAND = (30.0, 49.0)

#: run 11's baseline, asserted against the original in preflight()
BASELINE: dict[str, object] = {
    "learning_rate": 1e-3,
    "learning_rate_schedule": None,
    "min_learning_rate": 0.0,
    "gradient_steps": 10,
    "batch_size": 128,
    "policy_delay": 2,
    "gamma": 0.99,
    "tau": 0.005,
    "noise_sigma": 0.1,
    "noise_scale": 1,
    "noise_dt": 1,
    "action_noise_schedule": None,
    "target_policy_noise": 0.2,
    "target_noise_clip": 0.5,
    "episodes_collecting_initial_experience": 0,
    "load_replay_buffer": True,
    "save_replay_buffer": False,
    "early_stopping_steps": 1_000_000,
}

#: (foresight, action scale). act_share is filled in by act_share() below.
CONDITIONS: dict[str, dict[str, float]] = {
    "baseline": {"foresight": 24, "action_scale": 1.0},
    "foresight-6": {"foresight": 6, "action_scale": 1.0},
    "foresight-3": {"foresight": 3, "action_scale": 1.0},
    "act-x10": {"foresight": 24, "action_scale": 10.0},
    "act-x30": {"foresight": 24, "action_scale": 30.0},
}

SEEDS = [42, 1, 2]
EPISODES = 40


# --------------------------------------------------------------------- shared


def truncate_obs(obs: np.ndarray, k: int) -> np.ndarray:
    """The observation ASSUME builds at ``foresight = k``, from the 74-dim one."""
    if k == FULL_FORESIGHT:
        return obs
    f = FULL_FORESIGHT
    return np.concatenate(
        [
            obs[..., 0:k],  # residual-load forecast, forward
            obs[..., f: f + k],  # price forecast, forward
            obs[..., 3 * f - k: 3 * f],  # price history, backward -> keep the last k
            obs[..., 3 * f:],  # marginal cost, availability
        ],
        axis=-1,
    )


def truncated_buffer_path(k: int) -> Path:
    if k == FULL_FORESIGHT:
        return BUFFER
    return BUFFER.with_name(f"single_10ep_standard_f{k}.npz")


def build_truncated_buffer(k: int) -> Path:
    """Derive (once) the reduced-foresight copy of the shared starting buffer."""
    out = truncated_buffer_path(k)
    if k == FULL_FORESIGHT or out.exists():
        return out
    d = np.load(BUFFER)
    np.savez_compressed(
        out,
        observations=truncate_obs(d["observations"], k),
        actions=d["actions"],
        rewards=d["rewards"],
        pos=d["pos"],
        full=d["full"],
    )
    print(f"  derived {out.name}: obs_dim {3 * k + UNIQUE_OBS_DIM}")
    return out


def act_share(k: int, scale: float) -> float:
    """Action's share of total input std, the summary from the offline sweep."""
    d = np.load(BUFFER)
    n = int(d["pos"][0])
    obs = truncate_obs(d["observations"][:n, 0, :], k)
    sd_a = float(d["actions"][:n, 0, 0].std()) * scale
    return sd_a / (sd_a + float(obs.std(axis=0).sum()))


# ---------------------------------------------------------------------- child


def run_child(name: str, rest: list[str]) -> None:
    """Install the two patches, then hand over to the unmodified probe."""
    cond = CONDITIONS[name]
    k, scale = int(cond["foresight"]), float(cond["action_scale"])

    if scale != 1.0:
        from assume.reinforcement_learning.neural_network_architecture import CriticTD3

        original_forward = CriticTD3.forward
        original_q1 = CriticTD3.q1_forward

        # Q(s, S*a). The probe differentiates w.r.t. the *raw* action, so the
        # recorded gradient is the one the actor loss propagates -- but it is S
        # times larger than baseline's, so compare signs across conditions, never
        # magnitudes.
        def forward(self, obs, actions):
            return original_forward(self, obs, actions * scale)

        def q1_forward(self, obs, actions):
            return original_q1(self, obs, actions * scale)

        CriticTD3.forward = forward
        CriticTD3.q1_forward = q1_forward
        print(f"  patched CriticTD3: action input x{scale}")

    if k != FULL_FORESIGHT:
        from assume.strategies.learning_strategies import (
            EnergyLearningSingleBidRedispatchStrategy as Strategy,
        )

        original_init = Strategy.__init__

        def __init__(self, *args, **kwargs):
            kwargs["foresight"] = k
            original_init(self, *args, **kwargs)

        Strategy.__init__ = __init__

        import assume_training_probe as probe

        original_load = probe.load_observations
        probe.load_observations = lambda path, n: truncate_obs(original_load(path, n), k)
        print(f"  patched strategy: foresight {k}, obs_dim {3 * k + UNIQUE_OBS_DIM}")

    import assume_training_probe as probe

    sys.argv = ["assume_training_probe.py", *rest]
    probe.main()


# --------------------------------------------------------------------- parent


def result_path(out_dir: Path, name: str, seed: int) -> Path:
    return out_dir / f"assume_as_{name}_seed{seed}.npz"


def preflight() -> None:
    if not BUFFER.exists():
        raise SystemExit(f"no starting buffer at {BUFFER}")
    digest = hashlib.sha256(BUFFER.read_bytes()).hexdigest()
    if digest != BUFFER_SHA256:
        raise SystemExit(
            f"starting buffer changed:\n  expected {BUFFER_SHA256}\n  found    {digest}"
        )
    source = SHAPING_SOURCE.read_text(encoding="utf-8")
    if SHAPING_LIVE.search(source):
        raise SystemExit(
            "the reward shaping at learning_strategies.py:1583 is UNCOMMENTED. "
            "This run must be on the true reward; comment it back out first."
        )
    try:
        from assume_config_sweep import BASELINE as REFERENCE
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"  (could not cross-check run 11's BASELINE: {exc})")
        return
    if BASELINE != REFERENCE:
        raise SystemExit(
            "BASELINE has drifted from run 11's; the two sweeps would not be "
            f"comparable.\n  here:  {BASELINE}\n  run11: {REFERENCE}"
        )


#: training blocks per episode: the study case runs 24 h at train_freq 12h, and
#: the shared buffer means no collection episodes, so every episode leaves 2 frames
BLOCKS_PER_EPISODE = 2


def validate_result(path: Path, name: str, seed: int, episodes: int) -> None:
    """Refuse to treat a partial archive as a finished trial.

    ``assume_training_probe`` deliberately writes its film from ``finally``, so a
    run that died mid-way still leaves an ``.npz`` behind -- which is what makes a
    failure inspectable. The cost is that "the file exists" is not the same as
    "the trial finished", and this runner otherwise skips on existence alone.
    Run 11 has had this guard since its six TensorBoard failures; run 12 and 13
    did not, and the hazard grows with the number of parallel workers.
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


def launch(name: str, seed: int, args) -> tuple[str, int, int, float, Path]:
    out = result_path(args.out_dir, name, seed)
    if out.exists() and not args.rerun:
        validate_result(out, name, seed, args.episodes)
        return name, seed, 0, 0.0, out

    k = int(CONDITIONS[name]["foresight"])
    buffer_relative = f"buffers/{truncated_buffer_path(k).name}"

    tag = f"{name}_seed{seed}"
    scratch = args.out_dir / "scratch" / tag
    scratch.mkdir(parents=True, exist_ok=True)
    relative_save = Path("learned_strategies") / f"probe_as_{tag}"
    shutil.rmtree(SCENARIO / relative_save, ignore_errors=True)
    db = scratch / "probe.db"
    db.unlink(missing_ok=True)

    cmd = [
        sys.executable, str(SELF), "--child", name, "--",
        "--study-case", args.study_case,
        "--episodes", str(args.episodes),
        "--n-obs", str(args.n_obs),
        "--grid", str(args.grid),
        "--every", "1",
        "--seed", str(seed),
        "--threads", "1",
        "--disable-tensorboard",
        "--label", name,
        "--train-freq", "12h",
        "--load-buffer", buffer_relative,
        "--overrides-json", json.dumps(BASELINE, separators=(",", ":")),
        "--db-uri", f"sqlite:///{db}",
        "--save-path", str(relative_save),
        "--out", str(out),
    ]
    log = scratch / "run.log"
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=scratch, stdout=fh, stderr=subprocess.STDOUT)
    return name, seed, proc.returncode, time.perf_counter() - t0, out


def run(args) -> None:
    preflight()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k in sorted({int(c["foresight"]) for c in CONDITIONS.values()}):
        build_truncated_buffer(k)

    jobs = [(n, s) for n in args.conditions for s in args.seeds]
    print(f"\n  {len(jobs)} trials, {args.episodes} episodes, {args.workers} workers")
    print(f"  true reward (shaping commented out), starting buffer {BUFFER.name}\n")

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(launch, n, s, args) for n, s in jobs]
        for fut in concurrent.futures.as_completed(futures):
            name, seed, rc, secs, out = fut.result()
            done += 1
            if rc == 0 and out.exists():
                try:
                    validate_result(out, name, seed, args.episodes)
                    status = "ok"
                except RuntimeError as exc:
                    status = f"INCOMPLETE ({exc})"
            else:
                status = f"FAILED rc={rc}"
            print(f"  [{done}/{len(jobs)}] {name} seed {seed}: {status} "
                  f"({secs / 60:.1f} min)", flush=True)


# -------------------------------------------------------------------- reading


def report(args) -> None:
    from critic_coherence import argmax_disagreement
    from incdec_reward import PAPER_SMALL, reward_from_bid

    print(f"\nrun 12: live MATD3 on the TRUE reward, {args.episodes} episodes, "
          f"{len(args.seeds)} seeds")
    print("WARNING: every 'recon' column is reconstructed by applying the SURROGATE curve")
    print("    to the recorded bid -- it is NOT what the simulator paid. That curve")
    print("    matches the frozen buffer's stored rewards on 24.8 % of transitions")
    print("    (surrogate/incdec_reward.py, RUNS.md correction 15). Bids, critics and")
    print("    act_share are measured; the reward columns are a model of the reward.")
    print("band_neg = share of grid cells in [30, 49] with dQ1/d(bid) < 0 at the "
          "last frame (the surrogate's slope there is negative)")
    print("'disagree' = mean |argmax Q1 difference| over distinct pairs of the "
          "probed observations")
    print("'recon solved' = final mean RECONSTRUCTED reward over the probed "
          "observations >= +0.15\n")
    header = (f"{'condition':<13} {'act_share':>9} {'obs_dim':>7} {'final bid':>15} "
              f"{'recon reward':>13} {'recon best':>11} {'argmax Q1':>15} "
              f"{'disagree':>9} {'band_neg':>9} {'in band':>8} {'recon solved':>13}")
    print(header)
    print("-" * len(header))

    for name in args.conditions:
        k = int(CONDITIONS[name]["foresight"])
        scale = float(CONDITIONS[name]["action_scale"])
        rows = []
        for seed in args.seeds:
            path = result_path(args.out_dir, name, seed)
            if not path.exists():
                continue
            d = np.load(path, allow_pickle=False)
            greedy = d["greedy/MATD3"]        # (n_obs, frames)
            q1 = d["critic_q/MATD3"]          # (n_obs, frames, grid)
            grad = d["critic_grad/MATD3"]
            bids = d["critic_bids"]
            band = (bids >= BAND[0]) & (bids <= BAND[1])

            final_bid = float(np.median(greedy[:, -1]))
            rew = np.vectorize(lambda b: reward_from_bid(b, PAPER_SMALL))(greedy)
            argmax = bids[np.argmax(q1[:, -1, :], axis=1)]
            rows.append((
                final_bid,
                float(np.mean(rew[:, -1])),
                float(np.max(rew)),
                float(np.median(argmax)),
                float(argmax_disagreement(argmax)),
                float(np.mean((grad[:, -1, :][:, band] < 0))),
                float(np.mean([BAND[0] <= b <= BAND[1] for b in greedy[:, -1]])),
            ))
        if not rows:
            print(f"{name:<13} {'(no results)':>60}")
            continue
        a = np.array(rows)
        solved = int(np.sum(a[:, 1] >= 0.15))
        print(f"{name:<13} {act_share(k, scale):9.3f} {3 * k + UNIQUE_OBS_DIM:7d} "
              f"{a[:,0].mean():8.1f} +-{a[:,0].std():4.1f} "
              f"{a[:,1].mean():+13.3f} {a[:,2].mean():+11.3f} "
              f"{a[:,3].mean():8.1f} +-{a[:,3].std():4.1f} {a[:,4].mean():7.1f} "
              f"{a[:,5].mean():9.2f} {a[:,6].mean():8.2f} "
              f"{f'{solved}/{len(rows)}':>13}")

    print("\ncaveats: gradient magnitudes are not comparable across action-scale "
          "conditions (the recorded dQ/d(bid) carries the factor S); reduced-"
          "foresight runs also see a smaller observation, which in this scenario "
          "carries no reward information but in general would.")
    print("\nfor MEASURED reward, read each trial's own rl_params table under "
          "scratch/<condition>_seed<seed>/probe.db -- but note it holds only the "
          "first two products of each episode (RUNS.md correction 16), so it is an "
          "early-hours sample. RUNS.md section 12 tabulates both.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--study-case", default="inc_dec_learning_single_g0")
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--n-obs", type=int, default=6)
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path,
        default=OUT_DIR / "runs" / "data" / "12-actshare",
    )

    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        name = sys.argv[i + 1]
        rest = sys.argv[sys.argv.index("--", i) + 1:]
        run_child(name, rest)
        return

    args = parser.parse_args()
    unknown = set(args.conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    if not args.report_only:
        run(args)
    report(args)


if __name__ == "__main__":
    main()
