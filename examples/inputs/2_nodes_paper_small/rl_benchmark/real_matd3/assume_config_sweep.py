# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Staged stability sweep for ASSUME's real, unshaped MATD3.

Every child loads the same immutable ``single_10ep_standard.npz`` and then owns
its in-memory replay buffer. The runner refuses to start if reward shaping is
active or if the shared buffer's checksum differs from the reviewed input.

Examples::

    python real_matd3/assume_config_sweep.py --phase canary
    python real_matd3/assume_config_sweep.py --phase broad
    python real_matd3/assume_config_sweep.py --phase broad --report-only
    python real_matd3/assume_config_sweep.py --phase confirm \
        --configs baseline lr-1e-4 policy-delay-1 gamma-0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import RUNS, SCENARIO  # noqa: E402
from incdec_reward import PAPER_SMALL, reward_from_bid  # noqa: E402
from run_benchmark import INK, MUTED  # noqa: E402

PROBE = Path(__file__).resolve().parent / "assume_training_probe.py"
REPO = SCENARIO.parents[2]
SHAPING_SOURCE = REPO / "assume" / "strategies" / "learning_strategies.py"
SHAPING_LINE = re.compile(
    r"^\s*reward = \(price - marginal_cost\) / 100\s*$", re.MULTILINE
)

BUFFER_RELATIVE = "buffers/single_10ep_standard.npz"
BUFFER = SCENARIO / "learned_strategies" / BUFFER_RELATIVE
BUFFER_SHA256 = "5f1b80b4a2cb98c1c166c35e48931e87ae24f06e92f88c46c44f768954c16a72"

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
    # A sweep must use the requested horizon, not stop configurations at
    # different points because a short derived early-stopping window was built
    # from the study case's original 10 episodes.
    "early_stopping_steps": 1_000_000,
}

# One-at-a-time changes first, then explicitly named hypothesis combinations.
CONFIGS: dict[str, dict[str, object]] = {
    "baseline": {},
    "lr-3e-4": {"learning_rate": 3e-4},
    "lr-1e-4": {"learning_rate": 1e-4},
    "lr-3e-5": {"learning_rate": 3e-5},
    "batch-64": {"batch_size": 64},
    "batch-256": {"batch_size": 256},
    "batch-512": {"batch_size": 512},
    "policy-delay-1": {"policy_delay": 1},
    "policy-delay-4": {"policy_delay": 4},
    "policy-delay-8": {"policy_delay": 8},
    "grad-4": {"gradient_steps": 4},
    "grad-20": {"gradient_steps": 20},
    "grad-32": {"gradient_steps": 32},
    "gamma-0": {"gamma": 0.0},
    "gamma-0.90": {"gamma": 0.90},
    "gamma-0.95": {"gamma": 0.95},
    "tau-0.001": {"tau": 0.001},
    "tau-0.01": {"tau": 0.01},
    "target-noise-0": {"target_policy_noise": 0.0, "target_noise_clip": 0.0},
    "target-noise-0.1-clip-0.2": {
        "target_policy_noise": 0.1,
        "target_noise_clip": 0.2,
    },
    "sigma-0": {"noise_sigma": 0.0},
    "sigma-0.05": {"noise_sigma": 0.05},
    "sigma-0.2": {"noise_sigma": 0.2},
    "noise-linear": {"action_noise_schedule": "linear"},
    "lr-1e-4-delay-1": {"learning_rate": 1e-4, "policy_delay": 1},
    "lr-1e-4-grad-32": {"learning_rate": 1e-4, "gradient_steps": 32},
    "gamma-0-target-noise-0": {
        "gamma": 0.0,
        "target_policy_noise": 0.0,
        "target_noise_clip": 0.0,
    },
    "gamma-0-lr-3e-4": {"gamma": 0.0, "learning_rate": 3e-4},
    "lr-3e-4-delay-1": {"learning_rate": 3e-4, "policy_delay": 1},
    "lr-1e-4-batch-256": {"learning_rate": 1e-4, "batch_size": 256},
}

PHASE_DEFAULTS = {
    "canary": {"episodes": 4, "seeds": [42], "configs": ["baseline"]},
    "broad": {"episodes": 40, "seeds": [42, 1, 2], "configs": list(CONFIGS)},
    "confirm": {
        "episodes": 128,
        "seeds": [42, 1, 2, 3, 4, 5, 6, 7],
        "configs": ["baseline", "lr-1e-4", "policy-delay-1", "gamma-0"],
    },
}

SOLVED_REWARD = 0.15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolved_config(name: str) -> dict[str, object]:
    return {**BASELINE, **CONFIGS[name]}


def default_workers() -> int:
    # The machine has 14 physical / 20 logical CPUs. Previous investigation
    # found 16 workers beyond the throughput knee. Ten one-thread children leave
    # four physical cores and ten logical CPUs available to the system.
    logical = os.cpu_count() or 4
    return max(1, min(10, logical - 2))


def preflight() -> None:
    if not BUFFER.exists():
        raise SystemExit(f"shared replay buffer does not exist: {BUFFER}")
    actual = sha256(BUFFER)
    if actual != BUFFER_SHA256:
        raise SystemExit(
            f"shared replay buffer checksum changed:\n  expected {BUFFER_SHA256}"
            f"\n  actual   {actual}\nRefusing to mix starting points."
        )
    source = SHAPING_SOURCE.read_text(encoding="utf-8")
    if SHAPING_LINE.search(source):
        raise SystemExit(
            f"reward shaping is active in {SHAPING_SOURCE}; this sweep is "
            "unshaped-only and refuses to start"
        )


def result_path(root: Path, name: str, seed: int) -> Path:
    return root / f"assume_cfg_{name}_seed{seed}.npz"


def validate_result(path: Path, name: str, seed: int) -> None:
    data = np.load(path, allow_pickle=False)
    required = {
        "steps",
        "critic_bids",
        "critic_q/MATD3",
        "critic_grad/MATD3",
        "critic_q2/MATD3",
        "critic_grad2/MATD3",
        "greedy/MATD3",
        "config_json",
        "buffer_sha256",
    }
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
    if int(data["seed"]) != seed or str(data["label"]) != name:
        raise RuntimeError(f"{path.name} carries the wrong seed or label")
    if str(data["buffer_sha256"]) != BUFFER_SHA256:
        raise RuntimeError(f"{path.name} did not use the reviewed shared buffer")
    q1 = data["critic_q/MATD3"]
    q2 = data["critic_q2/MATD3"]
    g1 = data["critic_grad/MATD3"]
    greedy = data["greedy/MATD3"]
    if q1.shape != q2.shape or q1.shape != g1.shape:
        raise RuntimeError(f"{path.name} has inconsistent critic array shapes")
    if q1.shape[:2] != greedy.shape:
        raise RuntimeError(f"{path.name} has misaligned critic and actor films")


def launch(name: str, seed: int, args) -> tuple[str, int, int, float, Path, Path]:
    out = result_path(args.out_dir, name, seed)
    if out.exists() and not args.rerun:
        validate_result(out, name, seed)
        return name, seed, 0, 0.0, Path(), out

    tag = f"{name}_seed{seed}"
    scratch = args.out_dir / "scratch" / tag
    scratch.mkdir(parents=True, exist_ok=True)
    relative_save = Path("learned_strategies") / f"probe_cfg_{args.phase}_{tag}"
    shutil.rmtree(SCENARIO / relative_save, ignore_errors=True)
    db = scratch / "probe.db"
    if db.exists():
        db.unlink()

    cmd = [
        sys.executable,
        str(PROBE),
        "--study-case",
        args.study_case,
        "--episodes",
        str(args.episodes),
        "--n-obs",
        str(args.n_obs),
        "--grid",
        str(args.grid),
        "--every",
        str(args.every),
        "--seed",
        str(seed),
        "--threads",
        "1",
        "--disable-tensorboard",
        "--label",
        name,
        "--train-freq",
        "12h",
        "--load-buffer",
        BUFFER_RELATIVE,
        "--overrides-json",
        json.dumps(resolved_config(name), separators=(",", ":")),
        "--db-uri",
        f"sqlite:///{db}",
        "--save-path",
        str(relative_save),
        "--out",
        str(out),
    ]
    log = scratch / "run.log"
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(cmd, cwd=scratch, stdout=fh, stderr=subprocess.STDOUT)
    seconds = time.perf_counter() - t0
    if proc.returncode == 0:
        try:
            validate_result(out, name, seed)
        except Exception as exc:
            with log.open("a", encoding="utf-8") as fh:
                print(f"\nresult validation failed: {exc}", file=fh)
            return name, seed, 90, seconds, log, out
    return name, seed, proc.returncode, seconds, log, out


def run(args) -> None:
    preflight()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": args.phase,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "configs": {name: resolved_config(name) for name in args.configs},
        "buffer": str(BUFFER),
        "buffer_sha256": BUFFER_SHA256,
        "grid": args.grid,
        "n_obs": args.n_obs,
        "snapshot_every_blocks": args.every,
        "workers": args.workers,
        "python": sys.executable,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    tasks = [(name, seed) for name in args.configs for seed in args.seeds]
    print(
        f"\n  {args.phase}: {len(args.configs)} configs x {len(args.seeds)} seeds "
        f"= {len(tasks)} runs on {args.workers} one-thread workers\n"
        f"  episodes {args.episodes}, critic grid {args.grid}, observations {args.n_obs}\n"
        f"  shared buffer {BUFFER.name} ({BUFFER_SHA256[:12]}...)\n",
        flush=True,
    )
    failures: list[tuple[str, int, Path]] = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(launch, n, s, args): (n, s) for n, s in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            name, seed, code, seconds, log, _ = future.result()
            cached = seconds == 0.0
            status = "cached" if cached else ("ok" if code == 0 else f"FAILED {code}")
            print(
                f"  [{done:>3}/{len(tasks)}] {name:<28} seed {seed:<3} "
                f"{seconds / 60:6.1f} min  {status}",
                flush=True,
            )
            if code:
                failures.append((name, seed, log))
    if sha256(BUFFER) != BUFFER_SHA256:
        raise RuntimeError("shared buffer changed during the sweep")
    print(f"\n  wall time {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if failures:
        for name, seed, log in failures:
            print(f"  FAILED {name} seed {seed}: {log}")
        raise SystemExit(f"{len(failures)} run(s) failed")


def load_runs(root: Path, names: list[str]) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    runs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    paths = [
        path
        for name in names
        for path in sorted(root.glob(f"assume_cfg_{name}_seed*.npz"))
    ]
    # Recorder.save() deliberately writes a diagnostic film from ``finally``
    # even when training fails. That makes a failure inspectable, but such a
    # two-frame partial file must never be mistaken for a completed seed in the
    # stability plot. All successful trials in one phase have the same number
    # of training blocks, so the cohort maximum is the completed length.
    frame_counts = {
        path: len(np.load(path, allow_pickle=False)["steps"]) for path in paths
    }
    complete_frames = max(frame_counts.values(), default=0)
    for name in names:
        for path in sorted(root.glob(f"assume_cfg_{name}_seed*.npz")):
            if frame_counts[path] < complete_frames:
                print(
                    f"  exclude partial {path.name}: {frame_counts[path]}/"
                    f"{complete_frames} snapshots"
                )
                continue
            data = np.load(path, allow_pickle=False)
            seed = int(data["seed"])
            runs.setdefault(name, {})[seed] = {
                "steps": data["steps"],
                "greedy": data["greedy/MATD3"],
                "critic_bids": data["critic_bids"],
                "grad": data["critic_grad/MATD3"],
                "argmax": data["critic_bids"][
                    data["critic_q/MATD3"].argmax(axis=2)
                ],
            }
    return runs


def run_metrics(run_data: dict[str, np.ndarray]) -> dict[str, float]:
    from critic_coherence import argmax_disagreement, argmax_range

    greedy = run_data["greedy"]
    rewards = reward_from_bid(greedy, PAPER_SMALL)
    tail = slice(max(0, int(0.75 * greedy.shape[1])), None)
    final_rewards = rewards[:, -1]
    final_argmax = run_data["argmax"][:, -1]
    return {
        "final_reward": float(final_rewards.mean()),
        "positive_obs": float((final_rewards > 0).mean()),
        "tail_reward": float(rewards[:, tail].mean()),
        "final_bid": float(np.median(greedy[:, -1])),
        # shared with runs 10, 12 and 13 -- see analysis/critic_coherence.py.
        # This column used to be the range alone, which is roughly twice the
        # disagreement and was once compared against it across runs.
        "argmax_disagreement": float(argmax_disagreement(final_argmax)),
        "argmax_range": float(argmax_range(final_argmax)),
    }


def summarize(runs: dict[str, dict[int, dict[str, np.ndarray]]]) -> None:
    print(
        f"\n  recon solved = mean RECONSTRUCTED reward over fixed observations "
        f">= {SOLVED_REWARD:.2f}"
    )
    print(
        "  WARNING: the reward columns apply the SURROGATE curve to the recorded bid; it\n"
        "      agrees with the simulator's stored rewards on 24.8 % of transitions.\n"
        "      See surrogate/incdec_reward.py and RUNS.md correction 15.\n"
    )
    print(
        f"  {'config':<29}{'seeds':>6}{'recon solved':>14}{'recon reward':>22}"
        f"{'recon tail':>16}{'recon pos obs':>15}{'disagree':>11}{'range':>8}"
    )
    print("  " + "-" * 123)
    for name, seeds in runs.items():
        values = [run_metrics(run_data) for run_data in seeds.values()]
        if not values:
            continue
        final = np.array([v["final_reward"] for v in values])
        tail = np.array([v["tail_reward"] for v in values])
        positive = np.array([v["positive_obs"] for v in values])
        disagree = np.array([v["argmax_disagreement"] for v in values])
        spread = np.array([v["argmax_range"] for v in values])
        solved = int((final >= SOLVED_REWARD).sum())
        print(
            f"  {name:<29}{len(values):>6}{f'{solved}/{len(values)}':>14}"
            f"{final.mean():>+12.3f} +- {final.std():<6.3f}"
            f"{tail.mean():>+11.3f}"
            f"{positive.mean():>15.1%}{disagree.mean():>11.1f}{spread.mean():>8.1f}"
        )
    print()


def plot(runs: dict[str, dict[int, dict[str, np.ndarray]]], out: Path, phase: str) -> None:
    import matplotlib.pyplot as plt

    names = [name for name in CONFIGS if name in runs and runs[name]]
    if not names:
        return
    rows = (len(names) + 7) // 8
    fig = plt.figure(figsize=(16, 4.4 + 2.5 * rows))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.45, rows], hspace=0.34)
    facets = outer[1].subgridspec(rows, min(8, len(names)), wspace=0.12, hspace=0.55)
    blue = "#2a78d6"

    ax = fig.add_subplot(outer[0])
    ax.axvline(PAPER_SMALL.optimal_reward, ls="--", lw=1.2, color=INK)
    ax.axvline(0, ls=":", lw=1.2, color=MUTED)
    ax.axvspan(SOLVED_REWARD, PAPER_SMALL.optimal_reward, color="#1baf7a", alpha=0.1)
    rng = np.random.default_rng(0)
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names):
        values = np.array([run_metrics(r)["final_reward"] for r in runs[name].values()])
        ax.plot(values, y + rng.uniform(-0.15, 0.15, len(values)), "o", ms=6.5,
                color=blue, alpha=0.78, mec="white", mew=0.8)
        ax.plot([np.median(values)], [y], "|", ms=22, mew=2.2, color=INK)
        ax.text(PAPER_SMALL.optimal_reward + 0.008, y,
                f"{(values >= SOLVED_REWARD).sum()}/{len(values)}", va="center",
                fontsize=9, color=INK if (values >= SOLVED_REWARD).all() else MUTED)
    ax.set_yticks(ys, names, fontsize=9)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(-0.18, PAPER_SMALL.optimal_reward + 0.035)
    ax.set_xlabel("mean RECONSTRUCTED reward of the final greedy policy "
                  "(surrogate curve, not the simulator)")
    ax.set_title("every seed, every configuration  (| = median, right = seeds solved)", loc="left")

    facet_axes = []
    for i, name in enumerate(names):
        fa = fig.add_subplot(facets[i // 8, i % 8])
        facet_axes.append(fa)
        fa.axhspan(PAPER_SMALL.dec_threshold, PAPER_SMALL.eom_price,
                   color="#1baf7a", alpha=0.12, lw=0)
        fa.axhline(PAPER_SMALL.optimal_bid, ls="--", lw=1, color=INK)
        for run_data in runs[name].values():
            fa.plot(run_data["steps"], np.median(run_data["greedy"], axis=0),
                    lw=1.15, color=blue, alpha=0.58)
        last_step = max(r["steps"][-1] for r in runs[name].values())
        fa.set_xlim(0, last_step)
        fa.set_ylim(-100, 100)
        fa.set_xticks([0, last_step], ["0", f"{last_step / 1000:g}k"])
        fa.set_title(name, loc="left", fontsize=9)
        if i % 8:
            fa.tick_params(labelleft=False)
        else:
            fa.set_ylabel("median actor bid")

    for axis in (ax, *facet_axes):
        axis.set_axisbelow(True)
        axis.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle(
        f"How stable is ASSUME MATD3 on the true inc-dec reward? — {phase}",
        x=0.007, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    fig.text(
        0.007, 0.968,
        f"One immutable unshaped starting buffer ({BUFFER_SHA256[:12]}...), then "
        "an isolated online buffer per run. One torch thread per worker.",
        fontsize=9, color=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def plot_critic_sweep(
    runs: dict[str, dict[int, dict[str, np.ndarray]]], out: Path, phase: str
) -> None:
    """Main stability comparison with multiseed critic fields in the facets.

    Each heatmap cell is the median Q1 action-gradient over every available
    ``(seed, fixed observation)`` pair. The overlaid black lines are one actor
    trajectory per seed, each itself the median over the fixed observations.
    This keeps the full spatial/temporal critic structure while making the
    aggregation explicit; it does not pretend that a single observation is a
    seed or that an average actor trajectory was a run that really happened.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm

    from run_benchmark import DIVERGING

    names = [name for name in CONFIGS if name in runs and runs[name]]
    if not names:
        return
    rows = (len(names) + 7) // 8
    fig = plt.figure(figsize=(16, 4.6 + 2.75 * rows))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, rows], hspace=0.32)
    facets = outer[1].subgridspec(rows, min(8, len(names)), wspace=0.12, hspace=0.48)

    # Build the multiseed/multi-observation fields once. A shared colour scale
    # makes configurations comparable instead of making every facet look equally
    # strong after independent normalization.
    fields: dict[str, np.ndarray] = {}
    for name in names:
        stacked = np.concatenate([run_data["grad"] for run_data in runs[name].values()])
        fields[name] = np.median(stacked, axis=0)  # (snapshots, bid grid)
    vmax = max(float(np.abs(field).max()) for field in fields.values())
    vmax = max(vmax, 1e-8)
    norm = SymLogNorm(linthresh=1e-4, vmin=-vmax, vmax=vmax, base=10)

    # Same final-reward strip as the actor-only comparison, so the reader can
    # connect the critic field directly to the outcome above it.
    ax = fig.add_subplot(outer[0])
    ax.axvline(PAPER_SMALL.optimal_reward, ls="--", lw=1.2, color=INK)
    ax.axvline(0, ls=":", lw=1.2, color=MUTED)
    ax.axvspan(SOLVED_REWARD, PAPER_SMALL.optimal_reward,
               color="#1baf7a", alpha=0.1)
    rng = np.random.default_rng(0)
    ys = np.arange(len(names))[::-1]
    for y, name in zip(ys, names):
        values = np.array([run_metrics(r)["final_reward"] for r in runs[name].values()])
        ax.plot(values, y + rng.uniform(-0.15, 0.15, len(values)), "o", ms=6.5,
                color="#2a78d6", alpha=0.78, mec="white", mew=0.8)
        ax.plot([np.median(values)], [y], "|", ms=22, mew=2.2, color=INK)
        ax.text(PAPER_SMALL.optimal_reward + 0.008, y,
                f"{(values >= SOLVED_REWARD).sum()}/{len(values)}", va="center",
                fontsize=9, color=INK if (values >= SOLVED_REWARD).all() else MUTED)
    ax.set_yticks(ys, names, fontsize=9)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(-0.18, PAPER_SMALL.optimal_reward + 0.035)
    ax.set_xlabel("mean RECONSTRUCTED reward of the final greedy policy "
                  "(surrogate curve, not the simulator)")
    ax.set_title("outcome per seed  (| = median, right = seeds solved)", loc="left")

    heat_axes = []
    mesh = None
    for i, name in enumerate(names):
        fa = fig.add_subplot(facets[i // 8, i % 8])
        heat_axes.append(fa)
        exemplar = next(iter(runs[name].values()))
        steps = exemplar["steps"]
        bids = exemplar["critic_bids"]
        mesh = fa.pcolormesh(
            steps, bids, fields[name].T, cmap=DIVERGING, norm=norm,
            shading="nearest", rasterized=True,
        )
        for bid in (PAPER_SMALL.dec_threshold, PAPER_SMALL.eom_price):
            fa.axhline(bid, color=INK, lw=0.7, alpha=0.45, ls="--")
        for run_data in runs[name].values():
            actor = np.median(run_data["greedy"], axis=0)
            fa.plot(run_data["steps"], actor, color="white", lw=2.2, alpha=0.9)
            fa.plot(run_data["steps"], actor, color=INK, lw=0.9, alpha=0.72)
        fa.set_xlim(steps[0], steps[-1])
        fa.set_ylim(-100, 100)
        # One right-edge label per narrow facet; paired endpoint labels collide
        # across columns ("0.8k10") and make the time scale harder to read.
        fa.set_xticks([steps[-1]], [f"{steps[-1]/1000:g}k"])
        facet_name = "\n".join(textwrap.wrap(name, width=18))
        fa.set_title(facet_name, loc="left", fontsize=8.1, linespacing=0.9)
        if i % 8:
            fa.tick_params(labelleft=False)
        else:
            fa.set_ylabel("bid price")
        fa.spines[["top", "right"]].set_visible(False)
        fa.tick_params(colors=MUTED, labelsize=8)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=heat_axes, fraction=0.012, pad=0.012, aspect=45)
        cbar.set_label("median dQ1/d(bid), symlog autograd", fontsize=8.5, color=MUTED)
        cbar.ax.tick_params(colors=MUTED, labelsize=7.5)
    ax.set_axisbelow(True)
    ax.grid(True, color=MUTED, alpha=0.2, lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    fig.suptitle(
        f"ASSUME MATD3 stability — multiseed critic evolution, {phase}",
        x=0.007, y=0.995, ha="left", fontsize=14, fontweight="bold", color=INK,
    )
    seed_count = len(next(iter(runs.values())))
    fig.text(
        0.007, 0.968,
        f"Each field is the median Q1 action-gradient across {seed_count} seeds x "
        "6 fixed observations; black lines are the per-seed median actor bids. "
        "All facets share one colour scale.",
        fontsize=9, color=MUTED,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASE_DEFAULTS, default="broad")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--n-obs", type=int, default=6)
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--study-case", default="inc_dec_learning_single_g0")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--critic-out", type=Path, default=None,
        help="optional all-configuration multiseed critic-evolution figure",
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    defaults = PHASE_DEFAULTS[args.phase]
    args.configs = args.configs or list(defaults["configs"])
    args.seeds = args.seeds or list(defaults["seeds"])
    args.episodes = args.episodes or int(defaults["episodes"])
    unknown = set(args.configs) - set(CONFIGS)
    if unknown:
        raise SystemExit(f"unknown configurations: {sorted(unknown)}")
    args.out_dir = args.out_dir or (
        RUNS / "data" / "11-assume-config-stability" / args.phase
    )
    args.out = args.out or (
        RUNS / "img" / f"11-assume-config-stability-{args.phase}.png"
    )

    if not args.report_only:
        run(args)
    runs = load_runs(args.out_dir, args.configs)
    summarize(runs)
    if not args.no_plot:
        plot(runs, args.out, args.phase)
    if args.critic_out is not None:
        plot_critic_sweep(runs, args.critic_out, args.phase)


if __name__ == "__main__":
    main()
