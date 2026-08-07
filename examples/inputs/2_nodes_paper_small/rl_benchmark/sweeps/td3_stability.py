# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
How stable is the softsign TD3 result?

Run 05 turned TD3 from "pinned to the +100 ceiling" into "+0.173" by swapping one
activation, on two seeds of one configuration. That is a thin basis for a claim
about MATD3 in ASSUME, and the trajectory it produced suggests why: the actor
escapes the ceiling in a single burst shortly after warmup ends, while the critic
is still forming. If that is a *window* rather than a steady descent, the result
should be fragile in a specific direction -- anything that slows the actor down
or lets the critic converge first should break it, while noise and learning-rate
changes should matter less.

This script sweeps the configuration axes that bear on that, eight seeds each,
and reports how many seeds land in the profitable band.

    python td3_stability.py                      # full sweep, ~25 min on 20 cores
    python td3_stability.py --configs baseline warmup-3000 --seeds 4
    python td3_stability.py --replot             # table + figure from the npz

Each (config, seed) is an independent process pinned to one torch thread: the
networks are far too small for intra-op parallelism to pay, so 14 runs at one
thread beat one run at 14 threads by an order of magnitude.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import OUT_DIR, resolve  # noqa: E402  (also sets sys.path)
from incdec_reward import PAPER_SMALL, reward_from_bid  # noqa: E402
from run_benchmark import (  # noqa: E402
    INK,
    MUTED,
    RunConfig,
    _train_one,
    probe_grid,
)

#: Run 05's configuration, which is the one being stress-tested -- shortened from
#: 10 000 steps because every outcome here is decided in the first ~2500 and the
#: sweep is 100+ runs. Expect final bids ~1-2 EUR above run 05's, since the actor
#: is still creeping down at step 6000.
BASELINE = RunConfig(
    timesteps=6_000,
    eval_every=100,
    noise_schedule="linear",
    actor_activation="softsign",
)

#: Overrides on ``BASELINE``. Grouped by what they test, because the point is not
#: coverage of the hyperparameter space -- it is whether the *race* between actor
#: and critic is what decides the outcome.
SWEEP: dict[str, dict] = {
    # -- the claim itself, and its control ---------------------------------
    "baseline": {},
    "tanh": {"actor_activation": "tanh"},
    # Run 05's exact length. Not redundant with ``baseline``: ``noise_schedule
    # linear`` anneals sigma to zero at ``timesteps``, so a 6000-step run is a
    # different exploration schedule, not a truncated one.
    "run05-repro": {"timesteps": 10_000, "eval_every": 200},
    "lr-1e-4-10k": {"timesteps": 10_000, "eval_every": 200, "learning_rate": 1e-4},
    # -- slow the actor down: if the escape is a window, these should fail ---
    "policy-delay-8": {"policy_delay": 8},
    "policy-delay-64": {"policy_delay": 64},
    "lr-1e-4": {"learning_rate": 1e-4},
    # -- let the critic converge further before the actor starts ------------
    "warmup-250": {"warmup": 250},
    "warmup-3000": {"warmup": 3000},
    # -- exploration: the axis run 01 assumed was decisive ------------------
    "noise-const": {"noise_schedule": "const"},
    "sigma-0.05": {"sigma": 0.05},
    "sigma-0.3": {"sigma": 0.3},
    # -- ordinary optimisation knobs ---------------------------------------
    "lr-3e-3": {"learning_rate": 3e-3},
    "batch-128": {"batch_size": 128},
    "buffer-2000": {"buffer_size": 2_000},
    # -- ASSUME's own numbers ----------------------------------------------
    # policy_delay 8, train_freq 12h with 32 gradient steps: ~2.7 updates per
    # environment step where SB3's default gives 1.
    "assume-knobs": {"policy_delay": 8, "train_freq": 12, "gradient_steps": 32},
    # ...and ASSUME's actual budget: a 10-episode preloaded buffer (240 steps)
    # plus 10 training episodes, not 10 000 steps.
    "assume-budget": {
        "policy_delay": 8,
        "train_freq": 12,
        "gradient_steps": 32,
        "warmup": 240,
        "timesteps": 480,
        "eval_every": 12,
    },
}

BAND = (PAPER_SMALL.dec_threshold, PAPER_SMALL.eom_price)
#: A seed counts as solved if its final greedy bid earns at least this. 0.150
#: corresponds to bid 34.0 -- inside the band and past the point where the
#: remaining regret is a matter of where a converged actor settles, not of
#: whether it found the band at all.
SOLVED = 0.150


def config_for(name: str) -> RunConfig:
    return replace(BASELINE, **SWEEP[name])


def _init_worker() -> None:
    import torch

    torch.set_num_threads(1)


def _job(task: tuple[str, int, RunConfig]):
    name, seed, cfg = task
    t0 = time.perf_counter()
    steps, greedy, placed, _, _ = _train_one("TD3", seed, cfg)
    return name, seed, steps, greedy, placed, time.perf_counter() - t0


def train(names: list[str], seeds: int, workers: int) -> dict[str, dict]:
    tasks = [(n, s, config_for(n)) for n in names for s in range(seeds)]
    out: dict[str, dict] = {
        n: {"greedy": [None] * seeds, "placed": [None] * seeds} for n in names
    }

    print(f"  {len(tasks)} runs ({len(names)} configs x {seeds} seeds) on {workers} workers\n")
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        futures = [pool.submit(_job, t) for t in tasks]
        for fut in as_completed(futures):
            name, seed, steps, greedy, placed, secs = fut.result()
            expected = probe_grid(config_for(name))
            if not np.array_equal(steps, expected):
                raise RuntimeError(f"{name} seed {seed} probed off the expected grid")
            out[name]["steps"] = steps
            out[name]["greedy"][seed] = greedy
            out[name]["placed"][seed] = placed
            done += 1
            print(
                f"  [{done:>3}/{len(tasks)}] {name:<16} seed {seed}  "
                f"{secs:5.0f}s  -> bid {greedy[-1]:7.2f}",
                flush=True,
            )

    for name in names:
        out[name]["greedy"] = np.vstack(out[name]["greedy"])
        out[name]["placed"] = np.vstack(out[name]["placed"])
    print(f"\n  {time.perf_counter() - t0:.0f}s wall\n")
    return out


def metrics(steps: np.ndarray, greedy: np.ndarray, warmup: int) -> dict[str, np.ndarray]:
    """Per-seed summary of one configuration.

    ``greedy`` is ``(seeds, probes)`` noise-free bids in EUR/MWh. Everything here
    is measured on the greedy policy, which for TD3 is what would be deployed --
    the exploration noise is annealed away by the end anyway.
    """
    rewards = reward_from_bid(greedy, PAPER_SMALL)
    in_band = (greedy >= BAND[0]) & (greedy <= BAND[1])
    tail = slice(int(0.75 * greedy.shape[1]), None)

    #: first probe *after warmup* at which the greedy policy is inside the band
    t_enter = np.full(len(greedy), np.nan)
    #: first probe after which it never leaves again -- the honest "converged"
    t_settle = np.full(len(greedy), np.nan)
    for i, row in enumerate(in_band):
        post = row & (steps > warmup)
        if post.any():
            t_enter[i] = steps[np.argmax(post)]
            # walk back from the end while the policy stays in the band
            last_out = np.flatnonzero(~row)
            first_stable = (last_out[-1] + 1) if len(last_out) else 0
            if first_stable < len(steps):
                t_settle[i] = steps[first_stable]

    return {
        "end_bid": greedy[:, -1],
        "end_reward": rewards[:, -1],
        "tail_reward": rewards[:, tail].mean(axis=1),
        "best_reward": rewards.max(axis=1),
        "t_enter": t_enter,
        "t_settle": t_settle,
        "band_frac": in_band[:, tail].mean(axis=1),
    }


def summarize(runs: dict[str, dict]) -> None:
    print(
        f"  optimum bid {PAPER_SMALL.optimal_bid:.2f} -> {PAPER_SMALL.optimal_reward:+.3f}"
        f"   |   solved = final reward >= {SOLVED:.3f} (bid <= 34.0)\n"
    )
    print(
        f"  {'config':<16}{'solved':>8}{'end bid':>18}{'end reward':>19}"
        f"{'tail reward':>13}{'t_enter':>10}"
    )
    print("  " + "-" * 84)
    for name, r in runs.items():
        m = metrics(r["steps"], r["greedy"], config_for(name).warmup)
        n = len(m["end_bid"])
        solved = m["end_reward"] >= SOLVED
        t = m["t_enter"]
        t_txt = "never" if np.all(np.isnan(t)) else f"{np.nanmedian(t):.0f}"
        print(
            f"  {name:<16}{f'{solved.sum()}/{n}':>8}"
            f"{np.mean(m['end_bid']):>10.2f} +-{np.std(m['end_bid']):<6.2f}"
            f"{np.mean(m['end_reward']):>+11.3f} +-{np.std(m['end_reward']):<6.3f}"
            f"{np.mean(m['tail_reward']):>+13.3f}{t_txt:>10}"
        )
    print()


def save(path: Path, runs: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for name, r in runs.items():
        payload[f"steps/{name}"] = r["steps"]
        payload[f"greedy/{name}"] = r["greedy"]
        payload[f"placed/{name}"] = r["placed"]
    # one JSON blob rather than cfg/<name>/<field> keys -- the configs differ in
    # which fields they set, and a dict round-trips without a schema
    payload["configs"] = np.array(
        json.dumps({n: asdict(config_for(n)) for n in runs})
    )
    np.savez(path, **payload)
    print(f"  wrote {path}")


def load(paths: list[Path]) -> dict[str, dict]:
    """Merge one or more sweep files into a single config dict.

    Configurations carry their own ``steps``, so runs of different lengths merge
    without alignment -- which is what lets the 6000-step sweep and the two
    10 000-step configurations share one figure.
    """
    runs: dict[str, dict] = {}
    for path in paths:
        data = np.load(path, allow_pickle=False)
        for key in data.files:
            if not key.startswith("greedy/"):
                continue
            name = key.split("/", 1)[1]
            if name in runs:
                raise SystemExit(f"{name} appears in more than one results file")
            runs[name] = {
                "steps": data[f"steps/{name}"],
                "greedy": data[f"greedy/{name}"],
                "placed": data[f"placed/{name}"],
            }
    return runs


def plot(runs: dict[str, dict], out: Path) -> None:
    """Two questions, two panels: where does each configuration end up, and does
    the seed spread hide a bimodal outcome?

    Every mark is one seed. Nothing is averaged in the top panel and nothing is
    band-shaded in the bottom one, because the failure mode here is bimodal --
    a mean of "0.19 and 0.00" describes no run that ever happened.
    """
    import matplotlib.pyplot as plt

    p = PAPER_SMALL
    names = list(runs)
    blue = "#2a78d6"

    fig = plt.figure(figsize=(16, 4.2 + 2.6 * ((len(names) + 7) // 8)))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.34)
    facets = outer[1].subgridspec((len(names) + 7) // 8, min(len(names), 8), wspace=0.1, hspace=0.55)

    # --- top: final reward, one dot per seed ---------------------------------
    ax = fig.add_subplot(outer[0])
    ax.axvline(p.optimal_reward, ls="--", lw=1.2, color=INK, zorder=0)
    ax.axvline(0.0, ls=":", lw=1.2, color=MUTED, zorder=0)
    ax.axvspan(SOLVED, p.optimal_reward, color="#1baf7a", alpha=0.08, lw=0, zorder=0)

    ys = np.arange(len(names))[::-1]
    rng = np.random.default_rng(0)
    for y, name in zip(ys, names):
        r = reward_from_bid(runs[name]["greedy"][:, -1], p)
        ax.plot(
            r, y + rng.uniform(-0.16, 0.16, len(r)), "o", ms=6.5,
            color=blue, alpha=0.75, mec="white", mew=0.8, zorder=3,
        )
        ax.plot([np.median(r)], [y], "|", ms=22, mew=2.2, color=INK, zorder=4)
        ax.text(
            p.optimal_reward + 0.008, y, f"{(r >= SOLVED).sum()}/{len(r)}",
            va="center", fontsize=9, color=INK if (r >= SOLVED).all() else MUTED,
        )
    ax.set_yticks(ys, names, fontsize=9.5)
    ax.set_ylim(-0.7, len(names) - 0.3)
    ax.set_xlim(-0.02, p.optimal_reward + 0.03)
    ax.set_xlabel("reward of the final greedy policy")
    ax.set_title(
        "every seed, every configuration  (| = median, right column = seeds solved)",
        loc="left", fontsize=11, color=INK,
    )
    ax.annotate("optimum", xy=(p.optimal_reward, len(names) - 0.5), xytext=(-3, 0),
                textcoords="offset points", ha="right", fontsize=8.5, color=INK)
    ax.annotate("zero plateau", xy=(0.0, len(names) - 0.5), xytext=(4, 0),
                textcoords="offset points", fontsize=8.5, color=MUTED)

    # --- bottom: the trajectories behind those dots --------------------------
    facet_axes = []
    for i, name in enumerate(names):
        fa = fig.add_subplot(facets[i // 8, i % 8])
        facet_axes.append(fa)
        cfg = config_for(name)
        fa.axhspan(*BAND, color="#1baf7a", alpha=0.12, lw=0)
        fa.axhline(p.optimal_bid, ls="--", lw=1.0, color=INK, zorder=2)
        fa.axvline(cfg.warmup, ls=":", lw=1.0, color=INK, zorder=2)
        for row in runs[name]["greedy"]:
            fa.plot(runs[name]["steps"], row, lw=1.1, color=blue, alpha=0.55, zorder=3)
        fa.set_ylim(-p.max_bid_price, p.max_bid_price)
        fa.set_xlim(0, cfg.timesteps)
        fa.set_xticks([0, cfg.timesteps], ["0", f"{cfg.timesteps / 1000:g}k"])
        fa.set_title(name, loc="left", fontsize=9, color=INK)
        if i % 8:
            fa.tick_params(labelleft=False)
        else:
            fa.set_ylabel("greedy bid (EUR/MWh)")

    # anchored to the facet block rather than a fixed figure fraction, so the
    # caption stays put whether the sweep is 3 configs or 30
    fig.text(
        0.007, facet_axes[0].get_position().y1 + 0.035,
        "greedy bid over training — one line per seed, dotted line = warmup ends",
        fontsize=11, color=INK,
    )

    for a in (ax, *facet_axes):
        a.set_axisbelow(True)
        a.grid(True, color=MUTED, alpha=0.2, lw=0.7)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(MUTED)
        a.tick_params(colors=MUTED, labelsize=8.5)

    fig.suptitle(
        "How stable is softsign TD3 on the inc-dec landscape?",
        x=0.007, y=1.03, ha="left", fontsize=13.5, fontweight="bold", color=INK,
    )
    fig.text(
        0.007, 1.005,
        f"{len(runs[names[0]]['greedy'])} seeds per configuration, all deviations "
        "from run 05's settings applied one at a time. The two 10k configurations "
        "run for 10 000 steps, the rest for 6000.",
        fontsize=9, color=MUTED, ha="left",
    )
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=list(SWEEP))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[resolve("td3_stability.npz"), resolve("td3_stability_10k.npz")],
        help="one path to write to when training; one or more to merge when "
        "re-plotting. The default is the archived pair, so --replot needs no "
        "arguments",
    )
    parser.add_argument("--out", type=Path, default=OUT_DIR / "td3_stability.png")
    parser.add_argument("--replot", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="table only")
    args = parser.parse_args()

    unknown = set(args.configs) - set(SWEEP)
    if unknown:
        raise SystemExit(f"unknown configs: {sorted(unknown)}")

    if args.replot:
        runs = load(args.results)
    else:
        if len(args.results) != 1:
            raise SystemExit("--results takes a single path when training")
        runs = train(args.configs, args.seeds, args.workers)
        save(args.results[0], runs)

    summarize(runs)

    if not args.no_plot:
        plot(runs, args.out)


if __name__ == "__main__":
    main()
