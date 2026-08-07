# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Record ASSUME's own critic over a real learning run -- **without editing ASSUME**.

``assume_critic_probe.py`` reads the networks a finished run leaves behind, which
is one frame. Run 06 showed that on this landscape the frame that decides the
outcome lasts a few hundred critic updates, so a single end-state cannot tell a
critic that never flipped from one that flipped and was not followed. This script
records the whole film.

How it avoids touching ASSUME
-----------------------------
``matd3.TD3.update_policy`` is wrapped from the outside: the original runs untouched,
and afterwards the wrapper sweeps ``Q1`` over the bid axis at a fixed set of
observations. Nothing in ``assume/`` changes, the learning dynamics are identical,
and the wrapper is undone when the process exits.

One snapshot is taken per *training block*, i.e. every ``train_freq``. With the
scenario's ``train_freq: 12h`` and ``gradient_steps: 32`` that is a snapshot every
32 critic updates, which is the resolution these runs are filmed at.

``--train-freq`` and ``--gradient-steps`` exist as config overrides, but **leave
train_freq at 12h**: a 1h train_freq on this scenario dies partway through with
``AssumeException: No rewards were collected during evaluation run``. Finer
resolution than one frame per training block needs a hook inside the gradient
loop at ``matd3.py:510``, which is the one-line edit this script exists to avoid.

Output
------
An ``.npz`` in the schema the rest of this folder already reads, so::

    python assume_training_probe.py --study-case inc_dec_learning_single_g0 \\
        --episodes 40 --train-freq 1h --gradient-steps 3
    python critic_evolution.py --results <o>/assume_training_probe.npz

The **seeds axis carries probed observations**, not random seeds -- the real
scenario's 74-vector varies hour to hour, so ``critic_q/MATD3[k]`` is the film as
seen from observation ``k``. ``steps`` counts *critic gradient steps*, not
environment steps.

Requires a database, since ``run_learning`` reads evaluation rewards back out of
it. The default is the local SQLite file the examples use.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _layout import (
    OUT_DIR,  # noqa: E402  (also puts the folders on sys.path)
    SCENARIO,  # noqa: E402
)

#: repository root and the inputs folder load_scenario_folder() expects
REPO = SCENARIO.parents[2]
INPUTS = SCENARIO.parent

MAX_BID_PRICE = 100.0
#: Observations are borrowed from a collection buffer rather than built by hand --
#: see ``assume_critic_probe.SHARED_BUFFER`` for why that matters.
OBS_BUFFER = SCENARIO / "learned_strategies" / "buffers" / "single_10ep_standard.npz"


def load_observations(path: Path, n: int) -> np.ndarray:
    data = np.load(path)
    filled = len(data["observations"]) if bool(data["full"][0]) else int(data["pos"][0])
    pool = data["observations"][:filled, 0, :]
    return pool[np.linspace(0, len(pool) - 1, n).astype(int)]


class Recorder:
    """Sweeps ``Q1`` and the actor after every training block, and holds the film.

    Kept as an object rather than a closure so the accumulated arrays can be
    written out from the ``finally`` branch even if training raises.
    """

    def __init__(self, observations: np.ndarray, grid: int, every: int):
        self.obs = observations
        self.bids = np.linspace(-MAX_BID_PRICE, MAX_BID_PRICE, grid)
        self.every = every
        self.calls = 0
        #: Cumulative gradient steps. ``algorithm.n_updates`` cannot be used: the
        #: world -- and with it the algorithm object -- is rebuilt at the start of
        #: every episode, so that counter restarts from zero each time (it reads
        #: 32, 64, 32, 64, ... across a run).
        self.updates = 0
        self.steps: list[int] = []
        self.q: list[np.ndarray] = []
        self.grad: list[np.ndarray] = []
        self.actor_bids: list[np.ndarray] = []
        self.buffer_fill: list[int] = []

    def snapshot(self, algorithm) -> None:
        strategies = list(algorithm.learning_role.rl_strats.values())
        if len(strategies) != 1:
            raise RuntimeError(
                f"this probe assumes a single learning unit, found {len(strategies)}; "
                "the critic of a multi-agent run takes every agent's action at once"
            )
        strategy = strategies[0]

        obs_batch = th.as_tensor(
            np.repeat(self.obs[:, None, :], len(self.bids), axis=1).reshape(
                -1, self.obs.shape[1]
            ),
            dtype=th.float32,
        )
        act = th.as_tensor(
            np.tile(self.bids / MAX_BID_PRICE, len(self.obs))[:, None], dtype=th.float32
        )
        act.requires_grad_(True)

        # q1_forward is what matd3.py differentiates for the actor loss
        q = strategy.critics.q1_forward(obs_batch, act)
        (grad,) = th.autograd.grad(q.sum(), act)

        shape = (len(self.obs), len(self.bids))
        self.q.append(q.detach().numpy().reshape(shape))
        self.grad.append(grad.numpy().reshape(shape) / MAX_BID_PRICE)

        with th.no_grad():
            actions = strategy.actor(th.as_tensor(self.obs, dtype=th.float32))
        self.actor_bids.append(actions.numpy().ravel() * MAX_BID_PRICE)

        buf = algorithm.learning_role.buffer
        self.buffer_fill.append(
            len(buf.observations) if bool(buf.full) else int(buf.pos)
        )
        self.steps.append(self.updates)

    def save(self, path: Path, algo: str = "MATD3", label: str = "") -> None:
        if not self.steps:
            print("  nothing recorded -- did any training block run?")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            steps=np.array(self.steps),
            critic_bids=self.bids,
            **{
                f"critic_q/{algo}": np.stack(self.q, axis=1),
                f"critic_grad/{algo}": np.stack(self.grad, axis=1),
                f"greedy/{algo}": np.stack(self.actor_bids, axis=1),
            },
            observations=self.obs,
            buffer_fill=np.array(self.buffer_fill),
            # which reward the run was trained on. The shaping at
            # learning_strategies.py:1583 is unconditional, so nothing in the
            # config records this -- it has to be carried by the caller.
            label=np.array(label),
            # critic_evolution.py reads cfg/warmup to draw the warmup marker; there
            # is no separate warmup here, the buffer is preloaded
            **{"cfg/warmup": 0, "cfg/timesteps": self.steps[-1]},
        )
        print(f"  wrote {path}  ({len(self.steps)} snapshots, {self.steps[-1]} updates)")


def install(recorder: Recorder) -> None:
    """Wrap the algorithm's ``update_policy`` so each block leaves a frame behind."""
    # the class is named TD3 in matd3.py; "matd3" is the config's algorithm key
    from assume.reinforcement_learning.algorithms.matd3 import TD3

    original = TD3.update_policy

    def wrapped(self):
        original(self)
        recorder.calls += 1
        recorder.updates += int(self.learning_config.gradient_steps)
        if recorder.calls % recorder.every == 0:
            recorder.snapshot(self)

    TD3.update_policy = wrapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=SCENARIO.name)
    parser.add_argument("--study-case", default="inc_dec_learning_single_g0")
    parser.add_argument(
        "--db-uri",
        default=f"sqlite:///{REPO / 'examples' / 'local_db' / 'assume_training_probe.db'}",
        help="run_learning reads evaluation rewards back out of a database; a "
        "probe-specific file by default so the shared one is left alone",
    )
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument(
        "--n-obs", type=int, default=6,
        help="observations to film from; they become the npz's 'seeds' axis",
    )
    parser.add_argument(
        "--every", type=int, default=1, help="snapshot every N-th training block"
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--load-buffer",
        default=None,
        help="replay_buffer_load_path, relative to the scenario inputs folder. "
        "Matters more than it looks: the g0 study case preloads "
        "buffers/single_10ep_gradient.npz, whose stored rewards are SHAPED, so an "
        "otherwise-unshaped run still trains on shaped transitions. Pass "
        "buffers/single_10ep_standard.npz for a clean unshaped run",
    )
    parser.add_argument("--train-freq", default=None, help="e.g. 1h")
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="where the run writes its policies. Defaults to a probe-specific "
        "folder, and deliberately NOT the study case's own: with "
        "continue_learning false, run_learning rmtree's this path before "
        "starting (common/utils.py:885), which would delete the existing "
        "trained networks",
    )
    parser.add_argument(
        "--label",
        default="",
        help="free text stored in the npz. Use it to record which reward the run "
        "was trained on: the shaping at learning_strategies.py:1583 fires "
        "unconditionally, so no config value distinguishes a shaped run from an "
        "unshaped one",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "assume_training_probe.npz"
    )
    args = parser.parse_args()

    from assume import World
    from assume.scenario.loader_csv import load_scenario_folder, run_learning

    world = World(database_uri=args.db_uri, export_csv_path="")
    load_scenario_folder(
        world,
        inputs_path=str(INPUTS),
        scenario=args.scenario,
        study_case=args.study_case,
    )

    # A fresh run deletes trained_policies_save_path before it starts, so never
    # let it point at a folder that already holds results worth keeping. A path
    # that does not exist yet also means confirm_learning_save_path() returns
    # early instead of prompting, which matters when this runs unattended.
    relative = args.save_path or Path("learned_strategies") / f"probe_{args.study_case}"
    save_path = SCENARIO / relative
    if save_path.exists() and any(save_path.iterdir()):
        raise SystemExit(
            f"{save_path} already has contents and a fresh run would delete them.\n"
            "Pass --save-path to point somewhere new, or remove it yourself."
        )

    # setup_world() deep-copies world.scenario_data at the start of every episode,
    # so overriding it here reaches all of them -- run_learning writes train_freq
    # back into the same dict itself.
    overrides = {
        "training_episodes": args.episodes,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
    }
    lc = world.scenario_data["config"]["learning_config"]
    for key, value in overrides.items():
        if value is not None:
            print(f"  override {key}: {lc.get(key)} -> {value}")
            lc[key] = value
            setattr(world.learning_role.learning_config, key, value)

    # Path-valued config entries need both forms. replace_paths() prefixes them
    # with the scenario inputs path on every setup_world(), so scenario_data must
    # hold the *relative* one; but the live learning_role was built before this
    # override and run_learning reads its *already-resolved* value directly, so
    # that one needs the absolute path. Setting the relative form on both is the
    # bug that makes run_learning raise "no buffer file found".
    if args.load_buffer:
        rel_buffer = f"learned_strategies/{args.load_buffer}"
        print(f"  override replay_buffer_load_path: {lc.get('replay_buffer_load_path')}"
              f" -> {rel_buffer}")
        lc["replay_buffer_load_path"] = rel_buffer
        absolute = SCENARIO / rel_buffer
        if not absolute.exists():
            raise SystemExit(f"no replay buffer at {absolute}")
        world.learning_role.learning_config.replay_buffer_load_path = str(absolute)

    print(f"  override trained_policies_save_path: {lc.get('trained_policies_save_path')} -> {relative}")
    lc["trained_policies_save_path"] = str(relative).replace("\\", "/")
    world.learning_role.learning_config.trained_policies_save_path = str(save_path)

    recorder = Recorder(
        load_observations(OBS_BUFFER, args.n_obs), args.grid, args.every
    )
    install(recorder)

    print(f"  filming {args.study_case} from {args.n_obs} observations\n")
    try:
        run_learning(world)
    finally:
        recorder.save(args.out, label=args.label)


if __name__ == "__main__":
    main()
