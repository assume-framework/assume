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
import hashlib
import json
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
        self.q1: list[np.ndarray] = []
        self.q2: list[np.ndarray] = []
        self.grad1: list[np.ndarray] = []
        self.grad2: list[np.ndarray] = []
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

        # Q1 is what matd3.py differentiates for the actor loss. Record Q2 as
        # well so a saved run can later draw the complete twin-critic landscape
        # without retaining or reloading a model checkpoint.
        q1, q2 = strategy.critics(obs_batch, act)
        (grad1,) = th.autograd.grad(q1.sum(), act, retain_graph=True)
        (grad2,) = th.autograd.grad(q2.sum(), act)

        shape = (len(self.obs), len(self.bids))
        self.q1.append(q1.detach().numpy().reshape(shape))
        self.q2.append(q2.detach().numpy().reshape(shape))
        self.grad1.append(grad1.numpy().reshape(shape) / MAX_BID_PRICE)
        self.grad2.append(grad2.numpy().reshape(shape) / MAX_BID_PRICE)

        with th.no_grad():
            actions = strategy.actor(th.as_tensor(self.obs, dtype=th.float32))
        self.actor_bids.append(actions.numpy().ravel() * MAX_BID_PRICE)

        buf = algorithm.learning_role.buffer
        self.buffer_fill.append(
            len(buf.observations) if bool(buf.full) else int(buf.pos)
        )
        self.steps.append(self.updates)

    def save(
        self,
        path: Path,
        algo: str = "MATD3",
        label: str = "",
        seed: int | None = None,
        config: dict | None = None,
        buffer_path: Path | None = None,
    ) -> None:
        if not self.steps:
            print("  nothing recorded -- did any training block run?")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer_sha256 = (
            hashlib.sha256(buffer_path.read_bytes()).hexdigest()
            if buffer_path is not None
            else ""
        )
        np.savez_compressed(
            path,
            steps=np.array(self.steps),
            critic_bids=self.bids,
            **{
                # Existing analysis scripts consume these Q1 compatibility keys.
                f"critic_q/{algo}": np.stack(self.q1, axis=1),
                f"critic_grad/{algo}": np.stack(self.grad1, axis=1),
                # Twin-critic diagnostics use these additional arrays.
                f"critic_q2/{algo}": np.stack(self.q2, axis=1),
                f"critic_grad2/{algo}": np.stack(self.grad2, axis=1),
                f"greedy/{algo}": np.stack(self.actor_bids, axis=1),
            },
            observations=self.obs,
            buffer_fill=np.array(self.buffer_fill),
            # which reward the run was trained on. The shaping at
            # learning_strategies.py:1583 is unconditional, so nothing in the
            # config records this -- it has to be carried by the caller.
            label=np.array(label),
            # ASSUME's config seed for this run, -1 when it was left at the
            # default. The stability sweep groups films by it.
            seed=np.array(-1 if seed is None else seed),
            config_json=np.array(json.dumps(config or {}, sort_keys=True)),
            buffer_path=np.array(str(buffer_path or "")),
            buffer_sha256=np.array(buffer_sha256),
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
        "--overrides-json",
        default="{}",
        help="JSON object of LearningConfig fields to override. Values are "
        "applied to both scenario_data and the live learning role before "
        "run_learning constructs the episode networks.",
    )
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
        "--seed",
        type=int,
        default=None,
        help="ASSUME's own seed knob, applied again. loader_csv.py:555 calls "
        "set_random_seed(config.get('seed', 42)) once, while the scenario is "
        "being read, and nothing re-seeds afterwards -- so re-applying it here, "
        "after the scenario is loaded and before run_learning() builds the "
        "networks, is the same call with a different number. Default: leave "
        "ASSUME's 42 in place",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch.set_num_threads. Run 08 found the BLAS thread count alone "
        "moves a surrogate seed from +31.60 to -60.49, so a sweep whose runs "
        "share a machine should pin this rather than let each process pick a "
        "different effective width",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="replace only TensorBoard's asynchronous writer with a no-op. The "
        "probe records richer diagnostics itself, while concurrent Windows "
        "runs can otherwise lose the event directory during writer startup. "
        "Database logging and learning dynamics are unchanged.",
    )
    parser.add_argument(
        "--out", type=Path, default=OUT_DIR / "assume_training_probe.npz"
    )
    args = parser.parse_args()

    try:
        config_overrides = json.loads(args.overrides_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid --overrides-json: {exc}") from exc
    if not isinstance(config_overrides, dict):
        raise SystemExit("--overrides-json must decode to an object")

    if args.threads is not None:
        th.set_num_threads(args.threads)

    from assume import World
    from assume.common.utils import set_random_seed
    from assume.reinforcement_learning.learning_role import Learning
    from assume.scenario.loader_csv import load_scenario_folder, run_learning

    if args.disable_tensorboard:
        original_init_logging = Learning.init_logging

        class _NoTensorBoard:
            def update_tensorboard(self) -> None:
                pass

        def init_logging_without_tensorboard(self, *init_args, **init_kwargs):
            # Keep init_logging's db_addr, datetime, and update_steps setup; only
            # replace the logger before its first update creates SummaryWriter.
            original_init_logging(self, *init_args, **init_kwargs)
            self.tensor_board_logger = _NoTensorBoard()

        Learning.init_logging = init_logging_without_tensorboard

    world = World(database_uri=args.db_uri, export_csv_path="")
    load_scenario_folder(
        world,
        inputs_path=str(INPUTS),
        scenario=args.scenario,
        study_case=args.study_case,
    )

    # Reseed *after* the scenario is read. The forecaster and the unit tables are
    # deterministic CSV loads, so everything that a seed can still change --
    # network init (run_learning calls initialize_policy at loader_csv.py:1213),
    # the exploration noise, and the replay-buffer batch draws -- happens after
    # this line. The value is written back into scenario_data for the record
    # only: config["seed"] is read exactly once, at load time, and setup_world()
    # never looks at it again.
    if args.seed is not None:
        print(f"  override seed: {world.scenario_data['config'].get('seed', 42)}"
              f" -> {args.seed}")
        world.scenario_data["config"]["seed"] = args.seed
        set_random_seed(seed=args.seed, learning_mode=True)

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
        **config_overrides,
    }
    lc = world.scenario_data["config"]["learning_config"]
    for key, value in overrides.items():
        if value is not None:
            if not hasattr(world.learning_role.learning_config, key):
                raise SystemExit(f"unknown LearningConfig override: {key}")
            print(f"  override {key}: {lc.get(key)} -> {value}")
            lc[key] = value
            setattr(world.learning_role.learning_config, key, value)

    # These callables are derived in Learning.__init__, before the command-line
    # overrides above are applied. Rebuild them so schedule sweeps change the
    # live learner rather than merely changing the recorded dataclass.
    from assume.reinforcement_learning.learning_utils import (
        cosine_annealing_func,
        linear_schedule_func,
    )

    live = world.learning_role.learning_config
    if live.learning_rate_schedule == "linear":
        world.learning_role.calc_lr_from_progress = linear_schedule_func(
            live.learning_rate, live.min_learning_rate
        )
    elif live.learning_rate_schedule == "cosine":
        world.learning_role.calc_lr_from_progress = cosine_annealing_func(
            live.learning_rate, live.min_learning_rate
        )
    else:
        world.learning_role.calc_lr_from_progress = lambda _: live.learning_rate

    if live.action_noise_schedule == "linear":
        world.learning_role.calc_noise_from_progress = linear_schedule_func(
            live.noise_dt
        )
    else:
        world.learning_role.calc_noise_from_progress = lambda _: live.noise_dt

    # Path-valued config entries need both forms. replace_paths() prefixes them
    # with the scenario inputs path on every setup_world(), so scenario_data must
    # hold the *relative* one; but the live learning_role was built before this
    # override and run_learning reads its *already-resolved* value directly, so
    # that one needs the absolute path. Setting the relative form on both is the
    # bug that makes run_learning raise "no buffer file found".
    absolute_buffer = None
    if args.load_buffer:
        rel_buffer = f"learned_strategies/{args.load_buffer}"
        print(f"  override replay_buffer_load_path: {lc.get('replay_buffer_load_path')}"
              f" -> {rel_buffer}")
        lc["replay_buffer_load_path"] = rel_buffer
        absolute = SCENARIO / rel_buffer
        if not absolute.exists():
            raise SystemExit(f"no replay buffer at {absolute}")
        world.learning_role.learning_config.replay_buffer_load_path = str(absolute)
        absolute_buffer = absolute

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
        resolved_config = {
            key: getattr(world.learning_role.learning_config, key)
            for key in world.learning_role.learning_config.__dataclass_fields__
        }
        recorder.save(
            args.out,
            label=args.label,
            seed=args.seed,
            config=resolved_config,
            buffer_path=absolute_buffer,
        )


if __name__ == "__main__":
    main()
