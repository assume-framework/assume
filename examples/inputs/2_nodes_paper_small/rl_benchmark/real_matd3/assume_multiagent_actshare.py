# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Run 13 -- does run 12's ``act_share`` story survive the multi-agent case?

The question
------------
Run 12 (single learning unit) found that ASSUME's MATD3 fails on the true reward
because the action is **1 of 75 critic inputs**, and that raising

    act_share = sd(a_own) / (sd(a_own) + sum_j sd(other critic inputs_j))

from 0.030 to 0.479 solves the task 3/3. ``HANDOFF.md`` then lists the multi-agent
case as the interesting open one, because a centralised critic *lowers* each
actor's own share as agents are added: every extra agent contributes
``unique_obs_dim`` observation dimensions **and one more action dimension**, while
agent *i*'s own action stays a single scalar.

``inc_dec_learning`` is that case: all 11 units of ``powerplant_units_learning.csv``
learn, on a 72 h horizon, so each agent's critic sees

    obs   74 + 2*(11-1) = 94 dimensions
    act   11 dimensions, of which one is its own
    ---------------------------------------------
          105 inputs, the own action being one of them

Why the lever had to change
---------------------------
Run 12's action lever multiplies the critic's **whole** action input by ``S``
(``Q(s, S*a)``). With N agents that raises every agent's action together, so an
agent's *own* share saturates at ``1/N`` -- about 0.09 here -- and the requested
0.2 is unreachable that way. This script therefore scales **only the own-action
column of each agent's own critic**:

    critic_i fits  Q_i(s, a_1, ..., S*a_i, ..., a_N)

which is symmetric across agents (each critic upweights its own actor's action,
none is privileged), is exactly the input whose gradient the actor climbs
(``matd3.py:711`` differentiates ``q1_forward`` w.r.t. ``a_i``), and reduces to
run 12's patch when N = 1. Adam is scale-invariant, so the actor's *step size* is
unchanged even though its gradient is S times larger -- but the recorded
``dQ/d(bid)`` carries the factor S, so **signs are comparable across conditions
and magnitudes are not**, exactly as in run 12.

Nothing in ``assume/`` is edited: the patch is installed in the child process
before ``assume_training_probe.main()`` loads the scenario.

What is recorded
----------------
``assume_training_probe.py`` refuses to film a multi-agent run -- its recorder
sweeps ``critics(obs, a)`` with a single action column. ``MultiAgentRecorder``
below replaces it through the same monkeypatch route the run 12 sweep already
uses, and films, per training block and **per agent**:

* ``Q1`` and its autograd gradient over a bid grid for that agent's own action,
  with the other agents' actions held at their actors' current outputs and the
  critic's observation input assembled the way ``matd3.py:584-591`` assembles it;
* every agent's greedy bid at the probed observations;
* the mean stored reward per agent over the most recent episode of transitions;
* the per-dimension standard deviations of the live buffer, so each run records
  the evidence for its own ``act_share`` rather than inheriting a number.

The closed-form ``incdec_reward`` landscape does **not** apply here: it was
derived with the rest of the fleet bidding naively, and in this study case all 11
units learn. Rewards are therefore read from the run's own buffer, never
reconstructed from the bid.

Usage
-----
    # measure act_share (and the S that reaches a target) from a recorded run
    python real_matd3/assume_multiagent_actshare.py --measure <run.npz> --target 0.2

    # the two runs, in parallel
    python real_matd3/assume_multiagent_actshare.py --conditions baseline act-own-x15
    python real_matd3/assume_multiagent_actshare.py --report-only
"""

from __future__ import annotations

import argparse
import concurrent.futures
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

SELF = Path(__file__).resolve()
REPO = SCENARIO.parents[2]
SHAPING_SOURCE = REPO / "assume" / "strategies" / "learning_strategies.py"
#: the shaping is a source edit, not a config flag; this is its uncommented form
SHAPING_LIVE = re.compile(r"^\s{8}if reward > 0:", re.MULTILINE)

MAX_BID_PRICE = 100.0
STUDY_CASE = "inc_dec_learning"

#: Overrides applied to *both* conditions. The study case is otherwise run as
#: written (lr 1e-4, gradient_steps 10, batch 128, gamma .99, sigma .1,
#: train_freq 12h, 50 episodes, 5 collecting) -- this is the requested baseline.
#: early_stopping is disabled so both conditions are guaranteed the same number
#: of episodes; otherwise one could stop early and the two would not compare.
COMMON_OVERRIDES: dict[str, object] = {
    "early_stopping_steps": 1_000_000,
    "save_replay_buffer": False,
    "load_replay_buffer": False,
}

#: own-action scale S for each condition. S is set from a measured pilot; see
#: --measure. 1.0 is the study case exactly as given.
#: ``scale``    the factor S applied to the critic's action input
#: ``mode``     "own" scales only that critic's own action column; "all" is run
#:              12's lever verbatim, over the whole action vector
#: ``episodes`` per-condition budget, so a short look can run beside a full one
#:
#: S = 15 puts the mean *own*-action share at 0.197 under "own" (per agent
#: 0.178-0.220) and at 0.069 under "all", against a hard 1/N = 0.091 ceiling --
#: measured on a pilot buffer of 276 collection transitions, the same convention
#: run 12 used (uniform exploration actions). The pair therefore separates the
#: own action's share from the whole action block's share, which are 0.196/0.327
#: under "own" and 0.069/0.762 under "all": run 12 moved both at once and could
#: not tell them apart.
CONDITIONS: dict[str, dict] = {
    "baseline": {"scale": 1.0, "mode": "own", "episodes": 50},
    "act-own-x15": {"scale": 15.0, "mode": "own", "episodes": 50},
    "baseline-25": {"scale": 1.0, "mode": "own", "episodes": 25},
    "act-all-x15": {"scale": 15.0, "mode": "all", "episodes": 25},
    # the block-matched control: same action-block share as act-own-x15 (0.302
    # against 0.329) but an own share of 0.027, barely above baseline's 0.016.
    # If the own share is what the actor's gradient needs, this behaves like
    # baseline; if the action block's share is enough, it behaves like
    # act-own-x15. Run 12 moved both together and could not separate them.
    "act-all-x2": {"scale": 2.0, "mode": "all", "episodes": 25},
    # act-all-x2 at 25 episodes ends with `pulled left` = 1.00 at the actor in
    # 6/6 probed observations on both seeds, its plateau pull still rising and
    # its descent path still deepening -- run 12's exact signature for "the
    # budget ran out before it converged". This is the same condition with the
    # budget doubled, so that reading can be tested rather than inferred.
    "act-all-x2-50": {"scale": 2.0, "mode": "all", "episodes": 50},
}

#: the pair the default invocation runs
DEFAULT_CONDITIONS = ["baseline", "act-own-x15"]

SEEDS = [42]
EPISODES = 50


# ------------------------------------------------------------------ act_share


def act_share_from_sd(
    sd_obs: np.ndarray,
    sd_act: np.ndarray,
    unique_obs_dim: int,
    scale: float,
    mode: str = "own",
) -> np.ndarray:
    """Own-action share of the critic's total input std, one value per agent.

    ``sd_obs`` is ``(n_agents, obs_dim)`` and ``sd_act`` is ``(n_agents,)``, both
    measured over the replay buffer. Critic *i* sees agent *i*'s full observation
    plus the last ``unique_obs_dim`` entries of every other agent's, and all
    ``n_agents`` actions.

    ``mode="own"`` scales only critic *i*'s own action column; ``mode="all"`` is
    run 12's lever applied verbatim, ``Q(s, S*a)`` over the whole action vector,
    which raises every agent's action together and therefore **caps the own share
    at 1/N** no matter how large S is.
    """
    n = len(sd_act)
    unique_sum = sd_obs[:, -unique_obs_dim:].sum(axis=1)
    shares = np.empty(n)
    for i in range(n):
        obs_sum = sd_obs[i].sum() + (unique_sum.sum() - unique_sum[i])
        own = scale * sd_act[i]
        if mode == "all":
            act_sum = scale * sd_act.sum()
        else:
            act_sum = own + (sd_act.sum() - sd_act[i])
        shares[i] = own / (act_sum + obs_sum)
    return shares


def action_block_share(
    sd_obs: np.ndarray,
    sd_act: np.ndarray,
    unique_obs_dim: int,
    scale: float,
    mode: str = "own",
) -> np.ndarray:
    """Share carried by the *whole* action block, one value per agent.

    The companion of ``act_share_from_sd``: ``act-own`` and ``act-all`` move these
    two quantities in opposite proportions, which is what makes the pair a
    dissociation rather than two points on one curve.
    """
    n = len(sd_act)
    unique_sum = sd_obs[:, -unique_obs_dim:].sum(axis=1)
    shares = np.empty(n)
    for i in range(n):
        obs_sum = sd_obs[i].sum() + (unique_sum.sum() - unique_sum[i])
        if mode == "all":
            act_sum = scale * sd_act.sum()
        else:
            act_sum = scale * sd_act[i] + (sd_act.sum() - sd_act[i])
        shares[i] = act_sum / (act_sum + obs_sum)
    return shares


def scale_for_target(
    sd_obs: np.ndarray,
    sd_act: np.ndarray,
    unique_obs_dim: int,
    target: float,
    mode: str = "own",
) -> float:
    """The action scale S whose mean act_share is ``target``."""
    lo, hi = 1.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        if act_share_from_sd(sd_obs, sd_act, unique_obs_dim, mid, mode).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def measure(path: Path, target: float) -> None:
    d = np.load(path, allow_pickle=False)
    sd_obs, sd_act = d["buffer_sd_obs"], d["buffer_sd_act"]
    unique_obs_dim = int(d["unique_obs_dim"])
    units = [str(u) for u in d["unit_ids"]]
    n = len(units)

    print(f"\n  {path.name}: {n} agents, obs_dim {sd_obs.shape[1]}, "
          f"unique_obs_dim {unique_obs_dim}, buffer {int(d['buffer_fill'][-1])} transitions")
    print(f"  critic inputs per agent: {sd_obs.shape[1] + unique_obs_dim * (n - 1)} obs "
          f"+ {n} act = {sd_obs.shape[1] + unique_obs_dim * (n - 1) + n}\n")

    print(f"  {'S':>8} {'own act_share':>14} {'block share':>12}   "
          f"{'own (all-scaled)':>17} {'block (all)':>12}")
    for s in (1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 100.0, 1e6):
        own = act_share_from_sd(sd_obs, sd_act, unique_obs_dim, s, "own").mean()
        own_b = action_block_share(sd_obs, sd_act, unique_obs_dim, s, "own").mean()
        alls = act_share_from_sd(sd_obs, sd_act, unique_obs_dim, s, "all").mean()
        all_b = action_block_share(sd_obs, sd_act, unique_obs_dim, s, "all").mean()
        print(f"  {s:8.0f} {own:14.3f} {own_b:12.3f}   {alls:17.3f} {all_b:12.3f}")
    print(f"\n  scaling *every* action caps the own share at 1/N = {1 / n:.3f}; "
          f"scaling one column has no such cap")

    s = scale_for_target(sd_obs, sd_act, unique_obs_dim, target)
    sh = act_share_from_sd(sd_obs, sd_act, unique_obs_dim, s)
    print(f"\n  S = {s:.2f} gives mean act_share {sh.mean():.3f} "
          f"({sh.min():.3f}-{sh.max():.3f})")
    print(f"  per agent at S = {s:.1f}:")
    for u, v in zip(units, sh):
        print(f"    {u:<10} {v:.3f}")


# ---------------------------------------------------------------- the recorder


class MultiAgentRecorder:
    """Films every agent's critic over its own bid axis, once per training block.

    Drop-in replacement for ``assume_training_probe.Recorder``: same constructor
    signature, same ``snapshot``/``save`` interface, so the probe's own
    ``install()`` wrapper and ``finally`` branch work unchanged.
    """

    #: transitions averaged for the per-agent reward trace (one 72 h episode)
    REWARD_WINDOW = 62

    def __init__(self, observations: np.ndarray, grid: int, every: int):
        # the probe hands us whatever load_observations returned; for a
        # multi-agent run the observations have to be joint across agents, so
        # only the *count* is taken from it and the rest is sampled from the
        # live buffer at the first snapshot.
        self.n_obs = len(observations)
        self.obs: np.ndarray | None = None
        self.bids = np.linspace(-MAX_BID_PRICE, MAX_BID_PRICE, grid)
        self.every = every
        self.calls = 0
        #: cumulative gradient steps; algorithm.n_updates restarts every episode
        self.updates = 0
        self.steps: list[int] = []
        self.q1: list[np.ndarray] = []
        self.grad1: list[np.ndarray] = []
        self.actor_bids: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.buffer_fill: list[int] = []
        self.unit_ids: list[str] = []
        self.unique_obs_dim = 0
        self.sd_obs: np.ndarray | None = None
        self.sd_act: np.ndarray | None = None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _filled(buf) -> int:
        return len(buf.observations) if bool(buf.full) else int(buf.pos)

    def _sample_observations(self, buf, n_agents: int) -> np.ndarray:
        """n_obs joint observations, evenly spaced through the buffer so far."""
        fill = self._filled(buf)
        if fill < self.n_obs:
            raise RuntimeError(f"buffer holds {fill} transitions, need {self.n_obs}")
        idx = np.linspace(0, fill - 1, self.n_obs).astype(int)
        return np.asarray(buf.observations[idx], dtype=np.float64)

    def _critic_states(self, i: int) -> np.ndarray:
        """Critic *i*'s observation input, as ``matd3.py:584-591`` builds it."""
        own = self.obs[:, i, :]
        others = np.concatenate(
            [self.obs[:, :i, -self.unique_obs_dim:], self.obs[:, i + 1:, -self.unique_obs_dim:]],
            axis=1,
        ).reshape(self.n_obs, -1)
        return np.concatenate([own, others], axis=1)

    # -- the frame ----------------------------------------------------------

    def snapshot(self, algorithm) -> None:
        import torch as th

        strategies = list(algorithm.learning_role.rl_strats.items())
        n_agents = len(strategies)
        buf = algorithm.learning_role.buffer
        self.unique_obs_dim = int(algorithm.unique_obs_dim)
        if int(algorithm.act_dim) != 1:
            raise RuntimeError(
                f"this recorder sweeps a single bid axis per agent, but act_dim is "
                f"{algorithm.act_dim}"
            )

        if self.obs is None:
            self.obs = self._sample_observations(buf, n_agents)
            self.unit_ids = [u for u, _ in strategies]

        n_bids = len(self.bids)

        # every agent's greedy action at the probed observations; agent i's own
        # column is then replaced by the grid in i's own sweep
        with th.no_grad():
            base = th.stack(
                [
                    strategy.actor(th.as_tensor(self.obs[:, j, :], dtype=th.float32))
                    for j, (_, strategy) in enumerate(strategies)
                ],
                dim=1,
            )  # (n_obs, n_agents, act_dim)
        base = base.reshape(self.n_obs, -1)  # act_dim == 1 here

        q1_frame = np.empty((n_agents, self.n_obs, n_bids), dtype=np.float32)
        grad_frame = np.empty((n_agents, self.n_obs, n_bids), dtype=np.float32)

        for i, (_, strategy) in enumerate(strategies):
            states = th.as_tensor(
                np.repeat(
                    self._critic_states(i)[:, None, :], n_bids, axis=1
                ).reshape(self.n_obs * n_bids, -1),
                dtype=th.float32,
            )
            acts = base[:, None, :].repeat(1, n_bids, 1).clone()
            acts[:, :, i] = th.as_tensor(
                np.tile(self.bids / MAX_BID_PRICE, (self.n_obs, 1)), dtype=th.float32
            )
            acts = acts.reshape(self.n_obs * n_bids, -1)
            acts.requires_grad_(True)

            # Q1 is the objective matd3.py differentiates for the actor loss.
            q1 = strategy.critics.q1_forward(states, acts)
            (grad,) = th.autograd.grad(q1.sum(), acts)

            shape = (self.n_obs, n_bids)
            q1_frame[i] = q1.detach().numpy().reshape(shape)
            grad_frame[i] = grad[:, i].numpy().reshape(shape) / MAX_BID_PRICE

        self.q1.append(q1_frame)
        self.grad1.append(grad_frame)
        self.actor_bids.append(base.numpy() * MAX_BID_PRICE)  # (n_obs, n_agents)

        fill = self._filled(buf)
        window = slice(max(0, fill - self.REWARD_WINDOW), fill)
        self.rewards.append(np.asarray(buf.rewards[window]).mean(axis=0))
        self.buffer_fill.append(fill)
        self.steps.append(self.updates)

        # the evidence for this run's own act_share, refreshed each frame so the
        # last one describes the whole run
        self.sd_obs = np.asarray(buf.observations[:fill]).std(axis=0)
        self.sd_act = np.asarray(buf.actions[:fill]).std(axis=0).ravel()

    # -- output -------------------------------------------------------------

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
        np.savez_compressed(
            path,
            steps=np.array(self.steps),
            critic_bids=self.bids,
            # (n_agents, n_obs, frames, grid) -- the run 09-12 axis order, with
            # a leading agent axis. The 'seeds' axis of those files is this
            # schema's n_obs axis.
            **{
                f"critic_q/{algo}": np.stack(self.q1, axis=2),
                f"critic_grad/{algo}": np.stack(self.grad1, axis=2),
                # (n_agents, n_obs, frames)
                f"greedy/{algo}": np.stack(self.actor_bids, axis=2).transpose(1, 0, 2),
            },
            # (frames, n_agents): mean stored reward over the last episode of
            # transitions. The closed-form landscape does not apply here.
            rewards=np.stack(self.rewards, axis=0),
            observations=self.obs,
            unit_ids=np.array(self.unit_ids),
            unique_obs_dim=np.array(self.unique_obs_dim),
            buffer_sd_obs=self.sd_obs,
            buffer_sd_act=self.sd_act,
            buffer_fill=np.array(self.buffer_fill),
            label=np.array(label),
            seed=np.array(-1 if seed is None else seed),
            config_json=np.array(json.dumps(config or {}, sort_keys=True)),
            **{"cfg/warmup": 0, "cfg/timesteps": self.steps[-1]},
        )
        print(f"  wrote {path}  ({len(self.steps)} snapshots, {self.steps[-1]} updates, "
              f"{len(self.unit_ids)} agents)")


# ---------------------------------------------------------------------- child


def install_action_scale(scale: float, mode: str) -> None:
    """Scale the critic's action input by S.

    ``mode="own"``  critic *i* fits ``Q_i(s, ..., S*a_i, ...)`` -- its own column
                    only. The index is stamped onto every critic at the top of
                    each ``update_policy`` call rather than at construction,
                    because the world -- and with it the networks -- is rebuilt at
                    the start of every episode.
    ``mode="all"``  ``Q(s, S*a)`` over the whole action vector, i.e. run 12's
                    patch unchanged. No index is needed.
    """
    import torch as th

    from assume.reinforcement_learning.algorithms.matd3 import TD3
    from assume.reinforcement_learning.neural_network_architecture import CriticTD3

    original_forward = CriticTD3.forward
    original_q1 = CriticTD3.q1_forward

    def scaled(self, actions: "th.Tensor") -> "th.Tensor":
        if mode == "all":
            return actions * scale
        idx = getattr(self, "_own_action_index", None)
        if idx is None:
            return actions
        width = self.act_dim // getattr(self, "_n_agents", 1)
        out = actions.clone()
        out[:, idx * width: (idx + 1) * width] = (
            out[:, idx * width: (idx + 1) * width] * scale
        )
        return out

    def forward(self, obs, actions):
        return original_forward(self, obs, scaled(self, actions))

    def q1_forward(self, obs, actions):
        return original_q1(self, obs, scaled(self, actions))

    CriticTD3.forward = forward
    CriticTD3.q1_forward = q1_forward

    if mode == "own":
        original_update = TD3.update_policy

        def wrapped(self):
            strategies = list(self.learning_role.rl_strats.values())
            for i, strategy in enumerate(strategies):
                for net in (strategy.critics, strategy.target_critics):
                    net._own_action_index = i
                    net._n_agents = len(strategies)
            original_update(self)

        TD3.update_policy = wrapped
    print(f"  patched CriticTD3: {mode} action input x{scale}")


def run_child(name: str, rest: list[str]) -> None:
    cond = CONDITIONS[name]
    scale, mode = float(cond["scale"]), str(cond["mode"])

    # must be installed before the probe wraps update_policy, so that the index
    # stamping runs before the recorder's snapshot reads the critics
    if scale != 1.0:
        install_action_scale(scale, mode)

    import assume_training_probe as probe

    # the joint observations come from the live buffer; only the count is taken
    # from the probe's single-agent loader
    probe.load_observations = lambda path, n: np.empty((n, 0))
    probe.Recorder = MultiAgentRecorder

    sys.argv = ["assume_training_probe.py", *rest]
    probe.main()


# --------------------------------------------------------------------- parent


def result_path(out_dir: Path, name: str, seed: int) -> Path:
    return out_dir / f"assume_ma_{name}_seed{seed}.npz"


def preflight() -> None:
    source = SHAPING_SOURCE.read_text(encoding="utf-8")
    if SHAPING_LIVE.search(source):
        raise SystemExit(
            "the reward shaping at learning_strategies.py:1583 is UNCOMMENTED. "
            "This run must be on the true reward; comment it back out first."
        )


def launch(name: str, seed: int, args) -> tuple[str, int, int, float, Path]:
    out = result_path(args.out_dir, name, seed)
    if out.exists() and not args.rerun:
        return name, seed, 0, 0.0, out

    tag = f"{name}_seed{seed}"
    scratch = args.out_dir / "scratch" / tag
    scratch.mkdir(parents=True, exist_ok=True)
    relative_save = Path("learned_strategies") / f"probe_ma_{tag}"
    shutil.rmtree(SCENARIO / relative_save, ignore_errors=True)
    db = scratch / "probe.db"
    db.unlink(missing_ok=True)

    overrides = dict(COMMON_OVERRIDES)
    if args.collecting is not None:
        overrides["episodes_collecting_initial_experience"] = args.collecting
    if args.validation_interval is not None:
        overrides["validation_episodes_interval"] = args.validation_interval

    episodes = args.episodes or int(CONDITIONS[name].get("episodes", EPISODES))
    cmd = [
        sys.executable, str(SELF), "--child", name, "--",
        "--study-case", args.study_case,
        "--episodes", str(episodes),
        "--n-obs", str(args.n_obs),
        "--grid", str(args.grid),
        "--every", "1",
        "--seed", str(seed),
        "--threads", "1",
        "--disable-tensorboard",
        "--label", name,
        "--train-freq", "12h",
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
    return name, seed, proc.returncode, time.perf_counter() - t0, out


def run(args) -> None:
    preflight()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(n, s) for n in args.conditions for s in args.seeds]
    print(f"\n  {len(jobs)} trials, study case {args.study_case}, "
          f"true reward (shaping commented out)")
    for n in args.conditions:
        c = CONDITIONS[n]
        print(f"    {n:<14} {c['mode']}-action scale x{c['scale']:g}, "
              f"{args.episodes or c['episodes']} episodes")
    print()

    done = 0
    workers = args.workers or len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, n, s, args) for n, s in jobs]
        for fut in concurrent.futures.as_completed(futures):
            name, seed, rc, secs, out = fut.result()
            done += 1
            status = "ok" if rc == 0 and out.exists() else f"FAILED rc={rc}"
            print(f"  [{done}/{len(jobs)}] {name} seed {seed}: {status} "
                  f"({secs / 60:.1f} min)", flush=True)


# -------------------------------------------------------------------- reading


def report(args) -> None:
    for name in args.conditions:
        for seed in args.seeds:
            path = result_path(args.out_dir, name, seed)
            if not path.exists():
                print(f"\n{name} seed {seed}: (no results at {path.name})")
                continue
            d = np.load(path, allow_pickle=False)
            units = [str(u) for u in d["unit_ids"]]
            greedy = d[f"greedy/MATD3"]      # (n_agents, n_obs, frames)
            q1 = d["critic_q/MATD3"]         # (n_agents, n_obs, frames, grid)
            bids = d["critic_bids"]
            rewards = d["rewards"]           # (frames, n_agents)
            sd_obs, sd_act = d["buffer_sd_obs"], d["buffer_sd_act"]
            uod = int(d["unique_obs_dim"])
            scale, mode = CONDITIONS[name]["scale"], CONDITIONS[name]["mode"]
            share = act_share_from_sd(sd_obs, sd_act, uod, scale, mode)
            block = action_block_share(sd_obs, sd_act, uod, scale, mode)

            print(f"\n{name} seed {seed}: {len(units)} agents, "
                  f"{greedy.shape[2]} frames, {int(d['steps'][-1])} critic updates, "
                  f"mean act_share {share.mean():.3f}, "
                  f"action-block share {block.mean():.3f}")
            header = (f"  {'unit':<10} {'act_share':>9} {'first bid':>10} "
                      f"{'final bid':>10} {'argmax Q1':>10} {'obs spread':>11} "
                      f"{'reward last':>12}")
            print(header)
            print("  " + "-" * (len(header) - 2))
            for i, u in enumerate(units):
                argmax = bids[np.argmax(q1[i, :, -1, :], axis=1)]
                spread = float(np.mean([abs(a - b) for a in argmax for b in argmax]))
                print(f"  {u:<10} {share[i]:9.3f} "
                      f"{np.median(greedy[i, :, 0]):10.1f} "
                      f"{np.median(greedy[i, :, -1]):10.1f} "
                      f"{np.median(argmax):10.1f} {spread:11.1f} "
                      f"{rewards[-1, i]:+12.4f}")
            print(f"  {'TOTAL':<10} {'':>9} {'':>10} {'':>10} {'':>10} {'':>11} "
                  f"{rewards[-1].sum():+12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--study-case", default=STUDY_CASE)
    parser.add_argument("--episodes", type=int, default=None,
                        help="override every condition's own episode budget")
    parser.add_argument("--collecting", type=int, default=None,
                        help="override episodes_collecting_initial_experience")
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--workers", type=int, default=None,
                        help="concurrent trials; each holds ~0.8 GB, so cap this "
                             "when other runs already share the machine")
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--n-obs", type=int, default=6)
    parser.add_argument("--grid", type=int, default=401)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--measure", type=Path, default=None,
                        help="print act_share vs S from a recorded run's buffer stats")
    parser.add_argument("--target", type=float, default=0.2)
    parser.add_argument(
        "--out-dir", type=Path,
        default=OUT_DIR / "runs" / "data" / "13-multiagent-actshare",
    )

    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        name = sys.argv[i + 1]
        rest = sys.argv[sys.argv.index("--", i) + 1:]
        run_child(name, rest)
        return

    args = parser.parse_args()
    if args.measure is not None:
        measure(args.measure, args.target)
        return
    unknown = set(args.conditions) - set(CONDITIONS)
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")
    if not args.report_only:
        run(args)
    report(args)


if __name__ == "__main__":
    main()
