# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Workstream C -- film the critics of a *plain energy-only* run (``example_02a``-``02c``).

Why this exists
---------------
Everything in ``RUNS.md`` is one redispatch/multi-market scenario with a 90 %-flat
reward. ``HANDOFF.md`` § C asks whether any of it survives on an ordinary
energy-only market, and the cheapest way to find out is to point the same probe
at the three stock examples:

=======  ==================  ================================================
case     learning units      what it adds over the previous one
=======  ==================  ================================================
``02a``  ``pp_6``            single learner against a naive fleet
``02b``  ``pp_6``-``pp_10``  five learners, centralised critic
``02c``  ``pp_6``-``pp_15``  ten learners; a majority of the fleet
=======  ==================  ================================================

Each has a **single-bid twin** -- ``sb02a``, ``sb02b``, ``sb02c`` -- which is
the same fleet, demand, fuel prices and market bidding with
``EnergyLearningSingleBidStrategy`` (``act_dim`` 1, one bid for the unit's whole
``max_power``) instead of ``EnergyLearningStrategy`` (``act_dim`` 2, inflexible
+ flexible block). They live as three study cases of one scenario folder,
``examples/inputs/example_02_single_bid/``, differing only in
``powerplant_units``. See :data:`CASES`.

All six are one ``pay_as_clear`` EOM, no redispatch, no storage -- which is
also the only setting in which the exploitability metric is meaningful (see the
SCOPE note in ``assume/reinforcement_learning/exploitability.py``), so the two
readings can be taken off the same runs.

What is different from run 13's recorder
----------------------------------------
``MultiAgentRecorder`` (``assume_multiagent_actshare.py``) refuses to film
anything with ``act_dim != 1``: it sweeps one bid axis per agent. The stock EOM
examples use ``EnergyLearningStrategy``, which is **two** actions per unit -- the
inflexible block (P_min) and the flexible one (P_max - P_min), assigned by
``min``/``max`` of the two actions in ``calculate_bids``. This recorder is
therefore generic in ``act_dim`` and takes ``act_dim + 1`` sweeps per agent:

``a0``, ``a1``, ...   one action component swept over the bid grid, the agent's
                      other components and every other agent's actions held at
                      their actors' current greedy outputs;
``diag``              *all* of the agent's own components moved together, which
                      is the unit bidding one price for its whole capacity. This
                      is the axis directly comparable with runs 09-13's single
                      bid axis, and the one the figures default to.

At ``act_dim = 1`` -- the ``sb*`` cases -- the per-component and diagonal sweeps
coincide, so there is exactly one sweep and it is *named* ``diag``, which keeps
every reader working across both bid structures.

Only the **learning** units are filmed, which needs no filtering:
``learning_role.rl_strats`` holds exactly those, and the naive units of the fleet
(``pp_1``-``pp_5``, and the expensive peaker) never enter a critic's input.

Everything else follows run 13: nothing in ``assume/`` is edited, the recorder is
installed through ``assume_training_probe``'s monkeypatch route, one process per
(scenario, seed), and rewards are read from the run's own replay buffer -- the
closed-form ``incdec_reward`` landscape does not apply here at all.

Usage
-----
::

    # the single-bid trio, three seeds, locally
    python real_matd3/eom_critic_film.py --workers 3

    # the two-bid originals
    python real_matd3/eom_critic_film.py --cases 02a 02b 02c --workers 3

    # one trial, the shape a cluster array task uses
    python real_matd3/eom_critic_film.py --cases sb02b --seeds 42 \\
        --workers 1 --threads 1

    # smoke test: ten agents, 4 days, 10 episodes, ~1 min
    python real_matd3/eom_critic_film.py --cases sb02c --seeds 42 \\
        --study-case tiny --grid 51 --n-obs 3 --workers 1

    # read what was recorded, and draw it
    python real_matd3/eom_critic_film.py --report-only
    python analysis/eom_critic_evolution.py

⚠️ **File size is the one knob that matters.** The film is
``n_agents x (act_dim + 1) x n_obs x frames x grid`` floats, twice (Q and its
gradient). ``example_02c`` at ``--grid 401 --n-obs 6 --every 1`` is ~1.5 GB per
seed; the defaults here (``--grid 201 --n-obs 4 --every 4``) put it near 30 MB,
which is what makes the results scp-able. ``RUNS_Continuation.md`` house rules
ask for ``--critic-grid 401``; that rule is about the *surrogate* sweeps, where
the grid is free. Say which grid a run used in its section.
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
INPUTS = SCENARIO.parent
SHAPING_SOURCE = REPO / "assume" / "strategies" / "learning_strategies.py"
#: the inc-dec reward shaping is a source edit, not a config flag. It fires
#: unconditionally once uncommented, so it would silently reshape these EOM runs
#: too; this is its uncommented form.
SHAPING_LIVE = re.compile(r"^\s{8}if reward > 0:", re.MULTILINE)

#: ``max_bid_price`` in all three study cases. Actions live in [-1, 1] and are
#: multiplied by this to become EUR/MWh, so it is also the bid grid's half-width.
MAX_BID_PRICE = 100.0

#: The runnable cases, keyed by the short name that ends up in file names and
#: figures. ``02a``-``02c`` are the stock two-bid examples; ``sb02a``-``sb02c``
#: are the single-bid folder, one study case per fleet, and are the same three
#: fleets with ``EnergyLearningSingleBidStrategy`` (``act_dim`` 1, one bid for
#: the unit's whole ``max_power``) instead of ``EnergyLearningStrategy``
#: (``act_dim`` 2, inflexible + flexible block). Each ``sb`` case is a clean A/B
#: against the ``02x`` case above it -- same demand, same fuel prices, same
#: naive fleet, same market.
#:
#: ⚠️ The single-bid strategy also defaults ``foresight`` to 24 rather than 12,
#: so its observation is 50-dimensional against 26. The action count is not the
#: only thing that moves between the two columns; say so in any ``act_share``
#: comparison across them.
CASES: dict[str, dict[str, str]] = {
    "02a": {"scenario": "example_02a", "study_case": "base"},
    "02b": {"scenario": "example_02b", "study_case": "base"},
    "02c": {"scenario": "example_02c", "study_case": "base"},
    "sb02a": {"scenario": "example_02_single_bid", "study_case": "02a"},
    "sb02b": {"scenario": "example_02_single_bid", "study_case": "02b"},
    "sb02c": {"scenario": "example_02_single_bid", "study_case": "02c"},
}

#: what a bare invocation runs. The single-bid trio, because that is the open
#: question; the two-bid cases stay one ``--cases 02a 02b 02c`` away.
DEFAULT_CASES = ["sb02a", "sb02b", "sb02c"]

#: run 13's three seeds, so the two studies are read at the same seed count
SEEDS = [42, 1, 2]

# ------------------------------------------------------------ the stage game

#: Demand regimes, low to high, with the equilibrium each implies. Every
#: non-learning unit bids its marginal cost, so the merit order is known and the
#: one-shot game has a closed-form Nash equilibrium that **switches with
#: demand**. With ``C`` the cheap naive capacity below the learners, ``L`` the
#: learners' total and ``u`` one learner's:
#:
#: ``idle``      D <= C            learners not needed; nothing to exploit.
#: ``bertrand``  C < D <= C+L-u    at least one learner is completely
#:                                 undispatched and undercuts any price above
#:                                 cost, so NE is **everyone at marginal cost**.
#: ``pivotal``   C+L-u < D <= C+L  every learner runs and one only partly, so
#:                                 that unit is marginal, cannot be replaced by a
#:                                 peer, and NE is bidding up to the **backup
#:                                 generator's marginal cost**.
#: ``backup``    D > C+L           the backup sets the price; learners are
#:                                 inframarginal price-takers.
REGIMES: dict[str, str] = {
    "idle": "not dispatched",
    "bertrand": "marginal cost",
    "pivotal": "backup marginal cost",
    "backup": "price-taker",
}


def merit_order(case: str) -> dict:
    """Marginal costs, the learner set and the regime thresholds for one case.

    Marginal cost follows ``PowerPlant``'s convention — fuel price over
    efficiency, plus CO2 price times emission factor over efficiency, plus
    ``additional_cost``. It agrees with the recorded observations' last entry to
    0.1 EUR/MWh, so this is a restatement, not an independent derivation.
    """
    import pandas as pd

    scenario, study = CASES[case]["scenario"], CASES[case]["study_case"]
    folder = INPUTS / scenario
    # the single-bid folder carries one unit table per study case
    units_csv = next(p for p in (folder / f"powerplant_units_{study}.csv",
                                 folder / "powerplant_units.csv") if p.is_file())

    fuel = pd.read_csv(folder / "fuel_prices_df.csv", index_col=0).loc["price"].astype(float)
    pp = pd.read_csv(units_csv)
    pp["mc"] = (
        pp["fuel_type"].map(fuel).astype(float) / pp["efficiency"]
        + pp["emission_factor"] * float(fuel["co2"]) / pp["efficiency"]
        + pp["additional_cost"]
    )
    learn = pp["bidding_EOM"].str.contains("learning")
    learners = pp[learn]
    mc_l = float(learners["mc"].iloc[0])

    cheap = float(pp.loc[~learn & (pp["mc"] < mc_l), "max_power"].sum())
    total_l = float(learners["max_power"].sum())
    unit_mw = float(learners["max_power"].iloc[0])
    above = pp.loc[~learn & (pp["mc"] > mc_l)]

    return {
        "units": [str(u) for u in learners["name"]],
        "unit_mw": unit_mw,
        "mc": mc_l,
        "backup_mc": float(above["mc"].min()) if len(above) else float("inf"),
        # (upper bound, name); a timestep falls in the first band it fits
        "bands": [(cheap, "idle"),
                  (cheap + total_l - unit_mw, "bertrand"),
                  (cheap + total_l, "pivotal"),
                  (float("inf"), "backup")],
    }


def horizon(case: str, study_case: str | None = None) -> tuple[str, str]:
    """``(start_date, end_date)`` of the case's study case, from its config."""
    import yaml

    cfg = yaml.safe_load(
        (INPUTS / CASES[case]["scenario"] / "config.yaml").read_text(encoding="utf-8"))
    block = cfg[study_case or CASES[case]["study_case"]]
    return str(block["start_date"]), str(block["end_date"])


def demand(case: str):
    """Hourly demand of the case's scenario, over the whole series."""
    import pandas as pd

    s = pd.read_csv(INPUTS / CASES[case]["scenario"] / "demand_df.csv",
                    index_col=0, parse_dates=True)["demand_EOM"]
    return s.resample("1h").mean()


def live_bands(bands) -> list[tuple[float, str]]:
    """``bands`` with the zero-width ones removed.

    At a single learner ``L - u == 0``, so ``bertrand`` has no demand range at
    all: the one unit is marginal whenever it runs and no peer can undercut it.
    That is a fact about the case, not a degenerate edge to paper over.
    """
    kept, prev = [], -float("inf")
    for upper, name in bands:
        if upper > prev:
            kept.append((upper, name))
            prev = upper
    return kept


def regime_quantiles(case: str, start: str, end: str) -> dict[str, tuple[float, float]]:
    """Each regime's share of the horizon, as a quantile band of demand.

    The recorder cannot see the demand series — it only has the replay buffer,
    whose first observation entry is the **min-max scaled residual load** of the
    delivery hour. With no renewables in these scenarios residual load *is*
    demand, so a rank in the buffer is a rank in demand and a regime maps onto a
    quantile band. Returning quantiles rather than MW keeps the recorder free of
    the scaling factors, which live on the strategy and are not reachable from
    the algorithm object.
    """
    import numpy as np
    import pandas as pd

    d = demand(case).loc[start:end]
    out, lo = {}, 0.0
    for upper, name in live_bands(merit_order(case)["bands"]):
        hi = float((d <= upper).mean())
        if hi > lo:
            out[name] = (lo, min(hi, 1.0))
            lo = hi
    return out


#: Applied to every trial. ``early_stopping_steps`` is disabled so all scenarios
#: are guaranteed the same number of episodes -- ``example_02a``'s single learner
#: converges early and would otherwise stop with a different budget from
#: ``02c``'s ten, which is exactly the comparison being made. The buffer flags
#: keep a trial self-contained: no shared starting buffer exists for these
#: scenarios and none is written.
COMMON_OVERRIDES: dict[str, object] = {
    "early_stopping_steps": 1_000_000,
    "save_replay_buffer": False,
    "load_replay_buffer": False,
}


# ------------------------------------------------------------------ act_share


def act_share_from_sd(
    sd_obs: np.ndarray,
    sd_act: np.ndarray,
    unique_obs_dim: int,
) -> np.ndarray:
    """Own-action share of critic *i*'s total input std, one value per agent.

    Run 12's quantity, generalised from one action per agent to ``act_dim``:
    the numerator is the sum over agent *i*'s own action components. ``sd_obs``
    is ``(n_agents, obs_dim)`` and ``sd_act`` is ``(n_agents, act_dim)``, both
    measured over the live replay buffer. Critic *i* sees agent *i*'s full
    observation plus the last ``unique_obs_dim`` entries of every other agent's,
    and all agents' actions.

    ⚠️ ``act_share`` is a quantity invented in run 12, not a literature one, and
    ``HANDOFF.md`` § B says not to build on it before the offline architecture
    screen. It is recorded here because it is free -- the buffer std has to be
    taken anyway -- and because the whole point of these runs is whether the
    ordering it produced transfers off the inc-dec landscape.
    """
    n = sd_act.shape[0]
    unique_sum = sd_obs[:, -unique_obs_dim:].sum(axis=1)
    act_sum = float(sd_act.sum())
    shares = np.empty(n)
    for i in range(n):
        obs_sum = sd_obs[i].sum() + (unique_sum.sum() - unique_sum[i])
        shares[i] = sd_act[i].sum() / (act_sum + obs_sum)
    return shares


# ---------------------------------------------------------------- the recorder


class EomCriticRecorder:
    """Films every learning agent's critic over its own bid axes, per training block.

    Drop-in replacement for ``assume_training_probe.Recorder`` -- same
    constructor signature and the same ``snapshot``/``save`` interface -- so the
    probe's ``install()`` wrapper and its ``finally`` branch work unchanged.
    """

    def __init__(self, observations: np.ndarray, grid: int, every: int):
        # the probe hands us whatever load_observations returned; for a joint
        # critic the observations have to be joint across agents, so only the
        # *count* is taken from it and the rest is sampled from the live buffer
        # at the first snapshot.
        self.n_obs = len(observations)
        self.obs: np.ndarray | None = None
        #: quantile band per regime, set by run_child when --obs-regimes is
        #: passed; None means the plain evenly-spaced selection
        self.bands: dict[str, tuple[float, float]] | None = None
        self.obs_regime: list[str] = []
        self.bids = np.linspace(-MAX_BID_PRICE, MAX_BID_PRICE, grid)
        self.every = every
        self.calls = 0
        #: cumulative gradient steps; algorithm.n_updates restarts every episode
        #: because the world is rebuilt at the start of each one
        self.updates = 0
        self.steps: list[int] = []
        self.q1: list[np.ndarray] = []
        self.grad1: list[np.ndarray] = []
        self.actor_actions: list[np.ndarray] = []
        self.rewards: list[np.ndarray] = []
        self.buffer_fill: list[int] = []
        #: simulated time and episode index at each frame. Without these a frame
        #: cannot be placed in the horizon at all, and "critic updates" silently
        #: mixes training progress with position in the month -- see
        #: analysis/eom_critic_evolution.py's frame_schedule().
        self.frame_time: list[float] = []
        self.frame_episode: list[int] = []
        self.unit_ids: list[str] = []
        self.sweeps: list[str] = []
        self.unique_obs_dim = 0
        self.act_dim = 0
        self.sd_obs: np.ndarray | None = None
        self.sd_act: np.ndarray | None = None
        #: ring position at the previous frame, so the reward window is exactly
        #: "the transitions collected since the last frame" -- see _reward_window
        self._prev_pos: int | None = None

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _filled(buf) -> int:
        return len(buf.observations) if bool(buf.full) else int(buf.pos)

    def _sample_observations(self, buf) -> tuple[np.ndarray, list[str]]:
        """The joint observations to film from, and the regime label of each.

        Default (``bands`` unset): ``n_obs`` evenly spaced through the buffer,
        which is a spread over whatever hours happened to be collected and
        carries no regime meaning — every label is ``"any"``.

        With ``bands`` set the selection is **stratified by demand regime**.
        Entry 0 of an observation is the min-max scaled residual load of the
        delivery hour (``create_observation``'s forward window, first element),
        and with no renewables in these scenarios residual load is demand — so
        ranking the buffer on that column ranks it by demand, and a quantile
        band picks the hours of one regime. ``n_obs`` observations are then
        spread evenly *within* each band, so a film can be read separately for
        "the equilibrium is marginal cost" hours and "the equilibrium is the
        backup's cost" hours. This is the point of the whole exercise: whether
        one critic can hold both equilibria at once.
        """
        fill = self._filled(buf)
        if fill < self.n_obs:
            raise RuntimeError(f"buffer holds {fill} transitions, need {self.n_obs}")
        obs = np.asarray(buf.observations[:fill], dtype=np.float64)

        if not self.bands:
            idx = np.linspace(0, fill - 1, self.n_obs).astype(int)
            return obs[idx], ["any"] * self.n_obs

        # agent 0's column is enough: every agent sees the same market
        order = np.argsort(obs[:, 0, 0], kind="stable")
        picked, labels = [], []
        for name, (lo, hi) in self.bands.items():
            a, b = int(round(lo * fill)), int(round(hi * fill))
            if b - a < 1:
                print(f"  regime {name!r} has no transitions in the buffer -- skipped")
                continue
            take = min(self.n_obs, b - a)
            sel = order[a:b][np.linspace(0, (b - a) - 1, take).astype(int)]
            picked.append(obs[sel])
            labels += [name] * take
        if not picked:
            raise RuntimeError("no regime band matched any buffered transition")
        return np.concatenate(picked, axis=0), labels

    def _reward_window(self, buf) -> np.ndarray:
        """Indices of the transitions collected since the previous frame.

        Run 13 averaged a fixed count (69, one episode of its horizon). That
        constant cannot be carried over: these scenarios have 720-744 h horizons
        and ``train_freq: 100h``, so an episode spans several frames, and at
        ~74 000 transitions per 100-episode run the 50 000-slot buffer **wraps**
        -- after which ``buffer.full`` is permanently true and ``_filled`` is a
        constant, so any "last N of the fill" window silently starts averaging
        across the wrap. Tracking the ring position instead is exact on both
        sides of it.

        The first frame has no predecessor and falls back to the whole buffer.
        """
        size = len(buf.observations)
        pos = int(buf.pos)
        prev = self._prev_pos
        self._prev_pos = pos
        if prev is None or pos == prev:
            fill = self._filled(buf)
            return np.arange(max(0, fill - 1), dtype=int) if fill else np.zeros(0, int)
        if pos > prev:
            return np.arange(prev, pos, dtype=int)
        # wrapped since the last frame: tail of the ring, then its head
        return np.concatenate([np.arange(prev, size), np.arange(0, pos)]).astype(int)

    def _critic_states(self, i: int) -> np.ndarray:
        """Critic *i*'s observation input, as ``matd3.py:584-591`` builds it."""
        own = self.obs[:, i, :]
        others = np.concatenate(
            [
                self.obs[:, :i, -self.unique_obs_dim:],
                self.obs[:, i + 1:, -self.unique_obs_dim:],
            ],
            axis=1,
        ).reshape(self.n_obs, -1)
        return np.concatenate([own, others], axis=1)

    def _sweep_columns(self, i: int) -> list[tuple[str, list[int]]]:
        """Which action columns each sweep moves, for agent *i*.

        Columns are indexed into the flat ``(n_agents * act_dim)`` action vector
        the critic takes, in the order ``matd3.py`` concatenates it.
        """
        base = i * self.act_dim
        if self.act_dim == 1:
            # EnergyLearningSingleBidStrategy: one action, one bid for the whole
            # max_power. The single sweep is named "diag" rather than "a0" on
            # purpose -- it *is* the unit moving its whole bid, so every reader
            # ("draw the diag sweep") works unchanged across both bid structures.
            return [("diag", [base])]
        per_component = [(f"a{k}", [base + k]) for k in range(self.act_dim)]
        return per_component + [
            ("diag", [base + k for k in range(self.act_dim)])
        ]

    # -- the frame ----------------------------------------------------------

    def snapshot(self, algorithm) -> None:
        import torch as th

        strategies = list(algorithm.learning_role.rl_strats.items())
        n_agents = len(strategies)
        buf = algorithm.learning_role.buffer
        self.unique_obs_dim = int(algorithm.unique_obs_dim)
        self.act_dim = int(algorithm.act_dim)

        if self.obs is None:
            self.obs, self.obs_regime = self._sample_observations(buf)
            # a stratified selection returns one block per regime, so the count
            # is n_obs x (regimes present), not n_obs
            self.n_obs = len(self.obs)
            self.unit_ids = [u for u, _ in strategies]
            self.sweeps = [name for name, _ in self._sweep_columns(0)]
            if self.bands:
                counts = {r: self.obs_regime.count(r) for r in dict.fromkeys(self.obs_regime)}
                print(f"  probing {self.n_obs} observations by regime: {counts}")

        n_bids = len(self.bids)
        n_sweeps = len(self.sweeps)

        # every agent's greedy action at the probed observations; the swept
        # columns are then replaced by the grid inside each agent's own sweep
        with th.no_grad():
            base = th.stack(
                [
                    strategy.actor(th.as_tensor(self.obs[:, j, :], dtype=th.float32))
                    for j, (_, strategy) in enumerate(strategies)
                ],
                dim=1,
            )  # (n_obs, n_agents, act_dim)
        flat_base = base.reshape(self.n_obs, n_agents * self.act_dim)

        q1_frame = np.empty((n_agents, n_sweeps, self.n_obs, n_bids), dtype=np.float32)
        grad_frame = np.empty_like(q1_frame)

        grid = th.as_tensor(
            np.tile(self.bids / MAX_BID_PRICE, (self.n_obs, 1)), dtype=th.float32
        )

        for i, (_, strategy) in enumerate(strategies):
            states = th.as_tensor(
                np.repeat(self._critic_states(i)[:, None, :], n_bids, axis=1).reshape(
                    self.n_obs * n_bids, -1
                ),
                dtype=th.float32,
            )
            for s, (_, cols) in enumerate(self._sweep_columns(i)):
                acts = flat_base[:, None, :].repeat(1, n_bids, 1).clone()
                for c in cols:
                    acts[:, :, c] = grid
                acts = acts.reshape(self.n_obs * n_bids, -1)
                acts.requires_grad_(True)

                # Q1 is the objective matd3.py differentiates for the actor loss
                q1 = strategy.critics.q1_forward(states, acts)
                (grad,) = th.autograd.grad(q1.sum(), acts)

                shape = (self.n_obs, n_bids)
                q1_frame[i, s] = q1.detach().numpy().reshape(shape)
                # dQ/d(bid in EUR) along the swept direction: the columns move
                # together, so their gradients add before the EUR rescaling
                grad_frame[i, s] = (
                    grad[:, cols].sum(axis=1).numpy().reshape(shape) / MAX_BID_PRICE
                )

        self.q1.append(q1_frame)
        self.grad1.append(grad_frame)
        self.actor_actions.append(base.numpy() * MAX_BID_PRICE)  # (n_obs, n_ag, act)

        idx = self._reward_window(buf)
        rewards = np.asarray(buf.rewards)
        self.rewards.append(
            rewards[idx].mean(axis=0) if len(idx) else np.zeros(n_agents)
        )
        fill = self._filled(buf)
        self.buffer_fill.append(fill)
        self.steps.append(self.updates)

        # where in the horizon this frame sits. train_freq is snapped to divide
        # the horizon evenly (learning_role.sync_train_freq_with_simulation_
        # horizon), so blocks land on a fixed grid and every episode replays the
        # same calendar -- which is what makes the frame index alias.
        role = algorithm.learning_role
        self.frame_time.append(float(getattr(role.context, "current_timestamp", 0.0)))
        self.frame_episode.append(int(getattr(role, "episodes_done", -1)))

        # the evidence for this run's own act_share, refreshed each frame so the
        # last one describes the whole run
        self.sd_obs = np.asarray(buf.observations[:fill]).std(axis=0)
        self.sd_act = np.asarray(buf.actions[:fill]).std(axis=0)

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
            sweeps=np.array(self.sweeps),
            **{
                # (n_agents, n_sweeps, n_obs, frames, grid)
                f"critic_q/{algo}": np.stack(self.q1, axis=3),
                f"critic_grad/{algo}": np.stack(self.grad1, axis=3),
                # (n_agents, act_dim, n_obs, frames) in EUR/MWh
                f"greedy/{algo}": np.stack(self.actor_actions, axis=3).transpose(
                    1, 2, 0, 3
                ),
            },
            # (frames, n_agents): mean stored reward over the transitions
            # collected since the previous frame
            rewards=np.stack(self.rewards, axis=0),
            observations=self.obs,
            # (n_obs,) which demand regime each probed observation came from,
            # or all "any" when the selection was not stratified
            obs_regime=np.array(self.obs_regime or ["any"] * self.n_obs),
            unit_ids=np.array(self.unit_ids),
            unique_obs_dim=np.array(self.unique_obs_dim),
            act_dim=np.array(self.act_dim),
            buffer_sd_obs=self.sd_obs,
            buffer_sd_act=self.sd_act,
            buffer_fill=np.array(self.buffer_fill),
            # (frames,) unix seconds of simulated time, and the episode index.
            # Absent from anything recorded before 2026-08-12; the analysis
            # falls back to deriving them from the study case.
            frame_time=np.array(self.frame_time, dtype=float),
            frame_episode=np.array(self.frame_episode),
            label=np.array(label),
            seed=np.array(-1 if seed is None else seed),
            config_json=np.array(json.dumps(config or {}, sort_keys=True)),
            **{"cfg/warmup": 0, "cfg/timesteps": self.steps[-1]},
        )
        print(
            f"  wrote {path}  ({len(self.steps)} snapshots, {self.steps[-1]} updates, "
            f"{len(self.unit_ids)} agents, {path.stat().st_size / 1e6:.1f} MB)"
        )


# ---------------------------------------------------------------------- child


def run_child(rest: list[str], bands: dict | None = None) -> None:
    import assume_training_probe as probe

    # the joint observations come from the live buffer; only the count is taken
    # from the probe's single-agent loader, whose buffer belongs to another
    # scenario and has the wrong observation dimension
    probe.load_observations = lambda path, n: np.empty((n, 0))

    if bands:
        # the probe constructs the Recorder itself, so the bands are attached
        # through the class rather than the constructor signature, which has to
        # stay compatible with assume_training_probe.Recorder
        class _Stratified(EomCriticRecorder):
            def __init__(self, observations, grid, every):
                super().__init__(observations, grid, every)
                self.bands = bands

        probe.Recorder = _Stratified
    else:
        probe.Recorder = EomCriticRecorder

    sys.argv = ["assume_training_probe.py", *rest]
    probe.main()


# --------------------------------------------------------------------- parent


def result_path(out_dir: Path, case: str, seed: int) -> Path:
    return out_dir / f"eom_film_{case}_seed{seed}.npz"


def preflight(cases: list[str]) -> None:
    source = SHAPING_SOURCE.read_text(encoding="utf-8")
    if SHAPING_LIVE.search(source):
        raise SystemExit(
            "the reward shaping at learning_strategies.py:1583 is UNCOMMENTED. "
            "It fires unconditionally, so it would reshape these EOM runs too; "
            "comment it back out first."
        )
    for case in cases:
        scenario = CASES[case]["scenario"]
        if not (INPUTS / scenario / "config.yaml").is_file():
            raise SystemExit(f"no config.yaml for scenario {scenario} (case {case})")


def validate_result(path: Path, case: str, seed: int) -> None:
    """Refuse to treat a partial archive as a finished trial.

    ``assume_training_probe`` writes its film from ``finally``, so a crashed run
    still leaves an inspectable ``.npz`` -- "the file exists" does not mean "the
    trial finished". Frame count is not asserted here the way run 13 asserts it:
    ``--every`` and the study case's ``train_freq`` set it, and an early-stopped
    run is a legitimate short film. The last-frame check below is what catches a
    crash: a run killed mid-training has no reward row for its last snapshot.
    """
    d = np.load(path, allow_pickle=False)
    missing = {
        "steps", "critic_bids", "sweeps", "critic_q/MATD3", "critic_grad/MATD3",
        "greedy/MATD3", "rewards", "unit_ids", "act_dim",
    } - set(d.files)
    if missing:
        raise RuntimeError(f"{path.name} is missing {sorted(missing)}")
    if int(d["seed"]) != seed or str(d["label"]) != case:
        raise RuntimeError(f"{path.name} carries the wrong seed or label")
    n_agents = len(d["unit_ids"])
    if d["critic_q/MATD3"].shape[0] != n_agents or d["rewards"].shape[1] != n_agents:
        raise RuntimeError(f"{path.name} has an agent axis inconsistent with unit_ids")
    if len(d["steps"]) != d["rewards"].shape[0]:
        raise RuntimeError(f"{path.name} has fewer reward rows than frames -- partial run")


def launch(case: str, seed: int, args) -> tuple[str, int, int, float, Path]:
    out = result_path(args.out_dir, case, seed)
    if out.exists() and not args.rerun:
        validate_result(out, case, seed)
        return case, seed, 0, 0.0, out

    scenario = CASES[case]["scenario"]
    tag = f"{case}_seed{seed}"
    scratch = args.out_dir / "scratch" / tag
    scratch.mkdir(parents=True, exist_ok=True)
    # keyed by the case, not the scenario: example_02_single_bid carries three
    # cases in one folder and they must not share a policy directory
    relative_save = Path("learned_strategies") / f"probe_eom_{tag}"
    shutil.rmtree(INPUTS / scenario / relative_save, ignore_errors=True)
    db = scratch / "probe.db"
    db.unlink(missing_ok=True)

    overrides = dict(COMMON_OVERRIDES)
    if args.collecting is not None:
        overrides["episodes_collecting_initial_experience"] = args.collecting
    if args.validation_interval is not None:
        overrides["validation_episodes_interval"] = args.validation_interval

    stratify = []
    if args.obs_regimes:
        # resolved here rather than in the child: only the parent can read the
        # demand series and the merit order
        bands = regime_quantiles(case, *horizon(case, args.study_case))
        stratify = ["--bands", json.dumps(bands, separators=(",", ":"))]

    cmd = [
        sys.executable, str(SELF), "--child", *stratify, "--",
        "--scenario", scenario,
        "--study-case", args.study_case or CASES[case]["study_case"],
        "--n-obs", str(args.n_obs),
        "--grid", str(args.grid),
        "--every", str(args.every),
        "--seed", str(seed),
        "--threads", str(args.threads),
        "--disable-tensorboard",
        "--label", case,
        "--overrides-json", json.dumps(overrides, separators=(",", ":")),
        "--db-uri", f"sqlite:///{db}",
        "--save-path", str(relative_save),
        "--out", str(out),
    ]
    if args.episodes is not None:
        cmd += ["--episodes", str(args.episodes)]
    if args.train_freq is not None:
        cmd += ["--train-freq", args.train_freq]

    log = scratch / "run.log"
    t0 = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        fh.write(" ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.run(cmd, cwd=scratch, stdout=fh, stderr=subprocess.STDOUT)
    return case, seed, proc.returncode, time.perf_counter() - t0, out


def run(args) -> None:
    preflight(args.cases)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(c, seed) for c in args.cases for seed in args.seeds]
    print(f"\n  {len(jobs)} trials, true reward (shaping commented out), "
          f"grid {args.grid}, {args.n_obs} observations, every {args.every} blocks")
    for c in args.cases:
        print(f"    {c:<7} {CASES[c]['scenario']:<24} "
              f"study case {args.study_case or CASES[c]['study_case']}")
    print()

    done = 0
    workers = args.workers or len(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, c, seed, args) for c, seed in jobs]
        for fut in concurrent.futures.as_completed(futures):
            case, seed, rc, secs, out = fut.result()
            done += 1
            if rc == 0 and out.exists():
                try:
                    validate_result(out, case, seed)
                    status = "ok"
                except RuntimeError as exc:
                    status = f"INCOMPLETE ({exc})"
            else:
                status = f"FAILED rc={rc}"
            print(f"  [{done}/{len(jobs)}] {case} seed {seed}: {status} "
                  f"({secs / 60:.1f} min)", flush=True)


# -------------------------------------------------------------------- reading


def report(args) -> None:
    from critic_coherence import argmax_disagreement

    for case in args.cases:
        for seed in args.seeds:
            path = result_path(args.out_dir, case, seed)
            if not path.exists():
                print(f"\n{case} seed {seed}: (no results at {path.name})")
                continue
            d = np.load(path, allow_pickle=False)
            units = [str(u) for u in d["unit_ids"]]
            sweeps = [str(s) for s in d["sweeps"]]
            diag = sweeps.index("diag")
            act_dim = int(d["act_dim"])
            greedy = d["greedy/MATD3"]        # (n_agents, act_dim, n_obs, frames)
            q1 = d["critic_q/MATD3"]          # (n_agents, n_sweeps, n_obs, frames, grid)
            bids = d["critic_bids"]
            rewards = d["rewards"]
            share = act_share_from_sd(
                d["buffer_sd_obs"], d["buffer_sd_act"], int(d["unique_obs_dim"])
            )

            print(f"\n{case} ({CASES[case]['scenario']}/{CASES[case]['study_case']}) "
                  f"seed {seed}: {len(units)} learning units, act_dim {act_dim}, "
                  f"{greedy.shape[3]} frames, {int(d['steps'][-1])} critic updates, "
                  f"mean act_share {share.mean():.3f}")
            # act_dim 1 has one price per unit, so the two-bid columns collapse
            price_cols = (f"{'final infl':>11} {'final flex':>11}" if act_dim > 1
                          else f"{'final bid':>11}")
            header = (f"  {'unit':<8} {'act_share':>9} {'first bid':>10} "
                      f"{price_cols} {'argmax Q1':>10} "
                      f"{'obs disagree':>13} {'reward last':>12}")
            print(header)
            print("  " + "-" * (len(header) - 2))
            for i, u in enumerate(units):
                # min/max is how EnergyLearningStrategy assigns the two prices;
                # at act_dim 1 both reduce to the single action
                first = np.median(greedy[i, :, :, 0].min(axis=0))
                low = np.median(greedy[i, :, :, -1].min(axis=0))
                high = np.median(greedy[i, :, :, -1].max(axis=0))
                prices = (f"{low:11.1f} {high:11.1f}" if act_dim > 1
                          else f"{low:11.1f}")
                argmax = bids[np.argmax(q1[i, diag, :, -1, :], axis=1)]
                spread = float(argmax_disagreement(argmax))
                print(f"  {u:<8} {share[i]:9.3f} {first:10.1f} {prices} "
                      f"{np.median(argmax):10.1f} {spread:13.1f} "
                      f"{rewards[-1, i]:+12.4f}")
            pad = " " * len(price_cols)
            print(f"  {'TOTAL':<8} {'':>9} {'':>10} {pad} {'':>10} "
                  f"{'':>13} {rewards[-1].sum():+12.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES,
                        choices=list(CASES),
                        help="which of CASES to run; defaults to the single-bid "
                             "trio. Pass '02a 02b 02c' for the two-bid originals")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--study-case", default=None,
                        help="override every case's study case. Both folders "
                             "carry a 'tiny' case (4 days, 10 episodes), which "
                             "is the smoke test for this script")
    parser.add_argument("--episodes", type=int, default=None,
                        help="override the study case's training_episodes")
    parser.add_argument("--collecting", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--train-freq", default=None,
                        help="e.g. 24h. Finer means more frames and a bigger file; "
                             "the study cases use 100h")
    parser.add_argument("--workers", type=int, default=None,
                        help="concurrent trials; one task per trial on a cluster")
    parser.add_argument("--threads", type=int, default=1,
                        help="torch.set_num_threads in each child")
    parser.add_argument("--n-obs", type=int, default=4,
                        help="observations to film from. With --obs-regimes it "
                             "is per regime, so the film carries n_obs x "
                             "(regimes present)")
    parser.add_argument("--obs-regimes", action="store_true",
                        help="stratify the probed observations by demand regime "
                             "(idle / bertrand / pivotal), so the critic film can "
                             "be read separately for each Nash equilibrium. See "
                             "REGIMES and merit_order()")
    parser.add_argument("--grid", type=int, default=201)
    parser.add_argument("--every", type=int, default=4,
                        help="snapshot every N-th training block")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--out-dir", type=Path,
        default=OUT_DIR / "runs" / "data" / "14-eom-critic-evolution",
    )

    if "--child" in sys.argv:
        i = sys.argv.index("--child")
        # the parent resolves the quantile bands (it can read the demand series;
        # the recorder cannot) and hands them over as JSON
        bands = None
        if "--bands" in sys.argv[:i + 2]:
            bands = json.loads(sys.argv[sys.argv.index("--bands") + 1])
        run_child(sys.argv[sys.argv.index("--", i) + 1:], bands)
        return

    args = parser.parse_args()
    if not args.report_only:
        run(args)
    report(args)


if __name__ == "__main__":
    main()
