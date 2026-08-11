# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the parts of this benchmark whose failure would be **silent**.

Most of the code here produces a figure, and a figure that is wrong usually looks
wrong. These three groups cover the places where a wrong answer would look
entirely plausible and would change a conclusion in ``RUNS.md``:

0. ``MultiAgentRecorder`` vs ``matd3.TD3.update_policy`` -- the recorder films the
   critic by assembling its input by hand, following a convention it can only
   describe by quoting line numbers in ``assume/``. Every run 13 number is a
   function of that assembly. If ``matd3.py`` ever reorders the observation
   blocks, changes which agents' ``unique_obs_dim`` slice goes where, or flattens
   the action vector differently, the films keep rendering -- from the wrong
   input. These tests run one real gradient step through the real ``update_policy``
   with the real networks, capture the tensors the critic was actually handed, and
   assert the recorder reproduces them exactly.
1. ``install_action_scale`` -- run 13's lever. It scales one column of the
   critic's action input, and the column width it computes depends on
   ``CriticTD3.act_dim`` meaning ``act_dim * n_agents``. If that ever became the
   per-agent dimension, ``width`` would be ``1 // 11 == 0``, the slice would be
   empty, and ``act-own-x15`` would silently become a second copy of ``baseline``
   with no error anywhere. Finding 20 -- the own/block dissociation, and the whole
   reason run 13 needed a new lever -- rests on this one integer division.
2. ``act_share_from_sd`` / ``action_block_share`` -- the *independent variable* of
   runs 12 and 13. The formula is hand-rolled, its denominator has to match the
   critic's real input (own observation + every other agent's ``unique_obs_dim``
   + every action), and nothing else checks it. Here it is compared against the
   concatenated input matrix built the way ``matd3.py:585-591`` builds it.
3. ``critic_coherence`` -- the observation-disagreement statistic. It was
   reimplemented in five scripts with **two different definitions** (a range in
   runs 10 and 11, a mean pairwise difference in runs 12 and 13), and ``RUNS.md``
   once quoted one against the other. Both live in one module now; these tests pin
   each definition against a hand-computed value and assert the two stay distinct,
   so a future "simplification" cannot quietly merge them again.
4. ``MultiAgentRecorder.REWARD_WINDOW`` -- one episode of transitions, which is
   *not* the horizon in hours. It was 62 (inherited from the single-agent study
   case) against an actual 69 for ``inc_dec_learning``. Both numbers are derived
   here from ``config.yaml`` rather than asserted as literals, so changing a
   horizon or a market opening time fails loudly instead of quietly biasing the
   reward trace.

Run with::

    conda run -n assume python -m pytest examples/inputs/2_nodes_paper_small/rl_benchmark/test_rl_benchmark.py -v

The repository's own ``testpaths`` is ``tests/``, so these do not run with the
library suite; they are meant to be run from this folder, like everything else
here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _layout import SCENARIO  # noqa: E402  (also puts the four folders on sys.path)
from assume_multiagent_actshare import (  # noqa: E402
    MAX_BID_PRICE,
    MultiAgentRecorder,
    act_share_from_sd,
    action_block_share,
    install_action_scale,
    scale_for_target,
)
from critic_coherence import (  # noqa: E402
    argmax_disagreement,
    argmax_range,
    peak_bids,
)

th = pytest.importorskip("torch", reason="the levers patch torch modules")

from assume.reinforcement_learning.algorithms.matd3 import TD3  # noqa: E402
from assume.reinforcement_learning.neural_network_architecture import (  # noqa: E402
    CriticTD3,
)

CONFIG = SCENARIO / "config.yaml"

#: the shapes run 13 actually ran at
N_AGENTS = 11
OBS_DIM = 74
UNIQUE_OBS_DIM = 2
ACT_DIM = 1


# --------------------------------------------------------------------------- #
# fixtures


@pytest.fixture
def pristine_classes():
    """Undo the class-level monkeypatches the levers install.

    ``install_action_scale`` rebinds ``CriticTD3.forward``, ``CriticTD3.q1_forward``
    and ``TD3.update_policy`` on the *class*, which leaks into every later test in
    the same process. Yields the untouched functions so a test can compute the
    reference value the patch is supposed to reproduce.
    """
    saved = {
        "forward": CriticTD3.forward,
        "q1_forward": CriticTD3.q1_forward,
        "update_policy": TD3.update_policy,
    }
    yield saved
    CriticTD3.forward = saved["forward"]
    CriticTD3.q1_forward = saved["q1_forward"]
    TD3.update_policy = saved["update_policy"]


def make_critic(n_agents: int = N_AGENTS) -> CriticTD3:
    return CriticTD3(
        n_agents=n_agents,
        obs_dim=OBS_DIM,
        act_dim=ACT_DIM,
        unique_obs_dim=UNIQUE_OBS_DIM,
        float_type=th.float32,
    )


def random_batch(n_agents: int = N_AGENTS, batch: int = 8):
    critic = make_critic(n_agents)
    gen = th.Generator().manual_seed(0)
    obs = th.rand(batch, critic.obs_dim, generator=gen)
    act = th.rand(batch, critic.act_dim, generator=gen) * 2 - 1
    return critic, obs, act


# --------------------------------------------------------------------------- #
# 0. the recorder against the real update_policy


@pytest.fixture
def harness(pristine_classes):
    """One real MATD3 gradient step, with every critic and actor call recorded.

    Everything that touches the layout under test is the real thing: ``CriticTD3``,
    ``MLPActor``, ``ReplayBuffer`` and ``TD3.update_policy`` itself. Only the
    scaffolding ``update_policy`` reaches for -- the schedules, the noise object,
    the output writer -- is stubbed, and the replay sample is pinned so the batch
    is known rather than drawn.

    Deliberately built at N = 3 with an asymmetric ``obs_dim``: at N = 11 an
    off-by-one in the "every other agent's unique block" concatenation can still
    land inside the right array, and equal-sized blocks hide a transposition.
    """
    from torch.optim import AdamW

    from assume.common.base import LearningConfig
    from assume.reinforcement_learning.buffer import ReplayBuffer, ReplayBufferSamples
    from assume.reinforcement_learning.neural_network_architecture import MLPActor

    n_agents, obs_dim, unique_obs_dim, act_dim, batch = 3, 11, 2, 1, 5
    th.manual_seed(0)
    rng = np.random.default_rng(0)

    calls: list[tuple[str, object, tuple]] = []

    ref_forward = pristine_classes["forward"]
    ref_q1 = pristine_classes["q1_forward"]
    ref_actor = MLPActor.forward

    def spy_forward(self, obs, actions):
        calls.append(("critic", self, (obs.detach().clone(), actions.detach().clone())))
        return ref_forward(self, obs, actions)

    def spy_q1(self, obs, actions):
        calls.append(("q1", self, (obs.detach().clone(), actions.detach().clone())))
        return ref_q1(self, obs, actions)

    def spy_actor(self, obs):
        out = ref_actor(self, obs)
        calls.append(("actor", self, (obs.detach().clone(), out.detach().clone())))
        return out

    CriticTD3.forward = spy_forward
    CriticTD3.q1_forward = spy_q1
    MLPActor.forward = spy_actor

    class Strategy:
        def __init__(self, unit_id):
            self.unit_id = unit_id
            self.actor = MLPActor(obs_dim, act_dim, th.float32)
            self.actor_target = MLPActor(obs_dim, act_dim, th.float32)
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.actor.optimizer = AdamW(self.actor.parameters(), lr=1e-3)
            self.critics = CriticTD3(
                n_agents=n_agents, obs_dim=obs_dim, act_dim=act_dim,
                unique_obs_dim=unique_obs_dim, float_type=th.float32,
            )
            self.target_critics = CriticTD3(
                n_agents=n_agents, obs_dim=obs_dim, act_dim=act_dim,
                unique_obs_dim=unique_obs_dim, float_type=th.float32,
            )
            self.target_critics.load_state_dict(self.critics.state_dict())
            self.critics.optimizer = AdamW(self.critics.parameters(), lr=1e-3)
            self.action_noise = type("Noise", (), {"update_noise_decay": lambda s, v: None})()

    strategies = {f"unit_{i}": Strategy(f"unit_{i}") for i in range(n_agents)}

    buffer = ReplayBuffer(
        buffer_size=64, obs_dim=obs_dim, act_dim=act_dim,
        n_rl_units=n_agents, device="cpu", float_type=np.float32,
    )
    # distinctive per-agent scales, so a swapped agent axis cannot pass unnoticed
    obs_block = rng.normal(size=(20, n_agents, obs_dim)).astype(np.float32)
    obs_block *= (1.0 + np.arange(n_agents, dtype=np.float32))[None, :, None]
    act_block = rng.uniform(-1, 1, size=(20, n_agents, act_dim)).astype(np.float32)
    buffer.add(obs_block, act_block, rng.normal(size=(20, n_agents, 1)).astype(np.float32))

    # a pinned sample, so the batch update_policy sees is the batch we assert on
    idx = np.arange(batch)
    sample = ReplayBufferSamples(
        th.as_tensor(obs_block[idx]),
        th.as_tensor(act_block[idx]),
        th.as_tensor(obs_block[idx + 1]),
        th.as_tensor(rng.normal(size=(batch, n_agents)).astype(np.float32)),
    )
    buffer.sample = lambda _batch_size: sample

    class Role:
        def __init__(self):
            self.rl_strats = strategies
            self.buffer = buffer

        get_progress_remaining = staticmethod(lambda: 1.0)
        calc_lr_from_progress = staticmethod(lambda _p: 1e-3)
        calc_noise_from_progress = staticmethod(lambda _p: 1.0)
        write_rl_grad_params_to_output = staticmethod(lambda *_a, **_k: None)

    algorithm = TD3.__new__(TD3)
    algorithm.learning_role = Role()
    algorithm.learning_config = LearningConfig(
        gradient_steps=1, batch_size=batch, policy_delay=1,
        gamma=0.99, tau=0.005, target_policy_noise=0.2, target_noise_clip=0.5,
    )
    algorithm.obs_dim = obs_dim
    algorithm.act_dim = act_dim
    algorithm.unique_obs_dim = unique_obs_dim
    algorithm.n_updates = 0
    algorithm.grad_clip_norm = 1.0

    TD3.update_policy(algorithm)

    MLPActor.forward = ref_actor  # the class fixture only restores CriticTD3/TD3

    return {
        "algorithm": algorithm,
        "strategies": list(strategies.values()),
        "obs": obs_block[idx],
        "next_obs": obs_block[idx + 1],
        "actions": act_block[idx],
        "calls": calls,
        "n_agents": n_agents,
        "unique_obs_dim": unique_obs_dim,
        "batch": batch,
    }


def recorder_for(harness, grid: int = 5) -> MultiAgentRecorder:
    """A recorder primed with the same joint observations update_policy just saw."""
    rec = MultiAgentRecorder(np.empty((harness["batch"], 0)), grid=grid, every=1)
    rec.obs = harness["obs"].astype(np.float64)
    rec.unique_obs_dim = harness["unique_obs_dim"]
    rec.unit_ids = [s.unit_id for s in harness["strategies"]]
    return rec


def calls_on(harness, kind: str, module) -> list[tuple]:
    return [payload for k, obj, payload in harness["calls"] if k == kind and obj is module]


def test_recorder_observation_assembly_matches_the_critic_update(harness):
    """``_critic_states(i)`` must equal the ``all_states`` matd3 hands critic *i*.

    This is the assertion the whole run 13 archive rests on. The recorder builds
    "own full observation, then every other agent's last ``unique_obs_dim``, in
    index order skipping *i*" from a comment quoting ``matd3.py:585-591``; here it
    is checked against what that code actually did, on the same batch.
    """
    rec = recorder_for(harness)
    for i, strategy in enumerate(harness["strategies"]):
        online = calls_on(harness, "critic", strategy.critics)
        assert online, f"critic {i} was never called during the critic update"
        obs_seen, _ = online[0]

        expected = rec._critic_states(i)
        assert obs_seen.shape == expected.shape
        np.testing.assert_allclose(obs_seen.numpy(), expected, rtol=0, atol=1e-6)


def test_recorder_assembly_is_not_accidentally_symmetric(harness):
    """The agents must not all get the same input, or the test above proves nothing.

    With equal-sized observation blocks a wrong concatenation order can still
    produce the right shape and, if every agent's observation were alike, the
    right values. The per-agent scaling in the fixture rules that out; this makes
    the guarantee explicit.
    """
    rec = recorder_for(harness)
    states = [rec._critic_states(i) for i in range(harness["n_agents"])]
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            assert not np.allclose(states[i], states[j])


def test_target_critic_sees_the_same_layout_on_next_states(harness):
    """The bootstrap target uses the identical assembly, on ``next_observations``.

    Worth pinning separately: ``update_policy`` builds ``all_next_states`` in a
    second, near-duplicated block, so the two can drift apart from each other as
    well as from the recorder.
    """
    rec = recorder_for(harness)
    rec.obs = harness["next_obs"].astype(np.float64)
    for i, strategy in enumerate(harness["strategies"]):
        target = calls_on(harness, "critic", strategy.target_critics)
        assert target, f"target critic {i} was never called"
        obs_seen, _ = target[0]
        np.testing.assert_allclose(
            obs_seen.numpy(), rec._critic_states(i), rtol=0, atol=1e-6
        )


def test_agent_i_owns_action_column_i(harness):
    """The flattened action vector is agent-major, and the recorder writes column i.

    ``matd3`` replaces ``all_actions_clone[:, i, :]`` with actor *i*'s output and
    then flattens, so agent *i* owns columns ``[i*act_dim, (i+1)*act_dim)``. The
    recorder writes its bid grid into column ``i`` and reads ``grad[:, i]`` back.
    If the flattening ever became agent-minor, both would silently address another
    agent's action -- the films would still render, of the wrong unit.
    """
    for i, strategy in enumerate(harness["strategies"]):
        actor_calls = calls_on(harness, "actor", strategy.actor)
        assert actor_calls, f"actor {i} was never called"
        # the last actor call before the optimizer step is the actor-loss one
        _, actor_out = actor_calls[-1]

        q1_calls = calls_on(harness, "q1", strategy.critics)
        assert q1_calls, "the actor update did not reach q1_forward"
        _, actions_seen = q1_calls[-1]

        assert actions_seen.shape == (harness["batch"], harness["n_agents"])
        np.testing.assert_allclose(
            actions_seen[:, i].numpy(), actor_out.ravel().numpy(), rtol=0, atol=1e-6
        )
        # and every other column is still the stored behaviour action
        for j in range(harness["n_agents"]):
            if j == i:
                continue
            np.testing.assert_allclose(
                actions_seen[:, j].numpy(), harness["actions"][:, j, 0], rtol=0, atol=1e-6
            )


def test_recorder_sweeps_its_own_column_consistently(harness):
    """End to end: the recorded ``Q1`` is ``q1_forward`` at the swept own action.

    Combines the two assertions above -- right observation block, right action
    column -- into the value the film actually stores, so a change that broke both
    consistently could not cancel out.

    Note what this does **not** say: that the recorded surface is matd3's actor
    objective. It is not, at N > 1 -- see
    ``test_recorded_field_is_not_the_actor_objective_at_many_agents``.
    """
    rec = recorder_for(harness, grid=5)
    algorithm = harness["algorithm"]
    rec.snapshot(algorithm)

    q1_frame = rec.q1[-1]                     # (agents, obs, bids)
    assert q1_frame.shape == (harness["n_agents"], harness["batch"], 5)

    with th.no_grad():
        base = th.stack(
            [
                s.actor(th.as_tensor(harness["obs"][:, j, :], dtype=th.float32))
                for j, s in enumerate(harness["strategies"])
            ],
            dim=1,
        ).reshape(harness["batch"], -1)

    for i, strategy in enumerate(harness["strategies"]):
        states = th.as_tensor(rec._critic_states(i), dtype=th.float32)
        for b, bid in enumerate(rec.bids):
            acts = base.clone()
            acts[:, i] = float(bid) / MAX_BID_PRICE
            with th.no_grad():
                expected = strategy.critics.q1_forward(states, acts).ravel()
            np.testing.assert_allclose(
                q1_frame[i, :, b], expected.numpy(), rtol=1e-5, atol=1e-6
            )


def test_recorded_field_is_not_the_actor_objective_at_many_agents(harness):
    """Pin a known, deliberate divergence so it cannot be misread as parity.

    ``matd3`` builds the actor loss by cloning the **replay batch's** actions and
    replacing only column *i* (``matd3.py:704``), so the other agents sit at the
    behaviour actions that were actually stored. ``MultiAgentRecorder`` instead
    holds them at their **current actors' greedy outputs**. Both are legitimate
    slices of the same critic, and the recorder documents which one it takes -- but
    they are different surfaces, so run 13's window, ``pulled left`` and
    coherence readings describe the critic's response to the *current joint
    policy*, not the quantity the actor's gradient step is computed from.

    At N = 1 the distinction is empty: there are no other columns, so the recorder
    reproduces the actor objective exactly. That scopes this to run 13 and leaves
    runs 09-12 untouched.
    """
    n_agents = harness["n_agents"]
    assert n_agents > 1, "the divergence only exists with more than one agent"

    for i, strategy in enumerate(harness["strategies"]):
        _, actor_loss_actions = calls_on(harness, "q1", strategy.critics)[-1]

        rec = recorder_for(harness)
        with th.no_grad():
            recorder_actions = th.stack(
                [
                    s.actor(th.as_tensor(harness["obs"][:, j, :], dtype=th.float32))
                    for j, s in enumerate(harness["strategies"])
                ],
                dim=1,
            ).reshape(harness["batch"], -1)
        # put the same own action in both, so only the *other* columns differ
        recorder_actions[:, i] = actor_loss_actions[:, i]

        others = [j for j in range(n_agents) if j != i]
        assert not th.allclose(
            actor_loss_actions[:, others], recorder_actions[:, others], atol=1e-4
        ), "the two conventions coincided; this test's premise has changed"

        states = th.as_tensor(rec._critic_states(i), dtype=th.float32)
        with th.no_grad():
            q_actor = strategy.critics.q1_forward(states, actor_loss_actions)
            q_recorder = strategy.critics.q1_forward(states, recorder_actions)
        assert not th.allclose(q_actor, q_recorder, atol=1e-6), (
            "the difference is not merely cosmetic -- it moves Q1"
        )


def test_recorded_greedy_bid_is_the_actors_own_output(harness):
    """The bid trace must be actor *j*'s output on *j*'s own observation, in EUR."""
    rec = recorder_for(harness)
    rec.snapshot(harness["algorithm"])
    recorded = rec.actor_bids[-1]                     # (obs, agents)

    for j, strategy in enumerate(harness["strategies"]):
        with th.no_grad():
            expected = strategy.actor(
                th.as_tensor(harness["obs"][:, j, :], dtype=th.float32)
            ).ravel() * MAX_BID_PRICE
        np.testing.assert_allclose(recorded[:, j], expected.numpy(), rtol=1e-5, atol=1e-5)


def test_recorder_refuses_a_multi_dimensional_action(harness):
    """The sweep is one bid axis per agent, so ``act_dim > 1`` must raise, not alias.

    Silently sweeping only the first of several action dimensions would produce a
    film that looks entirely normal.
    """
    harness["algorithm"].act_dim = 2
    with pytest.raises(RuntimeError, match="act_dim"):
        recorder_for(harness).snapshot(harness["algorithm"])


# --------------------------------------------------------------------------- #
# 1. the run 13 lever


def test_critic_act_dim_is_the_whole_action_vector():
    """The premise ``width = act_dim // n_agents == 1`` depends on.

    ``CriticTD3`` stores ``act_dim * n_agents``, i.e. the width of the *joint*
    action input, not the per-agent action width. ``install_action_scale`` divides
    by ``_n_agents`` to recover the per-agent width; if this convention flipped,
    that division would floor to zero and the own-action lever would become a
    no-op. This is the assertion that would catch it.
    """
    critic = make_critic()
    assert critic.act_dim == ACT_DIM * N_AGENTS
    assert critic.obs_dim == OBS_DIM + UNIQUE_OBS_DIM * (N_AGENTS - 1)
    # the 105 inputs run 13's docstring quotes
    assert critic.obs_dim + critic.act_dim == 105
    assert critic.act_dim // N_AGENTS == ACT_DIM


@pytest.mark.parametrize("scale", [2.0, 15.0])
def test_own_mode_scales_exactly_the_stamped_column(pristine_classes, scale):
    """``mode="own"`` must fit ``Q_i(s, a_1, ..., S*a_i, ..., a_N)`` and nothing else."""
    critic, obs, act = random_batch()
    ref_forward, ref_q1 = pristine_classes["forward"], pristine_classes["q1_forward"]

    install_action_scale(scale, "own")
    critic._n_agents = N_AGENTS

    for i in range(N_AGENTS):
        critic._own_action_index = i

        expected_input = act.clone()
        expected_input[:, i] *= scale

        q1_ref = ref_q1(critic, obs, expected_input)
        q1_patched = critic.q1_forward(obs, act)
        assert th.equal(q1_patched, q1_ref), f"q1_forward wrong for agent {i}"

        both_ref = ref_forward(critic, obs, expected_input)
        both_patched = critic.forward(obs, act)
        assert th.equal(both_patched[0], both_ref[0])
        assert th.equal(both_patched[1], both_ref[1])

        # and it must NOT be the same as leaving the action alone, or as scaling
        # the whole vector -- the two failure modes that look identical in a plot
        assert not th.equal(q1_patched, ref_q1(critic, obs, act))
        assert not th.equal(q1_patched, ref_q1(critic, obs, act * scale))


def test_own_mode_is_inert_until_the_index_is_stamped(pristine_classes):
    """Without ``_own_action_index`` the patch is a no-op, by design.

    The index is stamped per ``update_policy`` call because the world -- and with
    it the networks -- is rebuilt every episode. Pinning this behaviour makes the
    ordering test below meaningful: if the patch silently scaled by S with no
    index, a stamping bug would be invisible.
    """
    critic, obs, act = random_batch()
    ref_q1 = pristine_classes["q1_forward"]

    install_action_scale(15.0, "own")
    assert not hasattr(critic, "_own_action_index")
    assert th.equal(critic.q1_forward(obs, act), ref_q1(critic, obs, act))


def test_all_mode_reproduces_run_12s_lever(pristine_classes):
    """``mode="all"`` is run 12's patch verbatim: ``Q(s, S*a)`` over the whole vector."""
    critic, obs, act = random_batch()
    ref_q1 = pristine_classes["q1_forward"]

    install_action_scale(15.0, "all")
    # no index stamped, and none needed
    assert th.equal(critic.q1_forward(obs, act), ref_q1(critic, obs, act * 15.0))


def test_own_mode_reduces_to_all_mode_at_one_agent(pristine_classes):
    """At N = 1 the two levers must coincide -- that is run 12's claim of continuity."""
    critic, obs, act = random_batch(n_agents=1)
    ref_q1 = pristine_classes["q1_forward"]

    install_action_scale(7.0, "own")
    critic._own_action_index, critic._n_agents = 0, 1
    assert th.equal(critic.q1_forward(obs, act), ref_q1(critic, obs, act * 7.0))


def test_indices_are_stamped_before_the_update_and_survive_it(pristine_classes):
    """Stamping must precede everything downstream, including the recorder's snapshot.

    ``assume_training_probe.install()`` wraps ``update_policy`` *outside* this
    patch, so the call order is: stamp -> real update -> snapshot. If the stamping
    ever moved after the original call, the first frame of every ``act-own`` run
    would be recorded from an unscaled critic and the crossing time would be wrong.
    """
    n_agents = 3

    class FakeNet:
        pass

    class FakeStrategy:
        def __init__(self):
            self.critics = FakeNet()
            self.target_critics = FakeNet()

    class FakeRole:
        def __init__(self):
            self.rl_strats = {f"u{i}": FakeStrategy() for i in range(n_agents)}

    class FakeAlgorithm:
        def __init__(self):
            self.learning_role = FakeRole()

    seen: dict[str, list] = {}

    def spy(self):
        seen["at_update"] = [
            (s.critics._own_action_index, s.critics._n_agents,
             s.target_critics._own_action_index)
            for s in self.learning_role.rl_strats.values()
        ]

    TD3.update_policy = spy
    install_action_scale(15.0, "own")

    algorithm = FakeAlgorithm()
    TD3.update_policy(algorithm)

    assert seen["at_update"] == [(0, n_agents, 0), (1, n_agents, 1), (2, n_agents, 2)]
    # still set afterwards, which is when the probe takes its frame
    after = [s.critics._own_action_index
             for s in algorithm.learning_role.rl_strats.values()]
    assert after == [0, 1, 2]


def test_all_mode_does_not_wrap_update_policy(pristine_classes):
    """``mode="all"`` needs no index, so it must leave the update path untouched.

    Wrapping it anyway would add a per-call loop to every gradient step of the
    ``act-all`` conditions and make them non-comparable with ``act-own`` for
    reasons that have nothing to do with ``act_share``.
    """
    original = TD3.update_policy
    install_action_scale(15.0, "all")
    assert TD3.update_policy is original


# --------------------------------------------------------------------------- #
# 2. act_share, against the critic input matrix it describes


def brute_force_shares(
    obs: np.ndarray, act: np.ndarray, unique_obs_dim: int, scale: float, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build critic *i*'s real input matrix and take per-column standard deviations.

    Independent of ``act_share_from_sd``: this concatenates the columns the way
    ``matd3.py:585-591`` does (own full observation, then every other agent's last
    ``unique_obs_dim`` in index order skipping *i*, then all ``N`` actions), scales
    whichever action columns the lever scales, and measures the result. Any error
    in the analytic formula's denominator -- a missing block, a double-counted own
    unique block, the wrong number of other agents -- shows up as a mismatch.
    """
    n_agents = obs.shape[1]
    own_shares, block_shares = [], []
    for i in range(n_agents):
        others = np.concatenate(
            [obs[:, j, -unique_obs_dim:] for j in range(n_agents) if j != i], axis=1
        )
        states = np.concatenate([obs[:, i, :], others], axis=1)

        actions = act[:, :, 0].copy()
        if mode == "all":
            actions = actions * scale
        else:
            actions[:, i] *= scale

        matrix = np.concatenate([states, actions], axis=1)
        sd = matrix.std(axis=0)
        total = sd.sum()
        own_shares.append(sd[states.shape[1] + i] / total)
        block_shares.append(sd[states.shape[1]:].sum() / total)
    return np.array(own_shares), np.array(block_shares)


def synthetic_buffer(n_steps: int = 400, n_agents: int = N_AGENTS, seed: int = 0):
    """A buffer with deliberately *unequal* per-agent action spread.

    Equal spreads would hide an indexing error -- every agent would get the same
    answer whichever column the formula picked. Run 13's real buffer is unequal
    too (sd(a) ranges 0.34-0.63 across the eleven units).
    """
    rng = np.random.default_rng(seed)
    obs = rng.normal(0.5, 0.2, size=(n_steps, n_agents, OBS_DIM))
    # give each observation dimension its own spread as well
    obs *= rng.uniform(0.3, 2.0, size=(1, n_agents, OBS_DIM))
    act = rng.uniform(-1, 1, size=(n_steps, n_agents, ACT_DIM))
    act *= rng.uniform(0.4, 1.0, size=(1, n_agents, 1))
    return obs, act


@pytest.mark.parametrize("mode", ["own", "all"])
@pytest.mark.parametrize("scale", [1.0, 2.0, 15.0])
def test_act_share_matches_the_real_critic_input(mode, scale):
    obs, act = synthetic_buffer()
    sd_obs = obs.std(axis=0)
    sd_act = act.std(axis=0).ravel()

    expected_own, expected_block = brute_force_shares(
        obs, act, UNIQUE_OBS_DIM, scale, mode
    )
    got_own = act_share_from_sd(sd_obs, sd_act, UNIQUE_OBS_DIM, scale, mode)
    got_block = action_block_share(sd_obs, sd_act, UNIQUE_OBS_DIM, scale, mode)

    np.testing.assert_allclose(got_own, expected_own, rtol=1e-12)
    np.testing.assert_allclose(got_block, expected_block, rtol=1e-12)
    # per-agent, not just on average -- the mean would survive a transposed index
    assert got_own.shape == (N_AGENTS,)
    assert len(set(np.round(got_own, 6))) > 1, "unequal spreads must give unequal shares"


def test_share_denominator_has_one_term_per_critic_input():
    """The share is over the critic's inputs, so the column count must match the net."""
    obs, act = synthetic_buffer()
    others = np.concatenate(
        [obs[:, j, -UNIQUE_OBS_DIM:] for j in range(N_AGENTS) if j != 0], axis=1
    )
    states = np.concatenate([obs[:, 0, :], others], axis=1)
    critic = make_critic()
    assert states.shape[1] == critic.obs_dim
    assert states.shape[1] + N_AGENTS == critic.obs_dim + critic.act_dim


def test_act_share_at_one_agent_is_run_12s_definition():
    """``act_share = sd(a) / (sd(a) + sum_j sd(obs_j))`` -- the number RUNS.md quotes."""
    obs, act = synthetic_buffer(n_agents=1)
    sd_obs, sd_act = obs.std(axis=0), act.std(axis=0).ravel()

    for scale in (1.0, 30.0):
        expected = (scale * sd_act[0]) / (scale * sd_act[0] + sd_obs[0].sum())
        for mode in ("own", "all"):
            got = act_share_from_sd(sd_obs, sd_act, UNIQUE_OBS_DIM, scale, mode)
            np.testing.assert_allclose(got, [expected], rtol=1e-12)


def test_scaling_every_action_caps_the_own_share_at_one_over_n():
    """Finding 20's reason the run 12 lever had to be replaced.

    Under ``act-all`` an agent's own share tends to ``sd(a_i) / sum_j sd(a_j)``,
    which is 1/N for equal spreads and cannot reach the 0.2 run 13 asked for.
    ``act-own`` has no such ceiling. If this inverted, the whole justification for
    the second lever would go with it.
    """
    obs, act = synthetic_buffer()
    sd_obs, sd_act = obs.std(axis=0), act.std(axis=0).ravel()

    huge = act_share_from_sd(sd_obs, sd_act, UNIQUE_OBS_DIM, 1e9, "all")
    np.testing.assert_allclose(huge, sd_act / sd_act.sum(), rtol=1e-6)
    assert huge.mean() < 1.0 / N_AGENTS + 1e-6

    # equal spreads: the cap is exactly 1/N
    equal = np.full(N_AGENTS, 0.5)
    np.testing.assert_allclose(
        act_share_from_sd(sd_obs, equal, UNIQUE_OBS_DIM, 1e9, "all"),
        np.full(N_AGENTS, 1.0 / N_AGENTS),
        rtol=1e-6,
    )
    # act-own has no ceiling
    assert act_share_from_sd(sd_obs, sd_act, UNIQUE_OBS_DIM, 1e6, "own").mean() > 0.99


def test_the_two_levers_move_own_and_block_share_in_opposite_proportions():
    """The dissociation finding 20 is built on, stated as an ordering.

    At a matched *block* share, ``act-own`` must carry the higher *own* share --
    that is the only thing separating "the critic must notice the action block"
    from "the critic must notice its own actor's action".
    """
    obs, act = synthetic_buffer()
    sd_obs, sd_act = obs.std(axis=0), act.std(axis=0).ravel()

    def shares(scale, mode):
        return (
            act_share_from_sd(sd_obs, sd_act, UNIQUE_OBS_DIM, scale, mode).mean(),
            action_block_share(sd_obs, sd_act, UNIQUE_OBS_DIM, scale, mode).mean(),
        )

    own_at_15, block_at_15 = shares(15.0, "own")
    # find the act-all scale with the same action-block share
    lo, hi = 1.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        if shares(mid, "all")[1] < block_at_15:
            lo = mid
        else:
            hi = mid
    own_matched, block_matched = shares((lo + hi) / 2, "all")

    np.testing.assert_allclose(block_matched, block_at_15, rtol=1e-6)
    assert own_at_15 > own_matched, (
        "at a matched action-block share act-own must still carry the larger own "
        "share, or run 13's control condition proves nothing"
    )


def test_scale_for_target_inverts_act_share():
    obs, act = synthetic_buffer()
    sd_obs, sd_act = obs.std(axis=0), act.std(axis=0).ravel()
    for target in (0.05, 0.2, 0.4):
        s = scale_for_target(sd_obs, sd_act, UNIQUE_OBS_DIM, target, "own")
        got = act_share_from_sd(sd_obs, sd_act, UNIQUE_OBS_DIM, s, "own").mean()
        assert abs(got - target) < 1e-6


# --------------------------------------------------------------------------- #
# 3. the coherence statistic, defined once


def test_disagreement_is_the_mean_over_distinct_pairs():
    """Pinned against a value that can be checked by hand.

    For ``[0, 10, 20]`` the three distinct pairs differ by 10, 10 and 20, so the
    mean is 40/3 = 13.33 -- not 40/9 = 4.44, which is what dividing by ``n**2``
    and counting the self-pairs would give, and not 20, which is the range.
    """
    peaks = np.array([0.0, 10.0, 20.0])
    assert argmax_disagreement(peaks) == pytest.approx(40.0 / 3.0)
    assert argmax_range(peaks) == pytest.approx(20.0)

    # the superseded n^2 normalisation, for the record: it is (n-1)/n of this
    old = float(np.mean([abs(a - b) for a in peaks for b in peaks]))
    assert old == pytest.approx(argmax_disagreement(peaks) * 2 / 3)


def test_disagreement_and_range_are_not_interchangeable():
    """The whole point of correction 14: quoting one where the other was measured.

    They agree only for two observations. At the six this archive probes, a wide
    spread reads about twice as high as a range -- which is exactly the size of the
    error that made run 12's ``baseline`` look more coherent than run 10's
    unshaped critic when it is in fact the same failure.
    """
    two = np.array([10.0, 40.0])
    assert argmax_disagreement(two) == pytest.approx(argmax_range(two))

    rng = np.random.default_rng(0)
    six = rng.uniform(0, 100, size=6)
    assert argmax_disagreement(six) < argmax_range(six)


def test_disagreement_is_independent_of_how_many_observations_were_probed():
    """Averaging over distinct pairs is what makes two runs comparable.

    Probing the same critic at more observations must not change the answer. The
    ``n**2`` version fails this: it scales by ``(n-1)/n``, so a run at
    ``--n-obs 4`` and one at ``--n-obs 12`` would not have been comparable.
    """
    base = np.array([0.0, 30.0])
    for repeats in (1, 2, 5):
        peaks = np.tile(base, repeats)
        assert argmax_disagreement(peaks) == pytest.approx(
            30.0 * repeats / (2 * repeats - 1)
        )
        old = float(np.mean([abs(a - b) for a in peaks for b in peaks]))
        assert old == pytest.approx(15.0)  # the n^2 version is flat at n/2 pairs

    # and on a genuinely resampled set the distinct-pair mean is stable
    rng = np.random.default_rng(1)
    draws = [argmax_disagreement(rng.uniform(0, 100, size=n)) for n in (200, 400, 800)]
    assert max(draws) - min(draws) < 4.0


def test_disagreement_broadcasts_over_a_film():
    """The figures apply it to ``(agents, obs, frames)`` arrays, one value per frame."""
    rng = np.random.default_rng(2)
    peaks = rng.uniform(0, 100, size=(11, 6, 30))

    got = argmax_disagreement(peaks, axis=1)
    assert got.shape == (11, 30)
    # spot-check one cell against the one-dimensional path
    assert got[3, 7] == pytest.approx(argmax_disagreement(peaks[3, :, 7]))

    assert argmax_range(peaks, axis=1).shape == (11, 30)


def test_disagreement_of_a_single_observation_is_undefined_not_zero():
    """Zero would read as "perfectly coherent", which is the opposite of unknown."""
    assert np.isnan(argmax_disagreement(np.array([42.0])))
    assert np.isnan(argmax_disagreement(np.zeros((1, 4)), axis=0)).all()


def test_every_script_uses_the_shared_definition():
    """No call site may reimplement it -- that is how the two versions diverged.

    Two shapes are banned: the inline double comprehension
    ``np.mean([abs(a - b) for a in x for b in x])`` that runs 12 and 13 used, and
    the broadcast form ``abs(p[:, :, None] - p[:, None, :]).sum(...)`` that the two
    figure scripts used. If either reappears, the archive is quoting two
    statistics again.
    """
    import re

    inline = re.compile(r"abs\(\s*\w+\s*-\s*\w+\s*\)\s+for\s+\w+\s+in\b.*\bfor\s+\w+\s+in\b")
    broadcast = re.compile(r"\[:,\s*:,\s*None\b.*-\s*\w+\[:,\s*None\b")

    offenders = []
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        if path.name == Path(__file__).name or path.name == "critic_coherence.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if inline.search(line) or broadcast.search(line):
                offenders.append(f"{path.relative_to(Path(__file__).parent)}:{lineno}")

    assert not offenders, (
        f"a pairwise-disagreement loop was reimplemented at {offenders}; import "
        "argmax_disagreement from analysis/critic_coherence.py instead"
    )


def test_peak_bids_reads_the_grid_axis():
    bids = np.linspace(-100, 100, 401)
    q = np.zeros((2, 5, 401))
    q[0, 0, 260] = 1.0  # bid +30
    q[1, 4, 65] = 1.0   # bid -67.5
    peaks = peak_bids(bids, q, axis=2)
    assert peaks.shape == (2, 5)
    assert peaks[0, 0] == pytest.approx(30.0)
    assert peaks[1, 4] == pytest.approx(bids[65])


# --------------------------------------------------------------------------- #
# 4. the surrogate is not the simulator's reward

FROZEN_BUFFER = SCENARIO / "learned_strategies" / "buffers" / "single_10ep_standard.npz"


def frozen_buffer():
    d = np.load(FROZEN_BUFFER)
    n = int(d["pos"][0])
    return d["actions"][:n, 0, 0] * 100.0, d["rewards"][:n, 0]


@pytest.mark.skipif(not FROZEN_BUFFER.exists(), reason="starting buffer is gitignored")
def test_surrogate_is_not_the_simulators_reward():
    """Pin the divergence, so nothing scores a real run with the closed form again.

    ``reward_from_bid`` is the SB3 surrogate's landscape and is exact *there* --
    ``IncDecEnv`` is defined from it. It is **not** the reward the simulator paid,
    and for a long time four ``real_matd3/`` scripts reconstructed "true reward",
    "regret" and "solved >= +0.15" for real ASSUME runs by applying it to a
    recorded bid.

    This asserts the disagreement rather than the agreement, deliberately: if
    someone later reconciles the two, this test fails and forces the archive's
    numbers to be revisited instead of quietly changing meaning.
    """
    from incdec_reward import PAPER_SMALL, reward_from_bid

    bid, stored = frozen_buffer()
    predicted = reward_from_bid(bid, PAPER_SMALL)

    agreement = float(np.mean(np.isclose(predicted, stored, atol=1e-6)))
    mae = float(np.mean(np.abs(predicted - stored)))
    assert len(stored) == 620
    assert agreement == pytest.approx(0.2484, abs=0.01), (
        f"surrogate/simulator agreement moved to {agreement:.2%}; RUNS.md "
        "correction 15 and every 'recon' column need rechecking"
    )
    assert mae == pytest.approx(0.0382, abs=0.005)
    assert float(np.max(np.abs(predicted - stored))) > 0.3


@pytest.mark.skipif(not FROZEN_BUFFER.exists(), reason="starting buffer is gitignored")
def test_the_simulators_loss_shelf_is_not_flat():
    """The structural reason for the mismatch: the EOM price varies by hour.

    The surrogate has one loss shelf at -0.17, from a fixed ``eom_price`` of 49.
    The simulator paid three, implying clearing prices of 48, 43 and 38. No
    rescaling of a single-shelf curve can represent that, which is why the fix is
    to stop using it for real runs rather than to retune its parameters.
    """
    from incdec_reward import PAPER_SMALL, reward_from_bid

    _, stored = frozen_buffer()
    shelves = np.unique(np.round(stored[stored < -0.05], 4))
    assert len(shelves) >= 3, f"expected several loss shelves, found {shelves}"

    surrogate_shelf = reward_from_bid(0.0, PAPER_SMALL)
    assert surrogate_shelf == pytest.approx(-0.17)
    assert not np.any(np.isclose(shelves, surrogate_shelf, atol=1e-3)), (
        f"the surrogate's shelf {surrogate_shelf} now matches a measured one "
        f"{shelves}; recheck correction 15"
    )


@pytest.mark.skipif(not FROZEN_BUFFER.exists(), reason="starting buffer is gitignored")
def test_the_profitable_region_is_not_exactly_the_surrogates_band():
    """`[30, 49)` is the surrogate's band, not the simulator's profitable region.

    Measured, profit runs from about 28 to about 47.4, so some bids below the
    surrogate's cliff pay and the top of its band does not. ``in band`` and
    ``band_neg`` are therefore approximations, which is worth knowing before
    quoting either as a success criterion.
    """
    bid, stored = frozen_buffer()
    profitable = bid[stored > 1e-9]
    assert profitable.min() < 30.0, "no profitable bid below the surrogate's cliff"
    assert profitable.max() < 49.0, "profit extended to the surrogate's band ceiling"


# --------------------------------------------------------------------------- #
# 5. one episode of transitions


def transitions_per_episode(study_case: str) -> int:
    """Products per episode for a study case, derived from ``config.yaml``.

    The learning agents place one bid per delivery product, so an episode holds
    ``horizon_hours`` transitions minus the hours the market opening offsets eat
    before the first delivery. Both offsets matter: the EOM's own start relative
    to the simulation start, and its ``first_delivery``.

    This is the arithmetic ``RUNS.md`` section 9 states as "hours - 10" -- true of
    the ``*single_case`` family, whose EOM opens at 07:00, and *not* of
    ``inc_dec_learning``, whose EOM opens with the simulation.
    """
    import pandas as pd

    cases = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    case = cases[study_case]

    # PyYAML leaves "2019-01-01 00:00" as a string (no seconds, so its timestamp
    # resolver does not fire); ASSUME's loader parses these with pandas too.
    start = pd.Timestamp(case["start_date"])
    end = pd.Timestamp(case["end_date"])
    horizon_hours = int((end - start).total_seconds() // 3600)

    eom = case["markets_config"]["EOM"]
    opening_offset = int(
        (pd.Timestamp(eom["start_date"]) - start).total_seconds() // 3600
    )
    first_delivery = int(
        pd.Timedelta(eom["products"][0]["first_delivery"]).total_seconds() // 3600
    )

    return horizon_hours - (opening_offset + first_delivery)


def test_config_derivation_reproduces_the_shared_buffers_62():
    """Validate the formula against a number the archive independently fixes.

    ``buffers/single_10ep_standard.npz`` holds 620 = 62 x 10 transitions, collected
    by ``inc_dec_collect_buffer`` over a 72 h horizon. If the derivation below
    cannot reproduce that 62, it cannot be trusted for the multi-agent case either.
    """
    assert transitions_per_episode("inc_dec_collect_buffer") == 62


def test_reward_window_is_one_episode_of_the_multi_agent_study_case():
    """``REWARD_WINDOW`` must be the study case's own episode length.

    It was 62, inherited from the single-agent case, against an actual 69 -- so
    the reported per-agent reward silently dropped the first seven hours of each
    episode (and, at intra-episode frames, mixed in the tail of the previous one).
    Nothing about the bids or the critic depended on it, but the reward column of
    run 13's table and its fleet total did.
    """
    expected = transitions_per_episode("inc_dec_learning")
    assert expected == 69, "config.yaml's inc_dec_learning horizon or EOM offset moved"
    assert MultiAgentRecorder.REWARD_WINDOW == expected


def test_the_two_study_cases_really_do_disagree():
    """Guard the reason the constant was wrong in the first place.

    The single-agent and multi-agent cases open their markets at different hours,
    so a constant that is right for one is wrong for the other. If this ever stops
    being true the two can share a number; until then they cannot.
    """
    assert transitions_per_episode("inc_dec_collect_buffer") != transitions_per_episode(
        "inc_dec_learning"
    )


# --- optional cross-check against the archive, when it is present -------------

ARCHIVE = (
    SCENARIO.parents[1]
    / "outputs"
    / SCENARIO.name
    / "rl_benchmark"
    / "runs"
    / "data"
    / "13-multiagent-actshare"
    / "assume_ma_baseline_seed42.npz"
)


@pytest.mark.skipif(not ARCHIVE.exists(), reason="run archive is gitignored")
def test_archived_run_confirms_the_episode_length():
    """The recorded ``buffer_fill`` is ground truth for the derivation above."""
    fill = np.load(ARCHIVE, allow_pickle=False)["buffer_fill"]
    expected = transitions_per_episode("inc_dec_learning")

    blocks_per_episode = 6  # 72 h horizon at train_freq 12h
    assert len(fill) % blocks_per_episode == 0

    # every training block adds 12 transitions except the first of each episode,
    # which adds 9 -- the market opening offset lands inside it
    added = np.diff(fill)
    assert set(np.unique(added)) == {9, 12}

    # frame 0 already sits at the end of an episode's first block, so the gaps
    # tile whole episodes from index 0; the trailing partial group is dropped
    whole = (len(added) // blocks_per_episode) * blocks_per_episode
    per_episode = added[:whole].reshape(-1, blocks_per_episode).sum(axis=1)
    assert set(per_episode.tolist()) == {expected}
