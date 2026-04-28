# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest

try:
    import torch as th

    from assume.reinforcement_learning.buffer import (
        RolloutBuffer,
        RolloutBufferSamples,
    )
except ImportError:
    pass

def make_rollout_buffer(
    buffer_size=8,
    obs_dim=3,
    act_dim=2,
    n_rl_units=2,
    gamma=0.99,
    gae_lambda=0.95,
):
    return RolloutBuffer(
        buffer_size=buffer_size,
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_rl_units=n_rl_units,
        device=th.device("cpu"),
        float_type=th.float32,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )


def fill_buffer(buf, n_steps=None, seed=0):
    rng = np.random.default_rng(seed)
    n = n_steps if n_steps is not None else buf.buffer_size
    for _ in range(n):
        obs = rng.random((buf.n_rl_units, buf.obs_dim)).astype(np.float32)
        act = rng.random((buf.n_rl_units, buf.act_dim)).astype(np.float32)
        rew = rng.random(buf.n_rl_units).astype(np.float32)
        done = np.zeros(buf.n_rl_units, dtype=np.float32)
        val = rng.random(buf.n_rl_units).astype(np.float32)
        lp = rng.random(buf.n_rl_units).astype(np.float32) - 1.0
        buf.add(obs, act, rew, done, val, lp)


@pytest.mark.require_learning
def test_rollout_buffer_init_shapes():
    buf = make_rollout_buffer(buffer_size=10, obs_dim=3, act_dim=2, n_rl_units=4)
    assert buf.observations.shape == (10, 4, 3)
    assert buf.actions.shape == (10, 4, 2)
    assert buf.rewards.shape == (10, 4)
    assert buf.values.shape == (10, 4)
    assert buf.log_probs.shape == (10, 4)
    assert buf.dones.shape == (10, 4)
    assert buf.advantages.shape == (10, 4)
    assert buf.returns.shape == (10, 4)


@pytest.mark.require_learning
def test_rollout_buffer_init_state():
    buf = make_rollout_buffer()
    assert buf.pos == 0
    assert buf.full is False
    assert buf.generator_ready is False
    assert buf.size() == 0


@pytest.mark.require_learning
def test_rollout_buffer_reset_clears_data():
    buf = make_rollout_buffer(buffer_size=4)
    fill_buffer(buf, n_steps=4)
    assert buf.pos == 4

    buf.reset()
    assert buf.pos == 0
    assert buf.full is False
    assert buf.generator_ready is False
    assert np.all(buf.observations == 0)
    assert np.all(buf.rewards == 0)
    assert np.all(buf.advantages == 0)
    assert np.all(buf.returns == 0)


@pytest.mark.require_learning
def test_rollout_buffer_add_increments_pos():
    buf = make_rollout_buffer(buffer_size=5)
    obs = np.ones((buf.n_rl_units, buf.obs_dim), dtype=np.float32)
    act = np.ones((buf.n_rl_units, buf.act_dim), dtype=np.float32)
    rew = np.ones(buf.n_rl_units, dtype=np.float32)
    done = np.zeros(buf.n_rl_units, dtype=np.float32)
    val = np.ones(buf.n_rl_units, dtype=np.float32)
    lp = np.zeros(buf.n_rl_units, dtype=np.float32)

    for i in range(1, 6):
        buf.add(obs, act, rew, done, val, lp)
        assert buf.pos == i
        assert buf.size() == i


@pytest.mark.require_learning
def test_rollout_buffer_add_stores_correct_values():
    buf = make_rollout_buffer(buffer_size=4, obs_dim=2, act_dim=2, n_rl_units=1)
    obs = np.array([[1.0, 2.0]], dtype=np.float32)
    act = np.array([[0.5, -0.5]], dtype=np.float32)
    rew = np.array([3.0], dtype=np.float32)
    done = np.array([0.0], dtype=np.float32)
    val = np.array([0.7], dtype=np.float32)
    lp = np.array([-1.2], dtype=np.float32)

    buf.add(obs, act, rew, done, val, lp)

    np.testing.assert_array_almost_equal(buf.observations[0, 0], [1.0, 2.0])
    np.testing.assert_array_almost_equal(buf.actions[0, 0], [0.5, -0.5])
    assert buf.rewards[0, 0] == pytest.approx(3.0)
    assert buf.dones[0, 0] == pytest.approx(0.0)
    assert buf.values[0, 0] == pytest.approx(0.7)
    assert buf.log_probs[0, 0] == pytest.approx(-1.2)


@pytest.mark.require_learning
def test_rollout_buffer_add_beyond_capacity_sets_full():
    buf = make_rollout_buffer(buffer_size=3)
    obs = np.zeros((buf.n_rl_units, buf.obs_dim), dtype=np.float32)
    act = np.zeros((buf.n_rl_units, buf.act_dim), dtype=np.float32)
    rew = np.zeros(buf.n_rl_units, dtype=np.float32)
    done = np.zeros(buf.n_rl_units, dtype=np.float32)
    val = np.zeros(buf.n_rl_units, dtype=np.float32)
    lp = np.zeros(buf.n_rl_units, dtype=np.float32)

    for _ in range(3):
        buf.add(obs, act, rew, done, val, lp)

    assert buf.pos == 3
    assert buf.size() == 3

    buf.add(obs, act, rew, done, val, lp)
    assert buf.full is True
    assert buf.size() == 3


@pytest.mark.require_learning
def test_gae_single_step_non_terminal():
    """For 1 step, 1 agent, non-terminal: advantage = TD error."""
    gamma, gae_lambda = 0.99, 0.95
    buf = make_rollout_buffer(
        buffer_size=1, obs_dim=1, act_dim=1, n_rl_units=1,
        gamma=gamma, gae_lambda=gae_lambda,
    )
    r, v, v_next = 1.0, 0.5, 0.8
    buf.add(
        obs=np.array([[0.0]]),
        action=np.array([[0.0]]),
        reward=np.array([r]),
        done=np.array([0.0]),
        value=np.array([v]),
        log_prob=np.array([0.0]),
    )

    buf.compute_returns_and_advantages(
        last_values=np.array([v_next]),
        dones=np.array([0.0]),
    )

    expected_advantage = r + gamma * v_next - v
    expected_return = expected_advantage + v

    assert buf.advantages[0, 0] == pytest.approx(expected_advantage, abs=1e-5)
    assert buf.returns[0, 0] == pytest.approx(expected_return, abs=1e-5)


@pytest.mark.require_learning
def test_gae_single_step_terminal():
    """For a terminal episode end, bootstrap value must not propagate."""
    gamma, gae_lambda = 0.99, 0.95
    buf = make_rollout_buffer(
        buffer_size=1, obs_dim=1, act_dim=1, n_rl_units=1,
        gamma=gamma, gae_lambda=gae_lambda,
    )
    r, v = 2.0, 1.0
    buf.add(
        obs=np.array([[0.0]]),
        action=np.array([[0.0]]),
        reward=np.array([r]),
        done=np.array([0.0]),
        value=np.array([v]),
        log_prob=np.array([0.0]),
    )

    # done=1 — so no bootstrapping from last_values
    buf.compute_returns_and_advantages(
        last_values=np.array([999.0]),
        dones=np.array([1.0]),
    )

    expected_advantage = r - v
    expected_return = expected_advantage + v  # = r

    assert buf.advantages[0, 0] == pytest.approx(expected_advantage, abs=1e-5)
    assert buf.returns[0, 0] == pytest.approx(expected_return, abs=1e-5)


@pytest.mark.require_learning
def test_gae_multi_step_manual():
    """Manually verify 2-step GAE for a single agent."""
    gamma, gae_lambda = 0.99, 0.95
    buf = make_rollout_buffer(
        buffer_size=2, obs_dim=1, act_dim=1, n_rl_units=1,
        gamma=gamma, gae_lambda=gae_lambda,
    )
    r0, v0 = 1.0, 0.4
    r1, v1 = 0.5, 0.6
    v_next = 0.8

    for r, v in [(r0, v0), (r1, v1)]:
        buf.add(
            obs=np.array([[0.0]]),
            action=np.array([[0.0]]),
            reward=np.array([r]),
            done=np.array([0.0]),
            value=np.array([v]),
            log_prob=np.array([0.0]),
        )

    buf.compute_returns_and_advantages(
        last_values=np.array([v_next]),
        dones=np.array([0.0]),
    )

    delta_1 = r1 + gamma * v_next - v1
    gae_1 = delta_1

    delta_0 = r0 + gamma * v1 - v0
    gae_0 = delta_0 + gamma * gae_lambda * gae_1

    assert buf.advantages[0, 0] == pytest.approx(gae_0, abs=1e-5)
    assert buf.advantages[1, 0] == pytest.approx(gae_1, abs=1e-5)
    assert buf.returns[0, 0] == pytest.approx(gae_0 + v0, abs=1e-5)
    assert buf.returns[1, 0] == pytest.approx(gae_1 + v1, abs=1e-5)


@pytest.mark.require_learning
def test_gae_lambda_zero_equals_td_error():
    """gae_lambda=0 reduces GAE to a 1-step TD advantage per step."""
    gamma, gae_lambda = 0.99, 0.0
    buf = make_rollout_buffer(
        buffer_size=3, obs_dim=1, act_dim=1, n_rl_units=1,
        gamma=gamma, gae_lambda=gae_lambda,
    )
    rewards = [1.0, 0.5, 2.0]
    values = [0.4, 0.6, 0.3]
    v_next = 0.8

    for r, v in zip(rewards, values):
        buf.add(
            obs=np.array([[0.0]]),
            action=np.array([[0.0]]),
            reward=np.array([r]),
            done=np.array([0.0]),
            value=np.array([v]),
            log_prob=np.array([0.0]),
        )

    buf.compute_returns_and_advantages(
        last_values=np.array([v_next]),
        dones=np.array([0.0]),
    )

    next_vals = [values[1], values[2], v_next]
    for step, (r, v, nv) in enumerate(zip(rewards, values, next_vals)):
        expected = r + gamma * nv - v
        assert buf.advantages[step, 0] == pytest.approx(expected, abs=1e-5)


@pytest.mark.require_learning
def test_gae_lambda_one_gamma_one_monte_carlo():
    """with gamma=1, gae_lambda=1, terminal, should return equal undiscounted reward sums."""
    gamma, gae_lambda = 1.0, 1.0
    T = 4
    buf = make_rollout_buffer(
        buffer_size=T, obs_dim=1, act_dim=1, n_rl_units=1,
        gamma=gamma, gae_lambda=gae_lambda,
    )
    rewards = [1.0, 1.0, 1.0, 1.0]
    values = [0.1] * T

    for r, v in zip(rewards, values):
        buf.add(
            obs=np.array([[0.0]]),
            action=np.array([[0.0]]),
            reward=np.array([r]),
            done=np.array([0.0]),
            value=np.array([v]),
            log_prob=np.array([0.0]),
        )

    buf.compute_returns_and_advantages(
        last_values=np.array([0.0]),
        dones=np.array([1.0]),
    )

    for t in range(T):
        assert buf.returns[t, 0] == pytest.approx(float(T - t), abs=1e-5)


@pytest.mark.require_learning
def test_gae_multi_agent_independence():
    """One agent's rewards must not cause issue with another agent's advantages."""
    gamma, gae_lambda = 0.99, 0.95
    buf = make_rollout_buffer(
        buffer_size=3, obs_dim=1, act_dim=1, n_rl_units=2,
        gamma=gamma, gae_lambda=gae_lambda,
    )

    for _ in range(3):
        buf.add(
            obs=np.zeros((2, 1), dtype=np.float32),
            action=np.zeros((2, 1), dtype=np.float32),
            reward=np.array([1.0, 0.0]),
            done=np.zeros(2, dtype=np.float32),
            value=np.array([0.5, 0.5]),
            log_prob=np.zeros(2, dtype=np.float32),
        )

    buf.compute_returns_and_advantages(
        last_values=np.array([0.5, 0.5]),
        dones=np.zeros(2),
    )

    for t in range(3):
        assert abs(buf.advantages[t, 1]) < abs(buf.advantages[t, 0]), (
            f"step {t}: agent-1 advantage {buf.advantages[t, 1]:.4f} should be "
            f"smaller than agent-0 advantage {buf.advantages[t, 0]:.4f}"
        )


@pytest.mark.require_learning
def test_gae_returns_equal_advantages_plus_values():
    """returns == advantages + values for every step and agent."""
    buf = make_rollout_buffer(buffer_size=6, n_rl_units=3)
    fill_buffer(buf, n_steps=6)

    last_values = np.random.rand(3).astype(np.float32)
    buf.compute_returns_and_advantages(last_values, dones=np.zeros(3, dtype=np.float32))

    np.testing.assert_array_almost_equal(
        buf.returns[: buf.pos],
        buf.advantages[: buf.pos] + buf.values[: buf.pos],
        decimal=5,
    )


@pytest.mark.require_learning
def test_rollout_buffer_get_raises_before_compute():
    """Calling get() before compute_returns_and_advantages must raise ValueError."""
    buf = make_rollout_buffer(buffer_size=4)
    fill_buffer(buf, n_steps=4)

    with pytest.raises(ValueError, match="compute_returns_and_advantages"):
        next(buf.get(batch_size=2))


@pytest.mark.require_learning
def test_rollout_buffer_get_full_batch():
    """get(batch_size=None) yields one batch with all steps and correct shapes."""
    buf = make_rollout_buffer(buffer_size=5, obs_dim=3, act_dim=2, n_rl_units=2)
    fill_buffer(buf, n_steps=5)
    buf.compute_returns_and_advantages(
        last_values=np.zeros(2, dtype=np.float32),
        dones=np.zeros(2, dtype=np.float32),
    )

    batches = list(buf.get(batch_size=None))
    assert len(batches) == 1

    batch = batches[0]
    assert isinstance(batch, RolloutBufferSamples)
    assert batch.observations.shape == (5, 2, 3)
    assert batch.actions.shape == (5, 2, 2)
    assert batch.old_values.shape == (5, 2)
    assert batch.old_log_probs.shape == (5, 2)
    assert batch.advantages.shape == (5, 2)
    assert batch.returns.shape == (5, 2)


@pytest.mark.require_learning
def test_rollout_buffer_get_mini_batches_cover_all_steps():
    """Mini-batch iteration must cover every step exactly once."""
    T = 8
    buf = make_rollout_buffer(buffer_size=T, obs_dim=2, act_dim=1, n_rl_units=1)
    fill_buffer(buf, n_steps=T)
    buf.compute_returns_and_advantages(
        last_values=np.zeros(1, dtype=np.float32),
        dones=np.zeros(1, dtype=np.float32),
    )

    total_samples = 0
    for batch in buf.get(batch_size=2):
        assert isinstance(batch, RolloutBufferSamples)
        total_samples += batch.observations.shape[0]

    assert total_samples == T


@pytest.mark.require_learning
def test_rollout_buffer_get_partial_fill():
    """A partially-filled buffer must only yield the filled steps."""
    buf = make_rollout_buffer(buffer_size=10, obs_dim=2, act_dim=1, n_rl_units=1)
    fill_buffer(buf, n_steps=4) 
    buf.compute_returns_and_advantages(
        last_values=np.zeros(1, dtype=np.float32),
        dones=np.zeros(1, dtype=np.float32),
    )

    batches = list(buf.get(batch_size=None))
    assert batches[0].observations.shape[0] == 4


@pytest.mark.require_learning
def test_full_episode_rollout():
    """fill -> GAE -> mini-batch epochs -> reset"""
    T, obs_dim, act_dim, n_agents = 16, 5, 3, 2
    buf = make_rollout_buffer(
        buffer_size=T, obs_dim=obs_dim, act_dim=act_dim, n_rl_units=n_agents,
        gamma=0.99, gae_lambda=0.95,
    )

    rng = np.random.default_rng(42)
    for _ in range(T):
        buf.add(
            obs=rng.random((n_agents, obs_dim)).astype(np.float32),
            action=rng.random((n_agents, act_dim)).astype(np.float32),
            reward=rng.random(n_agents).astype(np.float32),
            done=np.zeros(n_agents, dtype=np.float32),
            value=rng.random(n_agents).astype(np.float32),
            log_prob=-rng.random(n_agents).astype(np.float32),
        )

    assert buf.size() == T

    last_values = rng.random(n_agents).astype(np.float32)
    buf.compute_returns_and_advantages(last_values, dones=np.zeros(n_agents))

    # returns == advantages + values
    np.testing.assert_array_almost_equal(
        buf.returns, buf.advantages + buf.values, decimal=5
    )

    # Two PPO epochs over mini-batches of size 4
    for _epoch in range(2):
        samples_seen = 0
        for batch in buf.get(batch_size=4):
            assert batch.observations.shape == (4, n_agents, obs_dim)
            assert batch.actions.shape == (4, n_agents, act_dim)
            samples_seen += batch.observations.shape[0]
        assert samples_seen == T

    # Reset for next rollout
    buf.reset()
    assert buf.pos == 0
    assert buf.generator_ready is False
    assert buf.size() == 0
