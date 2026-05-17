# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os
from copy import copy, deepcopy
from datetime import datetime

import numpy as np
import pytest

from assume.common.base import LearningConfig, OnPolicyConfig

try:
    import torch as th

    from assume.common.base import LearningStrategy
    from assume.reinforcement_learning.algorithms.mappo import PPO
    from assume.reinforcement_learning.buffer import RolloutBuffer
    from assume.reinforcement_learning.learning_role import Learning


except ImportError:
    pass


start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.fixture
def base_learning_config() -> dict:
    foresight = 2
    unique_obs_dim = 2
    num_timeseries_obs_dim = 4
    return {
        "foresight": foresight,
        "act_dim": 3,
        "unique_obs_dim": unique_obs_dim,
        "num_timeseries_obs_dim": num_timeseries_obs_dim,
        "obs_dim": foresight * num_timeseries_obs_dim + unique_obs_dim,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="mappo",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=10,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
            learning_rate=1e-4,
            batch_size=10,
            gamma=0.99,
            on_policy=OnPolicyConfig(
                clip_ratio=0.2,
                entropy_coef=0.01,
                gae_lambda=0.95,
                max_grad_norm=0.5,
                vf_coef=0.5,
                n_epochs=2,
            ),
        ),
    }


@pytest.fixture(scope="function")
def learning_role_n(base_learning_config):
    config = copy(base_learning_config)
    learn = Learning(config["learning_config"], start, end)
    for agent_id in ("agent_0", "agent_1"):
        strategy = LearningStrategy(**config, learning_role=learn)
        strategy.unit_id = agent_id
        learn.rl_strats[agent_id] = strategy
    return learn


@pytest.fixture(scope="function")
def saved_n_agent_model(learning_role_n, tmp_path) -> tuple[str, dict]:
    """Save a 2-agent PPO model; return (save_dir, state_dict_snapshot)."""
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "saved_model_n"
    save_dir.mkdir(parents=True, exist_ok=True)
    learning_role_n.rl_algorithm.save_params(directory=str(save_dir))
    agent = learning_role_n.rl_strats["agent_0"]
    return str(save_dir), {
        "critic": agent.critics.state_dict(),
        "actor": agent.actor.state_dict(),
        "optimizer_critic": agent.critics.optimizer.state_dict(),
        "optimizer_actor": agent.actor.optimizer.state_dict(),
    }


def compare_state_dicts(dict1, dict2) -> bool:
    if dict1.keys() != dict2.keys():
        return False
    for k in dict1:
        v1, v2 = dict1[k], dict2[k]
        if isinstance(v1, th.Tensor):
            if not th.equal(v1, v2):
                return False
        elif isinstance(v1, dict):
            if not compare_state_dicts(v1, v2):
                return False
        else:
            if v1 != v2:
                return False
    return True


def _make_rollout_buffer(
    obs_dim: int,
    act_dim: int,
    n_agents: int,
    n_steps: int,
    device: str = "cpu",
) -> "RolloutBuffer":
    """Building and filling a RolloutBuffer with random data for update_policy tests."""
    buf = RolloutBuffer(
        buffer_size=n_steps + 10,
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_rl_units=n_agents,
        device=device,
        float_type=th.float32,
        gamma=0.99,
        gae_lambda=0.95,
    )
    rng = np.random.default_rng(42)
    for _ in range(n_steps):
        buf.add(
            obs=rng.random((n_agents, obs_dim)).astype(np.float32),
            action=rng.random((n_agents, act_dim)).astype(np.float32),
            reward=rng.random(n_agents).astype(np.float32),
            done=np.zeros(n_agents, dtype=np.float32),
            value=rng.random(n_agents).astype(np.float32),
            log_prob=(rng.random(n_agents).astype(np.float32) - 1.0),
        )
    return buf


def _setup_for_update(learning_role) -> None:
    """Setting minimal attributes needed."""
    learning_role.update_steps = 0
    learning_role.db_addr = None  # disables the context.schedule_instant_message path


@pytest.mark.require_learning
def test_mappo_algorithm_class(learning_role_n):
    """initialize_policy creates a PPO instance as the rl_algorithm."""
    learning_role_n.initialize_policy()
    assert isinstance(learning_role_n.rl_algorithm, PPO)


@pytest.mark.require_learning
def test_mappo_save_params_creates_files(learning_role_n, tmp_path):
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "model_save_test"

    learning_role_n.rl_algorithm.save_params(directory=str(save_dir))

    assert os.path.exists(save_dir / "critics" / "critic_agent_0.pt")
    assert os.path.exists(save_dir / "critics" / "critic_agent_1.pt")
    assert os.path.exists(save_dir / "actors" / "actor_agent_0.pt")
    assert os.path.exists(save_dir / "actors" / "actor_agent_1.pt")


@pytest.mark.require_learning
def test_mappo_save_params_u_id_order(learning_role_n, tmp_path):
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "uid_order_test"
    learning_role_n.rl_algorithm.save_params(directory=str(save_dir))

    order_file = save_dir / "critics" / "u_id_order.json"
    assert order_file.exists(), "u_id_order.json must be written alongside critic files"
    with open(order_file) as f:
        mapping = json.load(f)
    assert mapping.get("u_id_order") == ["agent_0", "agent_1"]


@pytest.mark.require_learning
def test_mappo_load_matching_n(base_learning_config, saved_n_agent_model):
    save_dir, original_states = saved_n_agent_model

    config_new = copy(base_learning_config)
    learn_new = Learning(config_new["learning_config"], start, end)
    learn_new.rl_strats["agent_0"] = LearningStrategy(
        **config_new, learning_role=learn_new
    )
    learn_new.rl_strats["agent_1"] = LearningStrategy(
        **config_new, learning_role=learn_new
    )
    learn_new.initialize_policy()
    learn_new.rl_algorithm.load_params(directory=save_dir)

    agent = learn_new.rl_strats["agent_0"]
    assert compare_state_dicts(original_states["critic"], agent.critics.state_dict())
    assert compare_state_dicts(original_states["actor"], agent.actor.state_dict())
    assert compare_state_dicts(
        deepcopy(original_states["optimizer_critic"]),
        deepcopy(agent.critics.optimizer.state_dict()),
    )
    assert compare_state_dicts(
        deepcopy(original_states["optimizer_actor"]),
        deepcopy(agent.actor.optimizer.state_dict()),
    )


@pytest.mark.require_learning
def test_mappo_initialize_policy_dimension_mismatch(base_learning_config):
    config = copy(base_learning_config)
    config["num_timeseries_obs_dim"] = 1

    learn = Learning(config["learning_config"], start, end)
    strat_0 = LearningStrategy(**config, learning_role=learn)

    config_bad = copy(config)
    config_bad["act_dim"] = 99
    strat_1 = LearningStrategy(**config_bad, learning_role=learn)

    learn.rl_strats["agent_0"] = strat_0
    learn.rl_strats["agent_1"] = strat_1

    with pytest.raises(ValueError, match="All action dimensions must be the same"):
        learn.rl_algorithm.initialize_policy()


@pytest.mark.require_learning
def test_mappo_initialize_policy_all_dimensions_match(base_learning_config):
    config = copy(base_learning_config)
    config["num_timeseries_obs_dim"] = 1

    learn = Learning(config["learning_config"], start, end)
    for agent_id in ("agent_0", "agent_1", "agent_2"):
        learn.rl_strats[agent_id] = LearningStrategy(**config, learning_role=learn)

    try:
        learn.rl_algorithm.initialize_policy()
    except Exception as e:
        pytest.fail(f"initialize_policy raised an unexpected error: {e}")


@pytest.mark.require_learning
def test_mappo_buffer_storage_uses_rl_strats_order(base_learning_config):
    """Regression test for the agent-ordering bug.

    The on-policy buffer-storage path used to call
    ``sorted(cache["obs"][timestamp].keys())`` to order agents, while
    ``mappo.PPO.update_policy`` iterates ``self.rl_strats.values()``.  When
    the unit ids do not happen to be alphabetically sorted (e.g.
    ``pp_6, pp_7, pp_8, pp_9, pp_10``) the two orders diverge and every
    agent is trained on a different agent's observations / actions / values,
    silently degrading MAPPO to noise.

    This test pins ``learning_role`` to use the ``rl_strats`` insertion order
    when filling the rollout buffer, exactly like the off-policy algorithms
    already do.
    """
    import asyncio
    from collections import defaultdict

    config = copy(base_learning_config)

    learn = Learning(config["learning_config"], start, end)
    insertion_order = ("pp_6", "pp_7", "pp_8", "pp_9", "pp_10")
    assert sorted(insertion_order) != list(insertion_order), (
        "test scenario must use unit ids whose sort order differs from "
        "insertion order; otherwise this regression test is trivially passing"
    )

    for agent_id in insertion_order:
        strat = LearningStrategy(**config, learning_role=learn)
        strat.unit_id = agent_id
        learn.rl_strats[agent_id] = strat

    learn.initialize_policy()

    n_agents = len(insertion_order)
    # ``LearningStrategy`` computes ``self.obs_dim`` from
    # ``num_timeseries_obs_dim * foresight + unique_obs_dim``, so we must
    # match that here for the fake centralized-critic input to align.
    obs_dim = (
        config["num_timeseries_obs_dim"] * config["foresight"]
        + config["unique_obs_dim"]
    )
    act_dim = config["act_dim"]

    # Build a fake rollout buffer large enough to hold one fake timestep.
    learn.buffer = RolloutBuffer(
        buffer_size=4,
        obs_dim=obs_dim,
        act_dim=act_dim,
        n_rl_units=n_agents,
        device="cpu",
        float_type=th.float32,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Craft a cache where each unit's observation/action/reward is a unique
    # constant equal to (1+i)*10, so we can assert that the row for agent i in
    # the buffer matches the i-th *insertion-order* unit, not the i-th
    # *sorted-order* unit.
    timestamp = "2023-07-01 00:00:00"
    cache = {
        "obs": {timestamp: {}},
        "actions": {timestamp: {}},
        "rewards": {timestamp: {}},
        "noises": {timestamp: {}},
        "regret": {timestamp: {}},
        "profit": {timestamp: {}},
        "values": {timestamp: defaultdict(list)},
        "log_probs": {timestamp: {}},
        "dones": {timestamp: {}},
    }
    for i, unit_id in enumerate(insertion_order):
        marker = float(i + 1)
        cache["obs"][timestamp][unit_id] = [
            th.full((obs_dim,), marker, dtype=th.float32)
        ]
        cache["actions"][timestamp][unit_id] = [
            th.full((act_dim,), marker, dtype=th.float32)
        ]
        cache["rewards"][timestamp][unit_id] = [marker]
        cache["noises"][timestamp][unit_id] = [
            th.zeros(act_dim, dtype=th.float32)
        ]
        cache["regret"][timestamp][unit_id] = [0.0]
        cache["profit"][timestamp][unit_id] = [0.0]
        cache["log_probs"][timestamp][unit_id] = [-marker]
        cache["dones"][timestamp][unit_id] = [0.0]
        # leave cache["values"][timestamp] empty - mappo recomputes values

    # Stash db_addr/update_steps so the logging path inside the algorithm is
    # safe to call.  We do NOT need an actual policy update for this test, so
    # we monkey-patch update_policy to a no-op.
    learn.db_addr = None
    learn.update_steps = 0
    learn.rl_algorithm.update_policy = lambda: None

    asyncio.run(
        learn._store_to_buffer_and_update_sync(cache, learn.device)
    )

    buf = learn.buffer
    # One timestamp -> one row in the buffer.
    assert buf.pos == 1, f"expected 1 transition, got {buf.pos}"

    stored_obs = buf.observations[0]
    stored_actions = buf.actions[0]
    stored_rewards = buf.rewards[0]
    stored_log_probs = buf.log_probs[0]

    for i in range(n_agents):
        expected = float(i + 1)
        assert np.allclose(stored_obs[i], expected), (
            f"row {i} of buffer.observations should match insertion-order "
            f"agent {insertion_order[i]} (value {expected}); got {stored_obs[i]}"
        )
        assert np.allclose(stored_actions[i], expected), (
            f"row {i} of buffer.actions should match insertion-order "
            f"agent {insertion_order[i]} (value {expected}); got {stored_actions[i]}"
        )
        assert np.allclose(stored_rewards[i], expected), (
            f"row {i} of buffer.rewards should match insertion-order "
            f"agent {insertion_order[i]} (value {expected}); got {stored_rewards[i]}"
        )
        assert np.allclose(stored_log_probs[i], -expected), (
            f"row {i} of buffer.log_probs should match insertion-order "
            f"agent {insertion_order[i]} (value {-expected}); got {stored_log_probs[i]}"
        )
