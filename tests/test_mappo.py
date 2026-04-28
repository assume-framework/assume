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
    from assume.reinforcement_learning.neural_network_architecture import (
        ActorPPO,
        CriticPPO,
    )

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
        strat = LearningStrategy(**config, learning_role=learn)
        strat.unit_id = agent_id
        learn.rl_strats[agent_id] = strat
    return learn


@pytest.fixture(scope="function")
def saved_n_agent_model(learning_role_n, tmp_path) -> tuple[str, dict]:
    """Save a 2-agent PPO model; return (save_dir, state_dict_snapshot).
    """
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
    """Setting minimal attributes needed.
    """
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


# @pytest.mark.require_learning
# def test_mappo_update_policy_skips_none_buffer(learning_role_n, monkeypatch):
#     learning_role_n.initialize_policy()
#     _setup_for_update(learning_role_n)
#     monkeypatch.setattr(learning_role_n, "get_progress_remaining", lambda: 1.0)

#     learning_role_n.buffer = None
#     learning_role_n.rl_algorithm.update_policy()
#     assert learning_role_n.rl_algorithm.n_updates == 0


# @pytest.mark.require_learning
# def test_mappo_update_policy_skips_empty_buffer(learning_role_n, monkeypatch):
#     learning_role_n.initialize_policy()
#     _setup_for_update(learning_role_n)
#     monkeypatch.setattr(learning_role_n, "get_progress_remaining", lambda: 1.0)

#     s = learning_role_n.rl_strats["agent_0"]
#     learning_role_n.buffer = RolloutBuffer(
#         buffer_size=50,
#         obs_dim=s.obs_dim,
#         act_dim=s.act_dim,
#         n_rl_units=2,
#         device="cpu",
#         float_type=th.float32,
#     )
#     learning_role_n.rl_algorithm.update_policy()
#     assert learning_role_n.rl_algorithm.n_updates == 0


# @pytest.mark.require_learning
# def test_mappo_update_policy_skips_insufficient_data(learning_role_n, monkeypatch):
#     learning_role_n.initialize_policy()
#     _setup_for_update(learning_role_n)
#     monkeypatch.setattr(learning_role_n, "get_progress_remaining", lambda: 1.0)

#     s = learning_role_n.rl_strats["agent_0"]
#     learning_role_n.buffer = _make_rollout_buffer(
#         obs_dim=s.obs_dim, act_dim=s.act_dim, n_agents=2, n_steps=1
#     )
#     learning_role_n.rl_algorithm.update_policy()
#     assert learning_role_n.rl_algorithm.n_updates == 0


# @pytest.mark.require_learning
# def test_mappo_update_policy_increments_n_updates(learning_role_n, monkeypatch):
#     learning_role_n.initialize_policy()
#     _setup_for_update(learning_role_n)
#     monkeypatch.setattr(learning_role_n, "get_progress_remaining", lambda: 1.0)

#     s = learning_role_n.rl_strats["agent_0"]
#     learning_role_n.buffer = _make_rollout_buffer(
#         obs_dim=s.obs_dim, act_dim=s.act_dim, n_agents=2, n_steps=20
#     )
#     learning_role_n.rl_algorithm.update_policy()
#     assert learning_role_n.rl_algorithm.n_updates == 1


# @pytest.mark.require_learning
# def test_mappo_update_policy_resets_buffer(learning_role_n, monkeypatch):
#     learning_role_n.initialize_policy()
#     _setup_for_update(learning_role_n)
#     monkeypatch.setattr(learning_role_n, "get_progress_remaining", lambda: 1.0)

#     s = learning_role_n.rl_strats["agent_0"]
#     learning_role_n.buffer = _make_rollout_buffer(
#         obs_dim=s.obs_dim, act_dim=s.act_dim, n_agents=2, n_steps=20
#     )
#     assert learning_role_n.buffer.pos > 0

#     learning_role_n.rl_algorithm.update_policy()
#     assert learning_role_n.buffer.pos == 0, (
#         "RolloutBuffer.reset() must be called at the end of every PPO update"
#     )


# @pytest.mark.require_learning
# def test_mappo_update_policy_multiple_epochs(base_learning_config, monkeypatch):
#     config = copy(base_learning_config)
#     config["learning_config"].on_policy.n_epochs = 3

#     learn = Learning(config["learning_config"], start, end)
#     for agent_id in ("agent_0", "agent_1"):
#         strat = LearningStrategy(**config, learning_role=learn)
#         strat.unit_id = agent_id
#         learn.rl_strats[agent_id] = strat
#     learn.initialize_policy()
#     _setup_for_update(learn)
#     monkeypatch.setattr(learn, "get_progress_remaining", lambda: 1.0)

#     s = learn.rl_strats["agent_0"]
#     learn.buffer = _make_rollout_buffer(
#         obs_dim=s.obs_dim, act_dim=s.act_dim, n_agents=2, n_steps=30
#     )

#     algo = learn.rl_algorithm
#     assert algo.n_epochs == 3
#     algo.update_policy()
#     assert algo.n_updates == 1


# @pytest.mark.require_learning
# def test_mappo_update_policy_actor_weights_change(learning_role_n, monkeypatch):
#     learning_role_n.initialize_policy()
#     _setup_for_update(learning_role_n)
#     monkeypatch.setattr(learning_role_n, "get_progress_remaining", lambda: 1.0)

#     s = learning_role_n.rl_strats["agent_0"]
#     pre_actor = deepcopy(s.actor.state_dict())
#     pre_critic = deepcopy(s.critics.state_dict())

#     learning_role_n.buffer = _make_rollout_buffer(
#         obs_dim=s.obs_dim, act_dim=s.act_dim, n_agents=2, n_steps=20
#     )
#     learning_role_n.rl_algorithm.update_policy()

#     post_actor = s.actor.state_dict()
#     post_critic = s.critics.state_dict()

#     actor_changed = any(
#         not th.equal(pre_actor[k], post_actor[k]) for k in pre_actor
#     )
#     critic_changed = any(
#         not th.equal(pre_critic[k], post_critic[k]) for k in pre_critic
#     )
#     assert actor_changed, "Actor weights must change after a PPO update"
#     assert critic_changed, "Critic weights must change after a PPO update"


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
