# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import os
from copy import copy, deepcopy
from datetime import datetime

import pytest

from assume.common.base import LearningConfig, OffPolicyConfig

try:
    import torch as th

    from assume.common.base import LearningStrategy
    from assume.reinforcement_learning.algorithms.maddpg import DDPG
    from assume.reinforcement_learning.learning_role import Learning
    from assume.reinforcement_learning.neural_network_architecture import CriticDDPG

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
            algorithm="maddpg",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=1,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
            learning_rate=1e-4,
            batch_size=100,
            gamma=0.99,
            off_policy=OffPolicyConfig(
                episodes_collecting_initial_experience=0,
                gradient_steps=1,
                tau=0.005,
                policy_delay=2, 
                target_policy_noise=0.2,
                target_noise_clip=0.5,
            ),
        ),
    }


@pytest.fixture(scope="function")
def learning_role_n(base_learning_config):
    config = copy(base_learning_config)
    learn = Learning(config["learning_config"], start, end)
    learn.rl_strats["agent_0"] = LearningStrategy(**config, learning_role=learn)
    learn.rl_strats["agent_1"] = LearningStrategy(**config, learning_role=learn)
    return learn


@pytest.fixture(scope="function")
def learning_role_n_plus_m(base_learning_config):
    config = copy(base_learning_config)
    learn = Learning(config["learning_config"], start, end)
    learn.rl_strats["agent_0"] = LearningStrategy(**config, learning_role=learn)
    learn.rl_strats["agent_1"] = LearningStrategy(**config, learning_role=learn)
    learn.rl_strats["agent_2"] = LearningStrategy(**config, learning_role=learn)
    return learn


@pytest.fixture(scope="function")
def saved_n_agent_model(learning_role_n, tmp_path) -> tuple[str, dict]:
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "saved_model_n"
    save_dir.mkdir(parents=True, exist_ok=True)
    learning_role_n.rl_algorithm.save_params(directory=str(save_dir))
    agent = learning_role_n.rl_strats["agent_0"]
    return str(save_dir), {
        "critic": agent.critics.state_dict(),
        "actor": agent.actor.state_dict(),
        "target_critic": agent.target_critics.state_dict(),
        "target_actor": agent.actor_target.state_dict(),
        "optimizer_critic": agent.critics.optimizer.state_dict(),
        "optimizer_actor": agent.actor.optimizer.state_dict(),
    }


@pytest.fixture(scope="function")
def saved_n_plus_m_agent_model(learning_role_n_plus_m, tmp_path) -> tuple[str, dict]:
    learning_role_n_plus_m.initialize_policy()
    save_dir = tmp_path / "saved_model_n_plus_m"
    save_dir.mkdir(parents=True, exist_ok=True)
    learning_role_n_plus_m.rl_algorithm.save_params(directory=str(save_dir))
    agent = learning_role_n_plus_m.rl_strats["agent_0"]
    return str(save_dir), {
        "critic": agent.critics.state_dict(),
        "actor": agent.actor.state_dict(),
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


@pytest.mark.require_learning
def test_maddpg_algorithm_class(learning_role_n):
    learning_role_n.initialize_policy()
    assert isinstance(learning_role_n.rl_algorithm, DDPG)


@pytest.mark.require_learning
def test_maddpg_save_params_creates_files(learning_role_n, tmp_path):
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "model_save_test"

    learning_role_n.rl_algorithm.save_params(directory=str(save_dir))

    assert os.path.exists(save_dir / "critics" / "critic_agent_0.pt")
    assert os.path.exists(save_dir / "critics" / "critic_agent_1.pt")
    assert os.path.exists(save_dir / "actors" / "actor_agent_0.pt")
    assert os.path.exists(save_dir / "actors" / "actor_agent_1.pt")


@pytest.mark.require_learning
def test_maddpg_save_params_u_id_order(learning_role_n, tmp_path):
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "u_id_order_test"

    learning_role_n.rl_algorithm.save_params(directory=str(save_dir))

    order_file = save_dir / "critics" / "u_id_order.json"
    assert order_file.exists(), "u_id_order.json must be written alongside critic files"
    with open(order_file) as f:
        mapping = json.load(f)
    assert mapping.get("u_id_order") == ["agent_0", "agent_1"]


@pytest.mark.require_learning
def test_maddpg_load_matching_n(base_learning_config, saved_n_agent_model):
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
        original_states["target_critic"], agent.target_critics.state_dict()
    )
    assert compare_state_dicts(
        original_states["target_actor"], agent.actor_target.state_dict()
    )
    assert compare_state_dicts(
        deepcopy(original_states["optimizer_critic"]),
        deepcopy(agent.critics.optimizer.state_dict()),
    )
    assert compare_state_dicts(
        deepcopy(original_states["optimizer_actor"]),
        deepcopy(agent.actor.optimizer.state_dict()),
    )


def make_state_dicts(
    obs_base: int,
    act_dim: int,
    unique_obs: int,
    old_id_order: list[str],
    new_id_order: list[str],
    hidden_dims: list[int],
):
    import torch as th

    class FakeModel:
        def __init__(self, sd):
            self._sd = sd

        def state_dict(self):
            return self._sd

    old_n = len(old_id_order)
    new_n = len(new_id_order)
    old_input_dim = obs_base + unique_obs * max(0, old_n - 1) + act_dim * old_n
    new_input_dim = obs_base + unique_obs * max(0, new_n - 1) + act_dim * new_n

    # Build baseline for new model
    baseline_new = {}
    prefix = "q_layers"
    baseline_new[f"{prefix}.0.weight"] = th.randn(hidden_dims[0], new_input_dim)
    baseline_new[f"{prefix}.0.bias"] = th.randn(hidden_dims[0])
    for i in range(1, len(hidden_dims)):
        baseline_new[f"{prefix}.{i}.weight"] = th.randn(hidden_dims[i], hidden_dims[i - 1])
        baseline_new[f"{prefix}.{i}.bias"] = th.randn(hidden_dims[i])

    # Build old_state with matching dims
    old_state = {}
    old_state[f"{prefix}.0.weight"] = th.randn(hidden_dims[0], old_input_dim) + 10.0
    old_state[f"{prefix}.0.bias"] = th.randn(hidden_dims[0]) + 20.0
    for i in range(1, len(hidden_dims)):
        old_state[f"{prefix}.{i}.weight"] = baseline_new[f"{prefix}.{i}.weight"].clone()
        old_state[f"{prefix}.{i}.bias"] = baseline_new[f"{prefix}.{i}.bias"].clone()

    model = FakeModel(baseline_new)
    return model, old_state, baseline_new


@pytest.mark.require_learning
def test_ddpg_load_transfer_n_plus_m(
    learning_role_n_plus_m, saved_n_agent_model, base_learning_config
):
    """Saving a 2-agent DDPG model and loading it into a 3-agent setup must
    transfer matching obs and action weight slices while leaving new-agent
    slices at their random initialisation.
    """
    save_dir, original_states = saved_n_agent_model
    n_agents_old = 2
    n_agents_new = 3

    learning_role_n_plus_m.initialize_policy()

    pre_state = deepcopy(
        learning_role_n_plus_m.rl_strats["agent_0"].critics.state_dict()
    )
    pre_opt_state = deepcopy(
        learning_role_n_plus_m.rl_strats["agent_0"].critics.optimizer.state_dict()
    )

    learning_role_n_plus_m.rl_algorithm.load_params(directory=save_dir)

    post_state = learning_role_n_plus_m.rl_strats["agent_0"].critics.state_dict()
    post_target = learning_role_n_plus_m.rl_strats["agent_0"].target_critics.state_dict()
    post_opt_state = (
        learning_role_n_plus_m.rl_strats["agent_0"].critics.optimizer.state_dict()
    )

    assert not compare_state_dicts(pre_state, post_state)

    obs_base = base_learning_config["obs_dim"]
    act_dim = base_learning_config["act_dim"]
    unique_obs = base_learning_config["unique_obs_dim"]

    old_total_obs = obs_base + unique_obs * (n_agents_old - 1)
    new_total_obs = obs_base + unique_obs * (n_agents_new - 1)
    copy_agent_count = min(n_agents_old, n_agents_new)
    copy_unique_obs_count = unique_obs * (copy_agent_count - 1)
    copy_obs_end_idx = obs_base + copy_unique_obs_count
    copy_action_count = act_dim * copy_agent_count

    for prefix in ["q_layers"]:
        w_key = f"{prefix}.0.weight"
        b_key = f"{prefix}.0.bias"
        assert th.equal(
            post_state[w_key][:, :obs_base],
            original_states["critic"][w_key][:, :obs_base],
        )
        # Matched unique-obs slices
        if copy_obs_end_idx > obs_base:
            assert th.equal(
                post_state[w_key][:, obs_base:copy_obs_end_idx],
                original_states["critic"][w_key][:, obs_base:copy_obs_end_idx],
            )
        # Matched action slices
        assert th.equal(
            post_state[w_key][:, new_total_obs : new_total_obs + copy_action_count],
            original_states["critic"][w_key][
                :, old_total_obs : old_total_obs + copy_action_count
            ],
        )

    # Target critic must copy critic after transfer
    assert compare_state_dicts(post_state, post_target)
    # Optimizer state is preserved
    assert compare_state_dicts(post_opt_state, pre_opt_state)


@pytest.mark.require_learning
def test_ddpg_load_transfer_n_minus_m(
    learning_role_n, saved_n_plus_m_agent_model, base_learning_config
):
    """Saving a 3-agent DDPG model and loading it into a 2-agent setup must
    transfer only the overlapping obs and action weight slices.
    """
    save_dir, original_states = saved_n_plus_m_agent_model
    n_agents_old = 3
    n_agents_new = 2

    learning_role_n.initialize_policy()

    pre_state = deepcopy(learning_role_n.rl_strats["agent_0"].critics.state_dict())
    learning_role_n.rl_algorithm.load_params(directory=save_dir)

    post_state = learning_role_n.rl_strats["agent_0"].critics.state_dict()
    post_target = learning_role_n.rl_strats["agent_0"].target_critics.state_dict()

    assert not compare_state_dicts(pre_state, post_state)

    obs_base = base_learning_config["obs_dim"]
    act_dim = base_learning_config["act_dim"]
    unique_obs = base_learning_config["unique_obs_dim"]

    old_total_obs = obs_base + unique_obs * (n_agents_old - 1)
    new_total_obs = obs_base + unique_obs * (n_agents_new - 1)
    copy_agent_count = min(n_agents_old, n_agents_new)
    copy_unique_obs_count = unique_obs * (copy_agent_count - 1)
    copy_obs_end_idx = obs_base + copy_unique_obs_count
    copy_action_count = act_dim * copy_agent_count

    for prefix in ["q_layers"]:
        w_key = f"{prefix}.0.weight"

        assert th.equal(
            post_state[w_key][:, :obs_base],
            original_states["critic"][w_key][:, :obs_base],
        )
        if copy_obs_end_idx > obs_base:
            assert th.equal(
                post_state[w_key][:, obs_base:copy_obs_end_idx],
                original_states["critic"][w_key][:, obs_base:copy_obs_end_idx],
            )
        assert th.equal(
            post_state[w_key][:, new_total_obs : new_total_obs + copy_action_count],
            original_states["critic"][w_key][
                :, old_total_obs : old_total_obs + copy_action_count
            ],
        )

    assert compare_state_dicts(post_state, post_target)


@pytest.mark.parametrize(
    "new_id_order",
    [
        ["pp_5", "pp_6", "pp_3", "pp_4", "st_1"],
        ["pp_3", "pp_4", "st_1", "pp_5", "pp_6"],
        ["pp_3", "pp_5", "pp_4", "pp_6", "st_1"],
        ["pp_3", "st_1"],
    ],
)
@pytest.mark.require_learning
def test_ddpg_transfer_weights_various_orders(new_id_order):
    import torch as th

    from assume.reinforcement_learning.learning_utils import transfer_weights

    obs_base = 10
    act_dim = 3
    unique_obs = 2
    hidden_dims = [5, 4]
    old_id_order = ["pp_3", "pp_4", "st_1"]

    model, old_state, baseline = make_state_dicts(
        obs_base, act_dim, unique_obs, old_id_order, new_id_order, hidden_dims
    )
    new_state = transfer_weights(
        model, old_state, old_id_order, new_id_order, obs_base, act_dim, unique_obs
    )
    assert isinstance(new_state, dict), "transfer_weights must return a dict for DDPG"

    old_n = len(old_id_order)
    new_n = len(new_id_order)
    old_obs_tot = obs_base + unique_obs * max(0, old_n - 1)
    new_obs_tot = obs_base + unique_obs * max(0, new_n - 1)

    prefix = "q_layers"
    w_old = old_state[f"{prefix}.0.weight"]
    w_base = baseline[f"{prefix}.0.weight"]
    w_new = new_state[f"{prefix}.0.weight"]

    # Shared obs_base copied from old
    assert th.equal(w_new[:, :obs_base], w_old[:, :obs_base])

    # unique_obs slices per agent
    for new_idx, u in enumerate(new_id_order):
        if new_idx == 0:
            continue
        s = obs_base + unique_obs * (new_idx - 1)
        e = s + unique_obs
        if u in old_id_order:
            old_idx = old_id_order.index(u)
            if old_idx > 0:
                os_ = obs_base + unique_obs * (old_idx - 1)
                oe = os_ + unique_obs
                assert th.equal(w_new[:, s:e], w_old[:, os_:oe])
        else:
            assert th.equal(w_new[:, s:e], w_base[:, s:e])

    # action slices per agent
    for new_idx, u in enumerate(new_id_order):
        s = new_obs_tot + act_dim * new_idx
        e = s + act_dim
        if u in old_id_order:
            old_idx = old_id_order.index(u)
            os_ = old_obs_tot + act_dim * old_idx
            oe = os_ + act_dim
            assert th.equal(w_new[:, s:e], w_old[:, os_:oe])
        else:
            assert th.equal(w_new[:, s:e], w_base[:, s:e])


@pytest.mark.require_learning
def test_maddpg_load_corrupted_critic(tmp_path, base_learning_config):
    config = copy(base_learning_config)
    learning = Learning(config["learning_config"], start, end)
    learning.rl_strats["agent_0"] = LearningStrategy(**config, learning_role=learning)
    learning.initialize_policy()

    original_state = deepcopy(learning.rl_strats["agent_0"].critics.state_dict())

    corrupted_dir = tmp_path / "critics"
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    corrupted_obj = {
        "critic": original_state,
        "critic_target": {
            k: v[:1] if isinstance(v, th.Tensor) and v.ndim > 0 else v
            for k, v in original_state.items()
        },
    }
    th.save(corrupted_obj, corrupted_dir / "critic_agent_0.pt")
    learning.rl_algorithm.load_critic_params(directory=str(tmp_path))

    loaded_state = learning.rl_strats["agent_0"].critics.state_dict()
    assert compare_state_dicts(loaded_state, original_state)


@pytest.mark.parametrize(
    "mod_field, mod_value, expected_error",
    [
        ("foresight", 99, "All foresight values must be the same"),
        ("act_dim", 99, "All action dimensions must be the same"),
        ("unique_obs_dim", 99, "All unique_obs_dim values must be the same"),
        (
            "num_timeseries_obs_dim",
            99,
            "All num_timeseries_obs_dim values must be the same",
        ),
    ],
)
@pytest.mark.require_learning
def test_initialize_policy_dimension_mismatch(
    base_learning_config, mod_field, mod_value, expected_error
):
    config = copy(base_learning_config)
    config["num_timeseries_obs_dim"] = 1

    learn = Learning(config["learning_config"], start, end)
    strat_0 = LearningStrategy(**config, learning_role=learn)

    config_mismatch = copy(config)
    config_mismatch[mod_field] = mod_value
    strat_1 = LearningStrategy(**config_mismatch, learning_role=learn)

    learn.rl_strats["agent_0"] = strat_0
    learn.rl_strats["agent_1"] = strat_1

    with pytest.raises(ValueError, match=expected_error):
        learn.rl_algorithm.initialize_policy()


@pytest.mark.require_learning
def test_initialize_policy_all_dimensions_match(base_learning_config):
    config = copy(base_learning_config)
    config["num_timeseries_obs_dim"] = 1

    learn = Learning(config["learning_config"], start, end)
    for agent_id in ("agent_0", "agent_1", "agent_2"):
        learn.rl_strats[agent_id] = LearningStrategy(**config, learning_role=learn)

    try:
        learn.rl_algorithm.initialize_policy()
    except Exception as e:
        pytest.fail(f"initialize_policy raised an unexpected error: {e}")