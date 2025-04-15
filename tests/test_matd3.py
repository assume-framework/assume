# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
from copy import deepcopy

import pytest

try:
    import torch as th

    from assume.common.base import LearningStrategy
    from assume.reinforcement_learning.learning_role import Learning
    from assume.reinforcement_learning.learning_utils import get_hidden_sizes

except ImportError:
    pass


@pytest.fixture
def base_learning_config():
    return {
        "obs_dim": 10,
        "act_dim": 3,
        "train_freq": "1h",
        "unique_obs_dim": 2,
        "algorithm": "matd3",
        "actor_architecture": "mlp",
        "learning_mode": True,
        "evaluation_mode": False,
        "training_episodes": 1,
        "episodes_collecting_initial_experience": 0,
        "continue_learning": False,
        "trained_policies_save_path": None,
        "early_stopping_steps": 10,
        "early_stopping_threshold": 0.05,
        "learning_rate": 1e-4,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "gradient_steps": 1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    }


@pytest.fixture(scope="function")
def learning_role_n(base_learning_config):
    config = base_learning_config.copy()
    learn = Learning(config)
    learn.rl_strats["agent_0"] = LearningStrategy(**config)
    learn.rl_strats["agent_1"] = LearningStrategy(**config)
    return learn


@pytest.fixture(scope="function")
def learning_role_n_plus_m(base_learning_config):
    config = base_learning_config.copy()
    learn = Learning(config)
    learn.rl_strats["agent_0"] = LearningStrategy(**config)
    learn.rl_strats["agent_1"] = LearningStrategy(**config)
    learn.rl_strats["agent_2"] = LearningStrategy(**config)
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


def compare_state_dicts(dict1, dict2):
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


# -------------------- Tests --------------------


@pytest.mark.require_learning
def test_td3_save_params(learning_role_n, tmp_path):
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "model_save_test"
    td3_n = learning_role_n.rl_algorithm

    td3_n.save_params(directory=str(save_dir))

    assert os.path.exists(save_dir / "critics" / "critic_agent_0.pt")
    assert os.path.exists(save_dir / "actors" / "actor_agent_0.pt")
    assert os.path.exists(save_dir / "critics" / "critic_agent_1.pt")
    assert os.path.exists(save_dir / "actors" / "actor_agent_1.pt")


@pytest.mark.require_learning
def test_td3_load_matching_n(base_learning_config, saved_n_agent_model):
    save_dir, original_states = saved_n_agent_model

    config_n_new = base_learning_config.copy()
    learn_n_new = Learning(config_n_new)
    learn_n_new.rl_strats["agent_0"] = LearningStrategy(**config_n_new)
    learn_n_new.rl_strats["agent_1"] = LearningStrategy(**config_n_new)
    learn_n_new.initialize_policy()

    learn_n_new.rl_algorithm.load_params(directory=save_dir)

    agent = learn_n_new.rl_strats["agent_0"]

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


@pytest.mark.require_learning
def test_td3_load_transfer_n_plus_m(
    learning_role_n_plus_m, saved_n_agent_model, base_learning_config
):
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
    post_target = learning_role_n_plus_m.rl_strats[
        "agent_0"
    ].target_critics.state_dict()
    post_opt_state = learning_role_n_plus_m.rl_strats[
        "agent_0"
    ].critics.optimizer.state_dict()

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
    hidden_arch_changed = get_hidden_sizes(
        original_states["critic"], "q1_layers"
    ) != get_hidden_sizes(post_state, "q1_layers")

    for prefix in ["q1_layers", "q2_layers"]:
        w_key = f"{prefix}.0.weight"
        b_key = f"{prefix}.0.bias"

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

        if not hidden_arch_changed:
            assert th.equal(post_state[b_key], original_states["critic"][b_key])

    assert compare_state_dicts(post_state, post_target)
    assert compare_state_dicts(post_opt_state, pre_opt_state)


@pytest.mark.require_learning
def test_td3_load_transfer_n_minus_m(
    learning_role_n, saved_n_plus_m_agent_model, base_learning_config
):
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
    hidden_arch_changed = get_hidden_sizes(
        original_states["critic"], "q1_layers"
    ) != get_hidden_sizes(post_state, "q1_layers")

    for prefix in ["q1_layers", "q2_layers"]:
        w_key = f"{prefix}.0.weight"
        b_key = f"{prefix}.0.bias"

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
        if not hidden_arch_changed:
            assert th.equal(post_state[b_key], original_states["critic"][b_key])

    assert compare_state_dicts(post_state, post_target)


@pytest.mark.require_learning
def test_td3_load_corrupted_or_incomplete_critic(tmp_path, base_learning_config):
    config = base_learning_config.copy()
    learning = Learning(config)
    learning.rl_strats["agent_0"] = LearningStrategy(**config)
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
        ("obs_dim", 99, "All observation dimensions must be the same"),
        ("act_dim", 99, "All action dimensions must be the same"),
        ("unique_obs_dim", 99, "All unique_obs_dim values must be the same"),
        (
            "num_timeseries_obs_dim",
            99,
            "All num_timeseries_obs_dim values must be the same",
        ),
    ],
)
def test_initialize_policy_dimension_mismatch(
    base_learning_config, mod_field, mod_value, expected_error
):
    """
    Test that mismatches in observation/action/unique/timeseries dims raise ValueErrors.
    """
    config = base_learning_config.copy()
    config["num_timeseries_obs_dim"] = 1  # Ensure field exists for valid check

    learn = Learning(config)

    # Create one agent with default config
    strat_0 = LearningStrategy(**config)
    # Create second agent with mismatching value
    config_mismatch = config.copy()
    config_mismatch[mod_field] = mod_value
    strat_1 = LearningStrategy(**config_mismatch)

    learn.rl_strats["agent_0"] = strat_0
    learn.rl_strats["agent_1"] = strat_1

    # This should raise a ValueError with the expected message
    with pytest.raises(ValueError, match=expected_error):
        learn.rl_algorithm.initialize_policy()


def test_initialize_policy_all_dimensions_match(base_learning_config):
    """
    Test that initialize_policy succeeds with all matching dimensions.
    """
    config = base_learning_config.copy()
    config["num_timeseries_obs_dim"] = 1  # Ensure the optional field is populated

    learn = Learning(config)
    learn.rl_strats["agent_0"] = LearningStrategy(**config)
    learn.rl_strats["agent_1"] = LearningStrategy(**config)
    learn.rl_strats["agent_2"] = LearningStrategy(**config)

    try:
        learn.rl_algorithm.initialize_policy()  # Should not raise
    except Exception as e:
        pytest.fail(f"initialize_policy raised an unexpected error: {e}")
