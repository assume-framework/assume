# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import json
import logging
import os
from copy import deepcopy

import pytest

try:
    import torch as th

    from assume.common.base import LearningStrategy
    from assume.reinforcement_learning.learning_role import Learning
    from assume.reinforcement_learning.learning_utils import (
        get_hidden_sizes,
        transfer_weights,
    )

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


class FakeModel:
    """Minimal model wrapper so model.state_dict() returns our baseline."""

    def __init__(self, baseline: dict):
        self._sd = baseline

    def state_dict(self):
        return self._sd


def make_state_dicts(
    obs_base: int,
    act_dim: int,
    unique_obs: int,
    old_id_order: list[str],
    new_id_order: list[str],
    hidden_dims: list[int],
):
    """
    Construct:
      - baseline_new: weights for the 'new' model (with new_input_dim)
      - old_state: weights for the 'old' model (with old_input_dim),
        offset by +10/+20 so we can detect copies.
    Returns (model, old_state, baseline_new).
    """
    old_n = len(old_id_order)
    new_n = len(new_id_order)

    # Compute exact input dims
    old_input_dim = obs_base + unique_obs * max(0, old_n - 1) + act_dim * old_n
    new_input_dim = obs_base + unique_obs * max(0, new_n - 1) + act_dim * new_n

    # Build baseline for new model
    baseline_new = {}
    for prefix in ("q1_layers", "q2_layers"):
        # input layer
        baseline_new[f"{prefix}.0.weight"] = th.randn(hidden_dims[0], new_input_dim)
        baseline_new[f"{prefix}.0.bias"] = th.randn(hidden_dims[0])
        # deeper layers
        for i in range(1, len(hidden_dims)):
            baseline_new[f"{prefix}.{i}.weight"] = th.randn(
                hidden_dims[i], hidden_dims[i - 1]
            )
            baseline_new[f"{prefix}.{i}.bias"] = th.randn(hidden_dims[i])

    # Build old_state with matching dims and offsets
    old_state = {}
    for prefix in ("q1_layers", "q2_layers"):
        # input layer
        w0 = th.randn(hidden_dims[0], old_input_dim) + 10.0
        b0 = th.randn(hidden_dims[0]) + 20.0
        old_state[f"{prefix}.0.weight"] = w0
        old_state[f"{prefix}.0.bias"] = b0
        # deeper layers: clone from baseline_new for shape consistency
        for i in range(1, len(hidden_dims)):
            old_state[f"{prefix}.{i}.weight"] = baseline_new[
                f"{prefix}.{i}.weight"
            ].clone()
            old_state[f"{prefix}.{i}.bias"] = baseline_new[f"{prefix}.{i}.bias"].clone()

    # FakeModel returns baseline_new for model.state_dict()
    model = FakeModel(baseline_new)
    return model, old_state, baseline_new


# -------------------- Tests --------------------


@pytest.mark.require_learning
def test_td3_save_params(learning_role_n, tmp_path):
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "model_save_test"
    td3_n = learning_role_n.rl_algorithm

    td3_n.save_params(directory=str(save_dir))

    assert os.path.exists(save_dir / "critics" / "critic_agent_0.pt")
    assert os.path.exists(save_dir / "critics" / "critic_agent_1.pt")

    assert os.path.exists(save_dir / "actors" / "actor_agent_0.pt")
    assert os.path.exists(save_dir / "actors" / "actor_agent_1.pt")

    # Verify u_id_order.json was written correctly
    order_file = save_dir / "critics" / "u_id_order.json"
    assert order_file.exists(), "u_id_order.json must exist"
    with open(order_file) as f:
        mapping = json.load(f)
    assert mapping.get("u_id_order") == ["agent_0", "agent_1"]


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
@pytest.mark.require_learning
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


@pytest.mark.require_learning
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


@pytest.mark.parametrize(
    "new_id_order",
    [
        # 1) new units inserted before existing
        ["pp_5", "pp_6", "pp_3", "pp_4", "st_1"],
        # 2) new units appended after existing
        ["pp_3", "pp_4", "st_1", "pp_5", "pp_6"],
        # 3) new units mixed in the middle
        ["pp_3", "pp_5", "pp_4", "pp_6", "st_1"],
        # 4) one old unit deleted (pp_4 removed)
        ["pp_3", "st_1"],
    ],
)
@pytest.mark.parametrize("prefix", ["q1_layers", "q2_layers"])
@pytest.mark.require_learning
def test_transfer_weights_various_orders(prefix, new_id_order):
    obs_base = 10
    act_dim = 3
    unique_obs = 2
    hidden_dims = [5, 4]
    old_id_order = ["pp_3", "pp_4", "st_1"]

    # Build our fake model and states
    model, old_state, baseline = make_state_dicts(
        obs_base, act_dim, unique_obs, old_id_order, new_id_order, hidden_dims
    )

    # Run transfer
    new_state = transfer_weights(
        model, old_state, old_id_order, new_id_order, obs_base, act_dim, unique_obs
    )
    assert isinstance(new_state, dict)

    # Recompute dims
    old_n = len(old_id_order)
    new_n = len(new_id_order)
    old_obs_tot = obs_base + unique_obs * max(0, old_n - 1)
    new_obs_tot = obs_base + unique_obs * max(0, new_n - 1)

    w_old = old_state[f"{prefix}.0.weight"]
    w_base = baseline[f"{prefix}.0.weight"]
    w_new = new_state[f"{prefix}.0.weight"]

    # 1) shared obs_base copied from old
    assert th.equal(w_new[:, :obs_base], w_old[:, :obs_base])

    # 2) unique_obs slices
    for new_idx, u in enumerate(new_id_order):
        if new_idx == 0:
            continue
        start = obs_base + unique_obs * (new_idx - 1)
        end = start + unique_obs

        if u in old_id_order:
            old_idx = old_id_order.index(u)
            if old_idx > 0:
                ostart = obs_base + unique_obs * (old_idx - 1)
                oend = ostart + unique_obs
                assert th.equal(w_new[:, start:end], w_old[:, ostart:oend])
        else:
            # brand-new unit: matches baseline
            assert th.equal(w_new[:, start:end], w_base[:, start:end])

    # 3) action slices
    for new_idx, u in enumerate(new_id_order):
        start = new_obs_tot + act_dim * new_idx
        end = start + act_dim

        if u in old_id_order:
            old_idx = old_id_order.index(u)
            ostart = old_obs_tot + act_dim * old_idx
            oend = ostart + act_dim
            assert th.equal(w_new[:, start:end], w_old[:, ostart:oend])
        else:
            assert th.equal(w_new[:, start:end], w_base[:, start:end])

    # 4) deeper layers copied wholesale from old_state
    for i in range(1, len(hidden_dims)):
        w_key = f"{prefix}.{i}.weight"
        b_key = f"{prefix}.{i}.bias"
        assert th.equal(new_state[w_key], old_state[w_key])
        assert th.equal(new_state[b_key], old_state[b_key])


@pytest.fixture
def clustering_config(base_learning_config):
    cfg = base_learning_config.copy()
    cfg.update(
        {
            "actor_architecture": "contextual_fusion",
            "actor_clustering_method": "k-means",
            "clustering_method_kwargs": {"n_clusters": 2, "random_state": 42},
            "context_dim": 2,
        }
    )
    return cfg


@pytest.mark.require_learning
def test_shared_actor_clustering_and_mapping(clustering_config):
    # 1) build Learning + 3 strategies
    learn = Learning(clustering_config)
    contexts = {
        "A": th.tensor([1.0, 0.0]),
        "B": th.tensor([1.0, 0.0]),
        "C": th.tensor([0.0, 1.0]),
    }
    for uid, ctx in contexts.items():
        strategy = LearningStrategy(**clustering_config)
        strategy.unit_id = uid
        strategy.context = ctx
        learn.rl_strats[uid] = strategy

    # 2) initialize → creates clusters on rl_algorithm
    learn.initialize_policy()

    alg = learn.rl_algorithm

    # 3) mapping covers all unit_ids
    assert set(alg.cluster_mapping.keys()) == set(contexts.keys())

    # 4) we expect two clusters by KMeans
    assert len(alg.clusters) == 2

    # 5) A and B in same cluster; C in a different one
    clA = alg.cluster_mapping["A"]
    clB = alg.cluster_mapping["B"]
    clC = alg.cluster_mapping["C"]
    assert clA == clB
    assert clC != clA

    # 6) clusters dict matches mapping
    assert {s.unit_id for s in alg.clusters[clA]} == {"A", "B"}
    assert {s.unit_id for s in alg.clusters[clC]} == {"C"}

    # 7) shared_actors keys match clusters, and instances are shared
    assert set(alg.shared_actors.keys()) == set(alg.clusters.keys())
    actorAB = alg.shared_actors[clA]
    actorC = alg.shared_actors[clC]
    assert learn.rl_strats["A"].actor is actorAB
    assert learn.rl_strats["B"].actor is actorAB
    assert learn.rl_strats["C"].actor is actorC
    assert actorAB is not actorC


@pytest.mark.require_learning
def test_load_shared_actor_params_mismatch_and_new_unit(
    clustering_config, tmp_path, caplog
):
    # --- 1) First run: create and save model with 3 units ---
    learn1 = Learning(clustering_config)
    contexts1 = {
        "u1": th.tensor([1.0, 0.0]),
        "u2": th.tensor([1.0, 0.0]),
        "u3": th.tensor([0.0, 1.0]),
    }
    for uid, ctx in contexts1.items():
        strategy = LearningStrategy(**clustering_config)
        strategy.unit_id = uid
        strategy.context = ctx
        learn1.rl_strats[uid] = strategy
    learn1.initialize_policy()

    save_dir = tmp_path / "shared_model"
    save_dir.mkdir()
    learn1.rl_algorithm.save_params(directory=str(save_dir))
    old_map = learn1.rl_algorithm.cluster_mapping.copy()

    # --- 2) Second run: initialize, then corrupt mapping + add new unit ---
    learn2 = Learning(clustering_config)
    for uid, ctx in contexts1.items():
        strategy = LearningStrategy(**clustering_config)
        strategy.unit_id = uid
        strategy.context = ctx
        learn2.rl_strats[uid] = strategy
    learn2.initialize_policy()

    alg2 = learn2.rl_algorithm
    # Force both u1 and u2 into u3's old cluster to create a real mismatch
    wrong_cluster = old_map["u3"]
    alg2.cluster_mapping["u1"] = wrong_cluster
    alg2.cluster_mapping["u2"] = wrong_cluster

    # Add a brand-new unit u4 in the cluster of old_map["u1"]
    strat4 = LearningStrategy(**clustering_config)
    strat4.unit_id = "u4"
    strat4.context = th.tensor([1.0, 0.0])
    learn2.rl_strats["u4"] = strat4
    alg2.cluster_mapping["u4"] = old_map["u1"]
    alg2._rebuild_clusters_from_mapping()

    caplog.set_level(logging.INFO)

    # --- 3) Load & reconcile ---
    alg2.load_shared_actor_params(directory=str(save_dir))

    # Should log that mismatches were detected and reverted
    assert "Reverting them to their original clusters" in caplog.text

    # u1 and u2 should be restored to their original clusters
    assert alg2.cluster_mapping["u1"] == old_map["u1"]
    assert alg2.cluster_mapping["u2"] == old_map["u2"]

    # The new unit u4 remains in its assigned cluster
    assert alg2.cluster_mapping["u4"] == old_map["u1"]

    # Clusters rebuilt accordingly
    restored_cl = alg2.cluster_mapping["u1"]
    assert any(s.unit_id == "u1" for s in alg2.clusters[restored_cl])

    # And the shared actor for that restored cluster was loaded
    actor = alg2.shared_actors[restored_cl]
    assert getattr(actor, "loaded", False) is True


@pytest.mark.require_learning
def test_load_shared_actor_params_no_previous_mapping(
    clustering_config, tmp_path, caplog
):
    """
    Case 1: actors/ exists but no cluster_mapping.json → warning, no weights loaded,
    all shared actors remain at loaded=False, and we log 'No previous cluster_mapping.json'.
    """
    # 1) Initialize policy so shared_actors exist
    learn = Learning(clustering_config)
    for uid in ["A", "B", "C"]:
        strategy = LearningStrategy(**clustering_config)
        strategy.unit_id = uid
        strategy.context = th.randn(clustering_config["context_dim"])
        learn.rl_strats[uid] = strategy
    learn.initialize_policy()
    alg = learn.rl_algorithm

    # 2) Prepare a save_dir with an empty actors/ subdir but no mapping
    save_dir = tmp_path / "no_map"
    actors_dir = save_dir / "actors"
    actors_dir.mkdir(parents=True)
    caplog.set_level(logging.WARNING)

    # 3) Call load_shared_actor_params
    alg.load_shared_actor_params(directory=str(save_dir))

    # 4) Check log for missing mapping
    assert "No previous cluster_mapping.json" in caplog.text

    # 5) After loading, none of the shared_actors should be marked loaded
    assert all(not getattr(a, "loaded", False) for a in alg.shared_actors.values())


@pytest.mark.require_learning
def test_load_shared_actor_params_no_actors_directory(
    clustering_config, tmp_path, caplog
):
    """
    Case 2: no actors/ subdirectory at all → warning and early return,
    shared_actors remain exactly as before.
    """
    learn = Learning(clustering_config)
    for uid in ["X", "Y"]:
        strategy = LearningStrategy(**clustering_config)
        strategy.unit_id = uid
        strategy.context = th.randn(clustering_config["context_dim"])
        learn.rl_strats[uid] = strategy
    learn.initialize_policy()
    alg = learn.rl_algorithm

    # record original actor objects
    before_shared = dict(alg.shared_actors)

    save_dir = tmp_path / "no_actors"  # no actors/ here
    save_dir.mkdir()
    caplog.set_level(logging.WARNING)

    alg.load_shared_actor_params(directory=str(save_dir))

    # should have early-returned; actors dict unchanged
    assert alg.shared_actors.keys() == before_shared.keys()
    for cluster, actor in alg.shared_actors.items():
        assert actor is before_shared[cluster]
    # check warning
    assert "No 'actors' folder found; skipping." in caplog.text


@pytest.mark.require_learning
def test_load_shared_actor_params_no_mismatches(clustering_config, tmp_path, caplog):
    """
    Case 3: new clustering exactly matches old mapping → logs 'No mapping mismatches detected'
    and all old clusters get loaded=True.
    """
    # First run: build & save
    learn1 = Learning(clustering_config)
    for uid in ["u1", "u2", "u3"]:
        strategy = LearningStrategy(**clustering_config)
        strategy.unit_id = uid
        strategy.context = th.randn(clustering_config["context_dim"])
        learn1.rl_strats[uid] = strategy
    learn1.initialize_policy()
    alg1 = learn1.rl_algorithm
    save_dir = tmp_path / "match"
    learn1.rl_algorithm.save_params(directory=str(save_dir))
    old_map = alg1.cluster_mapping.copy()
    old_clusters = set(old_map.values())

    # Second run: fresh init, then force mapping to old_map exactly
    learn2 = Learning(clustering_config)
    for uid in ["u1", "u2", "u3"]:
        strategy = LearningStrategy(**clustering_config)
        strategy.unit_id = uid
        strategy.context = th.randn(clustering_config["context_dim"])
        learn2.rl_strats[uid] = strategy
    learn2.initialize_policy()
    alg2 = learn2.rl_algorithm
    alg2.cluster_mapping = old_map.copy()
    alg2._rebuild_clusters_from_mapping()

    caplog.set_level(logging.INFO)
    alg2.load_shared_actor_params(directory=str(save_dir))

    assert "No mapping mismatches detected; keeping current clusters." in caplog.text
    # all old clusters should have been loaded
    for cluster in old_clusters:
        assert alg2.shared_actors[cluster].loaded is True


@pytest.mark.require_learning
def test_brand_new_clusters(clustering_config, tmp_path, caplog):
    """
    Case 4: first run with 2 clusters, second run with 3 requested →
    old clusters load=True, any brand-new clusters remain loaded=False.
    """
    # First run: request 2 clusters
    cfg2 = clustering_config.copy()
    cfg2["clustering_method_kwargs"] = {"n_clusters": 2, "random_state": 42}
    learn1 = Learning(cfg2)
    for uid, ctx in zip(
        ["A", "B", "C"],
        [th.tensor([1.0, 0.0]), th.tensor([1.0, 0.0]), th.tensor([0.0, 1.0])],
    ):
        strategy = LearningStrategy(**cfg2)
        strategy.unit_id = uid
        strategy.context = ctx
        learn1.rl_strats[uid] = strategy
    learn1.initialize_policy()
    alg1 = learn1.rl_algorithm
    save_dir = tmp_path / "two"
    save_dir.mkdir()
    alg1.save_params(directory=str(save_dir))
    old_clusters = set(alg1.cluster_mapping.values())

    # Second run: request 3 clusters
    cfg3 = clustering_config.copy()
    cfg3["clustering_method_kwargs"] = {"n_clusters": 3, "random_state": 42}
    learn2 = Learning(cfg3)
    for uid, ctx in zip(
        ["A", "B", "C"],
        [th.tensor([1.0, 0.0]), th.tensor([0.0, 1.0]), th.tensor([0.5, 0.5])],
    ):
        strategy = LearningStrategy(**cfg3)
        strategy.unit_id = uid
        strategy.context = ctx
        learn2.rl_strats[uid] = strategy
    learn2.initialize_policy()
    alg2 = learn2.rl_algorithm

    caplog.set_level(logging.INFO)
    alg2.load_shared_actor_params(directory=str(save_dir))

    new_clusters = set(alg2.shared_actors.keys())
    # 1) All old clusters must still be present
    assert old_clusters.issubset(new_clusters)

    # 2) Old clusters were loaded
    for cluster in old_clusters:
        assert alg2.shared_actors[cluster].loaded is True

    # 3) Any truly new clusters remain unloaded
    for cluster in new_clusters - old_clusters:
        assert alg2.shared_actors[cluster].loaded is False


@pytest.mark.require_learning
def test_single_strategy_clustering_fallback(clustering_config):
    """
    Case 5: only one strategy or identical contexts → always 1 cluster,
    even if n_clusters=3 requested.
    """
    # Force n_clusters=3 but identical contexts
    cfg = clustering_config.copy()
    cfg["clustering_method_kwargs"] = {"n_clusters": 3, "random_state": 0}
    learn = Learning(cfg)
    # one strategy
    strategy = LearningStrategy(**cfg)
    strategy.unit_id = "solo"
    strategy.context = th.randn(cfg["context_dim"])
    learn.rl_strats["solo"] = strategy
    learn.initialize_policy()
    alg = learn.rl_algorithm
    assert len(alg.clusters) == 1
    assert set(alg.shared_actors.keys()) == {0}

    # now two strategies with identical context
    learn2 = Learning(cfg)
    for uid in ["X", "Y"]:
        s = LearningStrategy(**cfg)
        s.unit_id = uid
        s.context = th.zeros(cfg["context_dim"])
        learn2.rl_strats[uid] = s
    learn2.initialize_policy()
    alg2 = learn2.rl_algorithm
    assert len(alg2.clusters) == 1
    assert set(alg2.shared_actors.keys()) == {0}


@pytest.mark.require_learning
def test_max_size_clustering(clustering_config):
    """
    Case 6: actor_clustering_method='max-size' with max_number_of_actors=1 over 3 strategies
    → 3 singleton clusters.
    """
    cfg = clustering_config.copy()
    cfg["actor_clustering_method"] = "max-size"
    cfg["clustering_method_kwargs"] = {"max_number_of_actors": 1, "random_state": 0}
    learn = Learning(cfg)
    for i in range(3):
        uid = f"S{i}"
        s = LearningStrategy(**cfg)
        s.unit_id = uid
        s.context = th.randn(cfg["context_dim"])
        learn.rl_strats[uid] = s

    learn.initialize_policy()
    alg = learn.rl_algorithm

    # Should have exactly 3 clusters (size ≤1 each)
    assert len(alg.clusters) == 3
    for cluster, strategies in alg.clusters.items():
        assert len(strategies) == 1
        # each cluster index appears exactly once in mapping
        assert list(alg.cluster_mapping.values()).count(cluster) == 1
