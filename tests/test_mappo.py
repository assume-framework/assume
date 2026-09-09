# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import json
import os
from copy import copy, deepcopy
from datetime import datetime, timedelta
from types import SimpleNamespace

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
        LSTMActor,
        LSTMActorPPO,
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
                clip_range_vf=0.15,
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
            value=rng.random(n_agents).astype(np.float32),
            log_prob=(rng.random(n_agents).astype(np.float32) - 1.0),
        )
    return buf


def _setup_for_update(learning_role) -> None:
    """Setting minimal attributes needed."""
    learning_role.update_steps = 0
    learning_role.db_addr = None  # disables the context.schedule_instant_message path
    learning_role._context = SimpleNamespace(current_timestamp=learning_role.start)


@pytest.mark.require_learning
def test_mappo_discards_rollout_without_bootstrap_observation(learning_role_n):
    learning_role_n.initialize_policy()
    learning_role_n.rl_algorithm.buffer = _make_rollout_buffer(
        obs_dim=learning_role_n.rl_algorithm.obs_dim,
        act_dim=learning_role_n.rl_algorithm.act_dim,
        n_agents=len(learning_role_n.rl_strats),
        n_steps=1,
    )

    learning_role_n.rl_algorithm.update_policy()

    assert learning_role_n.rl_algorithm.buffer.size() == 0


@pytest.mark.require_learning
def test_mappo_complete_policy_update_changes_actor_and_critic(learning_role_n):
    learning_role_n.initialize_policy()
    _setup_for_update(learning_role_n)
    algorithm = learning_role_n.rl_algorithm
    algorithm.buffer = _make_rollout_buffer(
        obs_dim=algorithm.obs_dim,
        act_dim=algorithm.act_dim,
        n_agents=len(learning_role_n.rl_strats),
        n_steps=6,
    )

    actor_before = {
        unit_id: [parameter.detach().clone() for parameter in strategy.actor.parameters()]
        for unit_id, strategy in learning_role_n.rl_strats.items()
    }
    critic_before = {
        unit_id: [
            parameter.detach().clone() for parameter in strategy.critics.parameters()
        ]
        for unit_id, strategy in learning_role_n.rl_strats.items()
    }

    algorithm.update_policy()

    for unit_id, strategy in learning_role_n.rl_strats.items():
        assert any(
            not th.equal(before, after)
            for before, after in zip(actor_before[unit_id], strategy.actor.parameters())
        )
        assert any(
            not th.equal(before, after)
            for before, after in zip(
                critic_before[unit_id], strategy.critics.parameters()
            )
        )
    assert algorithm.n_updates == 1
    assert learning_role_n.update_steps == 1
    assert algorithm.buffer.size() == 0


@pytest.mark.require_learning
def test_mappo_clears_rollout_between_episodes(learning_role_n):
    learning_role_n.initialize_policy()
    learning_role_n.rl_algorithm.buffer = _make_rollout_buffer(
        obs_dim=learning_role_n.rl_algorithm.obs_dim,
        act_dim=learning_role_n.rl_algorithm.act_dim,
        n_agents=len(learning_role_n.rl_strats),
        n_steps=1,
    )

    inter_episodic_data = learning_role_n.get_inter_episodic_data()

    assert inter_episodic_data["buffer"].size() == 0


@pytest.mark.require_learning
def test_mappo_algorithm_class(learning_role_n):
    """initialize_policy creates a PPO instance as the rl_algorithm."""
    learning_role_n.initialize_policy()
    assert isinstance(learning_role_n.rl_algorithm, PPO)
    assert learning_role_n.rl_algorithm.clip_range_vf == 0.15


@pytest.mark.require_learning
def test_mappo_progress_spans_all_training_episodes(learning_role_n):
    learning_role_n.episodes_done = 2
    learning_role_n._context = SimpleNamespace(
        current_timestamp=(learning_role_n.start + learning_role_n.end) / 2
    )

    progress_remaining = learning_role_n.rl_algorithm.get_progress_remaining()

    assert progress_remaining == pytest.approx(0.75)


@pytest.mark.require_learning
def test_mappo_value_head_uses_separate_initialization_gain():
    critic = CriticPPO(
        n_agents=1,
        obs_dim=3,
        float_type=th.float32,
        unique_obs_dim=1,
    )
    critic.v_layers = th.nn.ModuleList(
        [th.nn.Linear(3, 1), th.nn.ReLU(), th.nn.Linear(1, 1)]
    )

    critic._init_weights()

    assert th.isclose(
        critic.v_layers[0].weight.norm(),
        critic.v_layers[0].weight.new_tensor(np.sqrt(2)),
    )
    assert th.isclose(
        critic.v_layers[2].weight.norm(),
        critic.v_layers[2].weight.new_tensor(1.0),
    )
    assert th.count_nonzero(critic.v_layers[0].bias) == 0
    assert th.count_nonzero(critic.v_layers[2].bias) == 0


@pytest.mark.require_learning
def test_mappo_actors_use_consistent_squashed_gaussian_log_probs():
    actors = [
        ActorPPO(obs_dim=10, act_dim=3, float_type=th.float32),
        LSTMActorPPO(
            obs_dim=10,
            act_dim=3,
            float_type=th.float32,
            unique_obs_dim=2,
            num_timeseries_obs_dim=4,
        ),
    ]
    observations = th.randn(512, 10)

    for actor in actors:
        activation_input = th.tensor([-1.0, 0.0, 1.0])
        assert th.equal(actor.activation_function(activation_input), th.tanh(activation_input))

        th.manual_seed(42)
        actions, sampled_log_probs = actor.get_action_and_log_prob(observations)
        evaluated_log_probs, entropy = actor.evaluate_actions(observations, actions)

        assert th.all(actions > -1.0)
        assert th.all(actions < 1.0)
        assert th.all(th.isfinite(sampled_log_probs))
        assert th.allclose(sampled_log_probs, evaluated_log_probs, atol=1e-5)
        assert entropy.shape == evaluated_log_probs.shape
        assert th.all(th.isfinite(entropy))


@pytest.mark.require_learning
def test_lstm_actor_supports_non_default_unique_observation_dimension():
    actor = LSTMActor(
        obs_dim=11,
        act_dim=2,
        float_type=th.float32,
        unique_obs_dim=3,
        num_timeseries_obs_dim=4,
    )

    actions = actor(th.randn(5, 11))

    assert actor.FC1.in_features == 35
    assert actions.shape == (5, 2)


@pytest.fixture(params=["mlp", "lstm"])
def constant_ppo_actor(request):
    actor_type = ActorPPO if request.param == "mlp" else LSTMActorPPO
    actor = actor_type(
        obs_dim=10,
        act_dim=1,
        float_type=th.float32,
        unique_obs_dim=2,
        num_timeseries_obs_dim=4,
    )
    with th.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
    return actor


@pytest.mark.require_learning
@pytest.mark.parametrize("mean", [-10.0, 10.0])
def test_saturated_ppo_ratios_preserve_latent_samples(constant_ppo_actor, mean):
    actor = constant_ppo_actor
    with th.no_grad():
        actor.mean_layer.bias.fill_(mean)
        observations = th.zeros(16, 10)
        th.manual_seed(42)
        actions, old_log_probs, latents = actor.get_action_and_log_prob(
            observations, return_latent=True
        )
    assert (actions.abs() == 1).any()

    buffer = RolloutBuffer(
        buffer_size=1, obs_dim=10, act_dim=1, n_rl_units=1,
        device="cpu", float_type=th.float32,
    )
    for index in range(len(observations)):
        buffer.ensure_capacity(index + 1)
        buffer.add(
            observations[index].numpy(), actions[index].numpy(),
            np.zeros(1), np.zeros(1), old_log_probs[index].numpy(),
            latent_action=latents[index].numpy(),
        )
    # Growth and shuffled sampling must retain the original sample/log-prob pair.
    indices = np.random.default_rng(42).permutation(len(observations))
    batch = buffer.sample(indices)
    th.testing.assert_close(batch.latent_actions[:, 0], latents[indices])
    log_probs, _ = actor.evaluate_actions(
        batch.observations[:, 0], batch.actions[:, 0], batch.latent_actions[:, 0]
    )
    th.testing.assert_close(
        (log_probs - batch.old_log_probs[:, 0]).exp(), th.ones(16)
    )

    with th.no_grad():
        actor.mean_layer.bias.add_(0.2)
    changed_log_probs, _ = actor.evaluate_actions(
        batch.observations[:, 0], batch.actions[:, 0], batch.latent_actions[:, 0]
    )
    # The tanh Jacobian cancels from the importance ratio at the same sample.
    samples = batch.latent_actions[:, 0, 0]
    expected_log_ratio = -0.5 * ((samples - (mean + 0.2)) ** 2 - (samples - mean) ** 2)
    th.testing.assert_close(
        changed_log_probs - batch.old_log_probs[:, 0], expected_log_ratio,
        atol=3e-6, rtol=1e-5,
    )


@pytest.mark.require_learning
def test_squashed_entropy_gradient_matches_current_policy(constant_ppo_actor):
    actor = constant_ppo_actor
    with th.no_grad():
        actor.mean_layer.bias.fill_(2.0)
        actor.log_std.fill_(np.log(0.5))
    observations = th.zeros(2048, 10)
    fixed_old_actions = th.full((2048, 1), 0.1)
    th.manual_seed(42)
    noise = th.randn(2048, 1)
    samples = 2.0 + 0.5 * noise
    expected_mean_gradient = (-2 * samples.tanh()).mean()
    expected_log_std_gradient = (1 - 2 * samples.tanh() * (0.5 * noise)).mean()

    th.manual_seed(42)
    _, entropy = actor.evaluate_actions(observations, fixed_old_actions)
    mean_gradient, log_std_gradient = th.autograd.grad(
        entropy.mean(), (actor.mean_layer.bias, actor.log_std)
    )
    th.testing.assert_close(mean_gradient.squeeze(), expected_mean_gradient)
    th.testing.assert_close(log_std_gradient.squeeze(), expected_log_std_gradient)


@pytest.mark.require_learning
@pytest.mark.parametrize("clip_range, expected_loss", [(0.2, 0.445), (None, 0.125)])
def test_mappo_value_clipping_selects_loss_per_sample(
    learning_role_n, clip_range, expected_loss
):
    algorithm = learning_role_n.rl_algorithm
    algorithm.clip_range_vf = clip_range
    values = th.tensor([1.0, 0.5], requires_grad=True)
    loss = algorithm._compute_value_loss(values, th.zeros(2), th.tensor([1.0, 0.0]))
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    th.testing.assert_close(values.grad, th.tensor([0.0, 0.5]))


@pytest.mark.require_learning
def test_mappo_excludes_late_rewards_from_old_policy(learning_role_n, monkeypatch):
    learn = learning_role_n
    learn.initialize_policy()
    _setup_for_update(learn)
    algorithm = learn.rl_algorithm
    algorithm.buffer = algorithm.create_buffer("1h")
    for strategy in learn.rl_strats.values():
        strategy.learning_mode = True
        strategy.evaluation_mode = False
        strategy.float_type = learn.float_type

    stored_hours = []
    logged_hours = []
    original_store = algorithm.store_experience
    original_output = learn.write_rl_params_to_output

    def record_store(cache, device):
        original_store(cache, device)
        buffer = algorithm.buffer
        stored_hours.append(buffer.observations[:buffer.pos, 0, 0].tolist())
        batch = buffer.sample(np.arange(buffer.pos))
        for index, strategy in enumerate(learn.rl_strats.values()):
            log_probs, _ = strategy.actor.evaluate_actions(
                batch.observations[:, index], batch.actions[:, index],
                batch.latent_actions[:, index],
            )
            th.testing.assert_close(
                (log_probs - batch.old_log_probs[:, index]).exp(),
                th.ones(buffer.pos), atol=1e-5, rtol=1e-5,
            )

    def record_output(cache):
        logged_hours.append([timestamp.hour for timestamp in cache["rewards"]])
        original_output(cache)

    monkeypatch.setattr(algorithm, "store_experience", record_store)
    monkeypatch.setattr(learn, "write_rl_params_to_output", record_output)

    def add_actions(hour):
        timestamp = start + timedelta(hours=hour)
        for unit_id, strategy in learn.rl_strats.items():
            observation = th.full((algorithm.obs_dim,), float(hour))
            learn.add_observation_to_cache(unit_id, timestamp, observation)
            action, noise, extra = algorithm.get_action(strategy, observation)
            learn.add_actions_to_cache(unit_id, timestamp, action, noise, extra)

    def add_rewards(hour):
        for unit_id in learn.rl_strats:
            learn.add_reward_to_cache(unit_id, start + timedelta(hours=hour), 1.0, 0.0, 1.0)

    for hour in range(1, 5):
        add_actions(hour)
        if hour < 4:
            add_rewards(hour)
    pending_time = start + timedelta(hours=4)
    pending_latent = learn.cache["latent_actions"][pending_time]["agent_0"][0].clone()
    asyncio.run(learn.store_to_buffer_and_update())
    assert algorithm.n_updates == 1
    assert learn.cache["policy_versions"][pending_time]["agent_0"] == [0]
    th.testing.assert_close(
        learn.cache["latent_actions"][pending_time]["agent_0"][0], pending_latent
    )

    add_rewards(4)
    for hour in range(5, 9):
        add_actions(hour)
        if hour < 8:
            add_rewards(hour)
    asyncio.run(learn.store_to_buffer_and_update())
    assert algorithm.n_updates == 2
    assert stored_hours == [[1.0, 2.0, 3.0], [5.0, 6.0, 7.0]]
    assert logged_hours == [[1, 2, 3], [4, 5, 6, 7]]
    assert algorithm.buffer.pos == 0


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
@pytest.mark.parametrize("architecture", ["mlp", "lstm"])
def test_mappo_uses_configured_initial_std_and_restores_learned_std(
    learning_role_n, tmp_path, architecture
):
    learn = learning_role_n
    learn.learning_config.actor_architecture = architecture
    learn.learning_config.on_policy.action_std_init = 0.2
    learn.initialize_policy()

    for strategy in learn.rl_strats.values():
        mean, log_std = strategy.actor.get_distribution(th.zeros(2, strategy.obs_dim))
        th.testing.assert_close(log_std.exp(), th.full_like(mean, 0.2))
        assert strategy.actor.log_std.requires_grad
        # A checkpoint's learned variance must take precedence over initialization.
        with th.no_grad():
            strategy.actor.log_std.fill_(np.log(0.35))

    learn.rl_algorithm.save_params(str(tmp_path))
    learn.learning_config.on_policy.action_std_init = 0.1
    learn.initialize_policy()
    learn.rl_algorithm.load_params(str(tmp_path))
    for strategy in learn.rl_strats.values():
        th.testing.assert_close(
            strategy.actor.log_std.exp(), th.full_like(strategy.actor.log_std, 0.35)
        )


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
def test_mappo_transfers_critics_when_agent_order_changes(
    learning_role_n, base_learning_config, tmp_path
):
    learning_role_n.initialize_policy()
    save_dir = tmp_path / "saved_mappo"
    learning_role_n.rl_algorithm.save_params(directory=str(save_dir))
    saved_states = {
        unit_id: deepcopy(strategy.critics.state_dict())
        for unit_id, strategy in learning_role_n.rl_strats.items()
    }

    config = copy(base_learning_config)
    new_learning = Learning(config["learning_config"], start, end)
    for agent_id in ("agent_1", "agent_0", "agent_2"):
        strategy = LearningStrategy(**config, learning_role=new_learning)
        strategy.unit_id = agent_id
        new_learning.rl_strats[agent_id] = strategy
    new_learning.initialize_policy()
    initial_new_agent_slices = {
        unit_id: strategy.critics.state_dict()["v_layers.0.weight"][
            :, -strategy.unique_obs_dim :
        ].clone()
        for unit_id, strategy in new_learning.rl_strats.items()
        if unit_id != "agent_2"
    }

    new_learning.rl_algorithm.load_critic_params(str(save_dir))

    for unit_id in ("agent_0", "agent_1"):
        loaded = new_learning.rl_strats[unit_id].critics.state_dict()
        saved = saved_states[unit_id]
        assert th.equal(
            loaded["v_layers.0.weight"][:, : config["obs_dim"]],
            saved["v_layers.0.weight"][:, : config["obs_dim"]],
        )
        assert th.equal(loaded["v_layers.2.weight"], saved["v_layers.2.weight"])
        assert th.equal(loaded["v_layers.4.weight"], saved["v_layers.4.weight"])
        assert th.equal(
            loaded["v_layers.0.weight"][:, -config["unique_obs_dim"] :],
            initial_new_agent_slices[unit_id],
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
    learn.rl_algorithm.buffer = RolloutBuffer( # TODO: probably doesn't work
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
        "log_probs": {timestamp: {}},
        "latent_actions": {timestamp: {}},
        "policy_versions": {timestamp: {}},
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
        cache["latent_actions"][timestamp][unit_id] = [
            th.full((act_dim,), marker, dtype=th.float32)
        ]
        cache["policy_versions"][timestamp][unit_id] = [0]

    # Stash db_addr/update_steps so the logging path inside the algorithm is
    # safe to call.  We do NOT need an actual policy update for this test, so
    # we monkey-patch update_policy to a no-op.
    learn.db_addr = None
    learn.update_steps = 0
    learn.rl_algorithm.update_policy = lambda: None

    asyncio.run(
        learn._store_to_buffer_and_update_sync(cache, learn.device)
    )

    buf = learn.rl_algorithm.buffer
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
