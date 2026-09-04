# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from assume.common.base import LearningConfig, OffPolicyConfig, OnPolicyConfig


@pytest.fixture
def all_off_policy_values():
    return {
        "episodes_collecting_initial_experience": 3,
        "gradient_steps": 7,
        "noise_dt": 2,
        "noise_scale": 3,
        "noise_sigma": 0.4,
        "action_noise_schedule": "linear",
        "policy_delay": 4,
        "tau": 0.02,
        "target_policy_noise": 0.3,
        "target_noise_clip": 0.6,
        "replay_buffer_size": 1234,
    }


@pytest.mark.parametrize(
    ("algorithm", "section_name", "config_class"),
    [
        ("matd3", "off_policy", OffPolicyConfig),
        ("maddpg", "off_policy", OffPolicyConfig),
        ("mappo", "on_policy", OnPolicyConfig),
    ],
)
def test_accepts_nested_algorithm_config(algorithm, section_name, config_class):
    expected_values = asdict(config_class())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = LearningConfig(
            **{
                "algorithm": algorithm,
                section_name: expected_values,
            }
        )
    nested_config = getattr(config, section_name)

    assert isinstance(nested_config, config_class)
    assert asdict(nested_config) == expected_values


def test_migrates_all_legacy_off_policy_fields(all_off_policy_values):
    learning_dict = {"algorithm": "matd3", **all_off_policy_values}

    with pytest.warns(
        DeprecationWarning,
        match="must now be placed under 'off_policy'",
    ):
        config = LearningConfig(**learning_dict)

    assert isinstance(config.off_policy, OffPolicyConfig)
    assert asdict(config.off_policy) == all_off_policy_values


def test_does_not_modify_input(all_off_policy_values):
    learning_dict = {
        "algorithm": "matd3",
        **all_off_policy_values,
        "off_policy": {"gradient_steps": 11},
    }
    original_dict = deepcopy(learning_dict)

    with pytest.warns(DeprecationWarning):
        LearningConfig(**learning_dict)

    assert learning_dict == original_dict


@pytest.mark.parametrize(
    ("key", "nested_value"),
    list(asdict(OffPolicyConfig()).items()),
)
def test_nested_off_policy_value_takes_precedence(key, nested_value):
    legacy_value = object()

    with pytest.warns(DeprecationWarning):
        config = LearningConfig(
            **{
                "algorithm": "matd3",
                key: legacy_value,
                "off_policy": {
                    key: nested_value,
                },
            }
        )

    assert getattr(config.off_policy, key) == nested_value


@pytest.mark.parametrize(
    "unknown_key",
    [
        "unknown_parameter",
        "gradients_steps",
        "noise",
    ],
)
def test_rejects_unknown_keys(unknown_key):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        LearningConfig(
            **{
                "algorithm": "matd3",
                unknown_key: 123,
            }
        )


@pytest.mark.parametrize("section", ["off_policy", "on_policy"])
def test_rejects_unknown_nested_keys(section):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        LearningConfig(**{section: {"unknown_parameter": 123}})


def test_accepts_existing_nested_configs():
    off_policy = OffPolicyConfig(gradient_steps=7)
    on_policy = OnPolicyConfig(n_epochs=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = LearningConfig(off_policy=off_policy, on_policy=on_policy)
    assert config.off_policy is off_policy
    assert config.on_policy is on_policy


def test_existing_off_policy_config_takes_precedence(all_off_policy_values):
    off_policy = OffPolicyConfig()
    original_values = asdict(off_policy)
    with pytest.warns(DeprecationWarning):
        config = LearningConfig(off_policy=off_policy, **all_off_policy_values)
    assert asdict(config.off_policy) == original_values
    assert asdict(off_policy) == original_values


def test_explicit_none_is_a_legacy_value():
    with pytest.warns(DeprecationWarning, match="action_noise_schedule"):
        config = LearningConfig(action_noise_schedule=None)
    assert config.off_policy.action_noise_schedule is None


def test_migrated_values_are_validated():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="gradient_steps need to be positive"):
            LearningConfig(gradient_steps=0)


def test_legacy_values_are_not_stored_twice(all_off_policy_values):
    with pytest.warns(DeprecationWarning):
        config = LearningConfig(**all_off_policy_values)
    assert not set(all_off_policy_values).intersection(vars(config))
    assert not set(all_off_policy_values).intersection(asdict(config))


def test_default_configs_are_independent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        first = LearningConfig()
        second = LearningConfig()
    assert isinstance(first.off_policy, OffPolicyConfig)
    assert asdict(first.off_policy) == asdict(OffPolicyConfig())
    assert first.off_policy is not second.off_policy
    assert first.on_policy is not second.on_policy


@pytest.mark.parametrize("algorithm", ["matd3", "maddpg"])
def test_complete_legacy_yaml_config(algorithm, all_off_policy_values):
    legacy_config = yaml.safe_load(
        """
learning_mode: true
evaluation_mode: false
continue_learning: true
trained_policies_save_path: saved_policies
trained_policies_load_path: previous_policies
min_bid_price: -50
max_bid_price: 200
device: cpu
exploration_noise_std: 0.3
training_episodes: 40
validation_episodes_interval: 4
train_freq: 12h
batch_size: 64
learning_rate: 0.0005
learning_rate_schedule: linear
early_stopping_steps: 8
early_stopping_threshold: 0.02
gamma: 0.95
actor_architecture: lstm
"""
    )
    legacy_config.update(algorithm=algorithm, **all_off_policy_values)
    # Passing the full flat configuration through YAML as a legacy file would
    legacy_config = yaml.safe_load(yaml.safe_dump(legacy_config))
    original_config = deepcopy(legacy_config)

    with pytest.warns(DeprecationWarning) as recorded:
        config = LearningConfig(**legacy_config)

    assert len(recorded) == 1
    for key, expected in original_config.items():
        if key in all_off_policy_values:
            assert getattr(config.off_policy, key) == expected
            assert key in str(recorded[0].message)
        else:
            assert getattr(config, key) == expected
    assert asdict(config.on_policy) == asdict(OnPolicyConfig())
    assert legacy_config == original_config

    nested_config = {
        key: value
        for key, value in original_config.items()
        if key not in all_off_policy_values
    }
    nested_config["off_policy"] = all_off_policy_values.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        structured = LearningConfig(**nested_config)
    assert asdict(config) == asdict(structured)


@pytest.mark.parametrize("study_case", ["base", "base_lstm"])
def test_example_02b_legacy_config_matches_nested_config(study_case):
    config_path = (
        Path(__file__).resolve().parents[1] / "examples/inputs/example_02b/config.yaml"
    )
    with config_path.open() as config_file:
        nested_config = yaml.safe_load(config_file)[study_case]["learning_config"]

    # Recreating the flat off policy format without the newer on policy section
    nested_config.pop("on_policy", None)
    legacy_config = deepcopy(nested_config)
    legacy_config.update(legacy_config.pop("off_policy"))
    original_config = deepcopy(legacy_config)

    with pytest.warns(DeprecationWarning):
        legacy = LearningConfig(**legacy_config)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        structured = LearningConfig(**nested_config)

    assert asdict(legacy) == asdict(structured)
    assert legacy_config == original_config
