# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import warnings
from copy import deepcopy
from dataclasses import asdict

import pytest

from assume.common.base import LearningConfig, OffPolicyConfig, OnPolicyConfig


@pytest.fixture
def all_off_policy_values():
    return asdict(OffPolicyConfig())


@pytest.mark.parametrize(
    ("algorithm", "section_name", "config_class"),
    [
        ("matd3", "off_policy", OffPolicyConfig),
        ("maddpg", "off_policy", OffPolicyConfig),
        ("mappo", "on_policy", OnPolicyConfig),
    ],
)
def test_from_dict_accepts_nested_algorithm_config(
    algorithm, section_name, config_class
):
    expected_values = asdict(config_class())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = LearningConfig.from_dict(
            {
                "algorithm": algorithm,
                section_name: expected_values,
            }
        )
    nested_config = getattr(config, section_name)

    assert isinstance(nested_config, config_class)
    assert asdict(nested_config) == expected_values


def test_from_dict_migrates_all_legacy_off_policy_fields(all_off_policy_values):
    learning_dict = {"algorithm": "matd3", **all_off_policy_values}

    with pytest.warns(
        DeprecationWarning,
        match="must now be placed under 'off_policy'",
    ):
        config = LearningConfig.from_dict(learning_dict)

    assert isinstance(config.off_policy, OffPolicyConfig)
    assert asdict(config.off_policy) == all_off_policy_values


def test_from_dict_does_not_modify_input(all_off_policy_values):
    learning_dict = {
        "algorithm": "matd3",
        **all_off_policy_values,
    }
    original_dict = deepcopy(learning_dict)

    with pytest.warns(DeprecationWarning):
        LearningConfig.from_dict(learning_dict)

    assert learning_dict == original_dict


@pytest.mark.parametrize(
    ("key", "nested_value"),
    list(asdict(OffPolicyConfig()).items()),
)
def test_nested_off_policy_value_takes_precedence(key, nested_value):
    legacy_value = object()

    with pytest.warns(DeprecationWarning):
        config = LearningConfig.from_dict(
            {
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
def test_from_dict_rejects_unknown_keys(unknown_key):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        LearningConfig.from_dict(
            {
                "algorithm": "matd3",
                unknown_key: 123,
            }
        )
