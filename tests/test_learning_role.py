# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pytest

try:
    from assume.reinforcement_learning.learning_role import (
        Learning,
        LearningConfig,
        LearningStrategy,
    )
    from assume.reinforcement_learning.neural_network_architecture import (
        Actor,
        CriticTD3,
    )
except ImportError:
    pass

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.mark.require_learning
def test_learning_init():
    learning_config: LearningConfig = {
        "obs_dim": 3,
        "act_dim": 2,
        "train_freq": "1h",
        "unique_obs_dim": 0,
        "algorithm": "matd3",
        "actor_architecture": "mlp",
        "learning_mode": False,
        "perform_evaluation": False,
        "training_episodes": 3,
        "continue_learning": False,
        "trained_policies_save_path": None,
        "early_stopping_steps": 10,
        "early_stopping_threshold": 0.05,
    }
    # test init
    learn = Learning(learning_config)
    assert len(learn.rl_strats) == 0

    # we need to add learning strategies first
    learn.rl_strats["test_id"] = LearningStrategy(**learning_config)

    # test creating actors
    learn.initialize_policy()
    # now we have an actor for every strategy
    for strategy in learn.rl_strats.values():
        assert isinstance(strategy.actor, Actor)
        assert isinstance(strategy.actor_target, Actor)

    # now we have a critic for every strategy
    for strategy in learn.rl_strats.values():
        assert isinstance(strategy.critics, CriticTD3)
        assert isinstance(strategy.target_critics, CriticTD3)

    ac = learn.rl_algorithm.extract_policy()

    assert ac["target_critics"]["test_id"] == learn.rl_strats["test_id"].target_critics
    assert ac["critics"]["test_id"] == learn.rl_strats["test_id"].critics
