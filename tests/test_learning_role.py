# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import pytest

try:
    from assume.common.base import LearningConfig
    from assume.reinforcement_learning.learning_role import (
        Learning,
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
    config = {
        "foresight": 1,
        "act_dim": 2,
        "unique_obs_dim": 0,
        "learning_config": LearningConfig(
            train_freq="1h",
            algorithm="matd3",
            actor_architecture="mlp",
            learning_mode=True,
            evaluation_mode=False,
            training_episodes=3,
            episodes_collecting_initial_experience=1,
            continue_learning=False,
            trained_policies_save_path=None,
            early_stopping_steps=10,
            early_stopping_threshold=0.05,
        ),
    }

    # test init
    learn = Learning(config["learning_config"], start=start, end=end)
    assert len(learn.rl_strats) == 0

    # we need to add learning strategies first
    learn.rl_strats["test_id"] = LearningStrategy(**config, learning_role=learn)

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
