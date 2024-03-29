# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import numpy as np
import pytest

try:
    from assume.reinforcement_learning.learning_role import (
        Learning,
        LearningConfig,
        LearningStrategy,
    )
    from assume.reinforcement_learning.learning_utils import Actor, CriticTD3
except ImportError:
    pass

start = datetime(2023, 7, 1)
end = datetime(2023, 7, 2)


@pytest.mark.require_learning
def test_learning_init():
    learning_config: LearningConfig = {
        "observation_dimension": 3,
        "action_dimension": 2,
        "algorithm": "matd3",
        "learning_mode": False,
        "training_episodes": 3,
        "continue_learning": False,
        "trained_policies_save_path": None,
    }
    # test init
    l = Learning(learning_config, start, end)
    assert len(l.rl_strats) == 0

    # we need to add learning strategies first
    l.rl_strats["test_id"] = LearningStrategy(**learning_config)

    # test creating actors
    l.initialize_policy()
    # now we have an actor for every strategy
    for strategy in l.rl_strats.values():
        assert isinstance(strategy.actor, Actor)
        assert isinstance(strategy.actor_target, Actor)

    # now we have a critic for every strategy
    for str_id in l.rl_strats.keys():
        assert isinstance(l.critics[str_id], CriticTD3)
        assert isinstance(l.target_critics[str_id], CriticTD3)

    ac = l.rl_algorithm.extract_policy()

    assert ac["target_critics"] == l.target_critics
    assert ac["critics"] == l.critics
