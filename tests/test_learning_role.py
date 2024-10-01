# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import numpy as np
import pytest

try:
    from assume.common.utils import create_rrule
    from assume.reinforcement_learning.buffer import ReplayBuffer
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
    for str_id in learn.rl_strats.keys():
        assert isinstance(learn.critics[str_id], CriticTD3)
        assert isinstance(learn.target_critics[str_id], CriticTD3)

    ac = learn.rl_algorithm.extract_policy()

    assert ac["target_critics"] == learn.target_critics
    assert ac["critics"] == learn.critics


@pytest.mark.require_learning
def test_lr_progress_remaining():
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
        "episodes_collecting_initial_experience": 1,
        "continue_learning": False,
        "trained_policies_save_path": None,
    }
    # we need to add learning strategies first
    learn = Learning(learning_config)
    learn.rl_strats["test_id"] = LearningStrategy(**learning_config)
    learn.initialize_policy()

    # mock up total_simulation_steps because world is not initialized and run
    recurrency_task = create_rrule(
        start=start,
        end=end,
        freq=learning_config.get("train_freq", "24h"),
    )
    learn.total_simulation_steps = len(list(recurrency_task))

    # mock up replay buffer
    learn.buffer = ReplayBuffer(
        buffer_size=int(learning_config.get("replay_buffer_size", 5e5)),
        obs_dim=learn.rl_algorithm.obs_dim,
        act_dim=learn.rl_algorithm.act_dim,
        n_rl_units=len(learn.rl_strats),
        device=learn.device,
        float_type=learn.float_type,
    )
    learn.buffer.add(
        obs=np.array([0, 0, 0]), actions=np.array([1, 1]), reward=np.array([2])
    )

    # test remaining progress before training
    assert learn.get_progress_remaining() == 1.0

    # update policy 48 times (expected to update actor 24 times with policy delay = 2)
    for _ in range(48):
        learn.rl_algorithm.update_policy()

    # test final progress
    assert learn.get_progress_remaining() == 0.0
