# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os

import torch as th
from torch.nn import functional as F
from torch.optim import Adam

logger = logging.getLogger(__name__)

from assume.common.base import LearningStrategy
from assume.reinforcement_learning.algorithms.matd3 import TD3
from assume.reinforcement_learning.learning_utils import (
    CriticTD3,
    LSTM_Actor,
    polyak_update,
)


class LSTM_TD3(TD3):
    """
    Twin Delayed Deep Deterministic Policy Gradients (TD3).
    Addressing Function Approximation Error in Actor-Critic Methods.
    TD3 is a direct successor of DDPG and improves it using three major tricks:
    clipped double Q-Learning, delayed policy update and target policy smoothing.

    Open AI Spinning guide: https://spinningup.openai.com/en/latest/algorithms/td3.html

    Original paper: https://arxiv.org/pdf/1802.09477.pdf
    """

    def __init__(
        self,
        learning_role,
        learning_rate=1e-4,
        episodes_collecting_initial_experience=100,
        batch_size=1024,
        tau=0.005,
        gamma=0.99,
        gradient_steps=-1,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    ):
        super().__init__(
            learning_role,
            learning_rate,
            episodes_collecting_initial_experience,
            batch_size,
            tau,
            gamma,
            gradient_steps,
            policy_delay,
            target_policy_noise,
            target_noise_clip,
        )
        self.n_updates = 0

    def create_actors(self) -> None:
        """
        Create actor networks for reinforcement learning for each unit strategy.

        This method initializes actor networks and their corresponding target networks for each unit strategy.
        The actors are designed to map observations to action probabilities in a reinforcement learning setting.

        The created actor networks are associated with each unit strategy and stored as attributes.
        """

        obs_dim_list = []
        act_dim_list = []

        for _, unit_strategy in self.learning_role.rl_strats.items():
            unit_strategy.actor = LSTM_Actor(
                obs_dim=unit_strategy.obs_dim,
                act_dim=unit_strategy.act_dim,
                float_type=self.float_type,
            ).to(self.device)

            unit_strategy.actor_target = LSTM_Actor(
                obs_dim=unit_strategy.obs_dim,
                act_dim=unit_strategy.act_dim,
                float_type=self.float_type,
            ).to(self.device)
            unit_strategy.actor_target.load_state_dict(unit_strategy.actor.state_dict())
            unit_strategy.actor_target.train(mode=False)

            unit_strategy.actor.optimizer = Adam(
                unit_strategy.actor.parameters(), lr=self.learning_rate
            )  # TODO: Try LBFGS Optimizer

            obs_dim_list.append(unit_strategy.obs_dim)
            act_dim_list.append(unit_strategy.act_dim)

        if len(set(obs_dim_list)) > 1:
            raise ValueError(
                "All observation dimensions must be the same for all RL agents"
            )
        else:
            self.obs_dim = obs_dim_list[0]

        if len(set(act_dim_list)) > 1:
            raise ValueError("All action dimensions must be the same for all RL agents")
        else:
            self.act_dim = act_dim_list[0]
