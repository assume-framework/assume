# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from torch import nn

from assume.reinforcement_learning.neural_network_architecture import (
    LSTMActor,
    MLPActor,
)

from assume.reinforcement_learning.algorithms.base_algorithm import (
    ActorCriticAlgorithm,
    RLAlgorithm,
)

actor_architecture_aliases: dict[str, type[nn.Module]] = {
    "mlp": MLPActor,
    "lstm": LSTMActor,
}


__all__ = [
    # Base classes
    "RLAlgorithm",
    "ActorCriticAlgorithm",
    # Concrete algorithms
    "TD3",
    "DDPG",
    "PPO",
    # Actor architectures
    "actor_architecture_aliases",
    "MLPActor",
    "LSTMActor",
]
