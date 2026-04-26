# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from torch import nn

from assume.reinforcement_learning.neural_network_architecture import (
    LSTMActor,
    MLPActor,
)

actor_architecture_aliases: dict[str, type[nn.Module]] = {
    "mlp": MLPActor,
    "lstm": LSTMActor,
}

from assume.reinforcement_learning.algorithms.base_algorithm import ( 
    A2CAlgorithm,
    RLAlgorithm,
)
from assume.reinforcement_learning.algorithms.maddpg import DDPG 
from assume.reinforcement_learning.algorithms.mappo import PPO 
from assume.reinforcement_learning.algorithms.matd3 import TD3 

__all__ = [
    # Base classes
    "RLAlgorithm",
    "A2CAlgorithm",
    # Concrete algorithms
    "TD3",
    "DDPG",
    "PPO",
    # Actor architectures
    "actor_architecture_aliases",
    "MLPActor",
    "LSTMActor",
]
