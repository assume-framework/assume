# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from importlib.metadata import version

from assume.common import MarketConfig, MarketProduct
from assume.reinforcement_learning import (
    A2CAlgorithm,
    DDPG,
    LSTMActor,
    Learning,
    MLPActor,
    PPO,
    RLAlgorithm,
    ReplayBuffer,
    ReplayBufferSamples,
    RolloutBuffer,
    RolloutBufferSamples,
    TD3,
    actor_architecture_aliases,
)
from assume.scenario.loader_csv import (
    load_custom_units,
    load_scenario_folder,
    run_learning,
)
from assume.world import World

__version__ = version("assume-framework")

__author__ = "ASSUME Developers: Nick Harder, Kim Miskiw, Florian Maurer, Manish Khanra"
__copyright__ = "AGPL-3.0 License"

__all__ = [
    # Framework version
    "__version__",
    # World & scenario
    "World",
    "load_scenario_folder",
    "load_custom_units",
    "run_learning",
    # Market primitives
    "MarketConfig",
    "MarketProduct",
    # RL orchestration
    "Learning",
    # RL algorithm base classes
    "RLAlgorithm",
    "A2CAlgorithm",
    # RL concrete algorithms
    "TD3",
    "DDPG",
    "PPO",
    # RL actor architectures
    "MLPActor",
    "LSTMActor",
    "actor_architecture_aliases",
    # RL buffers
    "ReplayBuffer",
    "ReplayBufferSamples",
    "RolloutBuffer",
    "RolloutBufferSamples",
]
