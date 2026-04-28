# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for verification of the public API symbols are importable consistently.

Import layers which are covered are as follows:
- assume.reinforcement_learning.algorithms for algorithm-level package
- assume.reinforcement_learning for RL sub-package
- assume for top-level package
"""

import pytest

try:
    import torch as th

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Layer 1 – assume.reinforcement_learning.algorithms
# ---------------------------------------------------------------------------


@pytest.mark.require_learning
class TestAlgorithmsPackageExports:
    """All algorithm classes and helpers re-exported from the algorithms package."""

    def test_import_rl_algorithm_base(self):
        from assume.reinforcement_learning.algorithms import RLAlgorithm

        assert RLAlgorithm is not None

    def test_import_a2c_algorithm_base(self):
        from assume.reinforcement_learning.algorithms import A2CAlgorithm

        assert A2CAlgorithm is not None

    def test_import_td3(self):
        from assume.reinforcement_learning.algorithms import TD3

        assert TD3 is not None

    def test_import_ddpg(self):
        from assume.reinforcement_learning.algorithms import DDPG

        assert DDPG is not None

    def test_import_ppo(self):
        from assume.reinforcement_learning.algorithms import PPO

        assert PPO is not None

    def test_import_mlp_actor(self):
        from assume.reinforcement_learning.algorithms import MLPActor

        assert MLPActor is not None

    def test_import_lstm_actor(self):
        from assume.reinforcement_learning.algorithms import LSTMActor

        assert LSTMActor is not None

    def test_import_actor_architecture_aliases(self):
        from assume.reinforcement_learning.algorithms import actor_architecture_aliases

        assert "mlp" in actor_architecture_aliases
        assert "lstm" in actor_architecture_aliases

    def test_algorithm_hierarchy(self):
        """TD3, DDPG, PPO must all be subclasses of A2CAlgorithm → RLAlgorithm."""
        from assume.reinforcement_learning.algorithms import (
            A2CAlgorithm,
            DDPG,
            PPO,
            RLAlgorithm,
            TD3,
        )

        for cls in (TD3, DDPG, PPO):
            assert issubclass(cls, A2CAlgorithm), f"{cls.__name__} not subclass of A2CAlgorithm"
            assert issubclass(cls, RLAlgorithm), f"{cls.__name__} not subclass of RLAlgorithm"

    def test_actor_aliases_map_to_nn_modules(self):
        from torch import nn

        from assume.reinforcement_learning.algorithms import actor_architecture_aliases

        for name, cls in actor_architecture_aliases.items():
            assert issubclass(cls, nn.Module), f"alias '{name}' does not map to an nn.Module"


# ---------------------------------------------------------------------------
# Layer 2 – assume.reinforcement_learning
# ---------------------------------------------------------------------------


@pytest.mark.require_learning
class TestRLPackageExports:
    """All public symbols re-exported from the reinforcement_learning sub-package."""

    def test_import_learning(self):
        from assume.reinforcement_learning import Learning

        assert Learning is not None

    def test_import_rl_algorithm(self):
        from assume.reinforcement_learning import RLAlgorithm

        assert RLAlgorithm is not None

    def test_import_a2c_algorithm(self):
        from assume.reinforcement_learning import A2CAlgorithm

        assert A2CAlgorithm is not None

    def test_import_td3(self):
        from assume.reinforcement_learning import TD3

        assert TD3 is not None

    def test_import_ddpg(self):
        from assume.reinforcement_learning import DDPG

        assert DDPG is not None

    def test_import_ppo(self):
        from assume.reinforcement_learning import PPO

        assert PPO is not None

    def test_import_mlp_actor(self):
        from assume.reinforcement_learning import MLPActor

        assert MLPActor is not None

    def test_import_lstm_actor(self):
        from assume.reinforcement_learning import LSTMActor

        assert LSTMActor is not None

    def test_import_actor_architecture_aliases(self):
        from assume.reinforcement_learning import actor_architecture_aliases

        assert isinstance(actor_architecture_aliases, dict)

    def test_import_replay_buffer(self):
        from assume.reinforcement_learning import ReplayBuffer

        assert ReplayBuffer is not None

    def test_import_replay_buffer_samples(self):
        from assume.reinforcement_learning import ReplayBufferSamples

        assert ReplayBufferSamples is not None

    def test_import_rollout_buffer(self):
        from assume.reinforcement_learning import RolloutBuffer

        assert RolloutBuffer is not None

    def test_import_rollout_buffer_samples(self):
        from assume.reinforcement_learning import RolloutBufferSamples

        assert RolloutBufferSamples is not None

    def test_all_declared(self):
        """Every symbol listed in __all__ must actually be importable."""
        import assume.reinforcement_learning as rl_pkg

        for name in rl_pkg.__all__:
            assert hasattr(rl_pkg, name), f"__all__ entry '{name}' missing from module"

    def test_replay_buffer_and_rollout_buffer_are_distinct(self):
        from assume.reinforcement_learning import ReplayBuffer, RolloutBuffer

        assert ReplayBuffer is not RolloutBuffer

    def test_buffer_samples_are_distinct(self):
        from assume.reinforcement_learning import ReplayBufferSamples, RolloutBufferSamples

        assert ReplayBufferSamples is not RolloutBufferSamples


# ---------------------------------------------------------------------------
# Layer 3 – assume (top-level package)
# ---------------------------------------------------------------------------


@pytest.mark.require_learning
class TestTopLevelPackageRLExports:
    """RL symbols must be reachable directly from `import assume`."""

    def test_import_learning(self):
        import assume

        assert hasattr(assume, "Learning")

    def test_import_rl_algorithm(self):
        import assume

        assert hasattr(assume, "RLAlgorithm")

    def test_import_a2c_algorithm(self):
        import assume

        assert hasattr(assume, "A2CAlgorithm")

    def test_import_td3(self):
        import assume

        assert hasattr(assume, "TD3")

    def test_import_ddpg(self):
        import assume

        assert hasattr(assume, "DDPG")

    def test_import_ppo(self):
        import assume

        assert hasattr(assume, "PPO")

    def test_import_mlp_actor(self):
        import assume

        assert hasattr(assume, "MLPActor")

    def test_import_lstm_actor(self):
        import assume

        assert hasattr(assume, "LSTMActor")

    def test_import_actor_architecture_aliases(self):
        import assume

        assert hasattr(assume, "actor_architecture_aliases")

    def test_import_replay_buffer(self):
        import assume

        assert hasattr(assume, "ReplayBuffer")

    def test_import_replay_buffer_samples(self):
        import assume

        assert hasattr(assume, "ReplayBufferSamples")

    def test_import_rollout_buffer(self):
        import assume

        assert hasattr(assume, "RolloutBuffer")

    def test_import_rollout_buffer_samples(self):
        import assume

        assert hasattr(assume, "RolloutBufferSamples")

    def test_all_declared(self):
        """Every symbol in top-level __all__ must exist on the module."""
        import assume

        for name in assume.__all__:
            assert hasattr(assume, name), f"__all__ entry '{name}' missing from assume"

    def test_rl_symbols_consistent_across_layers(self):
        """The same class object must be reachable from all three import paths."""
        import assume
        import assume.reinforcement_learning as rl
        from assume.reinforcement_learning.algorithms import DDPG, PPO, TD3

        for name, algo_cls in [("TD3", TD3), ("DDPG", DDPG), ("PPO", PPO)]:
            assert getattr(rl, name) is algo_cls, f"rl.{name} is not the same object as algorithms.{name}"
            assert getattr(assume, name) is algo_cls, f"assume.{name} is not the same object as algorithms.{name}"

    def test_version_still_present(self):
        import assume

        assert hasattr(assume, "__version__")
        assert isinstance(assume.__version__, str)

    def test_non_rl_symbols_unchanged(self):
        """Core non-RL exports (World, MarketConfig, etc.) must still be present."""
        import assume

        for name in ("World", "MarketConfig", "MarketProduct", "load_scenario_folder", "run_learning"):
            assert hasattr(assume, name), f"Pre-existing export '{name}' missing after __init__ update"
