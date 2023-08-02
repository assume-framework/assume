from assume.common.base import BaseStrategy, LearningStrategy
from assume.strategies.extended import OTCStrategy
from assume.strategies.flexable import flexableEOM, flexableNegCRM, flexablePosCRM
from assume.strategies.flexable_storage import (
    flexableEOMStorage,
    flexableNegCRMStorage,
    flexablePosCRMStorage,
)

try:
    from assume.strategies.learning_strategies import RLStrategy
except ImportError:
    pass
from assume.strategies.naive_strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
)
