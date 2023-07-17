from typing import TypedDict

from assume.common.market_objects import MarketConfig


class SingleWindow(TypedDict):
    volume: int or float
    cost: int or float


class OperationalWindow(TypedDict):
    window: tuple
    ops: dict[str, SingleWindow]


class BaseStrategy:
    """A base class for a bidding strategy.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, *args, **kwargs):
        is_learning_strategy = False
        pass

    def calculate_bids(
        self,
        unit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
    ):
        raise NotImplementedError()
