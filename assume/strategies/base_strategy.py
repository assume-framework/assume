from typing import TypedDict

from assume.common.market_objects import MarketConfig, Product


class SingleWindow(TypedDict):
    volume: int or float
    cost: int or float


class OperationalWindow(TypedDict):
    window: Product
    ops: dict[str, SingleWindow]


class BaseStrategy:
    """A base class for a bidding strategy.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, *args, **kwargs):
        self.is_learning_strategy = False

    def calculate_bids(
        self,
        unit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
    ):
        raise NotImplementedError()

    def calculate_reward(
        self,
        start,
        end,
        product_type,
        clearing_price,
        unit,
    ):
        pass
