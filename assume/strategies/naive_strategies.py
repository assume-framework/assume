from assume.common.market_objects import MarketConfig
from assume.strategies.base_strategy import BaseStrategy, OperationalWindow
from assume.units.base_unit import BaseUnit


class NaiveStrategy(BaseStrategy):
    def __init__(self, *args, scale_firm_power_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale_firm_power_capacity

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
        **kwargs,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """
        price = operational_window["states"]["max_power"]["cost"]
        volume = operational_window["states"]["max_power"]["volume"]
        bids = [{"price": price, "volume": volume}]
        return bids


class NaivePosReserveStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
        **kwargs,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """
        price = 0
        volume = operational_window["states"]["pos_reserve"]["volume"]
        bids = [{"price": price, "volume": volume}]
        return bids


class NaiveNegReserveStrategy(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate_bids(
        self,
        unit: BaseUnit,
        operational_window: OperationalWindow,
        market_config: MarketConfig,
        **kwargs,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """
        price = 0
        volume = operational_window["states"]["neg_reserve"]["volume"]
        bids = [{"price": price, "volume": volume}]
        return bids
