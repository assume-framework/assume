from assume.common.market_objects import MarketConfig
from assume.strategies.base_strategy import BaseStrategy
from assume.units.base_unit import BaseUnit


class OTCStrategy(BaseStrategy):
    def __init__(self, *args, scale_firm_power_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale_firm_power_capacity

    def calculate_bids(
        self,
        unit: BaseUnit = None,
        market_config: MarketConfig = None,
        operational_window: dict = None,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """
        price = operational_window["max_power"]["marginal_cost"]
        volume = operational_window["max_power"]["power"]
        if "OTC" in market_config.name:
            volume *= self.scale
        bids = [{"price": price, "volume": volume}]
        return bids
