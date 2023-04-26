from assume.strategies.base_strategy import BaseStrategy
from assume.units.base_unit import BaseUnit


class NaiveStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def calculate_bids(
        self,
        unit: BaseUnit = None,
        operational_window: dict = None,
    ):
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """
        price = operational_window["max_power"]["marginal_cost"]
        volume = operational_window["max_power"]["power"]
        bids = [{"price": price, "volume": volume}]
        return bids
