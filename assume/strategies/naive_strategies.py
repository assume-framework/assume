from .base_strategy import BaseStrategy


class NaiveStrategyNoMarkUp(BaseStrategy):
    def __init__(self):
        super().__init__()

    def calculate_bids(self, market, operational_window):

        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """

        price = operational_window["max_power"]["marginal_cost"]
        volume = operational_window["max_power"]["power"]

        return volume, price


class NaiveStrategyMarkUp(BaseStrategy):
    def __init__(self):
        super().__init__()

    def calculate_bids(self, market, operational_window):

        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market, here a little mark up on the marginal
        costs is used as well

        Return: volume, price
        """
        price = operational_window["max_power"]["marginal_cost"] * 1.2
        volume = operational_window["max_power"]["power"]

        return volume, price
