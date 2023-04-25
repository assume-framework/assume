from assume.strategies.base_strategy import BaseStrategy


class NaiveStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()

    def calculate_bids(self, operational_window, current_time):

        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """

        price = operational_window["max_power"]["marginal_cost"]
        volume = operational_window["max_power"]["power"]

        return volume, price
