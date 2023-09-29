import numbers

import pandas as pd

from assume.common.base import SupportsMinMax


class Demand(SupportsMinMax):
    """
    A demand unit.

    :param id: unique identifier for the unit
    :type id: str
    :param unit_operator: the operator of the unit
    :type unit_operator: str
    :param technology: the technology of the unit
    :type technology: str
    :param bidding_strategies: the bidding strategies of the unit
    :type bidding_strategies: dict
    :param index: the index of the unit
    :type index: pd.DatetimeIndex
    :param max_power: the maximum power output of the unit (kW)
    :type max_power: float | pd.Series
    :param min_power: the minimum power output of the unit (kW)
    :type min_power: float | pd.Series
    :param node: the node of the unit
    :type node: str
    :param price: the price of the unit
    :type price: float | pd.Series
    :param location: the location of the unit (latitude, longitude)
    :type location: tuple[float, float]
    :param kwargs: additional keyword arguments
    :type kwargs: dict

    Methods
    -------
    """

    def __init__(
        self,
        id: str,
        unit_operator: str,
        technology: str,
        bidding_strategies: dict,
        index: pd.DatetimeIndex,
        max_power: float,
        min_power: float,
        node: str = "bus0",
        price: float | pd.Series = 3000.0,
        location: tuple[float, float] = (0.0, 0.0),
        **kwargs
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            index=index,
            node=node,
            **kwargs,
        )
        """Create a demand unit."""
        self.max_power = max_power
        self.min_power = min_power
        if max_power > 0 and min_power <= 0:
            self.max_power = min_power
            self.min_power = -max_power
        self.ramp_down = max(abs(min_power), abs(max_power))
        self.ramp_up = max(abs(min_power), abs(max_power))
        volume = self.forecaster[self.id]
        self.volume = -abs(volume)  # demand is negative
        if isinstance(price, numbers.Real):
            price = pd.Series(price, index=self.index)
        self.price = price
        self.location = location

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        Execute the current dispatch of the unit.
        Returns the volume of the unit within the given time range.

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :return: the volume of the unit within the given time range
        :rtype: pd.Series
        """
        return self.volume[start:end]

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate the minimum and maximum power output of the unit.
        Returns the bid volume as both the minimum and maximum power output of the unit.

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param end: the end time of the dispatch
        :type end: pd.Timestamp
        :param product_type: the product type of the unit
        :type product_type: str
        :return: the bid volume as both the minimum and maximum power output of the unit
        :rtype: tuple[pd.Series, pd.Series]
        """
        end_excl = end - self.index.freq
        bid_volume = (self.volume - self.outputs[product_type]).loc[start:end_excl]
        return bid_volume, bid_volume

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit.
        Returns the marginal cost of the unit based on the provided time and power.

        :param start: the start time of the dispatch
        :type start: pd.Timestamp
        :param power: the power output of the unit
        :type power: float
        :return: the marginal cost of the unit
        :rtype: float
        """
        return self.price.at[start]

    def as_dict(self) -> dict:
        """
        Return the unit as a dictionary.

        :return: the unit as a dictionary
        :rtype: dict
        """
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power": self.max_power,
                "min_power": self.min_power,
                "unit_type": "demand",
            }
        )

        return unit_dict
