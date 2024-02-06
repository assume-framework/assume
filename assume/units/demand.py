# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numbers

import pandas as pd

from assume.common.base import SupportsMinMax


class Demand(SupportsMinMax):
    """
    A demand unit.

    Parameters:
        id (str): The unique identifier of the unit.
        index (pd.DatetimeIndex): The index of the unit.
        max_power (float): The maximum power output capacity of the power plant in MW.
        min_power (float): The minimum power output capacity of the power plant in MW.
        volume (pd.Series): The volume of the unit.
        price (float): The price of the unit.
        forecaster (Forecaster): The forecaster of the unit.

    Args:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        index (pd.DatetimeIndex): The index of the unit.
        max_power (float): The maximum power output capacity of the power plant in MW.
        min_power (float, optional): The minimum power output capacity of the power plant in MW. Defaults to 0.0 MW.
        node (str, optional): The node of the unit. Defaults to "bus0".
        price (float): The price of the unit.
        location (tuple[float, float], optional): The location of the unit. Defaults to (0.0, 0.0).

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
            location=location,
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

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        Execute the current dispatch of the unit.
        Returns the volume of the unit within the given time range.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            end (pd.Timestamp): The end time of the dispatch.

        Returns:
            pd.Series: The volume of the unit within the gicen time range.
        """

        return self.volume[start:end]

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculates the minimum and maximum power output of the unit and returns the bid volume as both the minimum and maximum power output of the unit.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            end (pd.Timestamp): The end time of the dispatch.

        Returns:
            tuple[pd.Series, pd.Series]: The bid colume as both the minimum and maximum power output of the unit.
        """
        end_excl = end - self.index.freq
        bid_volume = (self.volume - self.outputs[product_type]).loc[start:end_excl]
        return bid_volume, bid_volume

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """
        Calculate the marginal cost of the unit returns the marginal cost of the unit based on the provided time and power.

        Args:
            start (pd.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        return self.price.at[start]

    def as_dict(self) -> dict:
        """
        Returns the unit as a dictionary.

        Returns:
            dict: The unit as a dictionary.
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
