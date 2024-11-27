# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime

import numpy as np

from assume.common.base import SupportsMinMax
from assume.common.forecasts import Forecaster


class Demand(SupportsMinMax):
    """
    A demand unit.

    Attributes:
        id (str): The unique identifier of the unit.
        unit_operator (str): The operator of the unit.
        technology (str): The technology of the unit.
        bidding_strategies (dict): The bidding strategies of the unit.
        index (pandas.DatetimeIndex): The index of the unit.
        max_power (float): The maximum power output capacity of the power plant in MW.
        min_power (float, optional): The minimum power output capacity of the power plant in MW. Defaults to 0.0 MW.
        node (str, optional): The node of the unit. Defaults to "node0".
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
        max_power: float,
        min_power: float,
        forecaster: Forecaster,
        node: str = "node0",
        price: float = 3000.0,
        location: tuple[float, float] = (0.0, 0.0),
        **kwargs,
    ):
        super().__init__(
            id=id,
            unit_operator=unit_operator,
            technology=technology,
            bidding_strategies=bidding_strategies,
            forecaster=forecaster,
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

        self.volume = -abs(self.forecaster[self.id])  # demand == negative
        # self.price = FastSeries(index=self.index, value=price)
        if self.id == "Denmark":
            self.price = self.forecaster["Denmark"]
        elif self.id == "France":
            self.price = self.forecaster["France"]
        elif self.id == "Netherlands":
            self.price = self.forecaster["Netherlands"]
        elif self.id == "Norway":
            self.price = self.forecaster["Norway"]
        elif self.id == "Austria":
            self.price = self.forecaster["electricity_price"]
        elif self.id == "Poland":
            self.price = self.forecaster["Poland"]
        elif self.id == "Sweden":
            self.price = self.forecaster["Sweden"]
        elif self.id == "Switzerland":
            self.price = self.forecaster["Switzerland"]
        elif self.id == "Sweden":
            self.price = self.forecaster["Czech Republic"]
        elif self.id == "export":
            self.price = self.forecaster["export"]
        else:
            self.price = 3000.0

    def execute_current_dispatch(
        self,
        start: datetime,
        end: datetime,
    ) -> np.array:
        """
        Execute the current dispatch of the unit.
        Returns the volume of the unit within the given time range.

        Args:
            start (datetime.datetime): The start time of the dispatch.
            end (datetime.datetime): The end time of the dispatch.

        Returns:
            np.array: The volume of the unit for the given time range.
        """

        return self.volume.loc[start:end]

    def calculate_min_max_power(
        self, start: datetime, end: datetime, product_type="energy"
    ) -> tuple[np.array, np.array]:
        """
        Calculates the minimum and maximum power output of the unit and returns the bid volume as both the minimum and maximum power output of the unit.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            end (pandas.Timestamp): The end time of the dispatch.

        Returns:
            tuple[pandas.Series, pandas.Series]: The bid colume as both the minimum and maximum power output of the unit.
        """
        end_excl = end - self.index.freq
        bid_volume = (self.volume - self.outputs[product_type]).loc[start:end_excl]

        return bid_volume, bid_volume

    def calculate_marginal_cost(self, start: datetime, power: float) -> float:
        """
        Calculate the marginal cost of the unit returns the marginal cost of the unit based on the provided time and power.

        Args:
            start (pandas.Timestamp): The start time of the dispatch.
            power (float): The power output of the unit.

        Returns:
            float: the marginal cost of the unit for the given power.
        """
        if self.id == "Denmark":
            return self.price.at[start]
        elif self.id == "France":
            return self.price.at[start]
        elif self.id == "Netherlands":
            return self.price.at[start]
        elif self.id == "Norway":
            return self.price.at[start]
        elif self.id == "Austria":
            return self.price.at[start]
        elif self.id == "Poland":
            return self.price.at[start]
        elif self.id == "Sweden":
            return self.price.at[start]
        elif self.id == "Switzerland":
            return self.price.at[start]
        elif self.id == "Sweden":
            return self.price.at[start]
        elif self.id == "export":
            return self.price.at[start]
        else:
            return 3000.0

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
