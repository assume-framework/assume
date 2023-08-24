import numbers

import pandas as pd

from assume.common.base import SupportsMinMax


class Demand(SupportsMinMax):
    """A demand unit.

    Attributes
    ----------
    id : str
        The ID of the unit.
    technology : str
        The technology of the unit.
    max_power: float
        The maximum power consumption capacity of the demand unit in MW
    min_power: float
        The minimum power consumption capacity of the demand unit in MW.
    node: str = "bus0"
        The identifier of the electrical bus or network node to which the demand unit is connected.(Defaults to "bus0".)
    price: float | pd.Series = 3000.0
        The price associated with the demand unit's consumption, represented as a single float or a pandas Series.(Defaults to 3000.0 monetary units)
    location: tuple[float, float] = (0.0, 0.0)
        The geographical coordinates (latitude and longitude) of the demand unit's location (Defaults to (0.0, 0.0).)

    Methods
    ---------
    reset()
        Reset the unit's energy consumption.
    execute_current_dispatch()
        Executes the current dispatch of the unit by returning its volume within the given time range.
    calculate_min_max_power() -> tuple[pd.Series, pd.Series]
        Calculates and returns the bid volume as both the minimum and maximum power output of the unit.
    calculate_marginal_cost() -> float
        Calculates and returns the marginal cost of the unit based on the provided time and power.
    as_dict(self) -> dict
        Return the unit as a dictionary.


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

    def reset(self):
        """Reset the unit."""
        self.outputs["energy"] = pd.Series(0, index=self.index)

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """Execute the current dispatch of the unit."""
        end_excl = end - self.index.freq
        return self.volume[start:end_excl]

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate the minimum and maximum power output of the unit."""
        bid_volume = (self.volume - self.outputs[product_type]).loc[start:end]
        return bid_volume, bid_volume

    def calculate_marginal_cost(self, start: pd.Timestamp, power: float) -> float:
        """Calculate the marginal cost of the unit."""
        return self.price.at[start]

    def as_dict(self) -> dict:
        """Return the unit as a dictionary."""
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power": self.max_power,
                "min_power": self.min_power,
                "unit_type": "demand",
            }
        )

        return unit_dict
