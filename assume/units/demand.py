import pandas as pd

from assume.strategies import BaseStrategy, OperationalWindow
from assume.units.base_unit import BaseUnit


class Demand(BaseUnit):
    """A demand unit.

    Attributes
    ----------
    id : str
        The ID of the unit.
    technology : str
        The technology of the unit.
    node : str
        The node of the unit.

    Methods
    -------
    calculate_operational_window(product)
        Calculate the operation window for the next time step.
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
        volume: float or pd.Series,
        node: str = "bus0",
        price: float or pd.Series = 3000.0,
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
        )

        self.max_power = max_power
        self.min_power = min_power
        self.volume = -volume  # demand is negative
        self.price = price
        self.location = location

    def reset(self):
        self.outputs["energy"] = pd.Series(0, index=self.index)

    def calculate_operational_window(
        self,
        product_type: str,
        product_tuple: tuple,
    ) -> OperationalWindow:
        start, end, only_hours = product_tuple
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        """Calculate the operation window for the next time step."""
        bid_volume = (self.volume - self.outputs[product_type]).loc[start:end].max()

        if type(self.price) == pd.Series:
            bid_price = self.price.loc[start:end].mean()
        else:
            bid_price = self.price

        return {
            "window": (start, end),
            "ops": {"max_power": {"volume": bid_volume, "cost": bid_price}},
        }

    def as_dict(self) -> dict:
        unit_dict = super().as_dict()
        unit_dict.update(
            {
                "max_power": self.max_power,
                "min_power": self.min_power,
                "unit_type": "demand",
            }
        )

        return unit_dict
