import pandas as pd

from assume.strategies import OperationalWindow
from assume.units.base_unit import OperationData, SupportsMinMax


class Demand(SupportsMinMax):
    """A demand unit.

    Attributes
    ----------
    id : str
        The ID of the unit.
    technology : str
        The technology of the unit.
    node : str
        The node of the unit.

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
        if isinstance(price, float):
            price = pd.Series(price, index=self.index)
        self.price = price
        self.location = location

    def reset(self):
        self.outputs["energy"] = pd.Series(0, index=self.index)

    def calculate_min_max_power(
        self, start: pd.Timestamp, end: pd.Timestamp, product_type="energy"
    ) -> dict:
        bid_volume = (self.volume - self.outputs[product_type]).loc[start:end].max()
        return bid_volume, bid_volume

    def calculate_marginal_cost(
        self, start: pd.Timestamp, power: float
    ) -> OperationData:
        return self.price.at[start]

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
