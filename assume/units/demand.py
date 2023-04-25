import pandas as pd

from assume.strategies import BaseStrategy
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
        technology: str = "inflexible_demand",
        node: str = None,
        price: float or pd.Series = 3000.0,
        volume: float or pd.Series = 1000,
        location: tuple[float, float] = None,
        bidding_strategy: BaseStrategy = None,
        **kwargs
    ):
        if bidding_strategy is None:
            bidding_strategy = {}
        super().__init__(
            id=id, technology=technology, node=node, bidding_strategy=bidding_strategy
        )

        self.price = price
        self.volume = volume
        self.location = location
        self.total_power_output = []

    def reset(self):
        self.current_time_step = 0

    def calculate_operational_window(self, product, current_time) -> dict:
        """Calculate the operation window for the next time step."""
        if type(self.volume) == pd.Series:
            bid_volume = self.volume.loc[current_time]
        else:
            bid_volume = self.volume

        if type(self.price) == pd.Series:
            bid_price = self.price.loc[current_time]
        else:
            bid_price = self.price

        return {"max_power": {"power": -bid_volume, "marginal_cost": bid_price}}  # MW
