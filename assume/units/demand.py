from ..strategies import BaseStrategy
from .base_unit import BaseUnit


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
        technology: str = None,
        node: str = None,
        price: int = 900,
        volume: int = -1000,
        location: tuple[float, float] = None,
        unit_operator_id: str = None,
        bidding_strategy: BaseStrategy = {},
        **kwargs
    ):
        super().__init__(id, technology, node, bidding_strategy=bidding_strategy)

        self.price = price
        self.volume = volume
        self.location = location
        self.total_power_output = []
        self.unit_operator = unit_operator_id

    def reset(self):
        self.current_time_step = 0

    def calculate_operational_window(self, product) -> dict:
        """Calculate the operation window for the next time step."""

        return {"max_power": {"power": self.volume, "marginal_cost": self.price}}  # MW
