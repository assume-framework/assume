import pandas as pd

from assume.strategies import BaseStrategy


class BaseUnit:
    """A base class for a unit.

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
        node: str,
    ):
        self.id = id
        self.unit_operator = unit_operator
        self.technology = technology
        self.node = node
        self.bidding_strategies: dict[str, BaseStrategy] = bidding_strategies
        self.index = index

    def calculate_operational_window(
        self,
        product_type: str,
        product_tuple: tuple,
    ) -> dict:
        """Calculate the operation window for the next time step."""

        raise NotImplementedError()

    def reset(self):
        """Reset the unit to its initial state."""
        raise NotImplementedError()

    def calculate_bids(
        self,
        product_type,
        product_tuple,
    ):
        """Calculate the bids for the next time step."""

        if product_type not in self.bidding_strategies:
            return None

        # get operational window for each unit
        operational_window = self.calculate_operational_window(
            product_type=product_type,
            product_tuple=product_tuple,
        )

        return self.bidding_strategies[product_type].calculate_bids(
            unit=self,
            operational_window=operational_window,
        )

    def get_dispatch_plan(self, dispatch_plan: dict, current_time: pd.Timestamp):
        """Get the dispatch plan for the next time step."""

        pass
