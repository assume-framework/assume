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
        technology: str = None,
        node: str = None,
        bidding_strategies: dict = None,
        index: pd.DatetimeIndex = None,
    ):
        self.id = id
        self.technology = technology
        self.node = node
        self.bidding_strategies = bidding_strategies
        self.index = index

    def calculate_operational_window(self, current_time) -> dict:
        """Calculate the operation window for the next time step."""

        raise NotImplementedError()

    def reset(self):
        """Reset the unit to its initial state."""
        raise NotImplementedError()

    def calculate_bids(
        self,
        unit,
        product_type,
        operational_window,
    ):
        """Calculate the bids for the next time step."""

        return self.bidding_strategies[product_type].calculate_bids(
            unit=unit, operational_window=operational_window
        )

    def get_dispatch_plan(self, dispatch_plan: dict, current_time: pd.Timestamp):
        """Get the dispatch plan for the next time step."""

        pass
