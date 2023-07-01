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
        self.total_power_output = pd.Series(0.0, index=self.index)

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
        market_config,
        product_tuple,
    ):
        """Calculate the bids for the next time step."""

        if market_config.product_type not in self.bidding_strategies:
            return None

        # get operational window for each unit
        operational_window = self.calculate_operational_window(
            product_type=market_config.product_type,
            product_tuple=product_tuple,
        )

        # check if operational window is valid
        if operational_window is None:
            return None

        return self.bidding_strategies[market_config.product_type].calculate_bids(
            unit=self,
            market_config=market_config,
            operational_window=operational_window,
        )

    def set_dispatch_plan(
        self,
        dispatch_plan: dict,
        start: pd.Timestamp,
        end: pd.Timestamp,
        product_type: str,
    ):
        """
        adds dispatch plan from current market result to total dispatch plan
        """
        pass

    def execute_current_dispatch(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ):
        """
        check if the total dispatch plan is feasible
        This checks if the market feedback is feasible for the given unit.
        And sets the closest dispatch if not.
        The end date is inclusive.
        """
        end_excl = end - self.index.freq
        return self.total_power_output[start:end_excl]

    def as_dict(self) -> dict:
        return {
            "technology": self.technology,
            "unit_operator": self.unit_operator,
            "unit_type": "base_unit",
        }
