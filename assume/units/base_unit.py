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
        technology: str,
        node: str,
        bidding_strategy: BaseStrategy,
    ):
        self.id = id
        self.technology = technology
        self.node = node
        self.bidding_strategy = bidding_strategy

    def calculate_operational_window(self, product) -> dict:
        """Calculate the operation window for the next time step."""

        raise NotImplementedError()

    def reset(self):
        """Reset the unit to its initial state."""
        raise NotImplementedError()
