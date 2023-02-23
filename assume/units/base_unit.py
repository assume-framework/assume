

class BaseUnit():
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
    calculate_operational_window()
        Calculate the operation window for the next time step.
    """
    
    def __init__(self,
                 id: str,
                 technology: str,
                 node: str):

        self.id = id
        self.technology = technology
        self.node = node

    def calculate_operational_window(self) -> dict:
        """Calculate the operation window for the next time step."""

        raise NotImplementedError




