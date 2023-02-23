from typing import TypedDict

class SingleWindow(TypedDict):
    power: float
    marginal_cost: float

class OperationalWindow(TypedDict):
    current_power: SingleWindow
    min_power: SingleWindow
    max_power: SingleWindow


class BaseUnit():
    def __init__(self,
                 id: str,
                 technology: str,
                 node: str):

        self.id = id
        self.technology = technology
        self.node = node

    def calculate_operational_window(self) -> OperationalWindow:
        """Calculate the operation window for the next time step."""

        raise NotImplementedError




