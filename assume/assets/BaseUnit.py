import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union, Type

from assume.common.policies import BasePolicy
from assume.common.bids import BaseBid

class BaseUnit(ABC):

    def __init__(self,
                 id: str,
                 technology: str,
                 node: str,
                 policy: Union[str, Type[BasePolicy]]):

        super().__init__()

        self.id = id
        self.technology = technology
        self.node = node
        self.policy = policy

        self.collected_bids = []

    @abstractmethod
    def reset(self) -> None:
        """Reset the unit to its initial state."""

        raise NotImplementedError()

    @abstractmethod
    def formulate_bids(self, market: str, bids: Type[BaseBid]) -> Type[BaseBid]:
        """Formulate bids for the given market."""
    
        return self.policy.formulate_bids(market)

    @staticmethod
    def collect_feedback(self, bids: Type[BaseBid]) -> None:
        """Collect feedback from the market."""

        self.collected_bids.append()
    
    @abstractmethod
    def step(self) -> None:
        """Perform a single time step."""

        raise NotImplementedError()







