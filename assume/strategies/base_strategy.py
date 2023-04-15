import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union


class BaseStrategy:
    """A base class for a bidding strategy.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self):
        super().__init__()

    def calculate_bids(self, operational_window):
        raise NotImplementedError()
