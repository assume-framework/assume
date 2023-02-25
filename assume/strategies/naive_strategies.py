import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union
from .base_strategy import BaseStrategy


class NaiveStrategyNoMarkUp(BaseStrategy):
    def __init__(self):
        super().__init__()

    def calculate_bids(self, operational_window):

        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market

        Return: volume, price
        """

        price = round(operational_window["max_power"]["marginal_cost"], 1)
        volume = round(operational_window["max_power"]["power"], 1)

        return volume, price


class NaiveStrategyMarkUp(BaseStrategy):
    def __init__(self):
        super().__init__()

    def calculate_bids(self, operational_window):

        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market, here a little mark up on the marginal
        costs is used as well

        Return: volume, price
        """
        price = round(operational_window["max_power"]["marginal_cost"] * 1.2, 1)
        volume = round(operational_window["max_power"]["power"], 1)

        return volume, price
