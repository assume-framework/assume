import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

class BaseStrategy():
    def __init__(self):
        super().__init__()

    def calculate_bids(self, marketconfig: MarketConfig, operational_window):

        """"
                Takes information from a unit that the unit operator manages and
                defines how it is dispatched to the market

                Return: volume, price
        """"

        #checking what clearing mechanism is used in market
        market_clearing = marketconfig.market_mechanism

        
        price = operational_window['max_power']['marginal_cost']
        volume = operational_window.['max_power']['power']

        return volume, price  

    def load_strategy():
        """"
                In case the strategy is learned with RL the policy (mapping of states to actions) need 
                to be loaded from current model

                Return: ?
        """"

        raise NotImplementedError