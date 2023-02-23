import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

class BaseStrategy():
    def __init__(self):
        super().__init__()

    def calculate_bids(self,  marketconfig: MarketConfig, operational_window):

        """
                Takes information from a unit that the unit operator manages and
                defines how it is dispatched to the market

                Return: volume, price
        """

        #checking what clearing mechanism is used in market
        market_clearing = marketconfig.market_mechanism

        if market_clearing=='pay_as_clear':

            price = operational_window['max_power']['marginal_cost'] 
            volume = operational_window['max_power']['power']
        
        elif market_clearing=='pay_as_bid':

            price = operational_window['max_power']['marginal_cost'] * 1.2
            volume = operational_window['max_power']['power']
        
        else:
            #TODO: To be implemented for Naive base line strategy 
            return 0, 0

        return volume, price  







class RLStrategy():
    def __init__(self):
        super().__init__()

    def load_strategy():
        """
                In case the strategy is learned with RL the policy (mapping of states to actions) it needs 
                to be loaded from current model

                Return: ?
        """

        raise NotImplementedError

    def calculate_bids(self,  marketconfig: MarketConfig, operational_window):

        """
                load current RL model to map state to action

                Return: volume, price
        """

        #checking what clearing mechanism is used in market
        market_clearing = marketconfig.market_mechanism

        if market_clearing=='pay_as_clear':
            
            #load model for market
            #sample action based on state
            
        
        elif market_clearing=='pay_as_bid':

            #load model for market
            #sample action based on state
        
        else:

            #load model for market
            #sample action based on state

        return NotImplementedError
    

    