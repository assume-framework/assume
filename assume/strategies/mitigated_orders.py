# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
from datetime import datetime, timedelta
from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product
from assume.strategies.naive_strategies import NaiveSingleBidStrategy
from itertools import groupby
from operator import itemgetter

# had to change type hint to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from assume.common.units_operator import UnitsOperator 

class MitigatedOrder(BaseStrategy):
    """
    A unit operator strategy that computes the bid of each unit according to its
    own BaseStrategy, but also considers market power mitigation rules. If the unit
    operator is found to be pivotal for a given product, the bids of its units for these  
    products are assessed through a conduct screening. The conduct screening checks
    whether the bids of the units are above a certain threshold that depend on their 
    marginal cost. If they are, two versions of the bid are produced: a 'not_mitigated'
    one and a 'mitigated' where the bid is replaced by the reference level (marginal cost)
    of the unit.
    -- 
    Note: this method performs pivotality and conduct checks at unit operator level.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        


    def tot_capacity(
            self,
            units_operator: "UnitsOperator", 
    ) -> dict[str,float]:

        #TODO: this should explicity consider supply unit only
        tot_capacity = {}
        for unit_id, unit in units_operator.units.items():
            for market_id, market_strategy in unit.bidding_strategies.items():
                tot_capacity[market_id] = tot_capacity.get(market_id, 0) + unit.max_power
        
        return tot_capacity


    

    def conduct_check(
            self,
            unit: SupportsMinMax,
            marketconfig: MarketConfig,
            mitigated_products: list[Product]
    ) -> Orderbook:
        """
        Conducts the screening on the bids.
        Args:
            unit (SupportsMinMax): The unit to screen.
            market_config (MarketConfig): The market configuration.
            mitigated_products (list[Product]): The products where screening is active.
        Returns:
            Orderbook: The screened bids.
        """

        checked_bids = Orderbook()

        bids_to_check = unit.calculate_bids(
                marketconfig,
                mitigated_products,
            )
        
        ref_levels = NaiveSingleBidStrategy().calculate_bids(
                unit,
                marketconfig,
                mitigated_products,
            )
        # sort and group bids by product,as unit can have multiple bids for the same product
        product_getter = itemgetter("start_time", "end_time", "only_hours")
        # there is only one reference level for each product, the unit marginal cost
        for prod, prod_bids in groupby(sorted(bids_to_check, key=product_getter), key=product_getter): 
            ref = next((r for r in ref_levels if product_getter(r) == prod), 0)
            #if bid is higher than conduct threshold, two version are produced - a mitigated and a not mitigated one
            for bid in prod_bids:
                if bid['price'] > self.conduct_threshold(ref['price']):
                    bid['status'] = 'not_mitigated'
                    bid_copy = bid.copy()
                    bid_copy['price'] = ref['price']
                    bid_copy['status'] = 'mitigated'
                    checked_bids.extend([bid, bid_copy])
                
                else:
                    bid['status'] = 'not_above_conduct'
                    checked_bids.append(bid)
        
        return checked_bids


    def init_forecaster(self, units_operator: "UnitsOperator", marketconfig: MarketConfig):
        """
        Notes: equivalent to prepare_observations in BaseLearningStrategy
        """
        unit = next(iter(units_operator.units.values()))
        self.pivotal_threshold = marketconfig.params.get("pivotal_threshold", 1)
        self.conduct_threshold = marketconfig.params.get("conduct_threshold", lambda ref: min(ref + 100, ref * 3))
        self.frequency = int(marketconfig.opening_duration / timedelta(hours=1))
        self.res_load_obs = unit.forecaster[f"residual_load__{marketconfig.market_id}"]


    def pivotality_check(self, units_operator: "UnitsOperator", marketconfig: MarketConfig, start: datetime, end: datetime):
        """
        Creates the next observation for the strategy.
        Returns:
            dict: The next observation.
        -- 
        Notes: equivalent to create_observation in BaseLearningStrategy
        """
        # --- 1. Residual load forecast (forward-looking) ---
        if not hasattr(self, "res_load_obs"):
            self.init_forecaster(units_operator, marketconfig)

        res_load_forecast = self.res_load_obs.window(
            start, self.frequency, direction="forward"
        )

        # --- 2. Market supply, net of units_operator supply (backward-looking) ---
        if not hasattr(marketconfig, "market_supply"):
            self.net_market_supply =  0 # for the first run, assume 0
        
        # --- 3. Compute RSI as the quotient between res load forecastand available supply w/o units_op ---
        residual_supply_index = self.net_market_supply / res_load_forecast
        pivotal_hours = residual_supply_index < self.pivotal_threshold
        
        return pivotal_hours
        
           
    
    def calculate_bids(
        self,
        units_operator: "UnitsOperator", 
        marketconfig: MarketConfig,
        product_tuples: list[Product],
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        
        --
        Note: 
        The orderbook contains redundant orders in case of mitigation (a not_mitigated and a mitigated bid).
        This is to allow the market to choose which one to accept.
        """

        
        ## Compute marginal costs ###
        #TODO: compute hours where mitigation is active
        pivotal_products = self.pivotality_check(units_operator, marketconfig)  #TODO: divide by total available capacity in the market 
        not_pivotal_products = [product for product in product_tuples if product not in pivotal_products]

        checked_op_bids = Orderbook()
        not_checked_op_bids = Orderbook()

        for unit_id, unit in units_operator.units.items():
            not_checked_bids = unit.calculate_bids(
                    marketconfig,
                    not_pivotal_products,
                )
        
            checked_bids = self.conduct_check(unit, pivotal_products)
            not_checked_op_bids.extend(not_checked_bids)
            checked_op_bids.extend(checked_bids)
            
        for op_bid in not_checked_op_bids:
            op_bid['status'] = 'not_checked'

        # merge the orderbooks, sort them by product, bid_id and status
        all_op_bids = [*checked_op_bids, *not_checked_op_bids]
        all_op_bids = sorted(all_op_bids, key=lambda p: (p["start_time"], p["bid_id"], p["status"]))

        return all_op_bids

    