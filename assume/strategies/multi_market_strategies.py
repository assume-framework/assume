import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch as th

from assume.common.base import (
    BaseUnit,
    LearningStrategy,
    MinMaxChargeStrategy,
    MinMaxStrategy,
    SupportsMinMax,
    SupportsMinMaxCharge,
)
from assume.common.fast_pandas import FastSeries
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import min_max_scale


class GenericEnergyMultiMarketStrategy:
    pass

class EnergyHeuristicRedispatchStrategy(GenericEnergyMultiMarketStrategy, MinMaxStrategy):

    def calculate_bids(self, unit, market_config, product_tuples, **kwargs):
        if market_config.market_mechanism == "redispatch":
            return self.calculate_redispatch_bids(
                unit, market_config, product_tuples, **kwargs
            )
        else:
            bids = self.calculate_EOM_bids(
                unit, market_config, product_tuples, **kwargs
            )
            return bids

    def calculate_redispatch_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        start = product_tuples[0][0]
        #end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.min_power, unit.max_power

        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]
            current_power = unit.outputs["energy"].at[start]
            
            # get the bid price from the EOM bids calculated before
            # the bid price on the redispatch market is based on the EOM bid price
            # as to represent regulated bids based on the spot market bids
            price = unit.outputs["eom_bids"].at[start]
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": product[2],
                    "price": price,
                    "volume": current_power,
                    "max_power": max_power,
                    "min_power": min_power,
                    "node": unit.node,
                }
            )

        unit.outputs["redispatch_bids"].loc[product_tuples[0][0]] = price

        return bids

    def calculate_EOM_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Generates a single price bid for the full available capacity (max_power).

        The method observes unit state, derives an action (bid price) from
        the actor network, and constructs one bid covering the entire capacity, without
        distinguishing between flexible and inflexible components.

        Notes
        -----
        The bid is written to the unit outputs for later reference during redispatch bidding as 'eom_bids'.

        Returns
        -------
        Orderbook
            A list containing one bid with start/end time, full volume, and calculated price.
        """
        start = product_tuples[0][0]
        end = product_tuples[0][1]

        # get technical bounds for the unit output from the unit
        min_power, max_power = unit.calculate_min_max_power(start, end)
        min_power = min_power[0]
        max_power = max_power[0]

        # =============================================================================
        # 1. Get the LMP forecast, which are the basis of the following decision
        # =============================================================================
        # ...

        # =============================================================================
        # 2. Create bids, based on the forecasted LMPs
        # =============================================================================
        # decide wether to take part in inc dec or not and for which hours
        # assuming a eom with pay-as-clear and a redispatch market with pay-as-bid
        # assuming bid_price_eom = bid_price_redispatch
        # mc: own marginal cost
        # LMP_n: forecasted LMP at own node n
        # LMP_Nm: forecasted LMP at neighboring nodes m
        # if mc > LMP_n and LMP_n < LMP_Nm -> generation pocket at own node, inc-dec could work -> offer own LMP_n
        # if mc > LMP_n and LMP_n > LMP_Nm -> load pocket at own node, no inc-dec opportunity -> bid mc
        # if mc < LMP_n and LMP_n < LMP_Nm -> generation pocket at own node and bidding mc would result in eom dispatch, no inc-dec opportunity -> offer own mc
        # if mc < LMP_n and LMP_n > LMP_Nm -> bid own LMP_n
        # else -> bid mc


        # actually formulate bids in orderbook format
        bids = [
            {
                "start_time": start,
                "end_time": end,
                "only_hours": None,
                "price": bid_price,
                "volume": max_power,
                "node": unit.node,
            },
        ]

        unit.outputs["eom_bids"].loc[product_tuples[0][0]] = bid_price

        return bids

    def calculate_profit(self, unit, marketconfig, orderbook):
        if marketconfig.market_mechanism == "redispatch":
            return self.calculate_redispatch_profit(unit, marketconfig, orderbook)
        else:
            return self.calculate_EOM_profit(unit, marketconfig, orderbook)

    def calculate_redispatch_profit(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the redispatch_profit for the unit and the total profit as sum of redispatch_profit, eom_profit and costs.

        Args
        ----
        unit : SupportsMinMax
            The unit for which to calculate the reward.
        marketconfig : MarketConfig
            Market configuration settings.
        orderbook : Orderbook
            Orderbook containing executed bids and details.

        Notes
        -----
        The reward is scaled and stored along with other outputs in the unit’s data to support learning.
        """
        revenue = 0
        costs = 0
        profit = 0

        reward = 0

        # iterate over all orders in the orderbook, to calculate order specific profit
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            # end includes the end of the last product, to get the last products' start time we deduct the frequency once
            end_excl = end - unit.index.freq

            # depending on way the unit calculates marginal costs we take costs
            marginal_cost = unit.outputs["marginal_cost"].at[start]

            duration = (end - start) / timedelta(hours=1)

            accepted_volume = order.get("accepted_volume", 0)
            accepted_price = order.get("accepted_price", 0)

            # calculate profit as income - running_cost from this event
            order_revenue = accepted_price * accepted_volume * duration
            order_cost = marginal_cost * accepted_volume * duration

            # collect profit and opportunity cost for all orders
            revenue += order_revenue
            costs += order_cost

        profit = revenue - costs

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["redispatch_profit"].loc[start:end_excl] = profit
        unit.outputs["redispatch_costs"].loc[start:end_excl] = costs

        unit.outputs['profit'].loc[start:end_excl] = unit.outputs["eom_profit"].loc[start:end_excl] + unit.outputs["redispatch_profit"].loc[start:end_excl]

    def calculate_EOM_profit(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates the reward for the unit on eom market based on profits and costs.

        Args
        ----
        unit : SupportsMinMax
            The unit for which to calculate the reward.
        marketconfig : MarketConfig
            Market configuration settings.
        orderbook : Orderbook
            Orderbook containing executed bids and details.

        Notes
        -----
        The profit is written to the unit outputs for later reference during reward calculation as 'eom_profit'.
        """
        # function is called after the market is cleared and we get the market feedback,
        # so we can calculate the profit

        product_type = marketconfig.product_type

        revenue = 0
        costs = 0
        profit = 0

        # iterate over all orders in the orderbook, to calculate order specific profit
        for order in orderbook:
            start = order["start_time"]
            end = order["end_time"]
            # end includes the end of the last product, to get the last products' start time we deduct the frequency once
            end_excl = end - unit.index.freq

            # depending on way the unit calculates marginal costs we take costs
            marginal_cost = unit.calculate_marginal_cost(
                start, unit.outputs[product_type].at[start]
            )
            unit.outputs["marginal_cost"].at[start] = marginal_cost

            duration = (end - start) / timedelta(hours=1)

            accepted_volume = order.get("accepted_volume", 0)
            accepted_price = order.get("accepted_price", 0)

            # calculate profit as income - running_cost from this event
            order_revenue = accepted_price * accepted_volume * duration
            order_cost = marginal_cost * accepted_volume * duration

            # collect profit and opportunity cost for all orders
            revenue += order_revenue
            costs += order_cost

        # consideration of start-up costs, which are evenly divided between the
        # upward and downward regulation events
        if (
            unit.outputs[product_type].at[start] != 0
            and unit.outputs[product_type].loc[start - unit.index.freq] == 0
        ):
            costs += unit.hot_start_cost / 2
        elif (
            unit.outputs[product_type].at[start] == 0
            and unit.outputs[product_type].loc[start - unit.index.freq] != 0
        ):
            costs += unit.hot_start_cost / 2

        profit = revenue - costs

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["eom_profit"].loc[start:end_excl] = profit
        unit.outputs["eom_costs"].loc[start:end_excl] = costs