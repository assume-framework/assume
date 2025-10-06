# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import random
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
from assume.common.base import BaseUnit, BaseStrategy
from assume.common.fast_pandas import FastSeries
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product


class ErevRothStrategy(BaseStrategy):
    """ An Erev-Roth bidding strategy for one unit"""

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        
        self.unit_id = kwargs["unit_id"]
        # defines bounds of actions space
        self.min_bid_price = kwargs.get("min_bid_price", 0)
        self.max_bid_price = kwargs.get("max_bid_price", 100)
        self.prior = kwargs.get("prior", "uniform")
        self.n_actions = kwargs.get("n_actions", 10)
        self.experiment = kwargs.get("experiment", 0.1)
        self.forget = kwargs.get("forget", 0.1)
        super().__init__()
     
    
    def init_action_space(self) -> list:
        """Initializes the action space as a set of equally distant bid 
        prices between `self.min_bid_price` and `self.max_bid_price`. 

        Returns:
            list: The list of discrete actions
        """
        
        actions = np.linspace(self.min_bid_price, self.max_bid_price, self.n_actions)
        return actions.tolist()
    

    def init_reward(self, unit:BaseUnit, market_config:MarketConfig) -> tuple[float,float]:
        """Initializes the minimum and maximum reward for the unit,
        based on the minimum and maximum bid price in the market configuration.
        
        Returns:
            tuple: The minimum and maximum reward
        """
        min_reward = unit.max_power * market_config.minimum_bid_price
        max_reward = unit.max_power * market_config.maximum_bid_price
        return min_reward, max_reward
    
    def init_propensity(self, unit:BaseUnit, product_tuples: list[Product]) -> list:
        """Initializes the propensity of performing actions for each product
        in the list of product tuples, where propensity[p][a] is the
        propensity of action a (i.e. bidding a specific price) for product p.
        
        Returns:
            list: List of propensity
        """
        if self.prior == "uniform":
            propensity = [[1/self.n_actions for a in range(self.n_actions)] for p in product_tuples]
        elif self.prior == "normal":
            start = product_tuples[0][0]
            mu = unit.calculate_marginal_cost(start,  unit.get_output_before(start))
            cumsum = norm.cdf(unit.actions, loc=mu, scale=self.n_actions)
            bins = [float(cumsum[0])] + [float(cumsum[i] - cumsum[i-1]) for i in range(1, len(cumsum) - 1)]
            bins += [1 - sum(bins)]
            propensity = [bins for p in product_tuples]
        else:
            raise ValueError(f"Unknown prior for propensity distribution: {self.prior}")
        return propensity
    


    def get_actions(self, unit:BaseUnit) -> int: 
        """Computes the next action given the propensity of performing each actions,
        drawing from a multinomial distribution. Returns a list of actions where 
        actions[p][a] is the index of an action a to be performed for product p.
        
        Args:
            unit (BaseUnit): The unit to be dispatched.
        
        Returns:
            list: list with indices of the action for each product
        """
        actions = []
        
        for prob in unit.propensity:
            idx = random.choices(range(self.n_actions), weights=prob, k=1)[0]
            actions.append(idx)
        return actions

    
    def calculate_bids(
        self,
        unit: BaseUnit,
        market_config: MarketConfig,
        product_tuples: list[Product],
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            unit (BaseUnit): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
                
        start_all = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product
        

        
        ## Compute marginal costs ###  
        previous_power = unit.get_output_before(
            start_all
        )  # power output of the unit before the start time of the first product
        op_time = unit.get_operation_time(start_all)
        min_power_values, max_power_values = unit.calculate_min_max_power(
            start_all, end_all
        )  # minimum and maximum power output of the unit between the start time of the first product and the end time of the last product

        bids = []
        
        # create or retrieve discretized action space
        if getattr(unit, "actions", None) is None: 
            unit.reward = self.init_reward(unit, market_config)
            unit.actions = self.init_action_space()
            unit.propensity = self.init_propensity(unit, product_tuples)

        actions = self.get_actions(unit)

        for i, (product, min_power, max_power) in enumerate(zip(
            product_tuples, min_power_values, max_power_values
        )):
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]
            current_power = unit.outputs["energy"].at[
                start
            ]  # power output of the unit at the start time of the current product

            volume = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )

            idx = actions[i]
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": unit.actions[idx],
                    "volume": volume,
                    "node": unit.node,
                }
            )
            unit.outputs["actions"].loc[start] = actions[i]

            if "node" in market_config.additional_fields:
                bids[-1]["max_power"] = unit.max_power if volume > 0 else unit.min_power
                bids[-1]["min_power"] = min_power if volume > 0 else unit.max_power

            previous_power = volume + current_power
            if previous_power > 0:
                op_time = max(op_time, 0) + 1
            else:
                op_time = min(op_time, 0) - 1

        if "node" in market_config.additional_fields:
            return bids
        else:
            return self.remove_empty_bids(bids)
        

    def get_propensity(self, 
                        propensity:list, 
                        action_idx:int, 
                        reward:float) -> list: 
        """
        Iterate over the items of a list of propensities
        for a given action and update it according to the
        Erev-Roth update rule.
        
        Args
            ----
            propensity : list
                The list of action propensities for a given product.
            action_idx : int
                The index of the action that was performed.
            reward: float
                The reward from performing that action (between 0 and 1).

        Returns:
            list: The updated list of action propensities.
        
        Notes:
        This method overwrites the propensity list at each iteration.
        The propensity list is normalized to sum to 1.
        
            """
        new_propensity = []

        for idx, prop in enumerate(propensity):
            if idx == action_idx:
                new_prop = (1 - self.forget) * prop + reward * (1 - self.experiment)
            else: 
                new_prop = (1 - self.forget) * prop + reward * self.experiment / (self.n_actions - 1)
            new_propensity.append(float(new_prop))
        
        new_propensity = [float(p) / sum(new_propensity) for p in new_propensity]
        return new_propensity
    
    
    def calculate_reward(
            self,
            unit: BaseUnit,
            marketconfig: MarketConfig,
            orderbook: Orderbook,
        ):
            """
            Calculates the reward for the unit based on profits.
            The reward is normalized to be between 0 and 1.

            Args
            ----
            unit : BaseUnit
                The unit for which to calculate the reward.
            marketconfig : MarketConfig
                Market configuration settings.
            orderbook : Orderbook
                Orderbook containing executed bids and details.

            Notes
            -----
            The propensity list is overwritten at each iteration.

            """
            # Function is called after the market is cleared, and we get the market feedback,
            # allowing us to calculate profit based on the realized transactions.
            product_type = marketconfig.product_type
            start = orderbook[0]["start_time"]
            end = orderbook[0]["end_time"]
            # `end_excl` marks the last product's start time by subtracting one frequency interval.
            end_excl = end - unit.index.freq

            # Depending on how the unit calculates marginal costs, retrieve cost values.
            marginal_cost = unit.calculate_marginal_cost(
                start, unit.outputs[product_type].at[start]
            )
            market_clearing_price = orderbook[0]["accepted_price"]
            duration = (end - start) / timedelta(hours=1)

            income = 0.0
            operational_cost = 0.0

            accepted_volume_total = 0
            offered_volume_total = 0

            # Iterate over all orders in the orderbook to calculate order-specific profit.

            for order_idx, order in enumerate(orderbook):
                accepted_volume = order.get("accepted_volume", 0)
                accepted_volume_total += accepted_volume

                offered_volume_total += order["volume"]

                # Calculate profit as income minus operational cost for this event.
                order_income = market_clearing_price * accepted_volume * duration
                order_cost = marginal_cost * accepted_volume * duration

                # Accumulate income and operational cost for all orders.
                income += order_income
                operational_cost += order_cost
                profit = income - operational_cost

                # Retrieve which action had been played and update propensities
                action_idx = unit.outputs["actions"].loc[order["start_time"]]
                propensity = unit.propensity[order_idx]
                # Normalize profit between 0 and 1 using min and max profit possible
                reward = (profit - unit.reward[0]) / (unit.reward[1] - unit.reward[0])
                new_propensity = self.get_propensity(propensity, action_idx, reward)
                unit.propensity[order_idx] = new_propensity

            # Store results in unit outputs, which are later written to the database by the unit operator.
            unit.outputs["profit"].loc[start:end_excl] += profit
