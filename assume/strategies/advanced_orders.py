# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_products_index


class flexableEOMBlock(BaseStrategy):
    """
    A strategy that bids on the EOM-market.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains eom_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market
        Returns a list of bids consisting of the start time, end time, only hours, price, volume and bid type.

        Args:
        - unit (SupportsMinMax): A unit that the unit operator manages
        - market_config (MarketConfig): A market configuration
        - product_tuples (list[Product]): A list of tuples containing the start and end time of each product
        - kwargs (dict): Additional arguments

        Returns:
        - Orderbook: A list of bids
        """

        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end)

        bids = []
        bid_quantity_block = {}
        bid_price_block = []
        op_time = unit.get_operation_time(start)
        avg_op_time, avg_down_time = unit.get_average_operation_times(start)

        for product in product_tuples:
            start = product[0]
            end = product[1]

            bid_quantity_flex, bid_price_flex = 0, 0
            bid_price_inflex, bid_quantity_inflex = 0, 0

            current_power = unit.outputs["energy"].at[start]

            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount and cost
            # =============================================================================

            # adjust for ramp speed
            max_power[start] = unit.calculate_ramp(
                op_time, previous_power, max_power[start], current_power
            )
            # adjust for ramp speed
            min_power[start] = unit.calculate_ramp(
                op_time, previous_power, min_power[start], current_power
            )

            bid_quantity_inflex = min_power[start]

            # =============================================================================
            # Calculating possible price
            # =============================================================================

            marginal_cost_inflex = unit.calculate_marginal_cost(
                start, current_power + bid_quantity_inflex
            )
            marginal_cost_flex = unit.calculate_marginal_cost(
                start, current_power + max_power[start]
            )

            # =============================================================================
            # Calculating possible price
            # =============================================================================
            if op_time > 0:
                bid_price_inflex = calculate_EOM_price_if_on(
                    unit,
                    start,
                    marginal_cost_flex,
                    bid_quantity_inflex,
                    self.foresight,
                    avg_down_time,
                )
            else:
                bid_price_inflex = calculate_EOM_price_if_off(
                    unit,
                    start,
                    marginal_cost_inflex,
                    bid_quantity_inflex,
                    op_time,
                    avg_op_time,
                )

            if unit.outputs["heat"][start] > 0:
                power_loss_ratio = (
                    unit.outputs["power_loss"][start] / unit.outputs["heat"][start]
                )
            else:
                power_loss_ratio = 0.0

            # Flex-bid price formulation
            if op_time <= -unit.min_down_time or op_time > 0:
                bid_quantity_flex = max_power[start] - bid_quantity_inflex
                bid_price_flex = (1 - power_loss_ratio) * marginal_cost_flex

            bid_quantity_block[product[0]] = bid_quantity_inflex
            if bid_quantity_inflex > 0:
                bid_price_block.append(bid_price_inflex)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": bid_price_flex,
                    "volume": bid_quantity_flex,
                    "bid_type": "SB",
                },
            )
            # calculate previous power with planned dispatch (bid_quantity)
            previous_power = bid_quantity_inflex + bid_quantity_flex + current_power
            op_time = max(op_time, 0) + 1 if previous_power > 0 else min(op_time, 0) - 1

        # calculate weighted average of prices
        volume = 0
        price = 0
        for i in range(len(bid_price_block)):
            price += bid_price_block[i] * list(bid_quantity_block.values())[i]
            volume += list(bid_quantity_block.values())[i]
        mean_price = price / volume

        bids.append(
            {
                "start_time": product_tuples[0][0],
                "end_time": product_tuples[-1][1],
                "only_hours": product_tuples[0][2],
                "price": mean_price,
                "volume": bid_quantity_block,
                "bid_type": "BB",
                "min_acceptance_ratio": 1,
                "accepted_volume": {product[0]: 0 for product in product_tuples},
            }
        )

        return bids

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        # TODO: Calculate profits over all markets

        calculate_reward_EOM(
            unit=unit,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )


class flexableEOMLinked(BaseStrategy):
    """
    A strategy that bids on the EOM-market.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains eom_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("eom_foresight", "12h"))

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market
        Returns a list of bids consisting of the start time, end time, only hours, price, volume and bid type.

        Args:
        - unit (SupportsMinMax): A unit that the unit operator manages
        - market_config (MarketConfig): A market configuration
        - product_tuples (list[Product]): A list of tuples containing the start and end time of each product
        - kwargs (dict): Additional arguments

        Returns:
        - Orderbook: A list of bids
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end)

        bids = []
        bid_quantity_block = {}
        bid_price_block = []
        op_time = unit.get_operation_time(start)
        avg_op_time, avg_down_time = unit.get_average_operation_times(start)

        block_id = unit.id + "_block"

        for product in product_tuples:
            start = product[0]
            end = product[1]

            bid_quantity_flex, bid_price_flex = 0, 0
            bid_price_inflex, bid_quantity_inflex = 0, 0

            current_power = unit.outputs["energy"].at[start]

            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount and cost
            # =============================================================================

            # adjust for ramp speed
            max_power[start] = unit.calculate_ramp(
                op_time, previous_power, max_power[start], current_power
            )
            # adjust for ramp speed
            min_power[start] = unit.calculate_ramp(
                op_time, previous_power, min_power[start], current_power
            )

            bid_quantity_inflex = min_power[start]

            # =============================================================================
            # Calculating possible price
            # =============================================================================

            marginal_cost_inflex = unit.calculate_marginal_cost(
                start, current_power + bid_quantity_inflex
            )
            marginal_cost_flex = unit.calculate_marginal_cost(
                start, current_power + max_power[start]
            )

            # =============================================================================
            # Calculating possible price
            # =============================================================================
            if op_time > 0:
                bid_price_inflex = calculate_EOM_price_if_on(
                    unit,
                    start,
                    marginal_cost_flex,
                    bid_quantity_inflex,
                    self.foresight,
                    avg_down_time,
                )
            else:
                bid_price_inflex = calculate_EOM_price_if_off(
                    unit,
                    start,
                    marginal_cost_inflex,
                    bid_quantity_inflex,
                    op_time,
                    avg_op_time,
                )

            if unit.outputs["heat"][start] > 0:
                power_loss_ratio = (
                    unit.outputs["power_loss"][start] / unit.outputs["heat"][start]
                )
            else:
                power_loss_ratio = 0.0

            # Flex-bid price formulation
            if op_time <= -unit.min_down_time or op_time > 0:
                bid_quantity_flex = max_power[start] - bid_quantity_inflex
                bid_price_flex = (1 - power_loss_ratio) * marginal_cost_flex

            bid_quantity_block[product[0]] = bid_quantity_inflex
            if bid_quantity_inflex > 0:
                bid_price_block.append(bid_price_inflex)
                parent_id = block_id
            else:
                parent_id = None

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": bid_price_flex,
                    "volume": {start: bid_quantity_flex},
                    "bid_type": "LB",
                    "parent_bid_id": parent_id,
                },
            )
            # calculate previous power with planned dispatch (bid_quantity)
            previous_power = bid_quantity_inflex + bid_quantity_flex + current_power
            op_time = max(op_time, 0) + 1 if previous_power > 0 else min(op_time, 0) - 1

        # calculate weighted average of prices
        volume = 0
        price = 0
        for i in range(len(bid_price_block)):
            price += bid_price_block[i] * list(bid_quantity_block.values())[i]
            volume += list(bid_quantity_block.values())[i]
        mean_price = price / volume

        bids.append(
            {
                "start_time": product_tuples[0][0],
                "end_time": product_tuples[-1][1],
                "only_hours": product_tuples[0][2],
                "price": mean_price,
                "volume": bid_quantity_block,
                "bid_type": "BB",
                "min_acceptance_ratio": 1,
                "accepted_volume": {product[0]: 0 for product in product_tuples},
                "bid_id": block_id,
            }
        )

        return bids

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        # TODO: Calculate profits over all markets

        calculate_reward_EOM(
            unit=unit,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )


def calculate_EOM_price_if_off(
    unit: SupportsMinMax,
    marginal_cost_inflex,
    bid_quantity_inflex,
    op_time,
    avg_op_time=1,
):
    """
    The powerplant is currently off and calculates a startup markup as an extra
    to the marginal cost
    Calculating the average uninterrupted operating period

    Args:
    - unit (SupportsMinMax): A unit that the unit operator manages
    - marginal_cost_inflex (float): The marginal cost of the unit
    - bid_quantity_inflex (float): The bid quantity of the unit
    - op_time (int): The operation time of the unit
    - avg_op_time (int): The average operation time of the unit

    Returns:
    - float: The inflexible bid price of the unit

    """
    starting_cost = unit.get_starting_costs(op_time)
    # if we split starting_cost across av_operating_time
    # we are never adding the other parts of the cost to the following hours

    if bid_quantity_inflex == 0:
        markup = starting_cost / avg_op_time
    else:
        markup = starting_cost / avg_op_time / bid_quantity_inflex

    bid_price_inflex = min(marginal_cost_inflex + markup, 3000.0)

    return bid_price_inflex


def calculate_EOM_price_if_on(
    unit: SupportsMinMax,
    start,
    marginal_cost_flex,
    bid_quantity_inflex,
    foresight,
    avg_down_time=-1,
):
    """
    Check the description provided by Thomas in last version, the average downtime is available here
    The powerplant is currently on

    Args:
    - unit (SupportsMinMax): A unit that the unit operator manages
    - start (datetime): The start time of the product
    - marginal_cost_flex (float): The marginal cost of the unit
    - bid_quantity_inflex (float): The bid quantity of the unit
    - foresight (timedelta): The foresight of the unit
    - avg_down_time (int): The average downtime of the unit

    Returns:
    - float: The inflexible bid price of the unit
    """

    if bid_quantity_inflex == 0:
        return 0

    t = start

    # TODO is it correct to bill for cold, hot and warm starts in one start?
    starting_cost = unit.get_starting_costs(avg_down_time)

    price_reduction_restart = starting_cost / -avg_down_time / bid_quantity_inflex

    if unit.outputs["heat"][t] > 0:
        heat_gen_cost = (
            unit.outputs["heat"][t]
            * (unit.forecaster.get_price("natural gas")[t] / 0.9)
        ) / bid_quantity_inflex
    else:
        heat_gen_cost = 0.0

    possible_revenue = get_specific_revenue(
        unit=unit,
        marginal_cost=marginal_cost_flex,
        t=start,
        foresight=foresight,
    )
    if possible_revenue >= 0 and unit.forecaster["price_EOM"][t] < marginal_cost_flex:
        marginal_cost_flex = 0

    bid_price_inflex = max(
        -price_reduction_restart - heat_gen_cost + marginal_cost_flex,
        -499.00,
    )

    return bid_price_inflex


def get_specific_revenue(
    unit: SupportsMinMax,
    marginal_cost: float,
    t: datetime,
    foresight: timedelta,
):
    """
    get the specific revenue of a unit depending on the foresight

    Args:
    - unit (SupportsMinMax): A unit that the unit operator manages
    - marginal_cost (float): The marginal cost of the unit
    - t (datetime): The start time of the product
    - foresight (timedelta): The foresight of the unit

    Returns:
    - float: The specific revenue of the unit
    """

    price_forecast = []

    if t + foresight > unit.forecaster["price_EOM"].index[-1]:
        price_forecast = unit.forecaster["price_EOM"][t:]
    else:
        price_forecast = unit.forecaster["price_EOM"][t : t + foresight]

    possible_revenue = (price_forecast - marginal_cost).sum()

    return possible_revenue


def calculate_reward_EOM(
    unit,
    marketconfig: MarketConfig,
    orderbook: Orderbook,
):
    """
    Calculate and write reward (costs and profit)

    Args:
    - unit (SupportsMinMax): A unit that the unit operator manages
    - marketconfig (MarketConfig): A market configuration
    - orderbook (Orderbook): An orderbook with accepted and rejected orders for the unit
    """
    # TODO: Calculate profits over all markets
    product_type = marketconfig.product_type
    products_index = get_products_index(orderbook)

    max_power = (
        unit.forecaster.get_availability(unit.id)[products_index] * unit.max_power
    )

    profit = pd.Series(0.0, index=products_index)
    reward = pd.Series(0.0, index=products_index)
    opportunity_cost = pd.Series(0.0, index=products_index)
    costs = pd.Series(0.0, index=products_index)

    for order in orderbook:
        start = order["start_time"]
        end = order["end_time"]
        end_excl = end - unit.index.freq

        order_times = pd.date_range(start, end_excl, freq=unit.index.freq)

        for start in order_times:
            marginal_cost = unit.calculate_marginal_cost(
                start, unit.outputs[product_type].loc[start]
            )

            if isinstance(order["accepted_volume"], dict):
                accepted_volume = order["accepted_volume"][start]
            else:
                accepted_volume = order["accepted_volume"]

            if isinstance(order["accepted_price"], dict):
                accepted_price = order["accepted_price"][start]
            else:
                accepted_price = order["accepted_price"]

            price_difference = accepted_price - marginal_cost

            # calculate opportunity cost
            # as the loss of income we have because we are not running at full power
            order_opportunity_cost = price_difference * (
                max_power[start] - unit.outputs[product_type].loc[start]
            )
            # if our opportunity costs are negative, we did not miss an opportunity to earn money and we set them to 0
            # don't consider opportunity_cost more than once! Always the same for one timestep and one market
            opportunity_cost[start] = max(order_opportunity_cost, 0)
            profit[start] += accepted_price * accepted_volume

    # consideration of start-up costs
    for start in products_index:
        op_time = unit.get_operation_time(start)

        marginal_cost = unit.calculate_marginal_cost(
            start, unit.outputs[product_type].loc[start]
        )
        costs[start] += marginal_cost * unit.outputs[product_type].loc[start]

        if unit.outputs[product_type].loc[start] != 0 and op_time < 0:
            start_up_cost = unit.get_starting_costs(op_time)
            costs[start] += start_up_cost

    profit += -costs
    scaling = 0.1 / unit.max_power
    regret_scale = 0.0
    reward = (profit - regret_scale * opportunity_cost) * scaling

    # store results in unit outputs which are written to database by unit operator
    unit.outputs["profit"].loc[products_index] = profit
    unit.outputs["reward"].loc[products_index] = reward
    unit.outputs["regret"].loc[products_index] = opportunity_cost
    unit.outputs["total_costs"].loc[products_index] = costs
