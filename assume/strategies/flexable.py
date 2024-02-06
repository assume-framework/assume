# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_products_index


class flexableEOM(BaseStrategy):
    """
    A strategy that bids on the EOM-market.

    Parameters:
        foresight (datetime.timedelta): The foresight of the unit.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
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
        Calculates bids for the EOM-market and returns a list of bids consisting of the start time, end time, only hours, price, volume and bid type.

        The bids take the following form:
        For each hour one inflexible and one flexible bid is formulated.
        The inflexible bid is the minimum power of the unit depending on the ramp up limitations.
        The price for this bid is calculated by the marginal cost of the unit plus a markup for the startup costs.
        The flexible bid is the maximum power of the unit depending on the ramp up limitations minus the inflexible bid.
        Here, the price is equal to the marginal costs of the unit.

        Args:
            unit (SupportsMinMax): A unit that the unit operator manages.
            market_config (MarketConfig): A market configuration.
            product_tuples (list[Product]): A list of tuples containing the start and end time of each product.
            kwargs (dict): Additional arguments.

        Returns:
            Orderbook: A list of bids.
        """

        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end)

        bids = []
        op_time = unit.get_operation_time(start)
        avg_op_time, avg_down_time = unit.get_average_operation_times(start)

        for product in product_tuples:
            bid_quantity_inflex, bid_price_inflex = 0, 0
            bid_quantity_flex, bid_price_flex = 0, 0

            start = product[0]
            end = product[1]

            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount and cost
            # =============================================================================

            current_power = unit.outputs["energy"].at[start]

            # adjust max_power for ramp speed
            max_power[start] = unit.calculate_ramp(
                op_time, previous_power, max_power[start], current_power
            )
            # adjust min_power for ramp speed
            min_power[start] = unit.calculate_ramp(
                op_time, previous_power, min_power[start], current_power
            )

            bid_quantity_inflex = min_power[start]

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

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": bid_price_inflex,
                    "volume": bid_quantity_inflex,
                }
            )
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": bid_price_flex,
                    "volume": bid_quantity_flex,
                },
            )
            # calculate previous power with planned dispatch (bid_quantity)
            previous_power = bid_quantity_inflex + bid_quantity_flex + current_power
            op_time = max(op_time, 0) + 1 if previous_power > 0 else min(op_time, 0) - 1

        return bids

    def calculate_reward(
        self,
        unit,
        marketconfig: MarketConfig,
        orderbook: Orderbook,
    ):
        """
        Calculates and writes the reward (costs and profit).

        Args:
            unit (SupportsMinMax): A unit that the unit operator manages.
            marketconfig (MarketConfig): A market configuration.
            orderbook (Orderbook): An orderbook with accepted and rejected orders for the unit.
        """
        # TODO: Calculate profits over all markets

        calculate_reward_EOM(
            unit=unit,
            marketconfig=marketconfig,
            orderbook=orderbook,
        )


class flexablePosCRM(BaseStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the CRM (reserve market).

    Parameters:
        foresight (datetime.timedelta): The foresight of the unit.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Calculates bids for the CRM-market.

        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        Args:
            unit (SupportsMinMax): A unit that the unit operator manages.
            market_config (MarketConfig): A market configuration.
            product_tuples (list[Product]): A list of tuples containing the start and end time of each product.
            kwargs (dict): Additional arguments.

        Returns:
            Orderbook: A list of bids.
        """

        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(
            start, end, market_config.product_type
        )  # get max_power for the product type

        bids = []
        for product in product_tuples:
            start = product[0]
            op_time = unit.get_operation_time(start)

            # calculate pos reserve volume
            current_power = unit.outputs["energy"].at[start]
            # max_power + current_power < previous_power + unit.ramp_up
            bid_quantity = unit.calculate_ramp(
                op_time, previous_power, max_power[start], current_power
            )

            if bid_quantity == 0:
                continue

            marginal_cost = unit.calculate_marginal_cost(
                start,
                previous_power + bid_quantity,
            )
            # Specific revenue if power was offered on the energy market
            specific_revenue = get_specific_revenue(
                unit=unit,
                marginal_cost=marginal_cost,
                t=start,
                foresight=self.foresight,
            )

            if specific_revenue >= 0:
                capacity_price = specific_revenue
            else:
                capacity_price = abs(specific_revenue) * unit.min_power / bid_quantity

            energy_price = marginal_cost

            if market_config.product_type == "capacity_pos":
                price = capacity_price
            elif market_config.product_type == "energy_pos":
                price = energy_price
            else:
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": price,
                    "volume": bid_quantity,
                }
            )
            previous_power = bid_quantity + current_power

        return bids


class flexableNegCRM(BaseStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the negative CRM(reserve market).

    Parameters:
        foresight (datetime.timedelta): The foresight of the unit.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = pd.Timedelta(kwargs.get("crm_foresight", "4h"))

        start = None

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        Args:
            unit (SupportsMinMax): A unit that the unit operator manages.
            market_config (MarketConfig): A market configuration.
            product_tuples (list[Product]): A list of tuples containing the start and end time of each product.
            kwargs (dict): Additional arguments.

        Returns:
            Orderbook: A list of bids.
        """

        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end)

        bids = []
        for product in product_tuples:
            start = product[0]
            op_time = unit.get_operation_time(start)
            current_power = unit.outputs["energy"].at[start]

            # min_power + current_power > previous_power - unit.ramp_down
            min_power[start] = unit.calculate_ramp(
                op_time, previous_power, min_power[start], current_power
            )
            bid_quantity = min_power[start] - previous_power
            if bid_quantity >= 0:
                continue

            # bid_quantity < 0
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power + bid_quantity
            )

            # Specific revenue if power was offered on the energy market
            specific_revenue = get_specific_revenue(
                unit=unit,
                marginal_cost=marginal_cost,
                t=start,
                foresight=self.foresight,
            )

            if specific_revenue < 0:
                capacity_price = (
                    abs(specific_revenue)
                    * (unit.min_power + bid_quantity)
                    / bid_quantity
                )
            else:
                capacity_price = 0.0

            energy_price = marginal_cost * (-1)

            if market_config.product_type == "capacity_neg":
                price = capacity_price
            elif market_config.product_type == "energy_neg":
                price = energy_price
            else:
                raise ValueError(
                    f"Product {market_config.product_type} is not supported by this strategy."
                )
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": price,
                    "volume": bid_quantity,
                }
            )
            previous_power = current_power + bid_quantity

        return bids


def calculate_EOM_price_if_off(
    unit: SupportsMinMax,
    marginal_cost_inflex,
    bid_quantity_inflex,
    op_time,
    avg_op_time=1,
):
    """
    The powerplant is currently off and calculates a startup markup as an extra
    to the marginal cost.

    The startup markup is calculated as follows:
    starting_cost / avg_operating_time / bid_quantity_inflex

    Args:
        unit (SupportsMinMax): A unit that the unit operator manages.
        marginal_cost_inflex (float): The marginal cost of the unit.
        bid_quantity_inflex (float): The bid quantity of the unit.
        op_time (int): The operation time of the unit.
        avg_op_time (int): The average operation time of the unit.

    Returns:
        float: The inflexible bid price of the unit.

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
    The powerplant is currently on and calculates a price reduction to prevent shutdowns.

    The price reduction is calculated as follows:
    starting_cost / -avg_down_time / bid_quantity_inflex
    If the unit is a CHP, the heat generation costs are added to the price reduction with the following formula:
    heat_gen_cost = (heat_output * (natural_gas_price / 0.9)) / bid_quantity_inflex
    If the estimated revenue for the time defined in foresight is positive,
    but the marginal costs are below the forecasted market clearing price, the marginal costs are set to 0.


    Args:
        unit (SupportsMinMax): A unit that the unit operator manages.
        start (datetime.datetime): The start time of the product.
        marginal_cost_flex (float): The marginal cost of the unit.
        bid_quantity_inflex (float): The bid quantity of the unit.
        foresight (datetime.timedelta): The foresight of the unit.
        avg_down_time (int): The average down time of the unit.

    Returns:
        float: The inflexible bid price of the unit.
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
    Calculates the specific revenue as difference between price forecast
    and marginal costs for the time defined by the foresight.

    Args:
        unit (SupportsMinMax): A unit that the unit operator manages.
        marginal_cost (float): The marginal cost of the unit.
        t (datetime.datetime): The start time of the product.
        foresight (datetime.timedelta): The foresight of the unit.

    Returns:
        float: The specific revenue of the unit.
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
    Calculates and writes reward (costs and profit) for EOM market.

    Args:
        unit (SupportsMinMax): A unit that the unit operator manages.
        marketconfig (MarketConfig): A market configuration.
        orderbook (Orderbook): An orderbook with accepted and rejected orders for the unit.
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
