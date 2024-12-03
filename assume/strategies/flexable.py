# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import numpy as np

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.common.utils import get_products_index, parse_duration


class flexableEOM(BaseStrategy):
    """
    A strategy that bids on the EOM-market.

    Attributes:
        foresight (datetime.timedelta): The foresight of the unit.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains eom_foresight argument
        self.foresight = parse_duration(kwargs.get("eom_foresight", "12h"))

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
        min_power_values, max_power_values = unit.calculate_min_max_power(start, end)

        op_time = unit.get_operation_time(start)

        bids = []
        for product, min_power, max_power in zip(
            product_tuples, min_power_values, max_power_values
        ):
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
            max_power = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )
            # adjust min_power for ramp speed
            min_power = unit.calculate_ramp(
                op_time, previous_power, min_power, current_power
            )

            bid_quantity_inflex = min_power

            marginal_cost_inflex = unit.calculate_marginal_cost(
                start, current_power + bid_quantity_inflex
            )
            marginal_cost_flex = unit.calculate_marginal_cost(
                start, current_power + max_power
            )

            # =============================================================================
            # Calculating possible price
            # =============================================================================
            if op_time > 0:
                bid_price_inflex = calculate_EOM_price_if_on(
                    unit=unit,
                    market_id=market_config.market_id,
                    start=start,
                    marginal_cost_flex=marginal_cost_flex,
                    bid_quantity_inflex=bid_quantity_inflex,
                    foresight=self.foresight,
                )
            else:
                bid_price_inflex = calculate_EOM_price_if_off(
                    unit=unit,
                    marginal_cost_inflex=marginal_cost_inflex,
                    bid_quantity_inflex=bid_quantity_inflex,
                    op_time=op_time,
                )

            if unit.outputs["heat"].at[start] > 0:
                power_loss_ratio = (
                    unit.outputs["power_loss"].at[start]
                    / unit.outputs["heat"].at[start]
                )
            else:
                power_loss_ratio = 0.0

            # Flex-bid price formulation
            if op_time <= -unit.min_down_time or op_time > 0:
                bid_quantity_flex = max_power - bid_quantity_inflex
                bid_price_flex = (1 - power_loss_ratio) * marginal_cost_flex

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": bid_price_inflex,
                    "volume": bid_quantity_inflex,
                    "node": unit.node,
                }
            )
            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": bid_price_flex,
                    "volume": bid_quantity_flex,
                    "node": unit.node,
                },
            )
            # calculate previous power with planned dispatch (bid_quantity)
            previous_power = bid_quantity_inflex + bid_quantity_flex + current_power
            op_time = max(op_time, 0) + 1 if previous_power > 0 else min(op_time, 0) - 1

        bids = self.remove_empty_bids(bids)

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

        Note:
            The reward is calculated as the profit minus the opportunity cost,
            which is the loss of income we have because we are not running at full power.
            The regret is the opportunity cost.
            Because the regret_scale is set to 0 the reward equals the profit.
            The profit is the income we have from the accepted bids.
            The total costs are the running costs and the start-up costs.

        """
        product_type = marketconfig.product_type
        products_index = get_products_index(orderbook)

        # Initialize intermediate results as numpy arrays for better performance
        profit = np.zeros(len(products_index))
        costs = np.zeros(len(products_index))

        # Map products_index to their positions for faster updates
        index_map = {time: i for i, time in enumerate(products_index)}

        for order in orderbook:
            start = order["start_time"]
            end_excl = order["end_time"] - unit.index.freq

            order_times = unit.index[start:end_excl]
            accepted_volume = order.get("accepted_volume", 0)
            accepted_price = order.get("accepted_price", 0)

            for start in order_times:
                idx = index_map.get(start)

                marginal_cost = unit.calculate_marginal_cost(
                    start, unit.outputs[product_type].at[start]
                )

                if isinstance(accepted_volume, dict):
                    accepted_volume = accepted_volume.get(start, 0)
                else:
                    accepted_volume = accepted_volume

                if isinstance(accepted_price, dict):
                    accepted_price = accepted_price.get(start, 0)
                else:
                    accepted_price = accepted_price

                profit[idx] += accepted_price * accepted_volume

        # consideration of start-up costs
        for i, start in enumerate(products_index):
            op_time = unit.get_operation_time(start)

            output = unit.outputs[product_type].at[start]
            marginal_cost = unit.calculate_marginal_cost(start, output)
            costs[i] += marginal_cost * output

            if output != 0 and op_time < 0:
                start_up_cost = unit.get_starting_costs(op_time)
                costs[i] += start_up_cost

        profit -= costs

        # store results in unit outputs which are written to database by unit operator
        unit.outputs["profit"].loc[products_index] = profit
        unit.outputs["total_costs"].loc[products_index] = costs

        # update average operation time
        update_avg_op_time(unit, product_type, products_index[0], products_index[-1])


class flexablePosCRM(BaseStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the CRM (reserve market).

    Attributes:
        foresight (datetime.timedelta): The foresight of the unit.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = parse_duration(kwargs.get("crm_foresight", "4h"))

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
        _, max_power_values = unit.calculate_min_max_power(
            start, end, market_config.product_type
        )  # get max_power for the product type

        bids = []
        for product, max_power in zip(product_tuples, max_power_values):
            start = product[0]
            op_time = unit.get_operation_time(start)

            # calculate pos reserve volume
            current_power = unit.outputs["energy"].at[start]
            # max_power + current_power < previous_power + unit.ramp_up
            bid_quantity = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )

            if bid_quantity == 0:
                continue

            marginal_cost = unit.calculate_marginal_cost(
                start,
                previous_power + bid_quantity,
            )
            # Specific revenue if power was offered on the energy market
            specific_revenue = get_specific_revenue(
                price_forecast=unit.forecaster[f"price_{market_config.market_id}"],
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

            # clip price by max and min bid price defined by the MarketConfig
            if price >= 0:
                price = min(price, market_config.maximum_bid_price)
            else:
                price = max(price, market_config.minimum_bid_price)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": price,
                    "volume": bid_quantity,
                    "node": unit.node,
                }
            )
            previous_power = bid_quantity + current_power

        bids = self.remove_empty_bids(bids)

        return bids


class flexableNegCRM(BaseStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the negative CRM(reserve market).

    Attributes:
        foresight (datetime.timedelta): The foresight of the unit.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check if kwargs contains crm_foresight argument
        self.foresight = parse_duration(kwargs.get("crm_foresight", "4h"))

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
        min_power_values, _ = unit.calculate_min_max_power(start, end)

        bids = []
        for product, min_power in zip(product_tuples, min_power_values):
            start = product[0]
            op_time = unit.get_operation_time(start)
            current_power = unit.outputs["energy"].at[start]

            # min_power + current_power > previous_power - unit.ramp_down
            min_power = unit.calculate_ramp(
                op_time, previous_power, min_power, current_power
            )
            bid_quantity = previous_power - min_power
            if bid_quantity <= 0:
                continue

            # bid_quantity < 0
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power + bid_quantity
            )

            # Specific revenue if power was offered on the energy market
            specific_revenue = get_specific_revenue(
                price_forecast=unit.forecaster[f"price_{market_config.market_id}"],
                marginal_cost=marginal_cost,
                t=start,
                foresight=self.foresight,
            )

            if specific_revenue < 0:
                capacity_price = (
                    abs(specific_revenue)
                    * (bid_quantity - unit.min_power)
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

            # clip price by max and min bid price defined by the MarketConfig
            if price >= 0:
                price = min(price, market_config.maximum_bid_price)
            else:
                price = max(price, market_config.minimum_bid_price)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": None,
                    "price": price,
                    "volume": bid_quantity,
                    "node": unit.node,
                }
            )
            previous_power = current_power + bid_quantity

        bids = self.remove_empty_bids(bids)

        return bids


def calculate_EOM_price_if_off(
    unit: SupportsMinMax,
    marginal_cost_inflex,
    bid_quantity_inflex,
    op_time,
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

    Returns:
        float: The inflexible bid price of the unit.

    """
    if bid_quantity_inflex == 0:
        return 0

    avg_operating_time = max(unit.avg_op_time, unit.min_operating_time)
    starting_cost = unit.get_starting_costs(op_time)
    # if we split starting_cost across av_operating_time
    # we are never adding the other parts of the cost to the following hours

    markup = starting_cost / avg_operating_time / bid_quantity_inflex

    bid_price_inflex = min(marginal_cost_inflex + markup, 3000.0)

    return bid_price_inflex


def calculate_EOM_price_if_on(
    unit: SupportsMinMax,
    market_id: str,
    start,
    marginal_cost_flex,
    bid_quantity_inflex,
    foresight,
):
    """
    The powerplant is currently on and calculates a price reduction to prevent shutdowns.

    The price reduction is calculated as follows:
    starting_cost / min_down_time / bid_quantity_inflex
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

    Returns:
        float: The inflexible bid price of the unit.
    """

    if bid_quantity_inflex == 0:
        return 0

    # check the starting cost if the unit were turned off for min_down_time
    starting_cost = unit.get_starting_costs(-unit.min_down_time)

    price_reduction_restart = starting_cost / unit.min_down_time / bid_quantity_inflex

    if unit.outputs["heat"].at[start] > 0:
        heat_gen_cost = (
            unit.outputs["heat"].at[start]
            * (unit.forecaster.get_price("natural gas").at[start] / 0.9)
        ) / bid_quantity_inflex
    else:
        heat_gen_cost = 0.0

    possible_revenue = get_specific_revenue(
        price_forecast=unit.forecaster[f"price_{market_id}"],
        marginal_cost=marginal_cost_flex,
        t=start,
        foresight=foresight,
    )
    if (
        possible_revenue >= 0
        and unit.forecaster[f"price_{market_id}"].at[start] < marginal_cost_flex
    ):
        marginal_cost_flex = 0

    bid_price_inflex = max(
        -price_reduction_restart - heat_gen_cost + marginal_cost_flex,
        -499.00,
    )

    return bid_price_inflex


def get_specific_revenue(
    price_forecast,
    marginal_cost: float,
    t: datetime,
    foresight: timedelta,
):
    """
    Calculates the specific revenue as difference between price forecast
    and marginal costs for the time defined by the foresight.

    Args:
        price_forecast (FastSeries): The price forecast.
        marginal_cost (float): The marginal cost of the unit.
        t (datetime.datetime): The start time of the product.
        foresight (datetime.timedelta): The foresight of the unit.

    Returns:
        float: The specific revenue of the unit.
    """

    if t + foresight > price_forecast.index[-1]:
        price_forecast = price_forecast[t:]
    else:
        price_forecast = price_forecast[t : t + foresight]

    possible_revenue = (price_forecast - marginal_cost).sum()

    return possible_revenue


def update_avg_op_time(unit, product_type, start, end):
    """
    Updates the average operation time for the unit based on the specified slice of outputs.

    Args:
        unit: The unit object containing `outputs`, `total_op_time`, and `avg_op_time`.
        product_type: The product type to update.
        start: The start index of the slice being updated.
        end: The end index of the slice being updated (inclusive).
    """
    # Get the current slice of outputs
    current_slice = unit.outputs[product_type].loc[start:end]

    # Increment total operation time for operating periods in the slice
    unit.total_op_time += (current_slice > 0).sum()

    # Update the average operation time
    total_periods = len(unit.index[:end]) + 1  # Total periods up to and including 'end'
    unit.avg_op_time = unit.total_op_time / total_periods
