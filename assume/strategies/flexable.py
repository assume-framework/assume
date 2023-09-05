from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product


class flexableEOM(BaseStrategy):
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
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        :param unit: A unit that the unit operator manages
        :type unit: SupportsMinMax
        :param market_config: A market configuration
        :type market_config: MarketConfig
        :param product_tuples: A list of tuples containing the start and end time of each product
        :type product_tuples: list[Product]
        :param kwargs: Additional arguments
        :type kwargs: dict
        :return: A list of bids
        :rtype: Orderbook
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end)

        bids = []
        for product in product_tuples:
            bid_quantity_inflex, bid_price_inflex = 0, 0
            bid_quantity_flex, bid_price_flex = 0, 0

            start = product[0]
            end = product[1]

            current_power = unit.outputs["energy"].at[start]

            # adjust for ramp down speed
            max_power[start] = unit.calculate_ramp(
                previous_power, max_power[start], current_power
            )
            # adjust for ramp up speed
            min_power[start] = unit.calculate_ramp(
                previous_power, min_power[start], current_power
            )

            bid_quantity_inflex = min_power[start]

            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount and cost
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
            if unit.get_operation_time(start) > 0:
                bid_price_inflex = calculate_EOM_price_if_on(
                    unit,
                    start,
                    marginal_cost_inflex,
                    bid_quantity_inflex,
                    self.foresight,
                )
            else:
                bid_price_inflex = calculate_EOM_price_if_off(
                    unit, start, marginal_cost_flex, bid_quantity_inflex
                )

            if unit.outputs["heat"][start] > 0:
                power_loss_ratio = (
                    unit.outputs["power_loss"][start] / unit.outputs["heat"][start]
                )
            else:
                power_loss_ratio = 0.0

            # Flex-bid price formulation
            if (
                unit.get_operation_time(start) <= -unit.min_down_time
                or unit.get_operation_time(start) > 0
            ):
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

        return bids


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
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        :param unit: A unit that the unit operator manages
        :type unit: SupportsMinMax
        :param market_config: A market configuration
        :type market_config: MarketConfig
        :param product_tuples: A list of tuples containing the start and end time of each product
        :type product_tuples: list[Product]
        :param kwargs: Additional arguments
        :type kwargs: dict
        :return: A list of bids
        :rtype: Orderbook
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]

        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end)

        bids = []
        bid_quantity_block = {}
        bid_price_block = []

        for product in product_tuples:
            bid_quantity_flex, bid_price_flex = 0, 0
            bid_quantity_inflex, bid_price_inflex = 0, 0

            start = product[0]
            end = product[1]

            current_power = unit.outputs["energy"].at[start]

            # adjust for ramp down speed
            max_power[start] = unit.calculate_ramp(
                previous_power, max_power[start], current_power
            )
            # adjust for ramp up speed
            min_power[start] = unit.calculate_ramp(
                previous_power, min_power[start], current_power
            )

            bid_quantity_inflex = min_power[start]

            # =============================================================================
            # Powerplant is either on, or is able to turn on
            # Calculating possible bid amount and cost
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
            if unit.get_operation_time(start) > 0:
                bid_price_inflex = calculate_EOM_price_if_on(
                    unit,
                    start,
                    marginal_cost_inflex,
                    bid_quantity_inflex,
                    self.foresight,
                )
            else:
                bid_price_inflex = calculate_EOM_price_if_off(
                    unit, start, marginal_cost_flex, bid_quantity_inflex
                )

            if unit.outputs["heat"][start] > 0:
                power_loss_ratio = (
                    unit.outputs["power_loss"][start] / unit.outputs["heat"][start]
                )
            else:
                power_loss_ratio = 0.0

            # Flex-bid price formulation
            if (
                unit.get_operation_time(start) <= -unit.min_down_time
                or unit.get_operation_time(start) > 0
            ):
                bid_quantity_flex = max_power[start] - bid_quantity_inflex
                bid_price_flex = (
                    1 - power_loss_ratio
                ) * marginal_cost_flex + bid_price_inflex  # hihger price than inflex

            bid_quantity_block[product[0]] = bid_quantity_inflex
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

        bids.append(
            {
                "start_time": product_tuples[0][0],
                "end_time": product_tuples[-1][1],
                "only_hours": product_tuples[0][2],
                "price": np.mean(bid_price_block),
                "volume": bid_quantity_block,
                "bid_type": "BB",
                "min_acceptance_ratio": 1,
                "accepted_volume": {product[0]: 0 for product in product_tuples},
            }
        )

        return bids


class flexablePosCRM(BaseStrategy):
    """
    A strategy that bids the energy_price or the capacity_price of the unit on the CRM (reserve market).
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
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        :param unit: A unit that the unit operator manages
        :type unit: SupportsMinMax
        :param market_config: A market configuration
        :type market_config: MarketConfig
        :param product_tuples: A list of tuples containing the start and end time of each product
        :type product_tuples: list[Product]
        :param kwargs: Additional arguments
        :type kwargs: dict
        :return: A list of bids
        :rtype: Orderbook
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

            # calculate pos reserve volume
            current_power = unit.outputs["energy"].at[start]
            # max_power + current_power < previous_power + unit.ramp_up
            bid_quantity = unit.calculate_ramp(
                previous_power, max_power[start], current_power
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
        defines how it is dispatched to the market
        Returns a list of bids consisting of the start time, end time, only hours, price and volume.

        :param unit: A unit that the unit operator manages
        :type unit: SupportsMinMax
        :param market_config: A market configuration
        :type market_config: MarketConfig
        :param product_tuples: A list of tuples containing the start and end time of each product
        :type product_tuples: list[Product]
        :param kwargs: Additional arguments
        :type kwargs: dict
        :return: A list of bids
        :rtype: Orderbook
        """
        start = product_tuples[0][0]
        end = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.calculate_min_max_power(start, end)

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]

            # min_power + current_power > previous_power - unit.ramp_down
            min_power[start] = unit.calculate_ramp(
                previous_power, min_power[start], current_power
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
    start,
    marginal_cost_inflex,
    bid_quantity_inflex,
):
    """
    The powerplant is currently off and calculates a startup markup as an extra
    to the marginal cost
    Calculating the average uninterrupted operating period

    :param unit: A unit that the unit operator manages
    :type unit: SupportsMinMax
    :param marginal_cost_inflex: The marginal cost of the unit
    :type marginal_cost_inflex: float
    :param bid_quantity_inflex: The bid quantity of the unit
    :type bid_quantity_inflex: float
    :return: The bid price of the unit
    :rtype: float
    """
    av_operating_time = max((unit.outputs["energy"][:start] > 0).mean(), 1)
    # 1 prevents division by 0
    op_time = unit.get_operation_time(start)
    starting_cost = unit.get_starting_costs(op_time)
    # if we split starting_cost across av_operating_time
    # we are never adding the other parts of the cost to the following hours

    if bid_quantity_inflex == 0:
        markup = starting_cost / av_operating_time
    else:
        markup = starting_cost / av_operating_time / bid_quantity_inflex

    bid_price_inflex = min(marginal_cost_inflex + markup, 3000.0)

    return bid_price_inflex


def calculate_EOM_price_if_on(
    unit: SupportsMinMax,
    start,
    marginal_cost_inflex,
    bid_quantity_inflex,
    foresight,
):
    """
    Check the description provided by Thomas in last version, the average downtime is not available
    The powerplant is currently on

    :param unit: A unit that the unit operator manages
    :type unit: SupportsMinMax
    :param start: The start time of the product
    :type start: datetime
    :param marginal_cost_inflex: The marginal cost of the unit
    :type marginal_cost_inflex: float
    :param bid_quantity_inflex: The bid quantity of the unit
    :type bid_quantity_inflex: float
    :return: The bid price of the unit
    :rtype: float
    """
    if bid_quantity_inflex == 0:
        return 0

    t = start
    op_time = unit.get_operation_time(start)
    # TODO is it correct to bill for cold, hot and warm starts in one start?
    starting_cost = unit.get_starting_costs(op_time)

    price_reduction_restart = starting_cost / unit.min_down_time / bid_quantity_inflex

    if unit.outputs["heat"][t] > 0:
        heat_gen_cost = (
            unit.outputs["heat"][t]
            * (unit.forecaster.get_price("natural gas")[t] / 0.9)
        ) / bid_quantity_inflex
    else:
        heat_gen_cost = 0.0

    possible_revenue = get_specific_revenue(
        unit=unit,
        marginal_cost=marginal_cost_inflex,
        t=start,
        foresight=foresight,
    )
    if (
        possible_revenue >= 0
        and unit.forecaster["price_forecast"][t] < marginal_cost_inflex
    ):
        marginal_cost_inflex = 0

    bid_price_inflex = max(
        -price_reduction_restart - heat_gen_cost + marginal_cost_inflex,
        -499.00,
    )

    return bid_price_inflex


def get_starting_costs(time, unit):
    """
    Calculates the starting costs of a unit

    :return: The starting costs of the unit
    :rtype: float
    """
    if time < unit.downtime_hot_start:
        return unit.hot_start_cost

    elif time < unit.downtime_warm_start:
        return unit.warm_start_cost

    else:
        return unit.cold_start_cost


def get_specific_revenue(
    unit: SupportsMinMax,
    marginal_cost: float,
    t: datetime,
    foresight: timedelta,
):
    """
    get the specific revenue of a unit

    :param unit: A unit that the unit operator manages
    :type unit: SupportsMinMax
    :param marginal_cost: The marginal cost of the unit
    :type marginal_cost: float
    :param t: The start time of the product
    :type t: datetime
    :param foresight: The foresight of the unit
    :type foresight: timedelta
    :return: The specific revenue of the unit
    :rtype: float
    """
    price_forecast = []

    if t + foresight > unit.forecaster["price_forecast"].index[-1]:
        price_forecast = unit.forecaster["price_forecast"][t:]
    else:
        price_forecast = unit.forecaster["price_forecast"][t : t + foresight]

    possible_revenue = (price_forecast - marginal_cost).sum()

    return possible_revenue
