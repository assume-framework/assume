# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Order, Orderbook, Product


class NaiveSingleBidStrategy(BaseStrategy):
    """
    A naive strategy that bids the marginal cost of the unit on the market.

    Methods
    -------
    """

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

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        start = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product
        previous_power = unit.get_output_before(
            start
        )  # power output of the unit before the start time of the first product
        op_time = unit.get_operation_time(start)
        min_power_values, max_power_values = unit.calculate_min_max_power(
            start, end_all
        )  # minimum and maximum power output of the unit between the start time of the first product and the end time of the last product

        bids = []
        for product, min_power, max_power in zip(
            product_tuples, min_power_values, max_power_values
        ):
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]
            current_power = unit.outputs["energy"].at[
                start
            ]  # power output of the unit at the start time of the current product
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power
            )  # calculation of the marginal costs
            volume = unit.calculate_ramp(
                op_time, previous_power, max_power, current_power
            )
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_cost,
                    "volume": volume,
                    "node": unit.node,
                }
            )

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


class NaiveProfileStrategy(BaseStrategy):
    """
    A naive strategy that bids the marginal cost of the unit as block bids over 24 hours on the day ahead market.
    """

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

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        start = product_tuples[0][0]
        end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        op_time = unit.get_operation_time(start)
        _, max_power = unit.calculate_min_max_power(start, end_all)

        current_power = unit.outputs["energy"].at[start]
        marginal_cost = unit.calculate_marginal_cost(start, previous_power)

        # calculate the ramp up volume using the initial maximum power
        volume = unit.calculate_ramp(
            op_time, previous_power, max_power[0], current_power
        )

        profile = {product[0]: volume for product in product_tuples}
        order: Order = {
            "start_time": start,
            "end_time": product_tuples[-1][1],
            "only_hours": product_tuples[0][2],
            "price": marginal_cost,
            "volume": profile,
            "bid_type": "BB",
            "node": unit.node,
        }

        bids = [order]

        bids = self.remove_empty_bids(bids)
        return bids


class NaiveDADSMStrategy(BaseStrategy):
    """
    A naive strategy of a Demand Side Management (DSM) unit. The bid volume is the optimal power requirement of
    the unit at the start time of the product. The bid price is the marginal cost of the unit at the start time of the product.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        # calculate the optimal operation of the unit
        unit.determine_optimal_operation_with_flex()

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]

            volume = unit.opt_power_requirement.at[start]
            marginal_price = 3000
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_price,
                    "volume": -volume,
                }
            )

        return bids


class NaiveRedispatchDSMStrategy(BaseStrategy):
    """
    A naive strategy of a Demand Side Management (DSM) unit that bids the available flexibility of the unit on the redispatch market.
    The bid volume is the flexible power requirement of the unit at the start time of the product. The bid price is the marginal cost of the unit at the start time of the product.
    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        # calculate the optimal operation of the unit according to the objective function
        unit.determine_optimal_operation_with_flex()

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]
            volume = unit.flex_power_requirement.at[start]
            marginal_price = unit.calculate_marginal_cost(start, volume)
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_price,
                    "volume": -volume,
                }
            )

        return bids


class NaiveRedispatchStrategy(BaseStrategy):
    """
    A naive strategy that simply submits all information about the unit and
    currently dispatched power for the following hours to the redispatch market.
    Information includes the marginal cost, the ramp up and down values, and the dispatch.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        :param unit: the unit to be dispatched
        :type unit: SupportsMinMax
        :param market_config: the market configuration
        :type market_config: MarketConfig
        :param product_tuples: list of all products the unit can offer
        :type product_tuples: list[Product]
        :return: the bids consisting of the start time, end time, only hours, price and volume.
        :rtype: Orderbook
        """
        start = product_tuples[0][0]
        # end_all = product_tuples[-1][1]
        previous_power = unit.get_output_before(start)
        min_power, max_power = unit.min_power, unit.max_power

        bids = []
        for product in product_tuples:
            start = product[0]
            current_power = unit.outputs["energy"].at[start]
            marginal_cost = unit.calculate_marginal_cost(
                start, previous_power
            )  # calculation of the marginal costs

            bids.append(
                {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": marginal_cost,
                    "volume": current_power,
                    "max_power": max_power,
                    "min_power": min_power,
                    "node": unit.node,
                }
            )

        return bids


class NaiveExchangeStrategy(BaseStrategy):
    """
    A naive strategy for an exchange unit that bids the defined import and export prices on the market.
    It submits two bids, one for import and one for export, with the respective prices and volumes.
    Export bids have negative volumes and are treated as demand on the market.
    Import bids have positive volumes and are treated as supply on the market.

    Methods
    -------
    """

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

        Args:
            unit (SupportsMinMax): The unit to be dispatched.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """

        bids = []
        for product in product_tuples:
            # for each product, calculate the marginal cost of the unit at the start time of the product
            # and the volume of the product. Dispatch the order to the market.
            start = product[0]

            # append import bid
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": unit.price_import,
                    "volume": unit.volume_import.at[start],
                    "node": unit.node,
                }
            )

            # append export bid
            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": unit.price_export,
                    "volume": unit.volume_export.at[start],
                    "node": unit.node,
                }
            )

        # clean up empty bids
        bids = self.remove_empty_bids(bids)

        return bids
