# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product


class OTCStrategy(BaseStrategy):
    """
    Strategy for OTC (over the counter trading) markets
    """

    def __init__(self, *args, scale_firm_power_capacity=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale = scale_firm_power_capacity

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

        Returns a list of bids that the unit operator will submit to the market
        :param unit: unit to dispatch
        :type unit: SupportsMinMax
        :param market_config: market configuration
        :type market_config: MarketConfig
        :param product_tuples: list of products to dispatch
        :type product_tuples: list[Product]
        :param kwargs: additional arguments
        :type kwargs: dict
        :return: orderbook
        :rtype: Orderbook
        """
        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]

            min_power, max_power = unit.calculate_min_max_power(
                start, end
            )  # max_power describes the maximum power output of the unit
            current_power = unit.outputs["energy"].at[
                start
            ]  # current power output describes the power output at the start of the product
            volume = max_power[start]
            if "OTC" in market_config.name:
                volume *= self.scale
            price = unit.calculate_marginal_cost(start, current_power + volume)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": product[2],
                    "price": price,
                    "volume": volume,
                }
            )
        return bids


class MarkupStrategy(BaseStrategy):
    """
    Strategy for Markup (over the counter trading) markets
    """

    def __init__(self, *args, abs_markup=0, rel_markup=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.abs_markup = abs_markup
        self.rel_markup = rel_markup

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

        Returns a list of bids that the unit operator will submit to the market.

        Args:
            unit (SupportsMinMax): Unit to dispatch.
            market_config (MarketConfig): Market configuration.
            product_tuples (list[Product]): List of products to dispatch.
            **kwargs (dict): Additional arguments.

        Returns:
            Orderbook: The orderbook.
        """
        bids = []
        for product in product_tuples:
            start = product[0]
            end = product[1]

            min_power, max_power = unit.calculate_min_max_power(
                start, end
            )  # max_power describes the maximum power output of the unit
            current_power = unit.outputs["energy"].at[
                start
            ]  # current power output describes the power output at the start of the product
            volume = max_power[start]
            price = unit.calculate_marginal_cost(start, current_power + volume)

            bids.append(
                {
                    "start_time": start,
                    "end_time": end,
                    "only_hours": product[2],
                    "price": price * self.rel_markup + self.abs_markup,
                    "volume": volume,
                }
            )
        return bids
