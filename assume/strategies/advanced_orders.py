# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

from assume.common.base import BaseStrategy, SupportsMinMax
from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.flexable import (
    calculate_EOM_price_if_off,
    calculate_EOM_price_if_on,
    calculate_reward_EOM,
)


class flexableEOMBlock(BaseStrategy):
    """
    A strategy that bids on the EOM-market with block bids.

    Parameters:
        foresight (pd.Timedelta): The foresight for the EOM-market.

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
        Calculates block bids for the EOM-market and returns a list of bids consisting of the start time, end time, only hours, price, volume and bid type.

        The bids take the following form:
        One block bid with the minimum acceptance ratio set to 1 spanning the total clearing period.
        It uses the inflexible power and the weighted average price of the inflexible power as the price.
        This price is based on the marginal cost of the inflexible power and the starting costs.
        The starting costs are split across inflexible power and the average operation or down time of the unit depending on the operation status before.
        Additionally, for every hour where the unit is on, a separate flexible bid is created using the flexible power and marginal costs as bidding price.

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

            # adjust max_power for ramp speed
            max_power[start] = unit.calculate_ramp(
                op_time, previous_power, max_power[start], current_power
            )
            # adjust min_power for ramp speed
            min_power[start] = unit.calculate_ramp(
                op_time, previous_power, min_power[start], current_power
            )

            bid_quantity_inflex = min_power[start]

            # =============================================================================
            # Calculating marginal cost
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

            # add volume and price to block bid
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


class flexableEOMLinked(BaseStrategy):
    """
    A strategy that bids on the EOM-market with block and linked bids.
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
        Calculates block and linked bids for the EOM-market and returns a list of bids consisting of the start time, end time, only hours, price, volume and bid type.

        The bids take the following form:
        One block bid with the minimum acceptance ratio set to 1 spanning the total clearing period.
        It uses the inflexible power and the weighted average price of the inflexible power as the price.
        This price is based on the marginal cost of the inflexible power and the starting costs.
        The starting costs are split across inflexible power and the average operation or down time of the unit depending on the operation status before.
        Additionally, for every hour where the unit is on, a separate flexible bid is created using the flexible power and marginal costs as bidding price.
        This bids are linked as children to the block bid.

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

            # adjust max_power for ramp speed
            max_power[start] = unit.calculate_ramp(
                op_time, previous_power, max_power[start], current_power
            )
            # adjust min_power for ramp speed
            min_power[start] = unit.calculate_ramp(
                op_time, previous_power, min_power[start], current_power
            )

            bid_quantity_inflex = min_power[start]

            # =============================================================================
            # Calculating marginal cost
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
