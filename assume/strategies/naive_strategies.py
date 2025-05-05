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

        # check if unit has opt_power_requirement attribute
        if not hasattr(unit, "opt_power_requirement"):
            unit.determine_optimal_operation_without_flex()

        bids = []
        for product in product_tuples:
            """
            for each product, calculate the marginal cost of the unit at the start time of the product
            and the volume of the product. Dispatch the order to the market.
            """
            start = product[0]

            volume = unit.opt_power_requirement.at[start]

            bids.append(
                {
                    "start_time": start,
                    "end_time": product[1],
                    "only_hours": product[2],
                    "price": 3000,
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


class ElasticDemandStrategy(BaseStrategy):
    """
    A bidding strategy for a demand unit that submits multiple bids to approximate
    a marginal utility curve, based on linear or isoelastic demand theory.
    P = Price, Q = Quantity, E = Elasticity.

    - Linear model: P = P_max + slope * Q (slope is only defined by P_max and Q_max, negative value)
    - Isoelastic model: P = (Q/Q_max) ** (1/E) (E is negative)
      (derived from log-log price elasticity of demand)

    See:
    - https://en.wikipedia.org/wiki/Price_elasticity_of_demand
    - Arnold, Fabian. 2023. https://hdl.handle.net/10419/286380
    - Hirth, Lion et al. 2024. https://doi.org/10.1016/j.eneco.2024.107652.

    """

    def calculate_bids(
        self,
        unit: SupportsMinMax,
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        bids = []

        for product in product_tuples:
            start, end, only_hours = product
            max_abs_power = max(abs(unit.min_power), abs(unit.max_power))

            if unit.elasticity_model == "isoelastic":
                # ISOELASTIC model (constant elasticity of demand):
                # P = (Q / Q_max)^(1 / E)
                # Derived from:
                #   E = (dQ/dP) * (P / Q(P)), E == constant
                #   E = Q'(P) * (P / Q(P))
                #   Q'(P) = E * 1/P * Q(P)
                #   dQ/dP = E * (1 / P) * Q
                #   integrate:
                #   \int 1/Q dQ = E * \int 1/P dP
                #   ln(Q) = E * ln(P) + C
                #   exp(ln(Q)) = exp(E * ln(p) + C)
                #   Q(p) = (exp(ln(p))^E) * exp(C)
                #   Q(p) = P^E * exp(C)
                #   possibly C = 0, C = 1 or C >= 1. We assume C >= 1.
                #   C shifts the demand curve (demand vs. price) up / down and
                #   can be interpreted as the demand that will always be served
                #   we set C = ln(Q_max) and derive the demand curve from there:
                #   P = Q^(1/E) * exp(-C/E) and C = ln(Q_max)
                #   => P = Q^(1/E) * exp(-ln(Q_max)/E) and because exp(-ln(Q_max)/E) = Q_max^(-1/E)
                #   => P = Q^(1/E) * Q_max^(-1/E)
                #   finally
                #   => P = (Q / Q_max)^(1/E)
                # This yields decreasing price as volume increases

                # calculate first bid in isoelastic model (the volume that is bid at max price)
                first_bid_volume = self.find_first_block_bid(
                    unit.elasticity, unit.max_price, max_abs_power
                )
                bids.append(
                    {
                        "start_time": start,
                        "end_time": end,
                        "only_hours": only_hours,
                        "price": unit.max_price,
                        "volume": -first_bid_volume,
                        "node": unit.node,
                    }
                )

                remaining_volume = max_abs_power - first_bid_volume
                if remaining_volume < 0:
                    raise ValueError(
                        "Negative remaining volume for bidding. Check max_power and elasticity."
                    )

                bid_volume = remaining_volume / (unit.num_bids - 1)
                # calculate the remaining bids in isoelastic model
                # P = (Q/Q_max) ** (1/E)
                for i in range(1, unit.num_bids):
                    ratio = (first_bid_volume + i * bid_volume) / max_abs_power
                    if ratio <= 0:
                        continue
                    bid_price = ratio ** (1 / unit.elasticity)

                    bids.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "only_hours": only_hours,
                            "price": bid_price,
                            "volume": -bid_volume,
                            "node": unit.node,
                        }
                    )

            elif unit.elasticity_model == "linear":
                # LINEAR model: P = P_max - slope * Q
                # where slope = -(P_max / Q_max)
                # This ensures price drops linearly with volume
                bid_volume = max_abs_power / unit.num_bids
                slope = -(unit.max_price / max_abs_power)
                for i in range(unit.num_bids):
                    bid_price = unit.max_price + (slope * i * bid_volume)

                    bids.append(
                        {
                            "start_time": start,
                            "end_time": end,
                            "only_hours": only_hours,
                            "price": bid_price,
                            "volume": -bid_volume,
                            "node": unit.node,
                        }
                    )

        return self.remove_empty_bids(bids)

    def find_first_block_bid(
        self, elasticity: float, max_price: float, max_power: float
    ) -> float:
        """
        Calculate the first block bid volume at max_price. P = Price, Q = Quantity, E = Elasticity.
        Assumes isoelastic demand:
            Q = Q_max * P^E

        The first block bid is the volume that is always bid at maximum price, because the
        willingness to pay for it is higher than the markets maximal price.
        The first block bid volume is calculated by finding the intersection of the isoelastic demand
        curve and the maximum price in the marginal utility plot. All demand left of the intersection
        is always bought at maximum price and is called Q_first.

            Q_first = Q
            Q_first = Q_max * P^E

        Therefore:
            Q_first = max_power * (max_price ** E)

        Returns:
            float: Volume > 0, demand that is always bought at max willingness to pay
        """
        volume = max_power * max_price**elasticity

        if abs(volume) > abs(max_power):
            raise ValueError(
                f"Calculated first block bid volume ({volume}) exceeds max power ({max_power})."
            )
        return volume
