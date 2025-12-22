# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.market_objects import MarketConfig, Orderbook, Product
from assume.strategies.naive_strategies import EnergyNaiveStrategy


class UnitOperatorStrategy:
    """
    The UnitOperatorStrategy is similar to the UnitStrategy.
    A UnitOperatorStrategy calculates the bids for all units of a units operator.
    """

    def __init__(self, *args, **kwargs):
        pass

    def calculate_bids(
        self,
        units_operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a units operator and
        defines how the units managed by it should bid.

        This gives a lot of flexibility to the market bids.

        Args:
            units_operator (UnitsOperator): The operator handling the units.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        return []

    def total_capacity(
        self,
        units_operator,  # type: UnitsOperator
    ) -> dict[str, dict[str, float]]:
        """
        Computes the total capacity of the units owned by a unit operator by market and technology.

        Args:
            units_operator (UnitsOperator): The operator that bids on the market(s).
        Returns:
            dict: a nested dictionary indexed by market and by technology.
        """

        total_capacity = {}

        for unit in units_operator.units.values():
            for market_id in unit.bidding_strategies.keys():
                total_capacity[market_id] = total_capacity.get(market_id, {})
                total_capacity[market_id][unit.technology] = (
                    total_capacity[market_id].get(unit.technology, 0) + unit.max_power
                )

        return total_capacity


class UnitsOperatorDirectStrategy(UnitOperatorStrategy):
    def calculate_bids(
        self,
        units_operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Formulates the bids to the market according to the bidding strategy of the each unit individually.
        This calls calculate_bids of each unit and returns the aggregated list of all individual bids of all units.

        Args:
            units_operator: The units operator whose units are queried
            market_config (MarketConfig): The market to formulate bids for.
            product_tuples (list[tuple]): The products to formulate bids for.

        Returns:
            OrderBook: The orderbook that is submitted as a bid to the market.
        """
        bids: Orderbook = []

        for unit_id, unit in units_operator.units.items():
            product_bids = unit.calculate_bids(
                market_config=market_config,
                product_tuples=product_tuples,
            )
            for i, order in enumerate(product_bids):
                order["agent_addr"] = units_operator.context.addr
                if market_config.volume_tick:
                    order["volume"] = round(order["volume"] / market_config.volume_tick)
                if market_config.price_tick:
                    order["price"] = round(order["price"] / market_config.price_tick)
                if "bid_id" not in order.keys() or order["bid_id"] is None:
                    order["bid_id"] = f"{unit_id}_{i + 1}"
                order["unit_id"] = unit_id
                bids.append(order)

        return bids


class UnitsOperatorEnergyHeuristicCournotStrategy(UnitOperatorStrategy):
    """
    A Cournot strategy that adds a markup to the marginal cost of each unit of
    the units operator. The marginal cost is computed with EnergyNaiveStrategy,
    and the markup depends on the total capacity of the unit operator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.markup = kwargs.get("markup", 0)

    def calculate_bids(
        self,
        units_operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a unit that the unit operator manages and
        defines how it is dispatched to the market.

        Args:
            units_operator (UnitsOperator): The units operator that bids on the market.
            market_config (MarketConfig): The configuration of the market.
            product_tuples (list[Product]): The list of all products open for bidding.

        Returns:
            Orderbook: The bids consisting of the start time, end time, price and volume.
        """

        max_power_by_technology = self.total_capacity(units_operator)[
            market_config.market_id
        ]
        max_power = sum(
            max_power_by_technology
        )  # TODO: divide by total available capacity in the market

        operator_bids = Orderbook()

        for unit_id, unit in units_operator.units.items():
            # Compute bids from marginal costs of a unit
            bids = EnergyNaiveStrategy().calculate_bids(
                unit,
                market_config,
                product_tuples,
            )
            # Apply Cournot mark-up
            for bid in bids:
                bid["price"] += self.markup * max_power
                bid["unit_id"] = unit_id

            operator_bids.extend(bids)

        return operator_bids
