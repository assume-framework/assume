# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume.common.market_objects import MarketConfig, Orderbook, Product


class UnitOperatorStrategy:
    """
    The base portfolio strategy

    Methods
    -------
    """

    def __init__(self, *args, **kwargs):
        pass

    def calculate_bids(
        self,
        operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Takes information from a units operator and
        defines how the units managed by it should bid.

        This gives a lot of flexibility to the market bids.

        Args:
            operator (UnitsOperator): The operator handling the units.
            market_config (MarketConfig): The market configuration.
            product_tuples (list[Product]): The list of all products the unit can offer.

        Returns:
            Orderbook: The bids consisting of the start time, end time, only hours, price and volume.
        """
        return []


class DirectUnitOperatorStrategy(UnitOperatorStrategy):
    def calculate_bids(
        self,
        operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        """
        Formulates the bid to the market according to the bidding strategy of the each unit individually.

        Args:
            market_config (MarketConfig): The market to formulate bids for.
            product_tuples (list[tuple]): The products to formulate bids for.

        Returns:
            OrderBook: The orderbook that is submitted as a bid to the market.
        """
        bids: Orderbook = []

        for unit_id, unit in operator.units.items():
            product_bids = unit.calculate_bids(
                market_config=market_config,
                product_tuples=product_tuples,
            )
            for i, order in enumerate(product_bids):
                order["agent_addr"] = operator.context.addr
                if market_config.volume_tick:
                    order["volume"] = round(order["volume"] / market_config.volume_tick)
                if market_config.price_tick:
                    order["price"] = round(order["price"] / market_config.price_tick)
                if "bid_id" not in order.keys() or order["bid_id"] is None:
                    order["bid_id"] = f"{unit_id}_{i+1}"
                order["unit_id"] = unit_id
                bids.append(order)

        return bids


class PortfolioBiddingStrategy(UnitOperatorStrategy):
    """
    A naive strategy that bids the marginal cost of the unit on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        operator,  # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        # TODO this should be adjusted

        for unit_id, unit in operator.units.items():
            pass

        bids = []
        return bids
