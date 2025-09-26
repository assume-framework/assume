# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from assume.common.market_objects import MarketConfig, Order, Orderbook, Product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assume.common.units_operator import UnitsOperator

class BasePortfolioStrategy:
    """
    The base portfolio strategy

    Methods
    -------
    """

    def calculate_bids(
        self,
        operator, # type: UnitsOperator
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

class SimplePortfolioStrategy(BasePortfolioStrategy):
    """
    A naive strategy that bids the marginal cost of the unit on the market.

    Methods
    -------
    """

    def calculate_bids(
        self,
        unit, # type: UnitsOperator
        market_config: MarketConfig,
        product_tuples: list[Product],
        **kwargs,
    ) -> Orderbook:
        # TODO this should be adjusted
        start = product_tuples[0][0]  # start time of the first product
        end_all = product_tuples[-1][1]  # end time of the last product

        for unit_id, unit in operator.units.items():
            pass

        bids = []
        return bids
