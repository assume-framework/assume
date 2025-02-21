# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import logging

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    MarketProduct,
    MetaDict,
    Orderbook,
)
from assume.common.units_operator import UnitsOperator
from assume.markets.clearing_algorithms.simple import PayAsClearRole

logger = logging.getLogger(__name__)


class PayAsClearIntermediateRole(PayAsClearRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def setup(self):
        super().setup()
        self.context.data["wait_events"] = {}

    async def opening(self):
        self.context.data["wait_events"]["upper"] = asyncio.Future()
        return await super().opening()

    async def clear_market(self, market_products: list[MarketProduct]):
        # iterate through subscribed markets
        # create events for clearing of those
        await self.context.data["wait_events"]["upper"]

        return await super().clear_market(market_products)


class PayAsClearIntermediateBidder(UnitsOperator):
    def setup(self):
        super().setup()
        self.context.data["wait_events"] = {}

    async def formulate_bids(
        self, market: MarketConfig, products: list[tuple]
    ) -> Orderbook:
        # hook in as an upper market opens
        # run the clear market - send rejected bids upwards
        rejected_orderbook = []

        # TODO this is a weird hack to call a function from another role
        for role in self.context._role_handler.roles:
            if isinstance(role, PayAsClearIntermediateRole):
                (accepted, rejected, _, _) = role.clear(
                    self.context.data["all_orders"], products
                )

                from itertools import groupby
                from operator import itemgetter

                market_getter = itemgetter("start_time", "end_time", "only_hours")
                for product, product_orders in groupby(rejected, market_getter):
                    if product in products:
                        rejected_orderbook.extend(product_orders)
                # only one role can be intermediate
                break

        for order in accepted:
            remaining = order["volume"] - order["accepted_volume"]
            if abs(remaining) > 1:
                o = order.copy()
                o["volume"] = remaining

                rejected_orderbook.append(o)

        # send the rejected bids to the upper market
        return rejected_orderbook

    def handle_market_feedback(self, content: ClearingMessage, meta: MetaDict) -> None:
        # set event of upper market ready - we can now clear our lower market, with the unified accepted bids
        for order in content["accepted_orders"]:
            # as we as a intermediate market received this bid, weg can bid the opposite on the lower market
            order["volume"] = -order["accepted_volume"]
            order.pop("accepted_volume", None)
            order.pop("accepted_price", None)
            self.context.data["all_orders"].append(order)
        self.context.data["wait_events"]["upper"].set_result("ready")
