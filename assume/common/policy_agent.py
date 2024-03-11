# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

from mango import Role
from mango.messages.message import Performatives

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    OpeningMessage,
    Order,
    Orderbook,
)
from assume.units import BaseUnit

logger = logging.getLogger(__name__)


class PolicyAgent(Role):
    def __init__(
        self, available_markets: list[MarketConfig], contracts: list[Contract]
    ):
        super().__init__()
        self.tender_volume = 10  # MW

        self.available_markets: list[MarketConfig] = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}
        self.valid_orders = []
        self.contracts = contracts

    def setup(self):
        self.id = self.context.aid
        self.context.subscribe_message(
            self,
            self.handle_opening,
            lambda content, meta: content.get("context") == "opening",
        )

        self.context.subscribe_message(
            self,
            self.handle_market_feedback,
            lambda content, meta: content.get("context") == "clearing",
        )

        for market in self.available_markets:
            if self.participate(market):
                self.register_market(market)
                self.registered_markets[market.name] = market

    def participate(self, market: MarketConfig):
        return "contract" in market.additional_fields

    def register_market(self, market):
        self.context.schedule_timestamp_task(
            self.context.send_acl_message(
                {"context": "registration", "market": market.name},
                market.addr,
                receiver_id=market.aid,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            ),
            1,  # register after time was updated for the first time
        )
        logger.debug(f"tried to register at market {market.name}")

    def handle_opening(self, opening: OpeningMessage, meta: dict[str, str]):
        logger.debug(
            f'Operator {self.id} received opening from: {opening["market_id"]} {opening["start"]}.'
        )
        logger.debug(f'Operator {self.id} can bid until: {opening["stop"]}')
        self.context.schedule_instant_task(coroutine=self.submit_bids(opening))

    def handle_market_feedback(self, content: ClearingMessage, meta: dict[str, str]):
        logger.debug(f"got market result: {content}")
        orderbook: Orderbook = content["orderbook"]
        for order in orderbook:
            order["market_id"] = content["market_id"]
        self.valid_orders.extend(orderbook)

    async def submit_bids(self, opening: OpeningMessage):
        """
        formulates an orderbook and sends it to the market.

        uses the given tender amount as a maximum

        Return:
        """
        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug(f"setting bids for {market.name}")
        orderbook = await self.formulate_bids(market, products)
        if len(orderbook) == 0:
            return
        acl_metadata = {
            "performative": Performatives.inform,
            "sender_id": self.context.aid,
            "sender_addr": self.context.addr,
            "conversation_id": "conversation01",
        }
        await self.context.send_acl_message(
            content={
                "context": "submit_bids",
                "market": market.name,
                "orderbook": orderbook,
            },
            receiver_addr=market.addr,
            receiver_id=market.aid,
            acl_metadata=acl_metadata,
        )

    async def formulate_bids(self, market: MarketConfig, products: list[tuple]):
        """
        Takes information from all units that the unit operator manages and
        formulates the bid to the market from that according to the bidding strategy.

        Return: OrderBook that is submitted as a bid to the market
        """

        orderbook: Orderbook = []
        product_type = market.product_type

        # only bid at energy markets
        if product_type != "energy":
            return orderbook

        # the given products just became available on our market
        # and we need to provide bids
        # [whole_next_hour, quarter1, quarter2, quarter3, quarter4]
        # algorithm should buy as much baseload as possible, then add up with quarters
        sorted_products = sorted(products, key=lambda p: (p[0] - p[1], p[0]))
        for product in sorted_products:
            order: Order = {
                "start_time": product[0],
                "end_time": product[1],
                "only_hours": product[2],
                "agent_id": (self.context.addr, self.context.aid),
                "contract": "market_premium",
                "volume": self.tender_volume,
                "price": 100,  # €/MW,
            }
            orderbook.append(order)
        return orderbook
