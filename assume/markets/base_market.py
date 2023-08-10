import calendar
import logging
from datetime import datetime
from itertools import groupby
from operator import itemgetter

from mango import Role

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    MarketProduct,
    OpeningMessage,
    Order,
    Orderbook,
)
from assume.common.utils import get_available_products

logger = logging.getLogger(__name__)


class MarketRole(Role):
    longitude: float
    latitude: float
    markets: list = []

    def __init__(self, marketconfig: MarketConfig):
        super().__init__()
        self.marketconfig: MarketConfig = marketconfig

    def setup(self):
        self.marketconfig.addr = self.context.addr
        self.marketconfig.aid = self.context.aid
        self.all_orders: list[Order] = []
        self.market_result: Orderbook = []
        self.registered_agents: list[tuple[str, str]] = []
        self.open_slots = []

        def accept_orderbook(content: dict, meta):
            if not isinstance(content, dict):
                return False
            name_match = content.get("market") == self.marketconfig.name
            orderbook_exists = content.get("orderbook") is not None
            return name_match and orderbook_exists

        def accept_registration(content: dict, meta):
            if not isinstance(content, dict):
                return False
            return (
                content.get("context") == "registration"
                and content.get("market") == self.marketconfig.name
            )

        def accept_get_unmatched(content: dict, meta):
            if not isinstance(content, dict):
                return False
            return (
                content.get("context") == "get_unmatched"
                and content.get("market") == self.marketconfig.name
            )

        self.context.subscribe_message(self, self.handle_orderbook, accept_orderbook)
        self.context.subscribe_message(
            self,
            self.handle_registration,
            accept_registration
            # TODO safer type check? dataclass?
        )

        if self.marketconfig.supports_get_unmatched:
            self.context.subscribe_message(
                self,
                self.handle_get_unmatched,
                accept_get_unmatched
                # TODO safer type check? dataclass?
            )

        current = datetime.utcfromtimestamp(self.context.current_timestamp)
        next_opening = self.marketconfig.opening_hours.after(current, inc=True)
        opening_ts = calendar.timegm(next_opening.utctimetuple())
        self.context.schedule_timestamp_task(self.opening(), opening_ts)

    async def opening(self):
        # scheduled to be opened now
        market_open = datetime.utcfromtimestamp(self.context.current_timestamp)
        market_closing = market_open + self.marketconfig.opening_duration
        products = get_available_products(
            self.marketconfig.market_products, market_open
        )
        if market_closing > self.marketconfig.opening_hours._until:
            # this market should not open, as the clearing is after the markets end time
            return

        opening_message: OpeningMessage = {
            "context": "opening",
            "market_id": self.marketconfig.name,
            "start": market_open,
            "stop": market_closing,
            "products": products,
        }

        for agent in self.registered_agents:
            agent_addr, agent_id = agent
            await self.context.send_acl_message(
                opening_message,
                agent_addr,
                receiver_id=agent_id,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                },
            )

        # schedule closing this market
        closing_ts = calendar.timegm(market_closing.utctimetuple())
        self.context.schedule_timestamp_task(self.clear_market(products), closing_ts)

        # schedule the next opening too
        next_opening = self.marketconfig.opening_hours.after(market_open)
        if next_opening:
            next_opening_ts = calendar.timegm(next_opening.utctimetuple())
            self.context.schedule_timestamp_task(self.opening(), next_opening_ts)
            logger.debug(
                f"market opening: {self.marketconfig.name} - {market_open} - {market_closing}"
            )
        else:
            logger.debug(f"market {self.marketconfig.name} - does not reopen")

    def handle_registration(self, content: dict, meta: dict):
        agent = meta["sender_id"]
        agent_addr = meta["sender_addr"]
        # TODO allow accessing agents properties?
        if self.marketconfig.eligible_obligations_lambda(agent):
            self.registered_agents.append((agent_addr, agent))

    def handle_orderbook(self, content: dict, meta: dict):
        orderbook: Orderbook = content["orderbook"]
        # TODO check if agent is allowed to bid
        agent_addr = meta["sender_addr"]
        agent_id = meta["sender_id"]
        try:
            max_price = self.marketconfig.maximum_bid_price
            min_price = self.marketconfig.minimum_bid_price
            max_volume = self.marketconfig.maximum_bid_volume

            if self.marketconfig.price_tick:
                # max and min should be in units
                max_price = round(max_price / self.marketconfig.price_tick)
                min_price = round(min_price / self.marketconfig.price_tick)
            if self.marketconfig.volume_tick:
                max_volume = round(max_volume / self.marketconfig.volume_tick)

            for order in orderbook:
                order["agent_id"] = (agent_addr, agent_id)
                if not order.get("only_hours"):
                    order["only_hours"] = None
                assert (
                    order["price"] <= max_price
                ), f"maximum_bid_price {order['price']}"
                assert (
                    order["price"] >= min_price
                ), f"minimum_bid_price {order['price']}"
                assert (
                    abs(order["volume"]) <= max_volume
                ), f"max_volume {order['volume']}"
                if self.marketconfig.price_tick:
                    assert isinstance(order["price"], int)
                if self.marketconfig.volume_tick:
                    assert isinstance(order["volume"], int)
                for field in self.marketconfig.additional_fields:
                    assert order[field], f"missing field: {field}"
                self.all_orders.append(order)
        except Exception as e:
            logger.error(f"error handling message from {agent_id} - {e}")
            self.context.schedule_instant_acl_message(
                content={"context": "submit_bids", "message": "Rejected"},
                receiver_addr=agent_addr,
                receiver_id=agent_id,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                    "reply_to": 1,
                },
            )

    def handle_get_unmatched(self, content: dict, meta: dict):
        """
        A handler which sends the orderbook with unmatched orders to an agent.
        Allows to query a subset of the orderbook.
        """
        order = content.get("order")
        agent_addr = meta["sender_addr"]
        agent_id = meta["sender_id"]
        if order:

            def order_matches_req(o):
                return (
                    o["start_time"] == order["start_time"]
                    and o["end_time"] == order["end_time"]
                    and o["only_hours"] == order["only_hours"]
                )

            available_orders = list(filter(order_matches_req, self.all_orders))
        else:
            available_orders = self.all_orders

        self.context.schedule_instant_acl_message(
            content={"context": "get_unmatched", "available_orders": available_orders},
            receiver_addr=agent_addr,
            receiver_id=agent_id,
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
                "reply_to": 1,
            },
        )

    async def clear_market(self, market_products: list[MarketProduct]):
        # print("clear market", len(self.all_orders))
        self.market_result, market_meta = self.marketconfig.market_mechanism(
            self, market_products
        )

        self.market_result.sort(key=itemgetter("agent_id"))
        for agent, accepted_orderbook in groupby(
            self.market_result, itemgetter("agent_id")
        ):
            addr, aid = agent
            meta = {"sender_addr": self.context.addr, "sender_id": self.context.aid}
            closing: ClearingMessage = {
                "context": "clearing",
                "market_id": self.marketconfig.name,
                "orderbook": list(accepted_orderbook),
            }
            await self.context.send_acl_message(
                closing,
                receiver_addr=addr,
                receiver_id=aid,
                acl_metadata=meta,
            )
        # store order book in db agent
        if not self.market_result:
            logger.warning(
                f"{self.context.current_timestamp} Market result {market_products} for market {self.marketconfig.name} are empty!"
            )
        await self.store_order_book(self.market_result)

        for meta in market_meta:
            logger.debug(
                f'clearing price for {self.marketconfig.name} is {meta["price"]:.2f}, volume: {meta["demand_volume"]}'
            )
            meta["market_id"] = self.marketconfig.name
            meta["time"] = meta["product_start"]

        await self.store_market_results(market_meta)

        return self.market_result, market_meta

    async def store_order_book(self, orderbook: Orderbook):
        # Send a message to the OutputRole to update data in the database
        message = {
            "context": "write_results",
            "type": "store_order_book",
            "sender": self.marketconfig.name,
            "data": orderbook,
        }
        db_aid = self.context.data_dict.get("output_agent_id")
        db_addr = self.context.data_dict.get("output_agent_addr")
        if db_aid and db_addr:
            await self.context.send_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content=message,
            )

    async def store_market_results(self, market_meta):
        # Send a message to the OutputRole to update data in the database
        message = {
            "context": "write_results",
            "type": "store_market_results",
            "sender": self.marketconfig.name,
            "data": market_meta,
        }
        db_aid = self.context.data_dict.get("output_agent_id")
        db_addr = self.context.data_dict.get("output_agent_addr")
        if db_aid and db_addr:
            await self.context.send_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content=message,
            )
