# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import calendar
import logging
import math
from datetime import datetime
from itertools import groupby
from operator import itemgetter

from mango import Role

from assume.common.market_objects import (
    ClearingMessage,
    DataRequestMessage,
    MarketConfig,
    MarketProduct,
    MetaDict,
    OpeningMessage,
    Orderbook,
    OrderBookMessage,
    RegistrationMessage,
    RegistrationReplyMessage,
)
from assume.common.utils import get_available_products, separate_orders

logger = logging.getLogger(__name__)


class MarketMechanism:
    """
    This class represents a market mechanism.

    It is different than the MarketRole, in the way that the functionality is unrelated to mango.
    The MarketMechanism is embedded into the general MarketRole, which takes care of simulation concerns.
    In the Marketmechanism, all data needed for the clearing is present.

    Parameters:
        all_orders (Orderbook): The list of all orders.
        marketconfig (MarketConfig): The configuration of the market.
        open_auctions (list[dict]): The list of open auctions.
        results (list[dict]): The list of market metadata.

    Args:
        marketconfig (MarketConfig): The configuration of the market.
    """

    all_orders: Orderbook
    marketconfig: MarketConfig
    open_auctions: list[dict]
    name: str

    def __init__(self, marketconfig: MarketConfig):
        super().__init__()
        self.marketconfig = marketconfig
        self.open_auctions = set()
        self.all_orders = []
        self.results = []

    def validate_registration(
        self, content: RegistrationMessage, meta: MetaDict
    ) -> bool:
        """
        Validates a given registration.
        Used to check if a participant is eligible to bid on this market.

        Args:
            content (RegistrationMessage): The content of the message.
            meta (MetaDict): The metadata of the message.

        Returns:
            bool: True if the registration is valid, False otherwise.
        """

        # simple check that 1 MW can be bid at least by  powerplants
        def requirement(unit: dict):
            return unit.get("unit_type") != "power_plant" or abs(unit["max_power"]) >= 1

        return all([requirement(info) for info in content["information"]])

    def validate_orderbook(self, orderbook: Orderbook, agent_tuple: tuple) -> None:
        """
        Validates a given orderbook.

        This is needed to check if all required fields for this mechanism are present.

        Args:
            orderbook (Orderbook): The orderbook to be validated.
            agent_tuple (tuple): The tuple of the agent.

        Raises:
            AssertionError: If the orderbook is invalid.
        """
        max_price = self.marketconfig.maximum_bid_price
        min_price = self.marketconfig.minimum_bid_price
        max_volume = self.marketconfig.maximum_bid_volume

        if self.marketconfig.price_tick:
            assert max_price is not None, "max_price unset"
            assert min_price is not None, "min_price unset"
            assert max_volume is not None, "max_volume unset"
            # max and min should be in units
            max_price = math.floor(max_price / self.marketconfig.price_tick)
            min_price = math.ceil(min_price / self.marketconfig.price_tick)
        if self.marketconfig.volume_tick:
            assert max_volume is not None, "max_volume unset"
            max_volume = math.floor(max_volume / self.marketconfig.volume_tick)

        for order in orderbook:
            order["agent_id"] = agent_tuple
            if not order.get("only_hours"):
                order["only_hours"] = None
            for field in self.marketconfig.additional_fields:
                assert field in order.keys(), f"missing field: {field}"

        sep_orders = separate_orders(orderbook.copy())
        for order in sep_orders:
            assert order["price"] <= max_price, f"maximum_bid_price {order['price']}"
            assert order["price"] >= min_price, f"minimum_bid_price {order['price']}"

            # check that the product is part of an open auction
            product = (order["start_time"], order["end_time"], order["only_hours"])
            assert product in self.open_auctions, "no open auction"

            if max_volume:
                assert (
                    abs(order["volume"]) <= max_volume
                ), f"max_volume {order['volume']}"

            if self.marketconfig.price_tick:
                assert isinstance(order["price"], int)
            if self.marketconfig.volume_tick:
                assert isinstance(order["volume"], int)

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        Clears the market.

        Args:
            orderbook (Orderbook): The orderbook to be cleared.
            market_products (list[MarketProduct]): The products to be traded.

        Returns:
            (Orderbook, Orderbook, list[dict]): The empty accepted orderbook, the empty rejected orderbook and the empty market metadata.
        """
        return [], [], []


class MarketRole(MarketMechanism, Role):
    """
    This is the base class for all market roles. It implements the basic functionality of a market role, such as
    registering agents, clearing the market and sending the results to the database agent.

    Parameters:
        longitude (float): The longitude of the market.
        latitude (float): The latitude of the market.
        marketconfig (MarketConfig): The configuration of the market.
        registered_agents (dict[tuple[str, str], dict]): The dictionary of registered agents.
        required_fields (list[str]): The list of required fields.

    Args:
        marketconfig (MarketConfig): The configuration of the market.

    Methods
    -------
    """

    longitude: float
    latitude: float
    marketconfig: MarketConfig
    registered_agents: dict[tuple[str, str], dict]
    required_fields: list[str] = []

    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)
        self.registered_agents = {}
        if marketconfig.price_tick:
            if marketconfig.maximum_bid_price % marketconfig.price_tick != 0:
                logger.warning(
                    f"{marketconfig.name} - max price not a multiple of tick size"
                )
            if marketconfig.minimum_bid_price % marketconfig.price_tick != 0:
                logger.warning(
                    f"{marketconfig.name} - min price not a multiple of tick size"
                )

        if marketconfig.volume_tick and marketconfig.maximum_bid_volume:
            if marketconfig.maximum_bid_volume % marketconfig.volume_tick != 0:
                logger.warning(
                    f"{marketconfig.name} - max volume not a multiple of tick size"
                )

    def setup(self):
        """
        This method sets up the initial configuration and subscriptions for the market role.

        It sets the address and agent ID of the market config to match the current context.
        It Defines three filter methods (accept_orderbook, accept_registration, and accept_get_unmatched)
        that serve as validation steps for different types of incoming messages.
        Subscribes the role to handle incoming order book messages using the handle_orderbook method.
        Subscribes the role to handle incoming registration messages using the handle_registration method
        If the market configuration supports "get unmatched" functionality, subscribes the role to handle
        such messages using the handle_get_unmatched.
        Schedules the opening() method to run at the next opening time of the market.

        Raises:
            AssertionError: If a required field is missing.
        """
        super().setup()
        self.marketconfig.addr = self.context.addr
        self.marketconfig.aid = self.context.aid

        for field in self.required_fields:
            assert field in self.marketconfig.additional_fields, "missing field"

        def accept_orderbook(content: OrderBookMessage, meta: MetaDict):
            if not isinstance(content, dict):
                return False

            if isinstance(meta["sender_addr"], list):
                meta["sender_addr"] = tuple(meta["sender_addr"])

            return (
                content.get("market_id") == self.marketconfig.name
                and content.get("orderbook") is not None
                and (meta["sender_addr"], meta["sender_id"])
                in self.registered_agents.keys()
            )

        def accept_registration(content: RegistrationMessage, meta: MetaDict):
            if not isinstance(content, dict):
                return False
            if isinstance(meta["sender_addr"], list):
                meta["sender_addr"] = tuple(meta["sender_addr"])

            return (
                content.get("context") == "registration"
                and content.get("market_id") == self.marketconfig.name
            )

        def accept_get_unmatched(content: dict, meta: MetaDict):
            if not isinstance(content, dict):
                return False
            if isinstance(meta["sender_addr"], list):
                meta["sender_addr"] = tuple(meta["sender_addr"])
            return (
                content.get("context") == "get_unmatched"
                and content.get("market_id") == self.marketconfig.name
            )

        def accept_data_request(content: dict, meta: MetaDict):
            return (
                content.get("context") == "data_request"
                and content.get("market_id") == self.marketconfig.name
            )

        self.context.subscribe_message(
            self, self.handle_data_request, accept_data_request
        )
        self.context.subscribe_message(self, self.handle_orderbook, accept_orderbook)
        self.context.subscribe_message(
            self, self.handle_registration, accept_registration
        )

        if self.marketconfig.supports_get_unmatched:
            self.context.subscribe_message(
                self, self.handle_get_unmatched, accept_get_unmatched
            )

        current = datetime.utcfromtimestamp(self.context.current_timestamp)
        next_opening = self.marketconfig.opening_hours.after(current, inc=True)
        opening_ts = calendar.timegm(next_opening.utctimetuple())
        self.context.schedule_timestamp_task(self.opening(), opening_ts)

    async def opening(self):
        """
        Sends an opening message to all registered agents, handles scheduling the clearing of the market and the next opening.

        """
        # scheduled to be opened now
        market_open = datetime.utcfromtimestamp(self.context.current_timestamp)
        market_closing = market_open + self.marketconfig.opening_duration
        products = get_available_products(
            self.marketconfig.market_products, market_open
        )
        until = self.marketconfig.opening_hours._until
        if until and market_closing > until:
            # this market should not open, as the clearing is after the markets end time
            return

        opening_message: OpeningMessage = {
            "context": "opening",
            "market_id": self.marketconfig.name,
            "start_time": market_open,
            "end_time": market_closing,
            "products": products,
        }

        self.open_auctions |= set(opening_message["products"])

        for agent in self.registered_agents.keys():
            agent_addr, agent_id = agent
            await self.context.send_acl_message(
                opening_message,
                receiver_addr=agent_addr,
                receiver_id=agent_id,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                    "reply_with": f"{self.marketconfig.name}_{market_open}",
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
                f"market opening: %s - %s - %s",
                self.marketconfig.name,
                market_open,
                market_closing,
            )
        else:
            logger.debug("market %s - does not reopen", self.marketconfig.name)

    def handle_registration(self, content: RegistrationMessage, meta: MetaDict):
        """
        This method handles incoming registration messages and adds the sender of the message to the list of registered agents.

        Args:
            content (RegistrationMessage): The content of the message.
            meta (MetaDict): The metadata of the message.
        """
        agent_id = meta["sender_id"]
        agent_addr = meta["sender_addr"]
        assert content["market_id"] == self.marketconfig.name
        if self.validate_registration(content, meta):
            self.registered_agents[(agent_addr, agent_id)] = content["information"]
            accepted = True
        else:
            accepted = False

        msg: RegistrationReplyMessage = {
            "context": "registration",
            "market_id": self.marketconfig.name,
            "accepted": accepted,
        }
        self.context.schedule_instant_acl_message(
            content=msg,
            receiver_addr=agent_addr,
            receiver_id=agent_id,
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
                "in_reply_to": meta.get("reply_with"),
            },
        )

    def handle_orderbook(self, content: OrderBookMessage, meta: MetaDict):
        """
        Handles incoming order book messages and validates th order book and adds it to the list of all orders.

        Args:
            content (OrderBookMessage): The content of the message.
            meta (MetaDict): The metadata of the message.

        Raises:
            AssertionError: If the order book is invalid.
        """
        orderbook: Orderbook = content["orderbook"]
        agent_addr = meta["sender_addr"]
        agent_id = meta["sender_id"]
        try:
            self.validate_orderbook(orderbook, (agent_addr, agent_id))
            for order in orderbook:
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
                    "in_reply_to": meta.get("reply_with"),
                },
            )

    def handle_data_request(self, content: DataRequestMessage, meta: MetaDict):
        """
        This method handles incoming data request messages.

        Args:
            content (DataRequestMessage): The content of the message.
            meta (MetaDict): The metadata of the message.

        Raises:
            AssertionError: If the order book is invalid.
        """
        metric_type = content["metric"]
        start = content["start_time"]
        end = content["end_time"]

        data = []
        try:
            import pandas as pd

            data = pd.DataFrame(self.results)
            data.index = data["time"]
            data = data[metric_type][start:end]
        except Exception:
            logger.exception("error handling data request")
        self.context.schedule_instant_acl_message(
            content={
                "context": "data_response",
                "data": data,
            },
            receiver_addr=meta["sender_addr"],
            receiver_id=meta["sender_id"],
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
                "in_reply_to": meta.get("reply_with"),
            },
        )

    def handle_get_unmatched(self, content: dict, meta: MetaDict):
        """
        A handler which sends the orderbook with unmatched orders to an agent and allows to query a subset of the orderbook.

        Args:
            content (dict): The content of the message.
            meta (MetaDict): The metadata of the message.

        Raises:
            AssertionError: If the order book is invalid.
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
                "in_reply_to": 1,
            },
        )

    async def clear_market(self, market_products: list[MarketProduct]):
        """
        This method clears the market and sends the results to the database agent.

        Args:
            market_products (list[MarketProduct]): The products to be traded.
        """
        (
            accepted_orderbook,
            rejected_orderbook,
            market_meta,
        ) = self.clear(self.all_orders, market_products)
        self.all_orders = []
        for order in rejected_orderbook:
            if isinstance(order["volume"], dict):
                order["accepted_volume"] = {
                    start: 0.0 for start in order["volume"].keys()
                }
                order["accepted_price"] = {
                    start: market_meta[i]["price"]
                    for i, start in enumerate(order["volume"].keys())
                }
            else:
                order["accepted_volume"] = 0.0
                order["accepted_price"] = market_meta[0]["price"]
        self.open_auctions - set(market_products)

        accepted_orderbook.sort(key=itemgetter("agent_id"))
        rejected_orderbook.sort(key=itemgetter("agent_id"))

        accepted_orders = {
            agent: list(bids)
            for agent, bids in groupby(accepted_orderbook, itemgetter("agent_id"))
        }
        rejected_orders = {
            agent: list(bids)
            for agent, bids in groupby(rejected_orderbook, itemgetter("agent_id"))
        }

        for agent in self.registered_agents.keys():
            addr, aid = agent
            meta = {"sender_addr": self.context.addr, "sender_id": self.context.aid}
            closing: ClearingMessage = {
                "context": "clearing",
                "market_id": self.marketconfig.name,
                "accepted_orders": accepted_orders.get(agent, []),
                "rejected_orders": rejected_orders.get(agent, []),
            }
            await self.context.send_acl_message(
                closing,
                receiver_addr=addr,
                receiver_id=aid,
                acl_metadata=meta,
            )
        # store order book in db agent
        if not accepted_orderbook:
            logger.warning(
                f"{self.context.current_timestamp} Market result {market_products} for market {self.marketconfig.name} are empty!"
            )
        all_orders = accepted_orderbook + rejected_orderbook
        await self.store_order_book(all_orders)

        for meta in market_meta:
            logger.debug(
                "clearing price for %s is %.2f, volume: %f",
                self.marketconfig.name,
                meta["price"],
                meta["demand_volume"],
            )
            meta["market_id"] = self.marketconfig.name
            meta["time"] = meta["product_start"]
            self.results.append(meta)

        await self.store_market_results(market_meta)

        return accepted_orderbook, market_meta

    async def store_order_book(self, orderbook: Orderbook):
        # Send a message to the OutputRole to update data in the database
        """
        Sends a message to the OutputRole to update data in the database.

        Args:
            orderbook (Orderbook): The order book to be stored.
        """

        db_aid = self.context.data.get("output_agent_id")
        db_addr = self.context.data.get("output_agent_addr")

        if db_aid and db_addr:
            message = {
                "context": "write_results",
                "type": "store_order_book",
                "market_id": self.marketconfig.name,
                "data": orderbook,
            }
            await self.context.send_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content=message,
            )

    async def store_market_results(self, market_meta):
        """
        Sends a message to the OutputRole to update data in the database.

        Args:
            market_meta: The metadata of the market.
        """

        db_aid = self.context.data.get("output_agent_id")
        db_addr = self.context.data.get("output_agent_addr")

        if db_aid and db_addr:
            message = {
                "context": "write_results",
                "type": "store_market_results",
                "market_id": self.marketconfig.name,
                "data": market_meta,
            }
            await self.context.send_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content=message,
            )
