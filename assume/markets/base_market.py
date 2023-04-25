import logging
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import pandas as pd
from mango import Role

from ..common.marketclasses import MarketConfig, MarketProduct, Order, Orderbook
from ..common.utils import get_available_products

logger = logging.getLogger(__name__)


# add role per Market
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
        self.registered_agents: list[str] = []
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

        current = datetime.fromtimestamp(self.context.current_timestamp)
        next_opening = self.marketconfig.opening_hours.after(current)
        market_closing = next_opening + self.marketconfig.opening_duration
        logger.info(
            f"first market opening: {self.marketconfig.name} - {next_opening} - {market_closing}"
        )
        self.context.schedule_timestamp_task(
            self.next_opening(), next_opening.timestamp()
        )

    async def next_opening(self):
        current = datetime.fromtimestamp(self.context.current_timestamp)
        next_opening = self.marketconfig.opening_hours.after(current)
        if not next_opening:
            logger.info(f"market {self.marketconfig.name} - does not reopen")
            return

        market_closing = next_opening + self.marketconfig.opening_duration
        products = get_available_products(
            self.marketconfig.market_products, next_opening
        )

        opening_message = {
            "context": "opening",
            "market_id": self.marketconfig.name,
            "start": next_opening,
            "stop": market_closing,
            "products": products,
        }
        self.context.schedule_timestamp_task(
            self.clear_market(products), market_closing.timestamp()
        )
        self.context.schedule_timestamp_task(
            self.next_opening(), next_opening.timestamp()
        )
        logger.info(
            f"next market opening: {self.marketconfig.name} - {next_opening} - {market_closing}"
        )

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
            max_price = self.marketconfig.maximum_bid
            min_price = self.marketconfig.minimum_bid
            max_volume = self.marketconfig.maximum_volume

            if self.marketconfig.price_tick:
                # max and min should be in units
                max_price = round(min_price / self.marketconfig.price_tick)
                min_price = round(min_price / self.marketconfig.price_tick)
            if self.marketconfig.volume_tick:
                max_volume = round(max_volume / self.marketconfig.maximum_volume)

            for order in orderbook:
                order["agent_id"] = (agent_addr, agent_id)
                if not order.get("only_hours"):
                    order["only_hours"] = None
                assert order["price"] <= max_price, "max_bid"
                assert order["price"] >= min_price, "min_bid"
                assert abs(order["volume"]) <= max_volume, "max_volume"
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

            available_orders = filter(order_matches_req, self.all_orders)
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
        self.market_result, market_meta = self.marketconfig.market_mechanism(
            self, market_products
        )

        self.market_result = sorted(self.market_result, key=itemgetter("agent_id"))
        for agent, accepted_orderbook in groupby(
            self.market_result, itemgetter("agent_id")
        ):
            addr, aid = agent
            meta = {"sender_addr": self.context.addr, "sender_id": self.context.aid}

            await self.context.send_acl_message(
                {
                    "context": "clearing",
                    "market_id": self.marketconfig.name,
                    "orderbook": list(accepted_orderbook),
                },
                receiver_addr=addr,
                receiver_id=aid,
                acl_metadata=meta,
            )
        # clear_price = sorted(self.market_result, lambda o: o['price'])[0]

        for meta in market_meta:
            logger.info(
                f'clearing price for {self.marketconfig.name} is {round(meta["price"],2)}, volume: {meta["demand_volume"]}'
            )
            meta["name"] = self.marketconfig.name
            meta["time"] = self.context.current_timestamp
        self.write_results(self.market_result, market_meta)

        return self.market_result, market_meta

    def write_results(self, market_result, market_meta):
        df = pd.DataFrame.from_dict(market_meta)
        export_csv_path = self.context.data_dict.get("export_csv")
        if export_csv_path:
            p = Path(export_csv_path)
            p.mkdir(parents=True, exist_ok=True)
            market_data_path = p.joinpath("market_meta.csv")
            df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())

        df.to_sql("market_meta", self.context.data_dict["db"].bind, if_exists="append")

        # TODO write market_result or other metrics
