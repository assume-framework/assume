import logging
from datetime import datetime, timedelta
from itertools import groupby
from pathlib import Path

import pandas as pd
from mango import Role

from ..common.marketclasses import MarketConfig, MarketProduct, Order, Orderbook
from ..common.utils import get_available_products, is_mod_close, round_digits

log = logging.getLogger(__name__)


# add role per Market
class MarketRole(Role):
    longitude: float
    latitude: float
    markets: list = []

    def __init__(self, marketconfig: MarketConfig):
        super().__init__()
        self.marketconfig: MarketConfig = marketconfig
        self.registered_agents: list[str] = []
        self.open_slots = []
        self.all_orders: list[Order] = []
        self.market_result: Orderbook = []

    def setup(self):
        self.marketconfig.addr = self.context.addr
        self.marketconfig.aid = self.context.aid

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

        self.context.subscribe_message(self, self.handle_orderbook, accept_orderbook)
        self.context.subscribe_message(
            self,
            self.handle_registration,
            accept_registration
            # TODO safer type check? dataclass?
        )
        current = datetime.fromtimestamp(self.context.current_timestamp)
        next_opening = self.marketconfig.opening_hours.after(current)
        market_closing = next_opening + self.marketconfig.opening_duration
        log.info(
            f"first market opening: {self.marketconfig.name} - {next_opening} - {market_closing}"
        )
        self.context.schedule_timestamp_task(
            self.next_opening(), next_opening.timestamp()
        )

    async def next_opening(self):
        current = datetime.fromtimestamp(self.context.current_timestamp)
        next_opening = self.marketconfig.opening_hours.after(current)
        if not next_opening:
            log.info(f"market {self.marketconfig.name} - does not reopen")
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
        log.info(
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

    def handle_registration(self, content: str, meta):
        agent = meta["sender_id"]
        agent_addr = meta["sender_addr"]
        # TODO allow accessing agents properties?
        if self.marketconfig.eligible_obligations_lambda(agent):
            self.registered_agents.append((agent_addr, agent))

    def handle_orderbook(self, content, meta):
        orderbook: Orderbook = content["orderbook"]
        # TODO check if agent is allowed to bid
        agent_addr = meta["sender_addr"]
        agent_id = meta["sender_id"]
        try:
            for order in orderbook:
                order["agent_id"] = (agent_addr, agent_id)

                assert is_mod_close(
                    order["volume"], self.marketconfig.amount_tick
                ), "amount_tick"
                order["volume"] = round_digits(
                    order["volume"], self.marketconfig.amount_tick
                )
                assert is_mod_close(
                    order["price"], self.marketconfig.price_tick
                ), "price_tick"
                order["price"] = round_digits(
                    order["price"], self.marketconfig.price_tick
                )
                if not order.get("only_hours"):
                    order["only_hours"] = None

                assert order["price"] <= self.marketconfig.maximum_bid, "max_bid"
                assert order["price"] >= self.marketconfig.minimum_bid, "min_bid"
                assert (
                    abs(order["volume"]) <= self.marketconfig.maximum_volume
                ), "max_volume"
                for field in self.marketconfig.additional_fields:
                    assert order[field], f"missing field: {field}"
                self.all_orders.append(order)
        except Exception as e:
            log.error(f"error handling message from {agent_id} - {e}")
            self.context.schedule_instant_acl_message(
                content={"context": "Rejected"},
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

        for agent, accepted_orderbook in groupby(
            self.market_result, lambda o: o["agent_id"]
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
        log.info(
            f'clearing price for {self.marketconfig.name} is {market_meta["price"]}, volume: {market_meta["volume"]}'
        )
        market_meta["name"] = self.marketconfig.name
        market_meta["time"] = self.context.current_timestamp
        df = pd.DataFrame.from_dict([market_meta])
        if self.context.data.export_csv:
            p = Path(self.context.data.export_csv)
            p.mkdir(parents=True, exist_ok=True)
            market_data_path = p.joinpath("market_meta.csv")
            df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
        df.to_sql("market_meta", self.context.data.db.bind, if_exists="append")
        return self.market_result
