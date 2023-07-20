import logging
from datetime import datetime
from itertools import groupby
from operator import itemgetter

import pandas as pd
from mango import Role
from mango.messages.message import Performatives

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    OpeningMessage,
    Order,
    Orderbook,
)
from assume.common.utils import aggregate_step_amount
from assume.strategies import BaseStrategy, LearningStrategy
from assume.units import BaseUnit

logger = logging.getLogger(__name__)

try:
    import torch as th
except ImportError:
    th = None


class UnitsOperator(Role):
    def __init__(
        self,
        available_markets: list[MarketConfig],
        opt_portfolio: tuple[bool, BaseStrategy] = None,
    ):
        super().__init__()

        self.bids_map = {}
        self.available_markets = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}
        self.last_sent_dispatch = 0

        if opt_portfolio is None:
            self.use_portfolio_opt = False
            self.portfolio_strategy = None
        else:
            self.use_portfolio_opt = opt_portfolio[0]
            self.portfolio_strategy = opt_portfolio[1]

        # should be a list per product_type
        self.valid_orders = []
        self.units: dict[str, BaseUnit] = {}

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

    async def add_unit(
        self,
        id: str,
        unit_class: type[BaseUnit],
        unit_params: dict,
        index: pd.DatetimeIndex,
    ):
        """
        Create a unit.
        """

        self.units[id] = unit_class(id=id, index=index, **unit_params)
        self.units[id].reset()

        db_aid = self.context.data_dict.get("output_agent_id")
        db_addr = self.context.data_dict.get("output_agent_addr")
        if db_aid and db_addr:
            # send unit data to db agent to store it
            message = {
                "context": "write_results",
                "type": "store_units",
                "data": self.units[id],
            }
            await self.context.send_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content=message,
            )

    def participate(self, market):
        # always participate at all markets
        return True

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
        logger.debug(f"{self.id} tried to register at market {market.name}")

    def handle_opening(self, opening: OpeningMessage, meta: dict[str, str]):
        logger.debug(
            f'{self.id} received opening from: {opening["market_id"]} {opening["start"]} until: {opening["stop"]}.'
        )
        self.context.schedule_instant_task(coroutine=self.submit_bids(opening))

    def handle_market_feedback(self, content: ClearingMessage, meta: dict[str, str]):
        logger.debug(f"{self.id} got market result: {content}")
        orderbook: Orderbook = content["orderbook"]
        for order in orderbook:
            order["market_id"] = content["market_id"]
            # map bid id to unit id
            order["unit_id"] = self.bids_map[order["bid_id"]]
        self.valid_orders.extend(orderbook)
        marketconfig = self.registered_markets[content["market_id"]]
        self.set_unit_dispatch(orderbook, marketconfig)
        self.write_actual_dispatch()

    def set_unit_dispatch(self, orderbook, marketconfig):
        """
        feeds the current market result back to the units
        this does not respect bids from multiple markets
        for the same time period
        """
        orderbook.sort(key=itemgetter("unit_id"))
        for unit_id, orders in groupby(orderbook, itemgetter("unit_id")):
            orders_l = list(orders)
            total_power = sum(map(itemgetter("volume"), orders_l))
            dispatch_plan = {"total_power": total_power}
            self.units[unit_id].set_dispatch_plan(
                dispatch_plan=dispatch_plan,
                start=orderbook[0]["start_time"],
                end=orderbook[0]["end_time"],
                product_type=marketconfig.product_type,
                clearing_price=orders_l[0]["price"],
            )

    def write_actual_dispatch(self):
        """
        sends the actual aggregated dispatch curve
        works across multiple markets
        sends dispatch at or after it actually happens
        """

        last = self.last_sent_dispatch
        if self.context.current_timestamp == last:
            # stop if we exported at this time already
            return
        self.last_sent_dispatch = self.context.current_timestamp

        now = datetime.utcfromtimestamp(self.context.current_timestamp)
        start = datetime.utcfromtimestamp(last)

        market_dispatch = aggregate_step_amount(
            self.valid_orders, start, now, groupby=["market_id", "unit_id"]
        )
        unit_dispatch_dfs = []
        for unit_id, unit in self.units.items():
            data = pd.DataFrame(
                unit.execute_current_dispatch(start, now),
                columns=["power"],
            )
            data["unit"] = unit_id
            unit_dispatch_dfs.append(data)
        unit_dispatch = pd.concat(unit_dispatch_dfs)

        db_aid = self.context.data_dict.get("output_agent_id")
        db_addr = self.context.data_dict.get("output_agent_addr")
        if db_aid and db_addr:
            self.context.schedule_instant_acl_message(
                receiver_id=db_aid,
                receiver_addr=db_addr,
                content={
                    "context": "write_results",
                    "type": "market_dispatch",
                    "data": market_dispatch,
                },
            )
            if unit_dispatch_dfs:
                unit_dispatch = pd.concat(unit_dispatch_dfs)
                self.context.schedule_instant_acl_message(
                    receiver_id=db_aid,
                    receiver_addr=db_addr,
                    content={
                        "context": "write_results",
                        "type": "unit_dispatch",
                        "data": unit_dispatch,
                    },
                )

        self.valid_orders = list(
            filter(lambda x: x["end_time"] >= now, self.valid_orders)
        )

        self.write_learning_params(now)

    async def submit_bids(self, opening: OpeningMessage):
        """
        formulates an orderbook and sends it to the market.
        This will handle optional portfolio processing

        Return:
        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug(f"{self.id} setting bids for {market.name} - {products}")

        # the given products just became available on our market
        # and we need to provide bids
        # [whole_next_hour, quarter1, quarter2, quarter3, quarter4]
        # algorithm should buy as much baseload as possible, then add up with quarters
        products.sort(key=lambda p: (p[0] - p[1], p[0]))
        if self.use_portfolio_opt:
            orderbook = await self.formulate_bids_portfolio(
                market=market,
                products=products,
            )
        else:
            orderbook = await self.formulate_bids(
                market=market,
                products=products,
            )
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

    async def formulate_bids_portfolio(
        self, market: MarketConfig, products: list[tuple]
    ) -> Orderbook:
        orderbook: Orderbook = []
        op_windows = []
        for unit_id, unit in self.units.items():
            # get operational window for each unit
            operational_window = unit.calculate_operational_window(
                product_type=market.product_type,
                product_tuples=products,
            )
            op_windows.append(operational_window)
            # TODO calculate bids from sum of op_windows

        return orderbook

    async def formulate_bids(
        self, market: MarketConfig, products: list[tuple]
    ) -> Orderbook:
        """
        Takes information from all units that the unit operator manages and
        formulates the bid to the market from that according to the bidding strategy.

        Return: OrderBook that is submitted as a bid to the market
        """

        orderbook: Orderbook = []

        for unit_id, unit in self.units.items():
            product_bids = unit.calculate_bids(
                market_config=market,
                product_tuples=products,
            )
            product = products[0]
            for i, bid in enumerate(product_bids):
                order: Order = {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "agent_id": (self.context.addr, self.context.aid),
                }
                price = bid["price"]
                volume = bid["volume"]

                if market.volume_tick:
                    volume = round(volume / market.volume_tick)
                if market.price_tick:
                    price = round(price / market.price_tick)

                order["volume"] = volume
                order["price"] = price
                order["bid_id"] = f"{unit_id}_{i+1}_{product}"
                orderbook.append(order)
                self.bids_map[order["bid_id"]] = unit_id

        return orderbook

    def write_learning_params(self, start):
        """
        sends the current rl_strategy update to the output agent
        """

        unit_rl_strategy_dfs = []
        for unit_id, unit in self.units.items():
            # rl only for energy market for now!
            if isinstance(unit.bidding_strategies.get("energy"), LearningStrategy):
                data = pd.DataFrame(
                    {
                        "profit": unit.outputs["profit"].loc[start],
                        "reward": unit.outputs["reward"].loc[start],
                        "regret": unit.outputs["regret"].loc[start],
                    },
                    index=[start],
                )
                data["unit"] = unit_id
                unit_rl_strategy_dfs.append(data)

        if len(unit_rl_strategy_dfs):
            learning_data = pd.concat(unit_rl_strategy_dfs)

            db_aid = self.context.data_dict.get("output_agent_id")
            db_addr = self.context.data_dict.get("output_agent_addr")
            if db_aid and db_addr:
                self.context.schedule_instant_acl_message(
                    receiver_id=db_aid,
                    receiver_addr=db_addr,
                    content={
                        "context": "write_results",
                        "type": "rl_learning_params",
                        "data": learning_data,
                    },
                )
