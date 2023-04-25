import pandas as pd
from mango import Role
from mango.messages.message import Performatives

from assume.strategies import BaseStrategy
from assume.units import BaseUnit
from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    OpeningMessage,
    Order,
    Orderbook,
)

import logging

logger = logging.getLogger(__name__)


class UnitsOperator(Role):
    def __init__(
        self,
        available_markets: list[MarketConfig],
        opt_portfolio: tuple[bool, BaseStrategy] = None,
    ):
        super().__init__()

        self.available_markets = available_markets
        self.registered_markets: dict[str, MarketConfig] = {}

        if opt_portfolio is None:
            self.use_portfolio_opt = False
            self.portfolio_strategy = None
        else:
            self.use_portfolio_opt = opt_portfolio[0]
            self.portfolio_strategy = opt_portfolio[1]

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

    def add_unit(
        self,
        id: str,
        unit_class: type[BaseUnit],
        unit_params: dict,
    ):
        """
        Create a unit.
        """
        self.units[id] = unit_class(id, **unit_params)
        self.units[id].reset()

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
        logger.debug(f"tried to register at market {market.name}")

    def handle_opening(self, opening: OpeningMessage, meta: dict[str, str]):
        logger.debug(
            f'Operator {self.id} received opening from: {opening["market_id"]} {opening["start"]}.'
        )
        logger.debug(f'Operator {self.id} can bid until: {opening["stop"]}')
        self.context.schedule_instant_task(coroutine=self.submit_bids(opening))

    def send_dispatch_plan(self):
        # todo group by unit_id
        for unit_id, unit in self.units.items():
            # unit.dispatch(valid_orders)
            unit.current_time_step += 1
            unit.total_power_output.append(self.valid_orders[0]["volume"])

    def handle_market_feedback(self, content: ClearingMessage, meta: dict[str, str]):
        logger.debug(f"got market result: {content}")
        orderbook: Orderbook = content["orderbook"]
        for bid in orderbook:
            self.valid_orders.append(bid)

        self.send_dispatch_plan()

    async def submit_bids(self, opening: OpeningMessage):
        """
        formulates an orderbook and sends it to the market.
        This will handle optional portfolio processing

        Return:
        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        logger.debug(f"setting bids for {market.name}")
        orderbook = await self.formulate_bids(market, products)
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
        current_time = pd.to_datetime(self.context.current_timestamp, unit="s")
        
        # the given products just became available on our market
        # and we need to provide bids
        # [whole_next_hour, quarter1, quarter2, quarter3, quarter4]
        # algorithm should buy as much baseload as possible, then add up with quarters
        sorted_products = sorted(products, key=lambda p: (p[0] - p[1], p[0]))

        for product in sorted_products:
            if self.use_portfolio_opt:
                op_windows = []
                for unit_id, unit in self.units.items():
                    # get operational window for each unit
                    operational_window = unit.calculate_operational_window(
                        product, current_time
                    )
                    op_windows.append(operational_window)
                    # TODO calculate bids from sum of op_windows
            else:
                order: Order = {
                    "start_time": product[0],
                    "end_time": product[1],
                    "only_hours": product[2],
                    "agent_id": (self.context.addr, self.context.aid),
                }
                for unit_id, unit in self.units.items():
                    order_c = order.copy()
                    # get operational window for each unit
                    operational_window = unit.calculate_operational_window(
                        product_type=product_type,
                        current_time=current_time,
                    )
                    # take price from bidding strategy
                    volume, price = unit.calculate_bids(
                        product_type=product_type,
                        operational_window=operational_window,
                        current_time=current_time,
                    )
                    if market.volume_tick:
                        volume = round(volume / market.volume_tick)
                    if market.price_tick:
                        price = round(price / market.price_tick)

                    order_c["volume"] = volume
                    order_c["price"] = price
                    orderbook.append(order_c)
        return orderbook
