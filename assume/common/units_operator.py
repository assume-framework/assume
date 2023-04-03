import asyncio
import logging

from mango import Role
from mango.messages.message import Performatives

from ..strategies import BaseStrategy
from ..units import BaseUnit
from .marketclasses import (
    ClearingMessage,
    MarketConfig,
    OpeningMessage,
    Order,
    Orderbook,
)

log = logging.getLogger(__name__)


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
        log.debug(f"tried to register at market {market.name}")

    def handle_opening(self, opening: OpeningMessage, meta: dict[str, str]):
        log.debug(
            f'Operator {self.id} received opening from: {opening["market_id"]} {opening["start"]}.'
        )
        log.debug(f'Operator {self.id} can bid until: {opening["stop"]}')

        self.context.schedule_instant_task(coroutine=self.submit_bids(opening))

    def send_dispatch_plan(self):
        valid_orders = self.valid_orders
        # todo group by unit_id
        for unit_id, unit in self.units.items():
            # unit.dispatch(valid_orders)
            unit.current_time_step += 1

    def handle_market_feedback(self, content: ClearingMessage, meta: dict[str, str]):
        log.debug(f"got market result: {content}")
        orderbook: Orderbook = content["orderbook"]
        for bid in orderbook:
            self.valid_orders.append(bid)

        self.send_dispatch_plan()

    async def submit_bids(self, opening: OpeningMessage):
        """
        send the formulated order book to the market. OPtion for further
        portfolio processing

        Return:
        """

        products = opening["products"]
        market = self.registered_markets[opening["market_id"]]
        log.debug(f"setting bids for {market.name}")
        orderbook = await self.formulate_bids(market, products)
        acl_metadata = {
            "performative": Performatives.inform,
            "sender_id": self.context.aid,
            "sender_addr": self.context.addr,
            "conversation_id": "conversation01",
        }
        await self.context.send_acl_message(
            content={
                "market": market.name,
                "orderbook": orderbook,
            },
            receiver_addr=market.addr,
            receiver_id=market.aid,
            acl_metadata=acl_metadata,
        )

    async def formulate_bids(self, market: MarketConfig, products):
        # sourcery skip: merge-dict-assign

        """
        Takes information from all units that the unit operator manages and
        formulates the bid to the market from that according to the bidding strategy.

        Return: OrderBook that is submitted as a bid to the market
        """

        orderbook: Orderbook = []
        for product in products:
            if self.use_portfolio_opt == False:
                for unit_id, unit in self.units.items():
                    order: Order = {}
                    order["start_time"] = product[0]
                    order["end_time"] = product[1]
                    order["only_hours"] = None
                    order["agent_id"] = (self.context.addr, self.context.aid)
                    # get operational window for each unit
                    operational_window = unit.calculate_operational_window()
                    # get used bidding strategy for the unit
                    unit_strategy = unit.bidding_strategy
                    # take price from bidding strategy
                    order["volume"], order["price"] = unit_strategy.calculate_bids(
                        market, operational_window
                    )

                    orderbook.append(order)

            else:
                raise NotImplementedError

        return orderbook

    def add_unit(
        self,
        id: str,
        unit_class: type[BaseUnit],
        unit_params: dict,
        bidding_strategy: type[BaseStrategy] = None,
    ):
        """
        Create a unit.
        """
        self.units[id] = unit_class(id, **unit_params)
        self.units[id].bidding_strategy = bidding_strategy

        if bidding_strategy is None and self.use_portfolio_opt == False:
            raise ValueError(
                "No bidding strategy defined for unit while not using portfolio optimization."
            )

        self.units[id].reset()

    # Needed data in the future
    """""
    def get_world_data(self, input_data):
        self.temperature = input_data.temperature
        self.wind_speed = input_data.wind_speed

    def location(self, coordinates: tuple(float, float)= (0,0), NUTS_0: str = None):
        self.x: int = 0
        self.y: int = 0
        NUTS_0: str = 0

    def get_temperature(self, location):
        if isinstance(location, tuple):
            # get lat lon table
            pass
        elif "NUTS" in location:
            # get nuts table
            pass
            
    def reset(self):
    
        #Reset the unit to its initial state.

    """
