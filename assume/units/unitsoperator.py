import asyncio
from datetime import datetime, timedelta
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from common.marketconfig import MarketConfig, MarketProduct

from mango import Role, create_container
from mango.messages.message import Performatives
from common.bids import (
    Order,
    Orderbook,
    OpeningMessage,
    ClearingMessage,
)
from units.base_unit import BaseUnit
from units.powerplant import PowerPlant
import logging


logger = logging.getLogger(__name__)


class UnitsOperatorRole(Role):
    def __init__(self, id:int= None, name:str=None,  units: dict={} ):
        super().__init__()
        self.id = id
        self.name = name
        self.available_markets: list[MarketConfig] = []
        self.registered_markets: dict[str, MarketConfig] = {}
        self.valid_orders = []
        
        self.build_units(units)

    def set_markets(self, available_markets: list[MarketConfig]):
        self.available_markets.extend(available_markets)
        
    def setup(self):
        self.context.volume = self.volume
        self.context.price = self.price
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

    def add_unit():
        pass

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

    def handle_opening(self, opening: OpeningMessage, meta: dict[str, str]):
        logger.debug(f'Received opening from: {opening["market"]} {opening["start"]}.')
        logger.debug(f'can bid until: {opening["stop"]}')

        self.context.schedule_instant_task(coroutine=self.submit_bids(opening))

    def send_dispatch_plan(self):
        valid_orders = self.valid_orders
        # todo group by unit_id
        for unit in self.units:
            unit.dispatch(valid_orders)
    
    def handle_market_feedback(self, content: ClearingMessage, meta: dict[str, str]):
        logger.debug(f"got market result: {content}")
        orderbook: Orderbook = content["orderbook"]
        for bid in orderbook:
            self.valid_orders.append(bid)
        
        self.send_dispatch_plan(sel)

    async def submit_bids(self, opening):
        """
            send the formulated order book to the market. OPtion for further
            portfolio processing

            Return:
        """

        products = opening["products"]
        market = self.registered_markets[opening["market"]]
        logger.debug(f"setting bids for {market.name}")
        orderbook = self.formulate_bid(market, products)
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

        """
            Takes information from all units that the unit operator manages and
            formulates the bid to the market from that according to the bidding strategy.

            Return: OrderBook that is submitted as a bid to the market
        """

        orderbook: Orderbook = []
        for product in products:
            for unit in self.units:
                order: Order = {}
                order["start_time"] = product[0]
                order["end_time"] = product[1]
                order["agent_id"] = (self.context.addr, self.context.aid)
                #get operational window for each unit
                operational_window= unit.calculate_operational_window()
                #get used bidding strategy for the unit
                used_strategy = unit['bidding_strategy'] 
                #take price from bidding strategy
                order["volume"], order["price"] = used_strategy.calculate_bids(market,  operational_window)

            orderbook.append(order)


        return orderbook

    def build_units(self, unit_dict: dict):
        """
        Instantiates all units assigned to the units operator.
        """
        self.units = []
        for id, unit_conf in unit_dict.items():

            # unit_creater_dict={"solar": SolarPlant()}
            
            if unit_conf["type"] == "solar":
                unit = SolarPlant()
            elif unit_conf["type"] == "powerplant":
                unit = PowerPlant(**unit_conf)
            elif unit_conf["type"] == "storage":
                unit = Storage(**unit_conf)
            elif unit_conf["type"] == "wind":
                unit = WindPowerPlant(**unit_conf)
            else:
                raise Exception("unknown unit type")
            self.units[id] = unit
        
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
    
        """Reset the unit to its initial state."""


        