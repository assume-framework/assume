import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from numbers import Number
from typing import Callable, NamedTuple, Optional, TypedDict

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import Agent, Role


class OnlyHours(NamedTuple):
    """
    Used for peak and off-peak bids.
    Allows to set a begin and hour of the peak.

    For example OnlyHours(8,16) would be used to have a bid which is valid for every day from 8 to 16 a clock.
    """

    begin_hour: int
    end_hour: int


# describes an order which can be either generation (volume > 0) or demand (volume < 0)
class Order(TypedDict):
    """
    Describes an order which can be either generation (volume > 0) or demand (volume < 0)

    :param bid_id: the id of the bid
    :type bid_id: str
    :param start_time: the start time of the order
    :type start_time: datetime
    :param end_time: the end time of the order
    :type end_time: datetime
    :param volume: the volume of the order
    :type volume: int
    :param price: the price of the order
    :type price: int
    :param only_hours: tuple of hours from which this order is available, on multi day products
    :type only_hours: OnlyHours | None
    :param agent_id: the id of the agent
    :type agent_id: str
    """

    bid_id: str
    start_time: datetime
    end_time: datetime
    volume: Number | dict[datetime, Number]  # positive if generation
    accepted_volume: Number | dict[datetime, Number]
    price: Number
    accepted_price: Number | dict[datetime, Number]
    agent_id: str
    only_hours: Optional[OnlyHours]


Orderbook = list[Order]
MarketOrderbook = dict[str, Orderbook]
eligible_lambda = Callable[[Agent], bool]


# describes the configuration of a market product which is available at a market
@dataclass
class MarketProduct:
    """
    Describes the configuration of a market product which is available at a market.

    :param duration: the duration of the product
    :type duration: rd | rr.rrule
    :param count: how many future durations can be traded, must be >= 1
    :type count: int
    :param first_delivery: when does the first delivery begin, in relation to market start
    :type first_delivery: rd
    :param only_hours: tuple of hours from which this order is available, on multi day products
    :type only_hours: OnlyHours | None
    :param eligible_lambda_function: lambda function which determines if an agent is eligible to trade this product
    :type eligible_lambda_function: eligible_lambda | None
    """

    duration: rd | rr.rrule  # quarter-hourly, half-hourly, hourly, 4hourly, daily, weekly, monthly, quarter-yearly, yearly
    count: int  # how many future durations can be traded, must be >= 1
    # count can also be given as a rrule with until
    first_delivery: rd = (
        rd()
    )  # when does the first delivery begin, in relation to market start
    # this should be a multiple of duration
    only_hours: OnlyHours | None = None
    # e.g. (8,20) - for peak trade, (20, 8) for off-peak, none for base
    eligible_lambda_function: eligible_lambda | None = None


class Product(NamedTuple):
    """
    an actual product with start and end

    :param start: the start time of the product
    :type start: datetime
    :param end: the end time of the product
    :type end: datetime
    :param only_hours: tuple of hours from which this order is available, on multi day products
    :type only_hours: OnlyHours | None
    """

    start: datetime
    end: datetime
    only_hours: OnlyHours | None = None


class MarketMechanism:
    pass


@dataclass
class MarketConfig:
    """
    Describes the configuration of a market.

    :param name: the name of the market
    :type name: str
    :param addr: the address of the market
    :type addr: str | None
    :param aid: automatic id of the market
    :type aid: str | None
    :param opening_hours: the opening hours of the market
    :type opening_hours: rr.rrule
    :param opening_duration: the duration of the opening hours
    :type opening_duration: timedelta
    :param market_mechanism: name of method used for clearing
    :type market_mechanism: str
    :param market_products: list of available products to be traded at the market
    :type market_products: list[MarketProduct]
    :param product_type: energy or capacity or heat
    :type product_type: str
    :param maximum_bid_volume: the maximum valid bid volume of the market
    :type maximum_bid_volume: float
    :param maximum_bid_price: the maximum bid price of the market
    :type maximum_bid_price: float
    :param minimum_bid_price: the minimum bid price of the market
    :type minimum_bid_price: float
    :param maximum_gradient: max allowed change between bids
    :type maximum_gradient: float | None
    :param additional_fields: additional fields of the market
    :type additional_fields: list[str]
    :param volume_unit: the volume unit of the market (e.g. MW)
    :type volume_unit: str
    :param volume_tick: step increments of volume (e.g. 0.1)
    :type volume_tick: float | None
    :param price_unit: the price unit of the market (e.g. €/MWh)
    :type price_unit: str
    :param price_tick: step increments of price (e.g. 0.1)
    :type price_tick: float | None
    :param supports_get_unmatched: whether the market supports get unmatched
    :type supports_get_unmatched: bool
    :param eligible_obligations_lambda: lambda function which determines if an agent is eligible to trade this product
    :type eligible_obligations_lambda: eligible_lambda | None
    """

    name: str
    addr = None
    aid = None

    # continuous markets are clearing just very fast and keep unmatched orders between clearings
    opening_hours: rr.rrule  # dtstart is start/introduction of market
    opening_duration: timedelta
    market_mechanism: str
    market_products: list[MarketProduct] = field(default_factory=list)
    product_type: str = "energy"
    maximum_bid_volume: float | None = 2000.0
    maximum_bid_price: float | None = 3000.0
    minimum_bid_price: float = -500.0
    maximum_gradient: float = None  # very specific - should be in market clearing
    additional_fields: list[str] = field(default_factory=list)
    volume_unit: str = "MW"
    volume_tick: float | None = None  # steps in which the amount can be increased
    price_unit: str = "€/MWh"
    price_tick: float | None = None  # steps in which the price can be increased
    supports_get_unmatched: bool = False
    eligible_obligations_lambda: eligible_lambda = lambda x: True
    # lambda: agent.payed_fee
    # obligation should be time-based
    # only allowed to bid regelenergie if regelleistung was accepted in the same hour for this agent by the market


class MarketMechanism:
    """
    This class represents a market mechanism.
    It is different thant the MarketRole, in the way that the functionality is unrelated to mango.
    The MarketMechanism is embedded into the general MarketRole, which takes care of simulation concerns.
    In the Marketmechanism, all data needed for the clearing is present.
    """

    all_orders: Orderbook
    marketconfig: MarketConfig
    open_auctions: list
    name: str

    def __init__(self, marketconfig: MarketConfig):
        self.marketconfig = marketconfig
        from queue import Queue

        self.open_auctions = Queue()
        self.all_orders = []

    def validate_registration(self, meta: dict) -> bool:
        """
        method to validate a given registration.
        Used to check if a participant is eligible to bid on this market
        """
        return True

    def validate_orderbook(self, orderbook: Orderbook, agent_tuple) -> None:
        """
        method to validate a given orderbook
        This is needed to check if all required fields for this mechanism are present
        """
        max_price = self.marketconfig.maximum_bid_price
        min_price = self.marketconfig.minimum_bid_price
        max_volume = self.marketconfig.maximum_bid_volume

        if self.marketconfig.price_tick:
            assert max_price is not None, "max_price unset"
            assert min_price is not None, "min_price unset"
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
            assert order["price"] <= max_price, f"maximum_bid_price {order['price']}"
            assert order["price"] >= min_price, f"minimum_bid_price {order['price']}"

            if max_volume:
                assert (
                    abs(order["volume"]) <= max_volume
                ), f"max_volume {order['volume']}"

            if self.marketconfig.price_tick:
                assert isinstance(order["price"], int)
            if self.marketconfig.volume_tick:
                assert isinstance(order["volume"], int)
            for field in self.marketconfig.additional_fields:
                assert field in order.keys(), f"missing field: {field}"

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        return [], [], []


class OpeningMessage(TypedDict):
    """
    Message which is sent to the market to open a market

    :param context: the context of the message
    :type context: str
    :param market_id: the id of the market
    :type market_id: str
    :param start: the start time of the market
    :type start: float
    :param stop: the stop time of the market
    :type stop: float
    :param products: list of products which are available at the market to be traded
    :type products: list[Product]
    """

    context: str
    market_id: str
    start: float
    stop: float
    products: list[Product]


class ClearingMessage(TypedDict):
    """
    Message which is sent to the market to clear a market

    :param context: the context of the message
    :type context: str
    :param market_id: the id of the market
    :type market_id: str
    :param orderbook: the orderbook of the market
    :type orderbook: Orderbook
    """

    context: str
    market_id: str
    orderbook: Orderbook


# Class for a Smart Contract which can contain something like:
# - Contract for Differences (CfD) -> based on market result
# - Market Subvention -> based on market result
# - Power Purchase Agreements (PPA) -> A buys everything B generates for price x
# - Swing Contract ->
contract_type = Callable[[Agent, Agent], None]
market_contract_type = Callable[[Agent, Agent, list], None]
