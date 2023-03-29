from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, TypedDict

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import Agent, Role

# describes an order which can be either generation (volume > 0) or demand (volume < 0)
class Order(TypedDict):
    start_time: datetime | float
    end_time: datetime | float
    volume: float  # positive if generation
    price: float
    only_hours: tuple[int, int]
    agent_id: str


Orderbook = list[Order]
MarketOrderbook = dict[str, Orderbook]

eligible_lambda = Callable[[Agent], bool]


# describes the configuration of a market product which is available at a market
@dataclass
class MarketProduct:
    duration: rd  # quarter-hourly, half-hourly, hourly, 4hourly, daily, weekly, monthly, quarter-yearly, yearly
    count: int  # how many future durations can be traded, must be >= 1
    # count can also be given as a rrule with until
    first_delivery_after_start: rd = (
        rd()
    )  # when does the first delivery begin, in relation to market start
    # this should be a multiple of duration
    only_hours: tuple[
        int, int
    ] | None = None  # e.g. (8,20) - for peak trade, (20, 8) for off-peak, none for base
    eligible_lambda_function: eligible_lambda | None = None


market_mechanism = Callable[[Role, list[MarketProduct]], tuple[Orderbook, dict]]


@dataclass
class MarketConfig:
    name: str
    addr = None
    aid = None
    # filled by market agent

    # continuous markets are clearing just very fast and keep unmatched orders between clearings
    opening_hours: rr.rrule  # dtstart ist start/introduction of market
    opening_duration: timedelta
    market_mechanism: market_mechanism  # market_mechanism determines wether old offers are deleted (continuous) or not (auction) after clearing
    # if continuous: one of [pay_as_bid, pay_as_ask] else: pay_as_clear

    maximum_bid: float = 9999
    minimum_bid: float = -500
    maximum_gradient: float = None  # very specific - should be in market clearing
    maximum_volume: int = 500
    additional_fields: list[str] = field(default_factory=list)
    market_products: list[MarketProduct] = field(default_factory=list)
    amount_unit: str = "MW"
    amount_tick: float = 0.1  # steps in which the amount can be increased
    price_unit: str = "€"
    price_tick: float = 0.1  # steps in which the price can be increased
    eligible_obligations_lambda: eligible_lambda = lambda x: True
    # lambda: agent.payed_fee
    # obligation should be time-based
    # only allowed to bid regelenergie if regelleistung was accepted in the same hour for this agent by the market


class OpeningMessage(TypedDict):
    context: str
    market_id: str
    start: float
    stop: float
    products: list[MarketProduct]


class ClearingMessage(TypedDict):
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


def ppa(buyer: Agent, seller: Agent):
    set_price = 26  # ct/kWh
    buyer.generation += seller.generation
    seller.revenue += seller.generation * set_price
    buyer.revenue -= seller.generation * set_price
    seller.generation = 0


def swingcontract(buyer: Agent, seller: Agent):
    maxDCQ = 100
    minDCQ = 80
    set_price = 26  # ct/kWh
    outer_price = 45  # ct/kwh
    if minDCQ < buyer.demand and buyer.demand < maxDCQ:
        cost = buyer.demand * set_price
    else:
        cost = outer_price
    buyer.revenue -= buyer.demand * cost
    seller.revenue += buyer.demand * cost


def cfd(buyer: Agent, seller: Agent, market_index):
    set_price = 26  # ct/kWh
    cost = set_price - market_index
    seller.revenue += cost
    buyer.revenue -= cost


def eeg(buyer: Agent, seller: Agent, market_index):
    set_price = 26  # ct/kWh
    cost = set_price - market_index
    if cost > 0:
        seller.revenue += cost
        buyer.revenue -= cost
