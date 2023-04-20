from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, TypedDict, Union

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import Agent, Role


# describes an order which can be either generation (volume > 0) or demand (volume < 0)
class Order(TypedDict):
    start_time: Union[datetime, float]
    end_time: Union[datetime, float]
    volume: int  # positive if generation
    price: int
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
    only_hours: Union[
        tuple[int, int], None
    ] = None  # e.g. (8,20) - for peak trade, (20, 8) for off-peak, none for base
    eligible_lambda_function: Union[eligible_lambda, None] = None


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
    market_mechanism: Union[
        market_mechanism, str
    ]  # market_mechanism determines wether old offers are deleted (auction) or not (continuous) after clearing

    maximum_bid: int = 9999
    minimum_bid: int = -500
    maximum_gradient: float = None  # very specific - should be in market clearing
    maximum_volume: int = 500
    additional_fields: list[str] = field(default_factory=list)
    market_products: list[MarketProduct] = field(default_factory=list)
    volume_unit: str = "MW"
    volume_tick: float = 0.1  # steps in which the amount can be increased
    price_unit: str = "€/MWh"
    price_tick: float = 0.1  # steps in which the price can be increased
    supports_get_unmatched: bool = False
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
