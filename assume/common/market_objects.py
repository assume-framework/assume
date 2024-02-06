# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from numbers import Number
from typing import Callable, NamedTuple, Optional, TypedDict

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import Agent


class OnlyHours(NamedTuple):
    """
    Used for peak and off-peak bids.
    Allows to set a begin and hour of the peak.

    For example OnlyHours(8,16) would be used to have a bid which is valid for every day from 8 to 16 a clock.
    """

    begin_hour: int
    end_hour: int


class Order(TypedDict):
    """
    Describes an order which can be either generation (volume > 0) or demand (volume < 0)

    Args:
        bid_id (str): the id of the bid
        start_time (datetime.datetime): the start time of the order
        end_time (datetime.datetime): the end time of the order
        volume (Number | dict[datetime, Number]): the volume of the order (positive if generation)
        accepted_volume (Number | dict[datetime, Number]): the accepted volume of the order
        price (Number): the price of the order
        accepted_price (Number | dict[datetime, Number]): the accepted price of the order
        agent_id (str): the id of the agent
        only_hours (OnlyHours | None): tuple of hours from which this order is available, on multi day products
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

    Args:
        duration (rd | rr.rrule): the duration of the product
        count (int): how many future durations can be traded, must be >= 1
        first_delivery (rd): when does the first delivery begin, in relation to market start
        only_hours (OnlyHours | None): tuple of hours from which this order is available, on multi day products
        eligible_lambda_function (eligible_lambda | None): lambda function which determines if an agent is eligible to trade this product
    """

    duration: rd | rr.rrule
    count: int
    first_delivery: rd = rd()
    only_hours: OnlyHours | None = None
    eligible_lambda_function: eligible_lambda | None = None


class Product(NamedTuple):
    """
    An actual product with start and end.

    Args:
        start (datetime.datetime): the start time of the product
        end (datetime.datetime): the end time of the product
        only_hours (OnlyHours | None): tuple of hours from which this order is available, on multi day products
    """

    start: datetime
    end: datetime
    only_hours: OnlyHours | None = None


@dataclass
class MarketConfig:
    """
    Describes the configuration of a market.

    Args:
        name (str): the name of the market
        opening_hours (rr.rrule): the opening hours of the market
        opening_duration (timedelta): the duration of the opening hours
        market_mechanism (str): name of method used for clearing
        market_products (list[MarketProduct]): list of available products to be traded at the market
        product_type (str): energy or capacity or heat
        maximum_bid_volume (float | None): the maximum valid bid volume of the market
        maximum_bid_price (float | None): the maximum bid price of the market
        minimum_bid_price (float): the minimum bid price of the market
        maximum_gradient (float | None): max allowed change between bids
        additional_fields (list[str]): additional fields of the market
        volume_unit (str): the volume unit of the market (e.g. MW)
        volume_tick (float | None): step increments of volume (e.g. 0.1)
        price_unit (str): the price unit of the market (e.g. €/MWh)
        price_tick (float | None): step increments of price (e.g. 0.1)
        supports_get_unmatched (bool): whether the market supports get unmatched
        eligible_obligations_lambda (eligible_lambda): lambda function which determines if an agent is eligible to trade this product
        addr (str): the address of the market
        aid (str): automatic id of the market
    """

    name: str = "market"
    opening_hours: rr.rrule = rr.rrule(rr.HOURLY)
    opening_duration: timedelta = timedelta(hours=1)
    market_mechanism: str = "pay_as_clear"
    market_products: list[MarketProduct] = field(default_factory=list)
    product_type: str = "energy"
    maximum_bid_volume: float | None = 2000.0
    maximum_bid_price: float | None = 3000.0
    minimum_bid_price: float = -500.0
    maximum_gradient: float | None = None
    additional_fields: list[str] = field(default_factory=list)
    volume_unit: str = "MW"
    volume_tick: float | None = None  # steps in which the amount can be increased
    price_unit: str = "€/MWh"
    price_tick: float | None = None  # steps in which the price can be increased
    supports_get_unmatched: bool = False
    eligible_obligations_lambda: eligible_lambda = lambda x: True

    addr: str = " "
    aid: str = " "


class OpeningMessage(TypedDict):
    """
    Message which is sent from the market to participating agent to open a market.

    Args:
        context (str): the context of the message
        market_id (str): the id of the market
        start_time (float): the start time of the market
        end_time (float): the stop time of the market
        products (list[Product]): list of products which are available at the market to be traded
    """

    context: str
    market_id: str
    start_time: float
    end_time: float
    products: list[Product]


class ClearingMessage(TypedDict):
    """
    Message which is sent from the market to agents to clear a market.

    Args:
        context (str): the context of the message
        market_id (str): the id of the market
        accepted_orders (Orderbook): the orders accepted by the market
        rejected_orders (Orderbook): the orders rejected by the market
    """

    context: str
    market_id: str
    accepted_orders: Orderbook
    rejected_orders: Orderbook


class OrderBookMessage(TypedDict):
    """
    Message containing the order book of a market.

    Args:
        context (str): the context of the message
        market_id (str): the id of the market
        orderbook (Orderbook): the order book of the market
    """

    context: str
    market_id: str
    orderbook: Orderbook


class RegistrationMessage(TypedDict):
    """
    Message for agent registration at a market.

    Args:
        context (str): the context of the message
        market_id (str): the id of the market
        information (dict): additional information for registration
    """

    context: str
    market_id: str
    information: dict


class RegistrationReplyMessage(TypedDict):
    """
    Reply message for agent registration at a market.

    Args:
        context (str): the context of the message
        market_id (str): the id of the market
        accepted (bool): whether the registration is accepted
    """

    context: str
    market_id: str
    accepted: bool


class DataRequestMessage(TypedDict):
    """
    Message for requesting data from a market.

    Args:
        context (str): the context of the message
        market_id (str): the id of the market
        metric (str): the specific metric being requested
        start_time (datetime.datetime): the start time of the data request
        end_time (datetime.datetime): the end time of the data request
    """

    context: str
    market_id: str
    metric: str
    start_time: datetime
    end_time: datetime


class MetaDict(TypedDict):
    """
    Message Meta of a FIPA ACL Message.

    Args:
        sender_addr (str | list): the address of the sender
        sender_id (str): the id of the sender
        reply_to (str): to which agent follow up messages should be sent
        conversation_id (str): the id of the conversation
        performative (str): the performative of the message
        protocol (str): the protocol used
        language (str): the language used
        encoding (str): the encoding used
        ontology (str): the ontology used
        reply_with (str): what the answer should contain as in_reply_to
        in_reply_to (str): str used to reference an earlier action
        reply_by (str): latest time to accept replies
    """

    sender_addr: str | list
    sender_id: str
    reply_to: str
    conversation_id: str
    performative: str
    protocol: str
    language: str
    encoding: str
    ontology: str
    reply_with: str
    in_reply_to: str
    reply_by: str


contract_type = Callable[[Agent, Agent], None]
market_contract_type = Callable[[Agent, Agent, list], None]
