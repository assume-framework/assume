# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import logging
import random
from collections.abc import Callable
from datetime import datetime
from itertools import groupby
from operator import itemgetter

import pandas as pd
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import addr, create_acl

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    MetaDict,
    Order,
    Orderbook,
)
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)


class PayAsBidContractRole(MarketRole):
    """
    This market role handles contracts between agents.
    Contracts can be long term agreements which are valid for a longer period of time.
    Each contract has an evaluation frequency in which the actual cashflow of the contract is executed.
    This can include a calculation depending on the market result and dispatched energy.
    It can be limited which agent can agree to a offered contract at this market by using an `eligible_lambda` in the contract properties.
    The available contracts can be configured through the `available_contracts` dictionary.

    Args:
        marketconfig (MarketConfig): The market configuration.
        limitation (str): a string for limitations - either being "only_co2emissionless" or "only_renewables"

    """

    required_fields = [
        "sender_id",
        "contract",
        "eligible_lambda",
        "evaluation_frequency",
    ]

    def __init__(
        self,
        marketconfig: MarketConfig,
        limitation: str = "only_co2emissionless",
    ):
        super().__init__(marketconfig)
        self.limitation = limitation
        self.futures = {}

    def setup(self):
        super().setup()

        def accept_data_response(content: dict, meta: MetaDict):
            return content.get("context") == "data_response"

        self.context.subscribe_message(
            self, self.handle_data_response, accept_data_response
        )

    def handle_data_response(self, content: dict, meta: MetaDict) -> None:
        """
        Handles the response data and finishes the async Future waiting for the data.

        Args:
            content (dict): message content with data property
            meta (MetaDict): message meta
        """
        if meta["in_reply_to"] not in self.futures:
            logger.error(f'data response {meta["in_reply_to"]} not in awaited futures')
        else:
            self.futures[meta["in_reply_to"]].set_result(content["data"])

    def validate_registration(self, content: dict, meta: MetaDict) -> bool:
        """
        validation function called by handle_registration
        Makes it possible to allow only a subset of agents to bid on this market
        by using self.limitation of the clearing mechanism.

        Args:
            content (dict): message content with registration message and agent information
            meta (MetaDict): message meta

        Returns:
            bool: True if agent fulfills requirements
        """
        if self.limitation:
            if self.limitation == "only_co2emissionless":
                requirement = lambda x: x in [
                    "demand",
                    "nuclear",
                    "wind",
                    "solar",
                    "biomass",
                ]
            elif self.limitation == "only_renewables":
                requirement = lambda x: x in ["demand", "wind", "solar", "biomass"]
            else:
                logger.error(f"unknown limitation {self.limitation}")
            return all(
                [requirement(info["technology"]) for info in content["information"]]
            )
        else:
            return True

    def check_working(self, supply_order: Order, demand_order: Order) -> bool:
        """
        Checks if a given supply_order fulfills the criteria of the demand_order and vice versa.
        Used to allow bidding on some policies only to some agents.

        Args:
            supply_order (Order): the supply order in question
            demand_order (Order): the demand order in question

        Returns:
            bool: True if the orders are compatible
        """
        s_information = self.registered_agents[supply_order["agent_addr"]]
        d_information = self.registered_agents[demand_order["agent_addr"]]
        return supply_order["eligible_lambda"](d_information) and demand_order[
            "eligible_lambda"
        ](s_information)

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> (Orderbook, Orderbook, list[dict]):
        market_getter = itemgetter(
            "start_time", "end_time", "only_hours", "contract", "evaluation_frequency"
        )
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
            accepted_demand_orders: Orderbook = []
            accepted_supply_orders: Orderbook = []
            if product[0:3] not in market_products:
                rejected_orders.extend(product_orders)
                # logger.debug(f'found unwanted bids for {product} should be {market_products}')
                continue

            accepted_product_orders = []

            product_orders = list(product_orders)
            demand_orders = list(filter(lambda x: x["volume"] < 0, product_orders))
            supply_orders = list(filter(lambda x: x["volume"] > 0, product_orders))

            # Sort supply orders by price with randomness for tie-breaking
            supply_orders.sort(key=lambda i: (i["price"], random.random()))
            # Sort demand orders by price in descending order with randomness for tie-breaking
            demand_orders.sort(
                key=lambda i: (i["price"], random.random()), reverse=True
            )
            dem_vol, gen_vol = 0, 0
            # the following algorithm is inspired by one bar for generation and one for demand
            # add generation for currents demand price, until it matches demand
            # generation above it has to be sold for the lower price (or not at all)
            for demand_order in demand_orders:
                if not supply_orders:
                    # if no more generation - reject left over demand
                    rejected_orders.append(demand_order)
                    continue

                dem_vol += -demand_order["volume"]
                to_commit: Orderbook = []

                while supply_orders and gen_vol < dem_vol:
                    supply_order = supply_orders.pop(0)
                    if supply_order["price"] <= demand_order[
                        "price"
                    ] and self.check_working(supply_order, demand_order):
                        supply_order["accepted_volume"] = supply_order["volume"]
                        to_commit.append(supply_order)
                        gen_vol += supply_order["volume"]
                    else:
                        rejected_orders.append(supply_order)
                # now we know which orders we need
                # we only need to see how to arrange it.

                diff = gen_vol - dem_vol

                if diff < 0:
                    # gen < dem
                    # generation is not enough - split demand
                    split_demand_order = demand_order.copy()
                    split_demand_order["accepted_volume"] = diff
                    demand_order["accepted_volume"] = demand_order["volume"] - diff
                    rejected_orders.append(split_demand_order)
                elif diff > 0:
                    # generation left over - split generation
                    supply_order = to_commit[-1]
                    split_supply_order = supply_order.copy()
                    split_supply_order["volume"] = diff
                    supply_order["accepted_volume"] = supply_order["volume"] - diff
                    # only volume-diff can be sold for current price
                    # add left over to supply_orders again
                    gen_vol -= diff

                    supply_orders.insert(0, split_supply_order)
                    demand_order["accepted_volume"] = demand_order["volume"]
                else:
                    # diff == 0 perfect match
                    demand_order["accepted_volume"] = demand_order["volume"]

                accepted_demand_orders.append(demand_order)
                # pay as bid
                for supply_order in to_commit:
                    supply_order["accepted_price"] = supply_order["price"]
                    demand_order["accepted_price"] = supply_order["price"]
                    supply_order["contractor_unit_id"] = demand_order["sender_id"]
                    supply_order["contractor_id"] = demand_order["agent_addr"]
                    demand_order["contractor_unit_id"] = supply_order["sender_id"]
                    demand_order["contractor_id"] = supply_order["agent_addr"]
                accepted_supply_orders.extend(to_commit)

            for order in supply_orders:
                rejected_orders.append(order)

            accepted_product_orders = accepted_demand_orders + accepted_supply_orders

            supply_volume = sum(map(itemgetter("volume"), accepted_supply_orders))
            demand_volume = sum(map(itemgetter("volume"), accepted_demand_orders))
            accepted_orders.extend(accepted_product_orders)
            prices = list(map(itemgetter("price"), accepted_supply_orders))
            if not prices:
                prices = [self.marketconfig.maximum_bid_price]

            meta.append(
                {
                    "supply_volume": supply_volume,
                    "demand_volume": demand_volume,
                    "price": sum(prices) / len(prices),
                    "max_price": max(prices),
                    "min_price": min(prices),
                    "node": None,
                    "product_start": product[0],
                    "product_end": product[1],
                    "only_hours": product[2],
                }
            )
            # demand for contracts is maximum generation capacity of the buyer
            # this is needed so that the seller of the contract can lower the volume

            from functools import partial

            for order in accepted_supply_orders:
                recurrency_task = rr.rrule(
                    freq=order["evaluation_frequency"],
                    dtstart=order["start_time"],
                    until=order["end_time"],
                    cache=True,
                )
                self.context.schedule_recurrent_task(
                    partial(self.execute_contract, contract=order), recurrency_task
                )

            # write flows if applicable
            flows = []

        # contract clearing (pay_as_bid) takes place
        return accepted_orders, rejected_orders, meta, flows

    async def execute_contract(self, contract: Order):
        """
        Scheduled async method which executes a contract in the future.
        For the execution, the actual generation of the selling agent is queried using the data_request mechanism.
        This timeseries is then used as an input to the contract which is used.

        If the contract relies on a market price signal, this is also queried before executing the contract function and sending the result to the buyer and seller.

        Args:
            contract (Order): the contract which gets executed
        """
        # contract must be executed
        # contract from supply is given
        buyer, seller = contract["contractor_unit_id"], contract["unit_id"]
        seller_agent = contract["agent_addr"]
        c_function: Callable[str, tuple[Orderbook, Orderbook]] = available_contracts[
            contract["contract"]
        ]

        end = datetime.utcfromtimestamp(self.context.current_timestamp)
        begin = end - rd(weeks=1)
        begin = max(contract["start_time"], begin)

        reply_with = f'{buyer}_{contract["start_time"]}'
        self.futures[reply_with] = asyncio.Future()
        self.context.schedule_instant_message(
            create_acl(
                {
                    "context": "data_request",
                    "unit": seller,
                    "metric": "energy",
                    "start_time": begin,
                    "end_time": end,
                },
                sender_addr=self.context.addr,
                receiver_addr=addr(seller_agent[0], seller_agent[1]),
                acl_metadata={
                    "reply_with": reply_with,
                },
            ),
            receiver_addr=addr(seller_agent[0], seller_agent[1]),
        )

        if contract["contract"] in contract_needs_market:
            reply_with_market = f'market_eom_{contract["start_time"]}'
            self.futures[reply_with_market] = asyncio.Future()
            self.context.schedule_instant_message(
                create_acl(
                    {
                        "context": "data_request",
                        # ID3 would be average price of orders cleared in last 3 hours before delivery
                        # monthly averages are used for EEG
                        # https://www.netztransparenz.de/de-de/Erneuerbare-Energien-und-Umlagen/EEG/Transparenzanforderungen/Marktpr%C3%A4mie/Marktwert%C3%BCbersicht
                        "market_id": "EOM",
                        "metric": "price",
                        "start_time": begin,
                        "end_time": end,
                    },
                    # TODO other market might not always be the same agent
                    receiver_addr=self.context.addr,
                    sender_addr=self.context.addr,
                    acl_metadata={
                        "reply_with": reply_with_market,
                    },
                ),
                receiver_addr=self.context.addr,
            )
            market_series = await self.futures[reply_with_market]
        else:
            market_series = None

        client_series = await self.futures[reply_with]
        buyer, seller = c_function(contract, market_series, client_series, begin, end)

        in_reply_to = f'{contract["contract"]}_{contract["start_time"]}'
        await self.send_contract_result(contract["contractor_id"], buyer, in_reply_to)
        await self.send_contract_result(contract["agent_addr"], seller, in_reply_to)

    async def send_contract_result(
        self, receiver: tuple, orderbook: Orderbook, in_reply_to: str
    ):
        """
        Send the result of a contract to the given receiver

        Args:
            receiver (tuple): the address and agent id of the receiver
            orderbook (Orderbook): the orderbook which is used as execution of the contract
            in_reply_to (str): the contract to which this is the resulting response
        """
        content: ClearingMessage = {
            # using a clearing message is a way of giving payments to the participants
            "context": "clearing",
            "market_id": self.marketconfig.name,
            "accepted_orders": orderbook,
            "rejected_orders": [],
        }
        await self.context.send_acl_message(
            content=content,
            receiver_addr=receiver[0],
            receiver_id=receiver[1],
            acl_metadata={
                "sender_addr": self.context.addr,
                "sender_id": self.context.aid,
                "in_reply_to": in_reply_to,
            },
        )


def ppa(
    contract: dict,
    market_index: pd.Series,
    future_generation_series: pd.Series,
    start: datetime,
    end: datetime,
):
    """
    The Power Purchase Agreement (PPA) is used to have an agreement where the total amount is not fixed, but the price is fixed.
    As the power is actually bought, the selling agent is not allowed to sell it on other markets.
    The buying agent has an uncertainty of the actual traded amount at the time of contracting.

    Args:
        contract (dict): the contract which is executed
        market_index (pd.Series): the market_index
        generation_series (pd.Series): the actual generation or demand of the agent
        start (datetime.datetime): the start time of the contract
        end (datetime.datetime): the end time of the contract

    Returns:
        tuple[dict, dict]: the buyer order and the seller order as a tuple
    """
    buyer_agent, seller_agent = contract["contractor_id"], contract["agent_addr"]
    volume = sum(future_generation_series.loc[start:end])
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": volume,
            "price": contract["price"],
            "accepted_volume": volume,
            "accepted_price": contract["price"],
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": -volume,
            "price": contract["price"],
            "accepted_volume": -volume,
            "accepted_price": contract["price"],
            "only_hours": None,
            "agent_addr": seller_agent,
        }
    ]
    return buyer, seller


def swingcontract(
    contract: dict,
    market_index: pd.Series,
    demand_series: pd.Series,
    start: datetime,
    end: datetime,
):
    """
    The swing contract is used to provide a band in which one price is paid, while the second (higher) price is paid, when the band is left.

    Args:
        contract (dict): the contract which is executed
        market_index (pd.Series): the market_index
        demand_series (pd.Series): the actual generation or demand of the agent
        start (datetime.datetime): the start time of the contract
        end (datetime.datetime): the end time of the contract

    Returns:
        tuple[dict, dict]: the buyer order and the seller order as a tuple
    """
    buyer_agent, seller_agent = contract["contractor_id"], contract["agent_addr"]

    minDCQ = 80  # daily constraint quantity
    maxDCQ = 100
    set_price = contract["price"]  # ct/kWh
    outer_price = contract["price"] * 1.5  # ct/kwh
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    demand = -demand_series.loc[start:end]
    normal = demand[minDCQ < demand and demand < maxDCQ] * set_price
    expensive = ~demand[minDCQ < demand and demand < maxDCQ] * outer_price
    price = sum(normal) + sum(expensive)
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": demand,
            "price": price,
            "accepted_volume": demand,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": -demand,
            "price": price,
            "accepted_volume": -demand,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": seller_agent,
        }
    ]
    return buyer, seller


def cfd(
    contract: dict,
    market_index: pd.Series,
    gen_series: pd.Series,
    start: datetime,
    end: datetime,
):
    """
    The Contract for Differences does rely on the market signal and does pay the difference between a set price and the actual price when the contract is due - retrospectively.

    Args:
        contract (dict): the contract which is executed
        market_index (pd.Series): the market_index
        demand_series (pd.Series): the actual generation or demand of the agent
        start (datetime.datetime): the start time of the contract
        end (datetime.datetime): the end time of the contract

    Returns:
        tuple[dict, dict]: the buyer order and the seller order as a tuple
    """
    buyer_agent, seller_agent = contract["contractor_id"], contract["agent_addr"]

    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    price_series = (market_index.loc[start:end] - contract["price"]) * gen_series.loc[
        start:end
    ]
    price_series = price_series.dropna()
    price = sum(price_series)
    volume = sum(gen_series.loc[start:end])
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": seller_agent,
        }
    ]
    return buyer, seller


def market_premium(
    contract: dict,
    market_index: pd.Series,
    gen_series: pd.Series,
    start: datetime,
    end: datetime,
):
    """
    The market premium calculates the difference of the market_index and the contracted price.
    As the seller already sold the energy on the index market,

    Args:
        contract (dict): the contract which is executed
        market_index (pd.Series): the market_index
        demand_series (pd.Series): the actual generation or demand of the agent
        start (datetime.datetime): the start time of the contract
        end (datetime.datetime): the end time of the contract

    Returns:
        tuple[dict, dict]: the buyer order and the seller order as a tuple
    """
    buyer_agent, seller_agent = contract["contractor_id"], contract["agent_addr"]
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    price_series = (market_index.loc[start:end] - contract["price"]) * gen_series.loc[
        start:end
    ]
    price_series = price_series.dropna()
    # sum only where market price is below contract price
    price = sum(price_series[price_series < 0])
    volume = sum(gen_series.loc[start:end])
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": seller_agent,
        }
    ]
    return buyer, seller


def feed_in_tariff(
    contract: dict,
    market_index: pd.Series,
    client_series: pd.Series,
    start: datetime,
    end: datetime,
):
    buyer_agent, seller_agent = contract["contractor_id"], contract["agent_addr"]
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    price_series = contract["price"] * client_series.loc[start:end]
    price = sum(price_series)
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    volume = sum(client_series)
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": start,
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": seller_agent,
        }
    ]
    return buyer, seller


available_contracts: dict[str, Callable] = {
    "ppa": ppa,
    "CFD": cfd,
    "FIT": feed_in_tariff,
    "MPFIX": market_premium,
}
contract_needs_market = ["CFD", "MPFIX"]
