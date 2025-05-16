# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import logging
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter

import pandas as pd
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from mango import AgentAddress, Performatives, create_acl

from assume.common.market_objects import (
    ClearingMessage,
    MarketConfig,
    MarketProduct,
    MetaDict,
    Order,
    Orderbook,
)
from assume.common.utils import timestamp2datetime
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)


freq_to_delta = {
    rr.YEARLY: rd(years=1),
    rr.MONTHLY: rd(months=1),
    rr.WEEKLY: rd(weeks=1),
    rr.DAILY: rd(days=1),
    rr.HOURLY: rd(hours=1),
    rr.MINUTELY: rd(minutes=1),
    rr.SECONDLY: rd(seconds=1),
}


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
    ):
        super().__init__(marketconfig)
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
            logger.debug("data response %s not in awaited futures", meta["in_reply_to"])
        else:
            self.futures[meta["in_reply_to"]].set_result(content["data"])

    def validate_orderbook(
        self, orderbook: Orderbook, agent_addr: AgentAddress
    ) -> None:
        super().validate_orderbook(orderbook, agent_addr)

        allowed_contracts = self.marketconfig.param_dict.get("allowed_contracts")

        if isinstance(allowed_contracts, list):
            for order in orderbook:
                if order["contract"] not in allowed_contracts:
                    contract = order["contract"]
                    raise ValueError(f"{contract} is not in {allowed_contracts}")

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
                    # if no more generation - reject left over demand at the end
                    continue

                dem_vol += -demand_order["volume"]
                to_commit: Orderbook = []

                while supply_orders and gen_vol < dem_vol:
                    supply_order = supply_orders.pop(0)
                    if supply_order["price"] <= demand_order["price"]:
                        supply_order["accepted_volume"] = supply_order["volume"]
                        to_commit.append(supply_order)
                        gen_vol += supply_order["volume"]
                    # if supply is not partially accepted before, reject it
                    elif not supply_order.get("accepted_volume"):
                        rejected_orders.append(supply_order)
                # now we know which orders we need
                # we only need to see how to arrange it.

                diff = gen_vol - dem_vol

                if diff < 0:
                    # gen < dem
                    # generation is not enough - accept partially
                    demand_order["accepted_volume"] = demand_order["volume"] - diff
                elif diff > 0:
                    # generation left over - accept generation bid partially
                    supply_order = to_commit[-1]
                    split_supply_order = supply_order.copy()
                    split_supply_order["volume"] = diff
                    supply_order["accepted_volume"] = supply_order["volume"] - diff

                    # changed supply_order is still part of to_commit and will be added
                    # only volume-diff can be sold for current price
                    gen_vol -= diff

                    # add left over to supply_orders again
                    supply_orders.insert(0, supply_order)
                    demand_order["accepted_volume"] = demand_order["volume"]
                else:
                    demand_order["accepted_volume"] = demand_order["volume"]

                if demand_order["accepted_volume"]:
                    accepted_demand_orders.append(demand_order)

                # pay as bid
                for supply_order in to_commit:
                    # The problem here is that cashflow is already calculated when the contracts are bid on
                    # the accepted_price = 0 therefore ensures that the contract itself is free of charge
                    # the accepted volume is the maximum amount which receives the support policy
                    supply_order["accepted_price"] = 0
                    demand_order["accepted_price"] = 0
                    supply_order["contract_price"] = supply_order["price"]
                    demand_order["contract_price"] = supply_order["price"]
                    supply_order["contractor_unit_id"] = demand_order["sender_id"]
                    supply_order["contractor_addr"] = demand_order["agent_addr"]
                    demand_order["contractor_unit_id"] = supply_order["sender_id"]
                    demand_order["contractor_addr"] = supply_order["agent_addr"]
                accepted_supply_orders.extend(to_commit)

            # if demand is fulfilled, we do have some additional supply orders
            # these will be rejected
            for order in supply_orders:
                # if the order was not accepted partially, it is rejected
                if not order.get("accepted_volume"):
                    rejected_orders.append(order)

            accepted_product_orders = accepted_demand_orders + accepted_supply_orders
            supply_volume = sum(
                map(itemgetter("accepted_volume"), accepted_supply_orders)
            )
            demand_volume = sum(
                map(itemgetter("accepted_volume"), accepted_demand_orders)
            )
            accepted_orders.extend(accepted_product_orders)
            prices = list(map(itemgetter("contract_price"), accepted_supply_orders))
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
        buyer_agent = contract["contractor_addr"]
        c_function: Callable[str, tuple[Orderbook, Orderbook]] = available_contracts[
            contract["contract"]
        ]

        end = contract["end_time"]
        end = min(end, timestamp2datetime(self.context.current_timestamp))
        begin = end - freq_to_delta[contract["evaluation_frequency"]]
        begin = max(contract["start_time"], begin)
        end -= timedelta(hours=1)

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
                receiver_addr=seller_agent,
                acl_metadata={
                    "reply_with": reply_with,
                    "performative": Performatives.request,
                },
            ),
            receiver_addr=seller_agent,
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
                        "performative": Performatives.request,
                    },
                ),
                receiver_addr=self.context.addr,
            )
            market_series = await self.futures[reply_with_market]
        else:
            market_series = None

        client_series = await self.futures[reply_with]
        o_buyer, o_seller = c_function(
            contract, market_series, client_series, begin, end
        )

        in_reply_to = f'{contract["contract"]}_{contract["start_time"]}'
        await self.send_contract_result(buyer_agent, o_buyer, in_reply_to)
        await self.send_contract_result(seller_agent, o_seller, in_reply_to)

    async def send_contract_result(
        self, receiver: AgentAddress, orderbook: Orderbook, in_reply_to: str
    ):
        """
        Send the result of a contract to the given receiver

        Args:
            receiver (mango.AgentAddress): the address and agent id of the receiver
            orderbook (Orderbook): the orderbook which is used as execution of the contract
            in_reply_to (str): the contract to which this is the resulting response
        """
        content: ClearingMessage = {
            # using a clearing message is a way of giving payments to the participants
            "context": "clearing",
            "market_id": self.marketconfig.market_id,
            "accepted_orders": orderbook,
            "rejected_orders": [],
        }
        await self.context.send_message(
            content=create_acl(
                content,
                receiver,
                self.context.addr,
                acl_metadata={
                    "sender_addr": self.context.addr,
                    "sender_id": self.context.aid,
                    "in_reply_to": in_reply_to,
                    "performative": Performatives.inform,
                },
            ),
            receiver_addr=receiver,
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
    buyer_agent, seller_agent = contract["contractor_addr"], contract["agent_addr"]
    max_volume = contract["accepted_volume"]
    price = sum(
        future_generation_series.clip(upper=max_volume).loc[start:end]
        * contract["contract_price"]
    )
    volume = 1
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
            "accepted_price": price,
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
    buyer_agent, seller_agent = contract["contractor_addr"], contract["agent_addr"]

    minDCQ = contract["accepted_volume"] * 0.8  # daily constraint quantity
    maxDCQ = contract["accepted_volume"] * 1.2

    set_price = contract["contract_price"]  # ct/kWh
    outer_price = contract["contract_price"] * 1.5  # ct/kwh
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    demand = demand_series.loc[start:end]
    normal = demand[(minDCQ <= demand) & (demand <= maxDCQ)] * set_price
    expensive = demand[(minDCQ > demand) | (demand > maxDCQ)] * outer_price
    price = sum(normal) + sum(expensive)
    volume = 1
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
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
    buyer_agent, seller_agent = contract["contractor_addr"], contract["agent_addr"]

    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    max_volume = contract["accepted_volume"]
    market_difference = contract["contract_price"] - market_index.loc[start:end]
    contracted_generation = gen_series.clip(upper=max_volume).loc[start:end]
    price_series = market_difference * contracted_generation
    price = price_series.dropna().sum()
    volume = 1
    # volume is hard to calculate with differing units?
    # unit is always €/MWh - calculation of sum is done in the base unit
    # as the product is energy_cashflow - the volume is already cash (€) - and volume is 0
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
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
    buyer_agent, seller_agent = contract["contractor_addr"], contract["agent_addr"]
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    # TODO add EEG condition to not pay if market_index is below 0 for 3 consecutive hours
    # https://www.gesetze-im-internet.de/eeg_2014/__51.html
    max_volume = contract["accepted_volume"]
    market_difference = contract["contract_price"] - market_index.loc[start:end]
    contracted_generation = gen_series.clip(upper=max_volume).loc[start:end]
    price_series = market_difference * contracted_generation
    price_series = price_series.dropna()

    # sum only where contract price is above market price
    price = price_series.clip(lower=0).sum()
    volume = 1
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
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
    buyer_agent, seller_agent = contract["contractor_addr"], contract["agent_addr"]
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    # multiply by a "hours_per_unit" factor to fix
    price_series = contract["contract_price"] * client_series.loc[start:end].sum()
    price = price_series.mean()
    # XXX the volume of 1 is required to have something to multiply the price with.
    # Another solution would be to switch the product_type to "financial_support" here
    # This would require changing product types
    volume = 1

    # buyer has negative volume, as he receives power
    buyer: Orderbook = [
        {
            "bid_id": contract["contractor_unit_id"],
            "unit_id": contract["contractor_unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": -volume,
            "price": price,
            "accepted_volume": -volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": buyer_agent,
        }
    ]
    seller: Orderbook = [
        {
            "bid_id": contract["unit_id"],
            "unit_id": contract["unit_id"],
            "start_time": end - timedelta(hours=1),
            "end_time": end,
            "volume": volume,
            "price": price,
            "accepted_volume": volume,
            "accepted_price": price,
            "only_hours": None,
            "agent_addr": seller_agent,
        }
    ]
    return buyer, seller


available_contracts: dict[str, Callable] = {
    "PPA": ppa,
    "CFD": cfd,
    "FIT": feed_in_tariff,
    "market_premium": market_premium,
    "MPVAR": market_premium,
    "MPFIX": market_premium,
    "swing": swingcontract,
}
contract_needs_market = ["CFD", "MPFIX", "MPVAR"]


if __name__ == "__main__":
    from dateutil import rrule as rr
    from dateutil.relativedelta import relativedelta as rd

    from assume.common.utils import get_available_products

    simple_dayahead_auction_config = MarketConfig(
        "simple_dayahead_auction",
        market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=datetime(2005, 6, 1),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        amount_unit="MW",
        amount_tick=0.1,
        price_unit="€/MW",
        market_mechanism="pay_as_clear",
    )
    mr: MarketRole = PayAsBidContractRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    print(products)
    start = products[0][0]
    end = products[0][1]
    only_hours = products[0][2]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_id": "gen1",
            "only_hours": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_id": "dem1",
            "only_hours": None,
        },
    ]

    accepted, rejected, meta = mr.clear(orderbook, products)
    import pandas as pd

    print(pd.DataFrame.from_dict(rejected))
    print(pd.DataFrame.from_dict(accepted))
    print(meta)
