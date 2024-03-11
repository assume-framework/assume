# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter
from typing import Callable

from assume.common.market_objects import (
    MarketConfig,
    MarketProduct,
    Order,
    Orderbook,
    market_mechanism,
)
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


def pay_as_bid_contract(market_agent: MarketRole, market_products: list[MarketProduct]):
    # make sure that all required fields do exist
    for field in ["sender_id", "contract", "eligible_lambda", "evaluation_frequency"]:
        assert field in market_agent.marketconfig.additional_fields
    accepted_contracts: Orderbook = market_agent.context.data_dict.get("contracts")

    market_getter = itemgetter(
        "start_time", "end_time", "only_hours", "contract", "evaluation_frequency"
    )
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        if product[0:3] not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        accepted_product_orders = []

        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)

        # generation
        sorted_supply_orders = sorted(supply_orders, key=lambda i: i["price"])
        # demand
        sorted_demand_orders = sorted(
            demand_orders, key=lambda i: i["price"], reverse=True
        )

        dem_vol, gen_vol = 0, 0
        # the following algorithm is inspired by one bar for generation and one for demand
        # add generation for currents demand price, until it matches demand
        # generation above it has to be sold for the lower price (or not at all)
        for i in range(len(sorted_demand_orders)):
            demand_order: Order = sorted_demand_orders[i]
            if not sorted_supply_orders:
                # if no more generation - reject left over demand
                rejected_orders.append(demand_order)
                continue

            dem_vol += -demand_order["volume"]
            to_commit: Orderbook = []

            while sorted_supply_orders and gen_vol < dem_vol:
                supply_order = sorted_supply_orders.pop(0)
                if supply_order["price"] <= demand_order["price"]:
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
                split_demand_order["volume"] = diff
                demand_order["volume"] -= diff
                rejected_orders.append(split_demand_order)
            elif diff > 0:
                # generation left over - split generation
                supply_order = to_commit[-1]
                split_supply_order = supply_order.copy()
                split_supply_order["volume"] = diff
                supply_order["volume"] -= diff
                # only volume-diff can be sold for current price
                # add left over to supply_orders again
                gen_vol -= diff

                sorted_supply_orders.insert(0, split_supply_order)
            # else: diff == 0 perfect match

            accepted_orders.append(demand_order)
            # pay as bid
            for supply_order in to_commit:
                supply_order["price"] = demand_order["price"]
            accepted_product_orders.extend(to_commit)

        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        accepted_demand_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        supply_volume = sum(map(itemgetter("volume"), accepted_supply_orders))
        demand_volume = sum(map(itemgetter("volume"), accepted_demand_orders))
        accepted_orders.extend(accepted_product_orders)
        prices = list(map(itemgetter("price"), accepted_supply_orders))
        if not prices:
            prices = [market_agent.marketconfig.maximum_bid]

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "price": sum(prices) / len(prices),
                "max_price": max(prices),
                "min_price": min(prices),
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )
        # demand for contracts is maximum generation capacity of the buyer
        # this is needed so that the seller of the contract can lower the volume

        # contract clearing (pay_as_bid) takes place

    # execute contracts which are due:
    now = datetime.fromtimestamp(market_agent.context.current_timestamp)
    for contract in accepted_contracts:
        if contract["start_time"] > now or contract["end_time"] < now:
            continue
        if contract.get("last_execution", now) + contract["evaluation_frequency"] < now:
            # contract must be executed
            begin = contract["last_execution"]
            end = now
            buyer, seller = contract["buyer_id"], contract["seller_id"]
            index = market_agent.context.data_dict["eom_index"]
            gen_series = market_agent.context.data_dict["generation"][buyer]
            c_function = available_contracts.get(contract["contract"])
            payment = c_function(contract, index, gen_series, begin, end)

            market_agent.context.send_acl_message(payment["buyer"], buyer)
            market_agent.context.send_acl_message(payment["seller"], seller)
            contract["last_execution"] = now

    market_agent.context.data_dict["contracts"] = accepted_contracts

    return accepted_orders, meta


# 1. multi-stage market -> clears locally, rejected_bids are pushed up a layer
# 2. nodal pricing -> centralized market which handles different node_ids different - can also be used for country coupling
# 3. nodal limited market -> clear by node_id, select cheapest generation orders from surrounding area up to max_capacity, clear market
# 4. one sided market? - fixed demand as special case of two sided market
# 5.


import pandas as pd
from mango import Agent


def ppa(
    contract: dict,
    market_index: pd.Series,
    future_generation_series: pd.Series,
    start: datetime,
    end: datetime,
):
    buyer, seller = contract["buyer_id"], contract["seller_id"]
    volume = sum(future_generation_series[start:end])
    return {
        "buyer": {
            "start_time": start,
            "end_time": end,
            "volume": volume,
            "price": contract["price"],
            "agent_id": buyer,
        },
        "seller": {
            "start_time": start,
            "end_time": end,
            "volume": -volume,
            "price": contract["price"],
            "agent_id": seller,
        },
    }


def swingcontract(
    contract: dict,
    market_index: pd.Series,
    demand_series: pd.Series,
    start: datetime,
    end: datetime,
):
    buyer, seller = contract["buyer_id"], contract["seller_id"]

    minDCQ = 80  # daily constraint quantity
    maxDCQ = 100
    set_price = contract["price"]  # ct/kWh
    outer_price = contract["price"] * 1.5  # ct/kwh
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    demand = -demand_series[seller][start:end]
    normal = demand[minDCQ < demand and demand < maxDCQ] * set_price
    expensive = ~demand[minDCQ < demand and demand < maxDCQ] * outer_price
    price = sum(normal) + sum(expensive)
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    return {
        "buyer": {
            "start_time": start,
            "end_time": end,
            "volume": 1,
            "price": price,
            "agent_id": buyer,
        },
        "seller": {
            "start_time": start,
            "end_time": end,
            "volume": -1,
            "price": price,
            "agent_id": seller,
        },
    }


def cfd(
    contract: dict,
    market_index: pd.Series,
    gen_series: pd.Series,
    start: datetime,
    end: datetime,
):
    buyer, seller = contract["buyer_id"], contract["seller_id"]

    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    price_series = (contract["price"] - market_index[start:end]) * gen_series[seller][
        start:end
    ]
    price = sum(price_series)
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    return {
        "buyer": {
            "start_time": start,
            "end_time": end,
            "volume": 1,
            "price": price,
            "agent_id": buyer,
        },
        "seller": {
            "start_time": start,
            "end_time": end,
            "volume": -1,
            "price": price,
            "agent_id": seller,
        },
    }


def market_premium(
    contract: dict,
    market_index: pd.Series,
    gen_series: pd.Series,
    start: datetime,
    end: datetime,
):
    buyer, seller = contract["buyer_id"], contract["seller_id"]
    # TODO does not work with multiple markets with differing time scales..
    # this only works for whole trading hours (as x MW*1h == x MWh)
    price_series = (contract["price"] - market_index[start:end]) * gen_series[seller][
        start:end
    ]
    # sum only where market price is below set_
    price = sum(price_series[price_series < 0])
    # volume is hard to calculate with differing units?
    # unit conversion is quite hard regarding the different intervals
    return {
        "buyer": {
            "start_time": start,
            "end_time": end,
            "volume": 1,
            "price": price,
            "agent_id": buyer,
        },
        "seller": {
            "start_time": start,
            "end_time": end,
            "volume": -1,
            "price": price,
            "agent_id": seller,
        },
    }


available_contracts: dict[str, Callable] = {
    "ppa": ppa,
    "swingcontract": swingcontract,
    "cfg": cfd,
    "market_premium": market_premium,
}


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

    mr = MarketRole(simple_dayahead_auction_config)
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
    simple_dayahead_auction_config.market_mechanism = clearing_mechanisms[
        simple_dayahead_auction_config.market_mechanism
    ]
    mr.all_orders = orderbook
    clearing_result, meta = simple_dayahead_auction_config.market_mechanism(
        mr, products
    )
    import pandas as pd

    print(pd.DataFrame.from_dict(mr.all_orders))
    print(pd.DataFrame.from_dict(clearing_result))
    print(meta)
