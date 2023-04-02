import logging
from itertools import groupby
from operator import itemgetter

from mango import Role

from .marketclasses import MarketConfig, MarketProduct, Order, Orderbook

log = logging.getLogger(__name__)


def cumsum(orderbook: Orderbook):
    sum_ = 0
    for order in orderbook:
        sum_ += order["volume"]
        order["cumsum"] = sum_
    return orderbook


def pay_as_clear(market_agent: Role, market_products: list[Order], **kwargs):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders = []
    rejected_orders = []
    price, demand = 0, 0
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue
        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid

        # generation
        sorted_supply_orders = sorted(supply_orders, key=lambda i: i["price"])

        # demand
        sorted_demand_orders = sorted(
            demand_orders, key=lambda i: i["price"], reverse=True
        )

        sorted_supply_orders = cumsum(sorted_supply_orders)
        sorted_demand_orders = cumsum(sorted_demand_orders)

        price, demand, i, j = 0, 0, 0, 0
        intersection_found = False
        for i in range(len(sorted_demand_orders)):
            total_vol = sorted_demand_orders[i]["cumsum"]
            # get first price to match demand (vol)
            for j in range(len(sorted_supply_orders)):
                # gen = total_generation + sorted_supply_orders[j]['volume']
                if sorted_supply_orders[j]["cumsum"] >= -demand:
                    assert price <= sorted_supply_orders[j]["price"], "wrong order"
                    if (
                        sorted_supply_orders[j]["price"]
                        < sorted_demand_orders[i]["price"]
                    ):
                        # generation is cheaper than demand
                        price = sorted_supply_orders[j]["price"]
                        demand = total_vol
                    else:
                        intersection_found = True
                    break
            if intersection_found:
                break

        accepted_orders.extend(sorted_demand_orders[:i])
        accepted_orders.extend(sorted_supply_orders[:j])
        rejected_orders.extend(sorted_demand_orders[i:])
        rejected_orders.extend(sorted_supply_orders[j:])

        if price == 0:
            price = market_agent.marketconfig.maximum_bid
    meta = {"volume": -demand, "price": price}
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta


# does not allow to have partial accepted bids
def pay_as_bid(market_agent: Role, market_products: list[MarketProduct]):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        # product_orders = list(product_orders)
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            product_orders = list(product_orders)
            demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
            supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
            # volume 0 is ignored/invalid

            # generation
            sorted_supply_orders = sorted(supply_orders, key=lambda i: i["price"])

            # demand
            sorted_demand_orders = sorted(
                demand_orders, key=lambda i: i["price"], reverse=True
            )

            min_len = min(len(sorted_supply_orders), len(sorted_demand_orders))
            i = 0
            for i in range(min_len):
                if sorted_supply_orders[i]["price"] <= sorted_demand_orders[i]["price"]:
                    # pay as bid - so the generator gets payed more than he needed to operate
                    sorted_supply_orders[i]["price"] = sorted_demand_orders[i]["price"]

                else:
                    # as we have sorted before, the other bids/supply_orders can't be matched either
                    # once we get here
                    break

            accepted_orders.extend(sorted_demand_orders[:i])
            accepted_orders.extend(sorted_supply_orders[:i])
            rejected_orders.extend(sorted_demand_orders[i:])
            rejected_orders.extend(sorted_supply_orders[i:])

    # TODO price and volume is wrong if multiple products exist
    if len(accepted_orders) > 0:
        price = sum(map(lambda order: order["price"], accepted_orders)) / len(
            accepted_orders
        )
    else:

        price = market_agent.marketconfig.maximum_bid
    volume = sum(map(lambda order: order["volume"], accepted_orders))
    meta = {"volume": volume, "price": price}
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta


# with partial accepted bids
def pay_as_bid_partial(market_agent: Role, market_products: list[MarketProduct]):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid

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

            if diff == 0:
                # perfect match
                accepted_orders.append(demand_order)
                # pay as bid
                for supply_order in to_commit:
                    supply_order["price"] = demand_order["price"]
                accepted_orders.extend(to_commit)
            elif diff < 0:
                # gen < dem
                # generation is not enough
                split_demand_order = demand_order.copy()
                split_demand_order["volume"] = diff
                demand_order["volume"] -= diff
                accepted_orders.append(demand_order)
                rejected_orders.append(split_demand_order)
                # pay as bid
                for supply_order in to_commit:
                    supply_order["price"] = demand_order["price"]
                accepted_orders.extend(to_commit)
            else:  # diff > 0
                # generation left over
                accepted_orders.append(demand_order)
                # split generation
                supply_order = to_commit[-1]
                split_supply_order = supply_order.copy()
                split_supply_order["volume"] = diff
                supply_order["volume"] -= diff
                # only volume-diff can be sold for current price
                # add left over to supply_orders again
                sorted_supply_orders.insert(0, split_supply_order)
                # pay as bid
                for supply_order in to_commit:
                    supply_order["price"] = demand_order["price"]

    price = sum(map(lambda order: order["price"], accepted_orders)) / len(
        accepted_orders
    )
    volume = sum(map(lambda order: order["volume"], accepted_orders))
    # TODO price and volume is wrong if multiple products exist
    if price == 0:
        price = market_agent.marketconfig.maximum_bid
    meta = {"volume": volume, "price": price}
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta


# 1. multi-stage market -> clears locally, rejected_bids are pushed up a layer
# 2. nodal pricing -> centralized market which handles different node_ids different - can also be used for country coupling
# 3. nodal limited market -> clear by node_id, select cheapest generation orders from surrounding area up to max_capacity, clear market
# 4. one sided market? - fixed demand as special case of two sided market
# 5.

available_clearing_strategies = {
    "pay_as_bid": pay_as_bid,
    "pay_as_bid_partial": pay_as_bid_partial,
    "pay_as_clear": pay_as_clear,
    "nodal_market": "TODO",
}

if __name__ == "__main__":
    from datetime import datetime, timedelta

    from dateutil import rrule as rr
    from dateutil.relativedelta import relativedelta as rd

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

    from ..common.utils import get_available_products
    from ..markets.base_market import MarketRole

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
    simple_dayahead_auction_config.market_mechanism = available_clearing_strategies[
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
