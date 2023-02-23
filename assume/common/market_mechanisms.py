
import logging
from datetime import datetime, timedelta
from itertools import groupby
from operator import itemgetter

from dateutil import relativedelta, rrule
from marketconfig import MarketConfig, MarketProduct, MarketOrderbook, Orderbook, Order

from mango import Role

logger = logging.getLogger(__name__)


def cumsum(orderbook: Orderbook):
    sum_ = 0
    for order in orderbook:
        sum_ += order["volume"]
        order["cumsum"] = sum_
    return orderbook


def twoside_clearing(market_agent: Role, market_products: list[MarketProduct]):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders = []
    rejected_orders = []
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # logger.debug(f'found unwanted bids for {product} should be {market_products}')
            continue
        product_orders = list(product_orders)
        bids = filter(lambda x: x["volume"] < 0, product_orders)
        asks = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid

        # generation
        sorted_asks = sorted(asks, key=lambda i: i["price"])

        # demand
        sorted_bids = sorted(bids, key=lambda i: i["price"], reverse=True)

        sorted_asks = cumsum(sorted_asks)
        sorted_bids = cumsum(sorted_bids)

        price, demand, i, j = 0, 0, 0, 0
        intersection_found = False
        for i in range(len(sorted_bids)):
            total_vol = sorted_bids[i]["cumsum"]
            # get first price to match demand (vol)
            for j in range(len(sorted_asks)):
                # gen = total_generation + sorted_asks[j]['volume']
                if sorted_asks[j]["cumsum"] >= -demand:
                    assert price <= sorted_asks[j]["price"], "wrong order"
                    if sorted_asks[j]["price"] < sorted_bids[i]["price"]:
                        # generation is cheaper than demand
                        price = sorted_asks[j]["price"]
                        demand = total_vol
                    else:
                        intersection_found = True
                    break
            if intersection_found:
                break

        accepted_orders.extend(sorted_bids[:i])
        accepted_orders.extend(sorted_asks[:j])
        rejected_orders.extend(sorted_bids[i:])
        rejected_orders.extend(sorted_asks[j:])
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
            # logger.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        # product_orders = list(product_orders)
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            bids = filter(lambda x: x["volume"] < 0, product_orders)
            asks = filter(lambda x: x["volume"] > 0, product_orders)
            # volume 0 is ignored/invalid

            # generation
            sorted_asks = sorted(asks, key=lambda i: i["price"])

            # demand
            sorted_bids = sorted(bids, key=lambda i: i["price"], reverse=True)

            min_len = min(len(sorted_asks, len(sorted_bids)))
            i = 0
            for i in range(min_len):
                if sorted_asks[i]['price'] <= sorted_bids[i]['price']:
                    # pay as bid - so the generator gets payed more than he needed to operate
                    sorted_asks[i]['price'] = sorted_bids[i]['price']
                    sorted_asks[i]['price']
                else:
                    # as we have sorted before, the other bids/asks can't be matched either
                    # once we get here
                    break

            accepted_orders.extend(sorted_bids[:i])
            accepted_orders.extend(sorted_asks[:j])
            rejected_orders.extend(sorted_bids[i:])
            rejected_orders.extend(sorted_asks[j:])

    price = sum(map(lambda order: order['price'], accepted_orders))/len(accepted_orders)
    volume = sum(map(lambda order: order['volume'], accepted_orders))
    # TODO price and volume is wrong if multiple products exist
    if price == 0:
        price = market_agent.marketconfig.maximum_bid
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
            # logger.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)
        bids = filter(lambda x: x["volume"] < 0, product_orders)
        asks = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid

        # generation
        sorted_asks = sorted(asks, key=lambda i: i["price"])
        # demand
        sorted_bids = sorted(bids, key=lambda i: i["price"], reverse=True)

        dem_vol, gen_vol = 0, 0
        # the following algorithm is inspired by one bar for generation and one for demand
        # add generation for currents demand price, until it matches demand
        # generation above it has to be sold for the lower price (or not at all)
        for i in range(len(sorted_bids)):
            bid: Order = sorted_bids[i]
            if not sorted_asks:
                # if no more generation - reject left over demand
                rejected_orders.append(bid)
                continue

            dem_vol += -bid["volume"]
            to_commit: Orderbook = []

            while sorted_asks and gen_vol < dem_vol:
                ask = sorted_asks.pop(0)
                if ask["price"] <= bid["price"]:
                    to_commit.append(ask)
                    gen_vol += ask["volume"]
                else:
                    rejected_orders.append(ask)
            # now we know which orders we need
            # we only need to see how to arrange it.

            diff = gen_vol - dem_vol

            if diff == 0:
                # perfect match
                accepted_orders.append(bid)
                # pay as bid
                for ask in to_commit:
                    ask["price"] = bid["price"]
                accepted_orders.extend(to_commit)
            elif diff < 0:
                # gen < dem
                # generation is not enough
                split_bid = bid.copy()
                split_bid["volume"] = diff
                bid["volume"] -= diff
                accepted_orders.append(bid)
                rejected_orders.append(split_bid)
                # pay as bid
                for ask in to_commit:
                    ask["price"] = bid["price"]
                accepted_orders.extend(to_commit)
            else:  # diff > 0
                # generation left over
                accepted_orders.append(bid)
                # split generation
                ask = to_commit[-1]
                split_ask = ask.copy()
                split_ask["volume"] = diff
                ask["volume"] -= diff
                # only volume-diff can be sold for current price
                # add left over to asks again
                sorted_asks.insert(0, split_ask)
                # pay as bid
                for ask in to_commit:
                    ask["price"] = bid["price"]

    price = sum(map(lambda order: order['price'], accepted_orders))/len(accepted_orders)
    volume = sum(map(lambda order: order['volume'], accepted_orders))
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
# 4. one sided market? - fixed demand?
# 5.

available_strategies = {
    "one_side_market": "TODO",
    "two_side_market": twoside_clearing,
    "pay_as_bid": pay_as_bid,  # TODO
    "pay_as_clear": twoside_clearing,
    "nodal_market": "TODO",
}