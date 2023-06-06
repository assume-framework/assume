import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Order, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


def pay_as_clear(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    clear_price = 0
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)
        demand_orders = filter(lambda x: x["volume"] < 0, product_orders)
        supply_orders = filter(lambda x: x["volume"] > 0, product_orders)
        # volume 0 is ignored/invalid

        # generation
        sorted_supply_orders = sorted(supply_orders, key=itemgetter("price"))
        # demand
        sorted_demand_orders = sorted(
            demand_orders, key=itemgetter("price"), reverse=True
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

            assert dem_vol == gen_vol
            # now add the next demand order
            dem_vol += -demand_order["volume"]
            to_commit: Orderbook = []

            # and add supply until the demand order is matched
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
                # generation is not enough - split last demand bid
                split_demand_order = demand_order.copy()
                split_demand_order["volume"] = diff
                demand_order["volume"] -= diff
                rejected_orders.append(split_demand_order)
            elif diff > 0:
                # generation left over - split last generation bid
                supply_order = to_commit[-1]
                split_supply_order = supply_order.copy()
                split_supply_order["volume"] = diff
                supply_order["volume"] -= diff
                # changed supply_order is still part of to_commit and will be added
                # only volume-diff can be sold for current price
                gen_vol -= diff

                # add left over to supply_orders again
                sorted_supply_orders.insert(0, split_supply_order)
            # else: diff == 0 perfect match

            accepted_product_orders.append(demand_order)
            accepted_product_orders.extend(to_commit)

        # set clearing price - merit order - uniform pricing
        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        if accepted_supply_orders:
            clear_price = max(map(itemgetter("price"), accepted_supply_orders))
        else:
            clear_price = 0
        for order in accepted_product_orders:
            order["original_price"] = order["price"]
            order["price"] = clear_price
        accepted_orders.extend(accepted_product_orders)

        accepted_supply_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        accepted_demand_orders = list(
            filter(lambda x: x["volume"] > 0, accepted_product_orders)
        )
        supply_volume = sum(map(itemgetter("volume"), accepted_supply_orders))
        demand_volume = sum(map(itemgetter("volume"), accepted_demand_orders))

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "price": clear_price,
                "max_price": clear_price,
                "min_price": clear_price,
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )

    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta


def pay_as_bid(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
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
                supply_order["original_price"] = supply_order["price"]
                demand_order["original_price"] = demand_order["price"]
                demand_order["price"] = supply_order["price"]
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
            prices = [0]

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
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta
