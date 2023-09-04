import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


def pay_as_clear(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    """
    Performs electricity market clearing using a pay-as-clear mechanism. This means that the clearing price is the
    highest price that is still accepted. The clearing price is the same for all accepted orders.

    :param market_agent: The market agent
    :type market_agent: MarketRole
    :param market_products: The products to be traded
    :type market_products: list[MarketProduct]
    :return: accepted_orders, rejected_orders, meta
    :rtype: tuple[Orderbook, Orderbook, list[dict]]
    """
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    clear_price = 0
    meta = []
    market_agent.all_orders.sort(key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        product_orders = list(product_orders)
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        supply_orders = [x for x in product_orders if x["volume"] > 0]
        demand_orders = [x for x in product_orders if x["volume"] < 0]
        # volume 0 is ignored/invalid

        # generation
        supply_orders.sort(key=itemgetter("price"))
        # demand
        demand_orders.sort(key=itemgetter("price"), reverse=True)
        dem_vol, gen_vol = 0, 0
        # the following algorithm is inspired by one bar for generation and one for demand
        # add generation for currents demand price, until it matches demand
        # generation above it has to be sold for the lower price (or not at all)
        for demand_order in demand_orders:
            if not supply_orders:
                # if no more generation - reject left over demand
                demand_order["accepted_volume"] = 0
                rejected_orders.append(demand_order)
                continue

            assert dem_vol == gen_vol
            # now add the next demand order
            dem_vol += -demand_order["volume"]
            demand_order["accepted_volume"] = demand_order["volume"]
            to_commit: Orderbook = []

            # and add supply until the demand order is matched
            while supply_orders and gen_vol < dem_vol:
                supply_order = supply_orders.pop(0)
                if supply_order["price"] <= demand_order["price"]:
                    supply_order["accepted_volume"] = supply_order["volume"]
                    to_commit.append(supply_order)
                    gen_vol += supply_order["volume"]
                else:
                    supply_order["accepted_volume"] = 0
                    rejected_orders.append(supply_order)
            # now we know which orders we need
            # we only need to see how to arrange it.

            diff = gen_vol - dem_vol

            if diff < 0:
                # gen < dem
                # generation is not enough - split last demand bid
                split_demand_order = demand_order.copy()
                split_demand_order["accepted_volume"] = diff
                demand_order["accepted_volume"] = demand_order["volume"] - diff
                rejected_orders.append(split_demand_order)
            elif diff > 0:
                # generation left over - split last generation bid
                supply_order = to_commit[-1]
                split_supply_order = supply_order.copy()
                split_supply_order["accepted_volume"] = diff
                supply_order["accepted_volume"] = supply_order["volume"] - diff
                # changed supply_order is still part of to_commit and will be added
                # only volume-diff can be sold for current price
                gen_vol -= diff

                # add left over to supply_orders again
                supply_orders.insert(0, split_supply_order)
            else:
                # diff == 0 perfect match
                demand_order["accepted_volume"] = demand_order["volume"]

            accepted_product_orders.append(demand_order)
            accepted_product_orders.extend(to_commit)

        # set clearing price - merit order - uniform pricing
        accepted_supply_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] > 0
        ]
        if accepted_supply_orders:
            clear_price = max(map(itemgetter("price"), accepted_supply_orders))
        else:
            clear_price = 0

        for order in accepted_product_orders:
            order["accepted_price"] = clear_price

        accepted_orders.extend(accepted_product_orders)

        accepted_supply_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] > 0
        ]
        accepted_demand_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] < 0
        ]
        supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
        demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
        duration_hours = (product[1] - product[0]).total_seconds() / 60 / 60
        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "demand_volume_energy": demand_volume * duration_hours,
                "supply_volume_energy": supply_volume * duration_hours,
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

    return accepted_orders, [], meta


def pay_as_bid(
    market_agent: MarketRole,
    market_products: list[MarketProduct],
):
    """
    Simulates electricity market clearing using a pay-as-bid mechanism.

    :param market_agent: The market agent
    :type market_agent: MarketRole
    :param market_products: The products to be traded
    :type market_products: list[MarketProduct]
    :return: accepted_orders, rejected_orders, meta
    :rtype: tuple[Orderbook, Orderbook, list[dict]]
    """
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders.sort(key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders = list(product_orders)
        supply_orders = [x for x in product_orders if x["volume"] > 0]
        demand_orders = [x for x in product_orders if x["volume"] < 0]
        # volume 0 is ignored/invalid

        # generation
        supply_orders.sort(key=itemgetter("price"))
        # demand
        demand_orders.sort(key=itemgetter("price"), reverse=True)

        dem_vol, gen_vol = 0, 0
        # the following algorithm is inspired by one bar for generation and one for demand
        # add generation for currents demand price, until it matches demand
        # generation above it has to be sold for the lower price (or not at all)
        for demand_order in demand_orders:
            if not supply_orders:
                # if no more generation - reject left over demand
                demand_order["accepted_volume"] = 0
                rejected_orders.append(demand_order)
                continue

            dem_vol += -demand_order["volume"]
            to_commit: Orderbook = []

            while supply_orders and gen_vol < dem_vol:
                supply_order = supply_orders.pop(0)
                if supply_order["price"] <= demand_order["price"]:
                    supply_order["accepted_volume"] = supply_order["volume"]
                    to_commit.append(supply_order)
                    gen_vol += supply_order["volume"]
                else:
                    supply_order["accepted_volume"] = 0
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
                split_supply_order["accepted_volume"] = diff
                supply_order["accepted_volume"] -= supply_order["volume"] - diff
                # only volume-diff can be sold for current price
                # add left over to supply_orders again
                gen_vol -= diff

                supply_orders.insert(0, split_supply_order)
            else:
                # diff == 0 perfect match
                demand_order["accepted_volume"] = demand_order["volume"]

            accepted_orders.append(demand_order)
            # pay as bid
            for supply_order in to_commit:
                supply_order["accepted_price"] = supply_order["price"]
                demand_order["accepted_price"] = supply_order["price"]
            accepted_product_orders.extend(to_commit)

        accepted_supply_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] > 0
        ]
        accepted_demand_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] < 0
        ]
        supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
        demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))

        avg_price = 0
        if supply_volume:
            weighted_price = [
                order["accepted_volume"] * order["accepted_price"]
                for order in accepted_supply_orders
            ]
            avg_price = sum(weighted_price) / supply_volume
        accepted_orders.extend(accepted_product_orders)
        prices = list(map(itemgetter("accepted_price"), accepted_supply_orders)) or [0]
        duration_hours = (product[1] - product[0]).total_seconds() / 60 / 60
        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "demand_volume_energy": demand_volume * duration_hours,
                "supply_volume_energy": supply_volume * duration_hours,
                "price": avg_price,
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

    return accepted_orders, [], meta
