import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


def cumsum(orderbook: Orderbook):
    sum_ = 0
    for order in orderbook:
        sum_ += order["volume"]
        order["cumsum"] = sum_
    return orderbook


# does not allow to have partially accepted bids
# all or nothing
def pay_as_clear_aon(market_agent: MarketRole, market_products: list[MarketProduct]):
    """
    This implements pay-as-clear where each bids volume needs an exactly matching order with the same volume.
    Partial clearing is not allowed here.
    This has the side effect, that the cleared price can be much higher if bids with different volume are accepted
    """
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    clear_price = market_agent.marketconfig.minimum_bid_price
    market_agent.all_orders.sort(key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        product_orders = list(product_orders)
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        # groupby does only group consecutive groups
        product_orders.sort(key=lambda x: abs(x["volume"]))
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            orders = list(orders)
            supply_orders = [x for x in product_orders if x["volume"] > 0]
            demand_orders = [x for x in product_orders if x["volume"] < 0]
            supply_orders.sort(key=itemgetter("price"))
            demand_orders.sort(key=itemgetter("price"), reverse=True)
            # volume 0 is ignored/invalid

            min_len = min(len(supply_orders), len(demand_orders))
            i = 0
            for i in range(min_len):
                if supply_orders[i]["price"] <= demand_orders[i]["price"]:
                    supply_orders[i]["accepted_volume"] = supply_orders[i]["volume"]
                    demand_orders[i]["accepted_volume"] = demand_orders[i]["volume"]

                    # pay as clear - all accepted receive the highest needed/cleared price
                    if clear_price < supply_orders[i]["price"]:
                        clear_price = supply_orders[i]["price"]
                else:
                    # as we have sorted before, the other bids/supply_orders can't be matched either
                    # once we get here
                    break
            # resulting i is the cut point
            accepted_product_orders.extend(demand_orders[:i])
            accepted_product_orders.extend(supply_orders[:i])

            for order in supply_orders[i:]:
                order["accepted_volume"] = 0
            for order in demand_orders[i:]:
                order["accepted_volume"] = 0

            rejected_orders.extend(demand_orders[i:])
            rejected_orders.extend(supply_orders[i:])

        for order in accepted_product_orders:
            order["accepted_price"] = clear_price

        accepted_supply_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] > 0
        ]
        accepted_demand_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] < 0
        ]
        supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
        demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
        accepted_orders.extend(accepted_product_orders)
        prices = list(map(itemgetter("accepted_price"), accepted_supply_orders))
        if not prices:
            prices = [market_agent.marketconfig.maximum_bid]

        meta.append(
            {
                "supply_volume": supply_volume,
                "demand_volume": demand_volume,
                "uniform_price": clear_price,
                "price": clear_price,
                "node_id": None,
                "product_start": product[0],
                "product_end": product[1],
                "only_hours": product[2],
            }
        )
    # remember unused orders - necessary if the same hour will be cleared again
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future
    return accepted_orders, [], meta


# does not allow to have partial accepted bids
def pay_as_bid_aon(market_agent: MarketRole, market_products: list[MarketProduct]):
    market_getter = itemgetter("start_time", "end_time", "only_hours")
    accepted_orders: Orderbook = []
    rejected_orders: Orderbook = []
    meta = []
    market_agent.all_orders.sort(key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        product_orders = list(product_orders)
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        product_orders.sort(key=lambda x: abs(x["volume"]))
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            orders = list(orders)
            supply_orders = [x for x in product_orders if x["volume"] > 0]
            demand_orders = [x for x in product_orders if x["volume"] < 0]
            # volume 0 is ignored/invalid
            supply_orders.sort(key=itemgetter("price"))
            demand_orders.sort(key=itemgetter("price"), reverse=True)

            min_len = min(len(supply_orders), len(demand_orders))
            i = 0
            for i in range(min_len):
                if supply_orders[i]["price"] <= demand_orders[i]["price"]:
                    supply_orders[i]["accepted_volume"] = supply_orders[i]["volume"]
                    demand_orders[i]["accepted_volume"] = demand_orders[i]["volume"]

                    # pay as bid - so the generator gets payed more than he needed to operate
                    supply_orders[i]["accepted_price"] = demand_orders[i]["price"]
                    demand_orders[i]["accepted_price"] = demand_orders[i]["price"]

                else:
                    # as we have sorted before, the other bids/supply_orders can't be matched either
                    # once we get here
                    break

            accepted_product_orders.extend(demand_orders[:i])
            accepted_product_orders.extend(supply_orders[:i])

            for order in supply_orders[i:]:
                order["accepted_volume"] = 0
            for order in demand_orders[i:]:
                order["accepted_volume"] = 0

            rejected_orders.extend(demand_orders[i:])
            rejected_orders.extend(supply_orders[i:])

        accepted_supply_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] > 0
        ]
        accepted_demand_orders = [
            x for x in accepted_product_orders if x["accepted_volume"] < 0
        ]
        supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
        demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
        accepted_orders.extend(accepted_product_orders)
        prices = list(map(itemgetter("accepted_price"), accepted_supply_orders))
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
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, [], meta
