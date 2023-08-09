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
    market_agent.all_orders = sorted(market_agent.all_orders, key=market_getter)
    for product, product_orders in groupby(market_agent.all_orders, market_getter):
        accepted_product_orders: Orderbook = []
        if product not in market_products:
            rejected_orders.extend(product_orders)
            # log.debug(f'found unwanted bids for {product} should be {market_products}')
            continue

        # groupby does only group consecutive groups
        product_orders = sorted(product_orders, key=lambda x: abs(x["volume"]))
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            orders = list(orders)
            demand_orders = filter(lambda x: x["volume"] < 0, orders)
            supply_orders = filter(lambda x: x["volume"] > 0, orders)
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
                    # pay as clear - all accepted receive the highest needed/cleared price
                    if clear_price < sorted_supply_orders[i]["price"]:
                        clear_price = sorted_supply_orders[i]["price"]
                else:
                    # as we have sorted before, the other bids/supply_orders can't be matched either
                    # once we get here
                    break
            # resulting i is the cut point
            accepted_product_orders.extend(sorted_demand_orders[:i])
            accepted_product_orders.extend(sorted_supply_orders[:i])
            rejected_orders.extend(sorted_demand_orders[i:])
            rejected_orders.extend(sorted_supply_orders[i:])

        for order in accepted_product_orders:
            order["price"] = clear_price

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
    return accepted_orders, meta


# does not allow to have partial accepted bids
def pay_as_bid_aon(market_agent: MarketRole, market_products: list[MarketProduct]):
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

        product_orders = sorted(product_orders, key=lambda x: abs(x["volume"]))
        for volume, orders in groupby(product_orders, lambda x: abs(x["volume"])):
            orders = list(orders)
            demand_orders = filter(lambda x: x["volume"] < 0, orders)
            supply_orders = filter(lambda x: x["volume"] > 0, orders)
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

            accepted_product_orders.extend(sorted_demand_orders[:i])
            accepted_product_orders.extend(sorted_supply_orders[:i])
            rejected_orders.extend(sorted_demand_orders[i:])
            rejected_orders.extend(sorted_supply_orders[i:])

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
    market_agent.all_orders = rejected_orders
    # accepted orders can not be used in future

    return accepted_orders, meta
