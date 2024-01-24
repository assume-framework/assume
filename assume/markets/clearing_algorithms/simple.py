# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import timedelta
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

log = logging.getLogger(__name__)


def calculate_meta(accepted_supply_orders, accepted_demand_orders, product):
    supply_volume = sum(map(itemgetter("accepted_volume"), accepted_supply_orders))
    demand_volume = -sum(map(itemgetter("accepted_volume"), accepted_demand_orders))
    prices = list(map(itemgetter("accepted_price"), accepted_supply_orders)) or [0]
    # can also be self.marketconfig.maximum_bid..?
    duration_hours = (product[1] - product[0]) / timedelta(hours=1)
    avg_price = 0
    if supply_volume:
        weighted_price = [
            order["accepted_volume"] * order["accepted_price"]
            for order in accepted_supply_orders
        ]
        avg_price = sum(weighted_price) / supply_volume
    return {
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


class PayAsClearRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        Performs electricity market clearing using a pay-as-clear mechanism. This means that the clearing price is the
        highest price that is still accepted. The clearing price is the same for all accepted orders.

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
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
            accepted_demand_orders: Orderbook = []
            accepted_supply_orders: Orderbook = []
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
                    rejected_orders.append(demand_order)
                    continue

                # assert dem_vol == gen_vol
                # now add the next demand order
                dem_vol += -demand_order["volume"]
                demand_order["accepted_volume"] = demand_order["volume"]
                # and add supply until the demand order is matched
                while supply_orders and gen_vol < dem_vol:
                    supply_order = supply_orders.pop(0)
                    if supply_order["price"] <= demand_order["price"]:
                        added = supply_order["volume"] - supply_order.get(
                            "accepted_volume", 0
                        )
                        should_insert = not supply_order.get("accepted_volume")
                        supply_order["accepted_volume"] = supply_order["volume"]
                        if should_insert:
                            accepted_supply_orders.append(supply_order)
                        gen_vol += added
                    else:
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
                    supply_order = accepted_supply_orders[-1]
                    supply_order["accepted_volume"] = supply_order["volume"] - diff

                    # changed supply_order is still part of to_commit and will be added
                    # only volume-diff can be sold for current price
                    gen_vol -= diff

                    # add left over to supply_orders again
                    supply_orders.insert(0, supply_order)
                    demand_order["accepted_volume"] = demand_order["volume"]
                else:
                    demand_order["accepted_volume"] = demand_order["volume"]

                accepted_demand_orders.append(demand_order)

            for order in supply_orders:
                # if the order was not accepted partially, it is rejected
                if not order.get("accepted_volume"):
                    rejected_orders.append(order)

            # set clearing price - merit order - uniform pricing
            if accepted_supply_orders:
                clear_price = float(
                    max(map(itemgetter("price"), accepted_supply_orders))
                )
            else:
                clear_price = 0

            accepted_product_orders = accepted_demand_orders + accepted_supply_orders
            for order in accepted_product_orders:
                order["accepted_price"] = clear_price
            accepted_orders.extend(accepted_product_orders)

            meta.append(
                calculate_meta(
                    accepted_supply_orders,
                    accepted_demand_orders,
                    product,
                )
            )

        return accepted_orders, rejected_orders, meta


class PayAsBidRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        Simulates electricity market clearing using a pay-as-bid mechanism.

        :param market_products: The products to be traded
        :type market_products: list[MarketProduct]
        :return: accepted_orders, rejected_orders, meta
        :rtype: tuple[Orderbook, Orderbook, list[dict]]
        """
        market_getter = itemgetter("start_time", "end_time", "only_hours")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
            accepted_demand_orders: Orderbook = []
            accepted_supply_orders: Orderbook = []
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
                accepted_supply_orders.extend(to_commit)

            for order in supply_orders:
                rejected_orders.append(order)

            accepted_product_orders = accepted_demand_orders + accepted_supply_orders

            accepted_orders.extend(accepted_product_orders)
            meta.append(
                calculate_meta(
                    accepted_supply_orders,
                    accepted_demand_orders,
                    product,
                )
            )
        return accepted_orders, rejected_orders, meta
