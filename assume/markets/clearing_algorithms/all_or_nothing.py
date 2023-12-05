# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from itertools import groupby
from operator import itemgetter

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

from .simple import calculate_meta

log = logging.getLogger(__name__)


def cumsum(orderbook: Orderbook):
    """
    This function adds a cumsum field to the orderbook.
    """
    sum_ = 0
    for order in orderbook:
        sum_ += order["volume"]
        order["cumsum"] = sum_
    return orderbook


# does not allow to have partially accepted bids
# all or nothing
class PayAsClearAonRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        This implements pay-as-clear where each bids volume needs an exactly matching order with the same volume.
        Partial clearing is not allowed here.
        This has the side effect, that the cleared price can be much higher if bids with different volume are accepted

        :param market_agent: The market agent
        :type market_agent: MarketRole
        :param market_products: The products to be traded
        :type market_products: list[MarketProduct]
        """
        market_getter = itemgetter("start_time", "end_time", "only_hours")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []
        clear_price = self.marketconfig.minimum_bid_price
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
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
                rejected_orders.extend(demand_orders[i:])
                rejected_orders.extend(supply_orders[i:])

            for order in accepted_product_orders:
                order["accepted_price"] = clear_price
            accepted_orders.extend(accepted_product_orders)

            accepted_supply_orders = [
                x for x in accepted_product_orders if x["accepted_volume"] > 0
            ]
            accepted_demand_orders = [
                x for x in accepted_product_orders if x["accepted_volume"] < 0
            ]
            meta.append(
                calculate_meta(
                    accepted_supply_orders,
                    accepted_demand_orders,
                    product,
                )
            )
        # accepted orders can not be used in future
        return accepted_orders, rejected_orders, meta


# does not allow to have partial accepted bids
class PayAsBidAonRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> (Orderbook, Orderbook, list[dict]):
        """
        This implements pay-as-bid where each bids volume needs an exactly matching order with the same volume.
        Partial clearing is not allowed here.

        :param market_agent: The market agent
        :type market_agent: MarketRole
        :param market_products: The products to be traded
        :type market_products: list[MarketProduct]
        """
        market_getter = itemgetter("start_time", "end_time", "only_hours")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
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
                rejected_orders.extend(demand_orders[i:])
                rejected_orders.extend(supply_orders[i:])

            accepted_orders.extend(accepted_product_orders)

            # meta calculation
            accepted_supply_orders = [
                x for x in accepted_product_orders if x["accepted_volume"] > 0
            ]
            accepted_demand_orders = [
                x for x in accepted_product_orders if x["accepted_volume"] < 0
            ]
            meta.append(
                calculate_meta(
                    accepted_supply_orders,
                    accepted_demand_orders,
                    product,
                )
            )
        return accepted_orders, rejected_orders, meta
