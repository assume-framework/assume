# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import logging
import random
from datetime import timedelta
from itertools import groupby
from operator import itemgetter

import numpy as np

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.markets.base_market import MarketRole

logger = logging.getLogger(__name__)


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
        "node": None,
        "product_start": product[0],
        "product_end": product[1],
        "only_hours": product[2],
    }


def clear_merit_order(
    supply_price: np.ndarray,
    supply_volume: np.ndarray,
    demand_price: np.ndarray,
    demand_volume: np.ndarray,
    rng: np.random.Generator | None = None,
    tolerance: float = 1e-9,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Clears a single product with a uniform (pay-as-clear) price.

    Supply bids are sorted by ascending price, demand bids by descending price. Both are then
    matched against each other as long as the supply price does not exceed the demand price.
    Price ties are traded, so the accepted volume is maximised on horizontal overlaps.

    The last accepted supply bid sets the clearing price. Only during supply scarcity the last
    accepted demand bid sets the price (never below the supply price), which is the case if

    - the marginal demand bid is only partially accepted, or
    - the supply is exhausted exactly with a demand bid, while further demand bids are left.

    Args:
        supply_price (np.ndarray): prices of the supply bids
        supply_volume (np.ndarray): positive volumes of the supply bids
        demand_price (np.ndarray): prices of the demand bids
        demand_volume (np.ndarray): positive volumes of the demand bids
        rng (np.random.Generator | None): breaks price ties randomly if given (anything with a
            ``random(size)`` method, e.g. ``np.random``), otherwise ties keep the input order
        tolerance (float): volume below which a bid counts as fully accepted (float residue);
            bids with a volume below it are ignored

    Returns:
        tuple[float, np.ndarray, np.ndarray]: clearing price, accepted supply volumes and accepted
        demand volumes (positive, in input order)
    """
    supply_price = np.asarray(supply_price, dtype=float)
    supply_volume = np.asarray(supply_volume, dtype=float)
    demand_price = np.asarray(demand_price, dtype=float)
    demand_volume = np.asarray(demand_volume, dtype=float)

    # indices of the bids taking part in the clearing, in merit order
    supply_idx = np.flatnonzero(supply_volume > tolerance)
    supply_idx = supply_idx[_argsort_prices(supply_price[supply_idx], rng)]
    demand_idx = np.flatnonzero(demand_volume > tolerance)
    demand_idx = demand_idx[_argsort_prices(-demand_price[demand_idx], rng)]

    clear_price, sorted_supply_accepted, sorted_demand_accepted = _match_sorted_bids(
        supply_price[supply_idx].tolist(),
        supply_volume[supply_idx].tolist(),
        demand_price[demand_idx].tolist(),
        demand_volume[demand_idx].tolist(),
        tolerance,
    )

    supply_accepted = np.zeros(len(supply_price))
    supply_accepted[supply_idx] = sorted_supply_accepted
    demand_accepted = np.zeros(len(demand_price))
    demand_accepted[demand_idx] = sorted_demand_accepted
    return clear_price, supply_accepted, demand_accepted


def _argsort_prices(price: np.ndarray, rng) -> np.ndarray:
    """Indices sorting the prices ascending, with ties broken randomly if a rng is given."""
    if rng is None:
        return np.argsort(price, kind="stable")
    # lexsort sorts by the last key first, the random key only decides among equal prices
    return np.lexsort((rng.random(len(price)), price))


def _match_sorted_bids(
    supply_price: list[float],
    supply_volume: list[float],
    demand_price: list[float],
    demand_volume: list[float],
    tolerance: float,
) -> tuple[float, list[float], list[float]]:
    """
    Matches the supply (ascending prices) against the demand (descending prices) bid by bid.

    Returns the clearing price and the accepted volumes in the given (sorted) order.
    """
    n_supply, n_demand = len(supply_price), len(demand_price)
    supply_accepted = [0.0] * n_supply
    demand_accepted = [0.0] * n_demand
    clear_price = 0.0

    # i and j point to the current supply and demand bid
    i = j = 0
    while i < n_supply and j < n_demand and supply_price[i] <= demand_price[j]:
        traded = min(
            supply_volume[i] - supply_accepted[i],
            demand_volume[j] - demand_accepted[j],
        )
        supply_accepted[i] += traded
        demand_accepted[j] += traded
        clear_price = supply_price[i]

        # move on to the next bid on each side which is fully accepted
        if supply_volume[i] - supply_accepted[i] <= tolerance:
            supply_accepted[i] = supply_volume[i]
            i += 1
        if demand_volume[j] - demand_accepted[j] <= tolerance:
            demand_accepted[j] = demand_volume[j]
            j += 1

    # supply scarcity: demand bid j is not fully served, then the demand sets the price
    if j < n_demand:
        if demand_accepted[j] > tolerance:
            # the marginal demand bid is partially accepted
            clear_price = max(clear_price, demand_price[j])
        elif i == n_supply and j > 0:
            # the supply was exhausted exactly with the previous demand bid
            clear_price = max(clear_price, demand_price[j - 1])

    return clear_price, supply_accepted, demand_accepted


class PayAsClearRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Performs electricity market clearing using a pay-as-clear mechanism. This means that the clearing price is the
        highest supply price that is still accepted. Only during supply scarcity, the last accepted demand order sets the price.
        The clearing price is the same for all accepted orders. See :func:`clear_merit_order` for the clearing itself.

        Args:
            orderbook (Orderbook): the orders to be cleared as an orderbook
            market_products (list[MarketProduct]): the list of products which are cleared in this clearing

        Returns:
            tuple: accepted orderbook, rejected orderbook and clearing meta data
        """
        market_getter = itemgetter("start_time", "end_time", "only_hours")
        accepted_orders: Orderbook = []
        rejected_orders: Orderbook = []
        meta = []
        orderbook.sort(key=market_getter)
        for product, product_orders in groupby(orderbook, market_getter):
            product_orders = list(product_orders)
            if product not in market_products:
                # logger.debug(f'found unwanted bids for {product} should be {market_products}')
                continue

            prices = np.array([x["price"] for x in product_orders], dtype=float)
            volumes = np.array([x["volume"] for x in product_orders], dtype=float)
            supply, demand = volumes > 0, volumes < 0
            # volume 0 is ignored/invalid

            # price ties are broken randomly, np.random is seeded by set_random_seed
            clear_price, supply_accepted, demand_accepted = clear_merit_order(
                supply_price=prices[supply],
                supply_volume=volumes[supply],
                demand_price=prices[demand],
                demand_volume=-volumes[demand],
                rng=np.random,
            )
            accepted_volumes = np.zeros(len(product_orders))
            accepted_volumes[supply] = supply_accepted
            # demand volumes are negative
            accepted_volumes[demand] -= demand_accepted

            accepted_supply_orders: Orderbook = []
            accepted_demand_orders: Orderbook = []
            for order, volume in zip(product_orders, accepted_volumes.tolist()):
                order["accepted_volume"] = volume
                order["accepted_price"] = clear_price
                if volume > 0:
                    accepted_supply_orders.append(order)
                elif volume < 0:
                    accepted_demand_orders.append(order)
                else:
                    rejected_orders.append(order)
            accepted_orders.extend(accepted_demand_orders + accepted_supply_orders)

            meta.append(
                calculate_meta(
                    accepted_supply_orders,
                    accepted_demand_orders,
                    product,
                )
            )

        # write network flows here if applicable
        flows = []

        return accepted_orders, rejected_orders, meta, flows


class PayAsBidRole(MarketRole):
    def __init__(self, marketconfig: MarketConfig):
        super().__init__(marketconfig)

    def clear(
        self, orderbook: Orderbook, market_products: list[MarketProduct]
    ) -> tuple[Orderbook, Orderbook, list[dict]]:
        """
        Simulates electricity market clearing using a pay-as-bid mechanism.

        Args:
            orderbook (Orderbook): the orders to be cleared as an orderbook
            market_products (list[MarketProduct]): the list of products which are cleared in this clearing

        Returns:
            tuple[Orderbook, Orderbook, list[dict]]: accepted orderbook, rejected orderbook and clearing meta data
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
                # logger.debug(f'found unwanted bids for {product} should be {market_products}')
                continue

            product_orders = list(product_orders)
            supply_orders = [x for x in product_orders if x["volume"] > 0]
            demand_orders = [x for x in product_orders if x["volume"] < 0]
            # volume 0 is ignored/invalid

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
                    # if no more generation - continue
                    # reject left over demand at the end
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
                    demand_order["accepted_volume"] = demand_order["volume"] - diff
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

                if demand_order["accepted_volume"]:
                    accepted_demand_orders.append(demand_order)
                # pay as bid
                for supply_order in to_commit:
                    supply_order["accepted_price"] = supply_order["price"]

                    demand_order["accepted_price"] = supply_order["price"]
                accepted_supply_orders.extend(to_commit)

            # if demand is fulfilled, we do have some additional supply orders
            # these will be rejected
            for order in product_orders:
                # if the order was not accepted partially, it is rejected
                if not order.get("accepted_volume") and order not in rejected_orders:
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

        # write network flows here if applicable
        flows = []

        return accepted_orders, rejected_orders, meta, flows
