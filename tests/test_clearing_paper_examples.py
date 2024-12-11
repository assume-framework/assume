# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import math
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct, Order
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import ComplexClearingRole

from .utils import extend_orderbook

simple_dayahead_auction_config = MarketConfig(
    market_id="simple_dayahead_auction",
    market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    additional_fields=["node"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        until=datetime(2005, 6, 2),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    volume_tick=0.1,
    price_unit="€/MW",
    market_mechanism="complex_clearing",
)
eps = 1e-4


def test_complex_clearing_whitepaper_a():
    """
    Example taken from Whitepaper electricity spot market design 2030-2050
    Bichler et al.
    2021
    See Figure 5 a)
    """
    market_config = copy.copy(simple_dayahead_auction_config)

    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 1

    """
    Create Orderbook with constant (for only one hour) order volumes and prices:
        - dem1: volume = -10, price = 300
        - dem2: volume = -14, price = 10
        - gen3: volume = 1, price = 40
        - block_gen4: volume = 11, price = 40
        - gen5: volume = 13, price = 100

    """
    orderbook = []
    orderbook = extend_orderbook(products, volume=-10, price=300, orderbook=orderbook)
    orderbook = extend_orderbook(products, volume=-14, price=10, orderbook=orderbook)
    orderbook = extend_orderbook(products, volume=1, price=40, orderbook=orderbook)
    orderbook = extend_orderbook(
        products, volume=11, price=40, orderbook=orderbook, bid_type="BB"
    )
    orderbook = extend_orderbook(products, volume=13, price=100, orderbook=orderbook)

    # bid gen1 and block_gen3 are from one unit
    orderbook[3]["agent_addr"] = orderbook[2]["agent_addr"]
    assert len(orderbook) == 5

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["supply_volume"], 10, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 40, abs_tol=eps)
    assert accepted_orders[0]["agent_addr"] == "dem1"
    assert math.isclose(accepted_orders[0]["accepted_volume"], -10, abs_tol=eps)
    assert accepted_orders[1]["agent_addr"] == "gen3"
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[0][0]], 10, abs_tol=eps
    )


def test_complex_clearing_whitepaper_d():
    """
    Example taken from Whitepaper electricity spot market design 2030-2050
    Bichler et al.
    2021
    See figure 5 d)
    """

    market_config = copy.copy(simple_dayahead_auction_config)
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 1

    """
    Create Orderbook with constant (for only one hour) order volumes and prices:
        - dem1: volume = -10, price = 300
        - dem2: volume = -14, price = 10
        - block_gen4: volume = 12, price = 40, mar=11/12
        - gen5: volume = 13, price = 100

    """
    orderbook = []
    orderbook = extend_orderbook(
        products,
        volume=-10,
        price=300,
        orderbook=orderbook,
        bid_type="SB",
        min_acceptance_ratio=0,
    )
    orderbook = extend_orderbook(products, -14, 10, orderbook, "SB")
    orderbook = extend_orderbook(
        products, 12, 40, orderbook, "BB", min_acceptance_ratio=11 / 12
    )
    orderbook = extend_orderbook(products, 13, 100, orderbook, "SB")

    assert len(orderbook) == 4

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["supply_volume"], 10, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 100, abs_tol=eps)
    assert accepted_orders[0]["agent_addr"] == "dem1"
    assert math.isclose(accepted_orders[0]["accepted_volume"], -10, abs_tol=eps)
    assert accepted_orders[1]["agent_addr"] == "gen4"
    assert math.isclose(accepted_orders[1]["accepted_volume"], 10, abs_tol=eps)


def test_clearing_non_convex_1():
    """
    Example taken from 'Pricing in non-convex markets: how to price electricity in the presence of demand response'
    Bichler et al.
    2021
    page 25
    5.1.1
    """

    market_config = copy.copy(simple_dayahead_auction_config)
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 3, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 3

    """
    Create Orderbook

    """
    orderbook_demand = []

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B1",
        "bid_id": "B1",
        "volume": {
            products[0][0]: -4,
            products[1][0]: -6,
            products[2][0]: -10,
        },
        "price": 100,
        "only_hours": None,
        "bid_type": "BB",
        "node": "node0",
    }
    orderbook_demand.append(order)

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B2",
        "bid_id": "B2",
        "volume": {
            products[0][0]: -3,
            products[1][0]: -6,
            products[2][0]: -12,
        },
        "price": 100,
        "only_hours": None,
        "bid_type": "BB",
        "node": "node0",
    }
    orderbook_demand.append(order)

    orderbook = orderbook_demand.copy()
    orderbook = extend_orderbook(
        products=products, volume=15, price=5, orderbook=orderbook
    )
    orderbook = extend_orderbook(products, volume=20, price=3, orderbook=orderbook)

    assert len(orderbook) == 8
    assert len(orderbook_demand) == 2

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["supply_volume"], 7, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 3, abs_tol=eps)
    assert math.isclose(meta[1]["supply_volume"], 12, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 3, abs_tol=eps)
    assert math.isclose(meta[2]["supply_volume"], 22, abs_tol=eps)
    assert math.isclose(meta[2]["price"], 5, abs_tol=eps)

    assert accepted_orders[0]["agent_addr"] == "B1"
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[0][0]], -4, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[1][0]], -6, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[2][0]], -10, abs_tol=eps
    )

    assert accepted_orders[1]["agent_addr"] == "B2"
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[0][0]], -3, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[1][0]], -6, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[2][0]], -12, abs_tol=eps
    )

    assert accepted_orders[2]["agent_addr"] == "gen6"
    assert math.isclose(accepted_orders[2]["accepted_volume"], 7, abs_tol=eps)

    assert accepted_orders[3]["agent_addr"] == "gen6"
    assert math.isclose(accepted_orders[3]["accepted_volume"], 12, abs_tol=eps)

    assert accepted_orders[4]["agent_addr"] == "gen3"
    assert math.isclose(accepted_orders[4]["accepted_volume"], 2, abs_tol=eps)

    assert accepted_orders[5]["agent_addr"] == "gen6"
    assert math.isclose(accepted_orders[5]["accepted_volume"], 20, abs_tol=eps)


def test_clearing_non_convex_2():
    """
    Introduce non-convexities
    5.1.2
    including mar, BB for gen7
    no load costs cannot be integrated here, so the results differ
    """

    market_config = copy.copy(simple_dayahead_auction_config)
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 3, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 3

    """
    Create Orderbook

    """
    orderbook = []

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B1",
        "bid_id": "B1",
        "volume": {
            products[0][0]: -4,
            products[1][0]: -6,
            products[2][0]: -11,
        },
        "price": 100,
        "only_hours": None,
        "bid_type": "BB",
        "min_acceptance_ratio": 0,
        "node": "node0",
    }
    orderbook.append(order)

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B2",
        "bid_id": "B2",
        "volume": {
            products[0][0]: -3,
            products[1][0]: -6,
            products[2][0]: -12,
        },
        "price": 100,
        "only_hours": None,
        "bid_type": "BB",
        "min_acceptance_ratio": 0,
        "node": "node0",
    }
    orderbook.append(order)

    orderbook = extend_orderbook(
        products=products,
        volume=15,
        price=5,
        orderbook=orderbook,
        bid_type="SB",
        min_acceptance_ratio=2 / 15,
    )
    orderbook = extend_orderbook(
        products=products,
        volume=20,
        price=3,
        orderbook=orderbook,
        bid_type="SB",
        min_acceptance_ratio=10 / 20,
    )

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert math.isclose(meta[0]["supply_volume"], 7, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 5, abs_tol=eps)
    assert math.isclose(meta[1]["supply_volume"], 12, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 3, abs_tol=eps)
    assert math.isclose(meta[2]["supply_volume"], 23, abs_tol=eps)
    assert math.isclose(meta[2]["price"], 5, abs_tol=eps)

    assert accepted_orders[0]["agent_addr"] == "B1"
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[0][0]], -4, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[1][0]], -6, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[2][0]], -11, abs_tol=eps
    )

    assert accepted_orders[1]["agent_addr"] == "B2"
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[0][0]], -3, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[1][0]], -6, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[2][0]], -12, abs_tol=eps
    )

    assert accepted_orders[2]["agent_addr"] == "gen3"
    assert math.isclose(accepted_orders[2]["accepted_volume"], 7, abs_tol=eps)
    assert accepted_orders[3]["agent_addr"] == "gen6"
    assert math.isclose(accepted_orders[3]["accepted_volume"], 12, abs_tol=eps)
    assert accepted_orders[4]["agent_addr"] == "gen3"
    assert math.isclose(accepted_orders[4]["accepted_volume"], 3, abs_tol=eps)
    assert accepted_orders[5]["agent_addr"] == "gen6"
    assert math.isclose(accepted_orders[5]["accepted_volume"], 20, abs_tol=eps)


def test_clearing_non_convex_3():
    """
    5.1.3 price sensitive demand

    half of the demand bids are elastic
    """

    market_config = copy.copy(simple_dayahead_auction_config)
    market_config.market_products = [
        MarketProduct(timedelta(hours=1), 3, timedelta(hours=1))
    ]
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
    ]
    next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
    products = get_available_products(market_config.market_products, next_opening)
    assert len(products) == 3

    """
    Create Orderbook

    """
    orderbook = []

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B1",
        "bid_id": "B1_1",
        "volume": {
            products[0][0]: -2,
            products[1][0]: -3,
            products[2][0]: -5.5,
        },
        "price": 100,
        "only_hours": None,
        "bid_type": "BB",
        "min_acceptance_ratio": 0,
        "node": "node0",
    }
    orderbook.append(order)

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B2",
        "bid_id": "B2_1",
        "volume": {
            products[0][0]: -1.5,
            products[1][0]: -3,
            products[2][0]: -6,
        },
        "price": 100,
        "only_hours": None,
        "bid_type": "BB",
        "min_acceptance_ratio": 0,
        "node": "node0",
    }
    orderbook.append(order)

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B1",
        "bid_id": "B1_2",
        "volume": {
            products[0][0]: -2,
            products[1][0]: -3,
            products[2][0]: -5.5,
        },
        "price": 10,
        "only_hours": None,
        "bid_type": "BB",
        "min_acceptance_ratio": 0,
        "node": "node0",
    }
    orderbook.append(order)

    order: Order = {
        "start_time": products[0][0],
        "end_time": products[0][1],
        "agent_addr": "B2",
        "bid_id": "B2_2",
        "volume": {
            products[0][0]: -1.5,
            products[1][0]: -3,
            products[2][0]: -6,
        },
        "price": 2,
        "only_hours": None,
        "bid_type": "BB",
        "min_acceptance_ratio": 0,
        "node": "node0",
    }
    orderbook.append(order)

    orderbook = extend_orderbook(
        products=products,
        volume=15,
        price=5,
        orderbook=orderbook,
        bid_type="SB",
        min_acceptance_ratio=2 / 15,
    )
    orderbook = extend_orderbook(
        products=products,
        volume=20,
        price=3,
        orderbook=orderbook,
        bid_type="SB",
        min_acceptance_ratio=10 / 20,
    )

    mr = ComplexClearingRole(market_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)

    assert math.isclose(meta[0]["supply_volume"], 5.5, abs_tol=eps)
    assert math.isclose(meta[0]["price"], 5, abs_tol=eps)
    assert math.isclose(meta[1]["supply_volume"], 9, abs_tol=eps)
    assert math.isclose(meta[1]["price"], 5, abs_tol=eps)
    assert math.isclose(meta[2]["supply_volume"], 17, abs_tol=eps)
    assert math.isclose(meta[2]["price"], 3, abs_tol=eps)

    assert accepted_orders[0]["bid_id"] == "B1_1"
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[0][0]], -2, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[1][0]], -3, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[0]["accepted_volume"][products[2][0]], -5.5, abs_tol=eps
    )

    assert accepted_orders[1]["bid_id"] == "B2_1"
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[0][0]], -1.5, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[1][0]], -3, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[1]["accepted_volume"][products[2][0]], -6, abs_tol=eps
    )

    assert accepted_orders[2]["bid_id"] == "B1_2"
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[0][0]], -2, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[1][0]], -3, abs_tol=eps
    )
    assert math.isclose(
        accepted_orders[2]["accepted_volume"][products[2][0]], -5.5, abs_tol=eps
    )

    assert accepted_orders[3]["agent_addr"] == "gen5"
    assert math.isclose(accepted_orders[3]["accepted_volume"], 5.5, abs_tol=eps)
    assert accepted_orders[4]["agent_addr"] == "gen5"
    assert math.isclose(accepted_orders[4]["accepted_volume"], 9, abs_tol=eps)
    assert accepted_orders[5]["agent_addr"] == "gen8"
    assert math.isclose(accepted_orders[5]["accepted_volume"], 17, abs_tol=eps)

    """
    5.1.4 Flexible demand
    can only be implemented with exclusive bids
    """


if __name__ == "__main__":
    pass
    # from assume.common.utils import plot_orderbook
    # clearing_result, meta, flows = test_market_mechanism()
    # only works with per node clearing
    # fig, ax = plot_orderbook(clearing_result, meta)
    # fig.show()
