# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import PayAsClearRole, clearing_mechanisms

from .utils import create_orderbook, extend_orderbook

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
    market_mechanism="pay_as_clear",
)


def test_market():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    print(products)

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -1000, 3000)
    orderbook = extend_orderbook(products, 1000, 100, orderbook)
    orderbook = extend_orderbook(products, 900, 50, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    import pandas as pd

    print(pd.DataFrame(mr.all_orders))
    print(pd.DataFrame(accepted))
    print(meta)


async def test_simple_market_mechanism():
    for name, role in clearing_mechanisms.items():
        skip = False
        for skip_name in ["complex", "nodal", "redispatch", "contract"]:
            if skip_name in name:
                skip = True
        if skip:
            continue

        print(name)
        market_config = copy.copy(simple_dayahead_auction_config)
        market_config.market_mechanism = name
        next_opening = market_config.opening_hours.after(datetime(2005, 6, 1))
        products = get_available_products(market_config.market_products, next_opening)
        assert len(products) == 1
        order = {
            "start_time": products[0][0],
            "end_time": products[0][1],
            "only_hours": products[0][2],
        }

        orderbook = create_orderbook(order, node_ids=[0, 1, 2])
        mr = role(simple_dayahead_auction_config)
        accepted, rejected, meta, flows = mr.clear(orderbook, products)
        assert meta[0]["supply_volume"] > 0
        assert meta[0]["price"] > 0
        # import pandas as pd
        # print(pd.DataFrame(mr.all_orders))
        # print(pd.DataFrame(clearing_result))
        # print(meta)

    # return mr.all_orders, meta


def test_market_pay_as_clear():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -400, 3000)
    orderbook = extend_orderbook(products, -100, 3000, orderbook)
    orderbook = extend_orderbook(products, 300, 100, orderbook)
    orderbook = extend_orderbook(products, 200, 50, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 4
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 500
    assert meta[0]["demand_volume"] == 500
    assert meta[0]["price"] == 100
    for bid in accepted:
        assert bid["volume"] == bid["accepted_volume"]


def test_market_pay_as_clears_single_demand():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -700, 3000)
    orderbook = extend_orderbook(products, 300, 100, orderbook)
    orderbook = extend_orderbook(products, 200, 50, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 3
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 500
    assert meta[0]["demand_volume"] == 500
    assert meta[0]["price"] == 100
    assert accepted[0]["volume"] == -700
    assert accepted[0]["accepted_volume"] == -500


def test_market_pay_as_clears_single_demand_more_generation():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -1000, price = 3000
        - gen1: volume = 1000, price = 100
        - gen2: volume = 900, price = 50
    """
    orderbook = extend_orderbook(products, -400, 3000)
    orderbook = extend_orderbook(products, 300, 100, orderbook)
    orderbook = extend_orderbook(products, 200, 50, orderbook)
    orderbook = extend_orderbook(products, 230, 60, orderbook)

    mr = PayAsClearRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 3
    assert len(rejected) == 1
    assert meta[0]["supply_volume"] == 400
    assert meta[0]["demand_volume"] == 400
    assert meta[0]["price"] == 60
    assert accepted[0]["volume"] == -400
    assert accepted[0]["accepted_volume"] == -400
