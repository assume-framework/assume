# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
from datetime import datetime, timedelta

import pytest
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import PayAsClearRole, clearing_mechanisms, PayAsBidBuildingRole, PayAsBidRole

from .utils import create_orderbook, extend_orderbook

simple_dayahead_auction_config = MarketConfig(
    market_id="simple_dayahead_auction",
    market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
    additional_fields=["node"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    volume_tick=0.1,
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)

simple_building_auction_config = MarketConfig(
    market_id="simple_building_auction",
    market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
    additional_fields=["node"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    volume_tick=0.1,
    price_unit="€/MW",
    market_mechanism="pay_as_bid_building",
)


def test_market():
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
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
        next_opening = market_config.opening_hours.after(datetime.now())
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
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
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
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
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
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
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


def test_bidding_demand_within_community_supply_external():
    # Example of demand from a building (within community) and supply from an external energy provider
    next_opening = simple_building_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(simple_building_auction_config.market_products, next_opening)

    # Create the orderbook with demand from the community (a building) and supply from an external energy provider
    orderbook = extend_orderbook(products, -500, 100, "building_demand")
    orderbook = extend_orderbook(products, 500, 130, "external_supply", orderbook)

    mr = PayAsBidBuildingRole(simple_building_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)

    assert len(accepted) == 2
    assert len(rejected) == 0

    # Check that the accepted price is set based on the supply price
    assert accepted[0]["accepted_price"] == accepted[1]["accepted_price"] == 130
    assert accepted[0]["volume"] == -500
    assert accepted[1]["volume"] == 500


def test_bidding_supply_within_community_demand_external():
    # Example of supply from a building (within community) and demand from an external energy provider
    next_opening = simple_building_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(simple_building_auction_config.market_products, next_opening)

    # Create the orderbook with supply from the community (a building) and demand from an external energy provider
    orderbook = extend_orderbook(products, -500, 70, "external_demand")
    orderbook = extend_orderbook(products, 500, 90, "building_supply", orderbook)

    mr = PayAsBidBuildingRole(simple_building_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)

    assert len(accepted) == 2
    assert len(rejected) == 0

    # Check that the accepted price is set based on the demand price (external energy provider's price)
    assert accepted[0]["accepted_price"] == accepted[1]["accepted_price"] == 70
    assert accepted[0]["volume"] == -500
    assert accepted[1]["volume"] == 500

def test_sorting_with_building_priority():
    # Create an orderbook with both building and non-building bids
    next_opening = simple_building_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(simple_building_auction_config.market_products, next_opening)

    # Creating a mix of building and non-building orders
    orderbook = extend_orderbook(products, 500, 130, "external_supply")
    orderbook = extend_orderbook(products, -400, 68, "external_demand", orderbook)
    orderbook = extend_orderbook(products, -0.2, 100, "building_demand", orderbook)
    orderbook = extend_orderbook(products, 0.3, 90,  "building_supply", orderbook)

    mr = PayAsBidBuildingRole(simple_dayahead_auction_config)
    accepted, rejected, meta, flows = mr.clear(orderbook, products)

    # accepted for the trade within the community and one additional with the 0.1 surplus with the energy provider
    assert len(accepted) == 4
    # The energy provider, does not trade with itself
    assert len(rejected) == 1

    # Ensure that building orders are prioritized over non-building orders
    assert accepted[0]["unit_id"] == "bid_building_demand"
    assert accepted[1]["unit_id"] == "bid_external_demand"
    assert accepted[2]["unit_id"] == "bid_building_supply"
    assert accepted[3]["unit_id"] == "bid_building_supply"

    # Ensure the pricing is correct
    assert accepted[0]["accepted_price"] == 90
    assert accepted[1]["accepted_price"] == 68
    assert accepted[2]["accepted_price"] == 90
    assert accepted[3]["accepted_price"] == 68

    # Ensure the volumes are correct
    assert round(accepted[0]["accepted_volume"], 1) == -0.2
    assert round(accepted[1]["accepted_volume"], 1) == -0.1
    assert round(accepted[2]["accepted_volume"], 1) == 0.2
    assert round(accepted[3]["accepted_volume"], 1) == 0.1
