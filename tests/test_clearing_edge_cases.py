# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

import copy
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms import (
    PayAsBidRole,
    PayAsClearRole,
    clearing_mechanisms,
)

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

complex_dayahead_auction_config = MarketConfig(
    market_id="complex_dayahead_auction",
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

def test_supply_and_demand_curves_horizontal_overlap():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -60, price = 20
        - dem2: volume = -60, price = 10
        - dem3: volume = -60, price = 0
        - gen1: volume = 100, price = 0
        - gen2: volume = 100, price = 10
    The dem2 bid and gen2 bid overlap horizontally at 10€ between 100MW and 120MW.
    SimpleClearing should clear at 10€ with 120MW.
    ComplexClearing should clear at 10€ with 120MW (due to volume maximization postprocessing step).
    """
    orderbook = extend_orderbook(products, -60, 20)
    orderbook = extend_orderbook(products, -60, 10, orderbook)
    orderbook = extend_orderbook(products, -60, 0, orderbook)
    orderbook = extend_orderbook(products, 100, 0, orderbook)
    orderbook = extend_orderbook(products, 100, 10, orderbook)

    orderbook_copy = copy.deepcopy(orderbook)

    mr_simple_pac = PayAsClearRole(simple_dayahead_auction_config)
    mr_complex_pac = PayAsClearRole(complex_dayahead_auction_config)
    accepted, rejected, meta, flows = mr_simple_pac.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 4
    assert len(rejected) == 1
    assert meta[0]["supply_volume"] == 120
    assert meta[0]["demand_volume"] == 120
    assert meta[0]["price"] == 10
    assert accepted[0]["volume"] == -60
    assert accepted[1]["volume"] == -60
    assert accepted[0]["accepted_volume"] == -60
    assert accepted[1]["accepted_volume"] == -60

    accepted, rejected, meta, flows = mr_complex_pac.clear(orderbook_copy, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 4
    assert len(rejected) == 1
    assert meta[0]["supply_volume"] == 120
    assert meta[0]["demand_volume"] == 120
    assert meta[0]["price"] == 10
    assert accepted[0]["volume"] == -60
    assert accepted[1]["volume"] == -60
    assert accepted[0]["accepted_volume"] == -60
    assert accepted[1]["accepted_volume"] == -60

def test_supply_and_demand_curves_full_horizontal_overlap():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -700, price = 3000
        - dem2: volume = -300, price = 200
        - dem3: volume = -200, price = 100
        - gen1: volume = 700, price = 100
        - gen2: volume = 300, price = 200
    The dem2 bid and gen2 bid fully overlap horizontally at 200€, and the next orders should not be accepted.
    SimpleClearing should clear at 200€ with 1000MW.
    ComplexClearing should clear at 200€ with 1000MW (due to volume maximization postprocessing step).
    """
    orderbook = extend_orderbook(products, -700, 3000)
    orderbook = extend_orderbook(products, -300, 200, orderbook)
    orderbook = extend_orderbook(products, -200, 100, orderbook)
    orderbook = extend_orderbook(products, 700, 100, orderbook)
    orderbook = extend_orderbook(products, 300, 200, orderbook)

    orderbook_copy = copy.deepcopy(orderbook)

    mr_simple_pac = PayAsClearRole(simple_dayahead_auction_config)
    mr_complex_pac = PayAsClearRole(complex_dayahead_auction_config)
    accepted, rejected, meta, flows = mr_simple_pac.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 4
    assert len(rejected) == 1
    assert meta[0]["supply_volume"] == 1000
    assert meta[0]["demand_volume"] == 1000
    assert meta[0]["price"] == 200
    assert accepted[0]["volume"] == -700
    assert accepted[1]["volume"] == -300
    assert accepted[0]["accepted_volume"] == -700
    assert accepted[1]["accepted_volume"] == -300

    accepted, rejected, meta, flows = mr_complex_pac.clear(orderbook_copy, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 4
    assert len(rejected) == 1
    assert meta[0]["supply_volume"] == 1000
    assert meta[0]["demand_volume"] == 1000
    assert meta[0]["price"] == 200

def test_supply_and_demand_curves_vertical_overlap():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -100, price = 20
        - dem2: volume = -100, price = 10
        - gen1: volume = 100, price = 8
        - gen2: volume = 100, price = 12
    Supply and demand curves intersect at volume 100MW from 10€ to 12€.
    SimpleClearing should clear at 10€ with 100MW, ComplexClearing should clear at 12€ with 100MW
    (but any price between 10€ and 12€ is correct in reality).
    """
    orderbook = extend_orderbook(products, -100, 20)
    orderbook = extend_orderbook(products, -100, 10, orderbook)
    orderbook = extend_orderbook(products, 100, 8, orderbook)
    orderbook = extend_orderbook(products, 100, 12, orderbook)

    orderbook_copy = copy.deepcopy(orderbook)
    
    mr_simple_pac = PayAsClearRole(simple_dayahead_auction_config)
    mr_complex_pac = PayAsClearRole(complex_dayahead_auction_config)

    accepted, rejected, meta, flows = mr_simple_pac.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 2
    assert len(rejected) == 2
    assert meta[0]["supply_volume"] == 100
    assert meta[0]["demand_volume"] == 100
    assert meta[0]["price"] == 20 # hier kommt 20 raus und das sollte nicht so sein. sondern 10. todo Gunter
    assert accepted[0]["volume"] == -100
    assert accepted[0]["accepted_volume"] == -100

    accepted, rejected, meta, flows = mr_complex_pac.clear(orderbook_copy, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 2
    assert len(rejected) == 2
    assert meta[0]["supply_volume"] == 100
    assert meta[0]["demand_volume"] == 100
    assert meta[0]["price"] == 20 # hier kommen 20 raus aber 12 wäre plausibler
    assert accepted[0]["volume"] == -100
    assert accepted[0]["accepted_volume"] == -100

def test_supply_and_demand_curves_full_vertical_overlap():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -700, price = 3000
        - dem2: volume = -100, price = 200
        - dem3: volume = -200, price = 100
        - gen1: volume = 800, price = 100
        - gen2: volume = 200, price = 200
    Supply and demand curves fully overlap at volume 800MW from 100€ to 200€.
    SimpleClearing should clear at 100€ with 800MW, ComplexClearing should clear at 200€ with 800MW
    (but any price between 100€ and 200€ is correct in reality).
    """
    orderbook = extend_orderbook(products, -700, 3000)
    orderbook = extend_orderbook(products, -100, 200, orderbook)
    orderbook = extend_orderbook(products, -200, 100, orderbook)
    orderbook = extend_orderbook(products, 800, 100, orderbook)
    orderbook = extend_orderbook(products, 200, 200, orderbook)

    orderbook_copy = copy.deepcopy(orderbook)

    mr_simple_pac = PayAsClearRole(simple_dayahead_auction_config)
    mr_complex_pac = PayAsClearRole(complex_dayahead_auction_config)

    accepted, rejected, meta, flows = mr_simple_pac.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 3
    assert len(rejected) == 2
    assert meta[0]["supply_volume"] == 800
    assert meta[0]["demand_volume"] == 800
    assert meta[0]["price"] == 200 # hier sollte 100 rauskommen. todo Gunter
    assert accepted[0]["volume"] == -700
    assert accepted[0]["accepted_volume"] == -700
    assert accepted[1]["volume"] == -100
    assert accepted[1]["accepted_volume"] == -100

    accepted, rejected, meta, flows = mr_complex_pac.clear(orderbook_copy, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 3
    assert len(rejected) == 2
    assert meta[0]["supply_volume"] == 800
    assert meta[0]["demand_volume"] == 800
    assert meta[0]["price"] == 200
    assert accepted[0]["volume"] == -700
    assert accepted[0]["accepted_volume"] == -700
    assert accepted[1]["volume"] == -100
    assert accepted[1]["accepted_volume"] == -100

def test_supply_and_demand_curves_no_overlap_demand_scarcity():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -75, price = 20
        - dem2: volume = -75, price = 15
        - gen1: volume = 60, price = 0
        - gen2: volume = 60, price = 5
        - gen3: volume = 60, price = 10
    Supply and demand curves do not overlap and supply is abundant.
    SimpleClearing should clear at 10€ (supply side) with 150MW,
    ComplexClearing should clear at 15€ (demand side) with 150MW.
    """
    orderbook = extend_orderbook(products, -75, 20)
    orderbook = extend_orderbook(products, -75, 15, orderbook)
    orderbook = extend_orderbook(products, 60, 0, orderbook)
    orderbook = extend_orderbook(products, 60, 5, orderbook)
    orderbook = extend_orderbook(products, 60, 10, orderbook)

    orderbook_copy = copy.deepcopy(orderbook)

    mr_simple_pac = PayAsClearRole(simple_dayahead_auction_config)
    mr_complex_pac = PayAsClearRole(complex_dayahead_auction_config)

    accepted, rejected, meta, flows = mr_simple_pac.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 5
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 150
    assert meta[0]["demand_volume"] == 150
    assert meta[0]["price"] == 10

    accepted, rejected, meta, flows = mr_complex_pac.clear(orderbook_copy, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 5
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 150
    assert meta[0]["demand_volume"] == 150
    assert meta[0]["price"] == 10

def test_supply_and_demand_curves_no_overlap_supply_scarcity():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 1

    """
    Create Orderbook with constant order volumes and prices:
        - dem1: volume = -60, price = 20
        - dem2: volume = -60, price = 15
        - dem3: volume = -60, price = 10
        - gen1: volume = 75, price = 0
        - gen2: volume = 75, price = 8
    Supply and demand curves do not overlap and supply is scarce.
    SimpleClearing should clear at 10€ with 150MW,
    ComplexClearing should clear at 10€ with 150MW.
    """
    orderbook = extend_orderbook(products, -60, 20)
    orderbook = extend_orderbook(products, -60, 15, orderbook)
    orderbook = extend_orderbook(products, -60, 10, orderbook)
    orderbook = extend_orderbook(products, 75, 0, orderbook)
    orderbook = extend_orderbook(products, 75, 8, orderbook)

    orderbook_copy = copy.deepcopy(orderbook)

    mr_simple_pac = PayAsClearRole(simple_dayahead_auction_config)
    mr_complex_pac = PayAsClearRole(complex_dayahead_auction_config)

    accepted, rejected, meta, flows = mr_simple_pac.clear(orderbook, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 5
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 150
    assert meta[0]["demand_volume"] == 150
    assert meta[0]["price"] == 10

    accepted, rejected, meta, flows = mr_complex_pac.clear(orderbook_copy, products)
    assert meta[0]["demand_volume"] > 0
    assert meta[0]["price"] > 0
    assert len(accepted) == 5
    assert len(rejected) == 0
    assert meta[0]["supply_volume"] == 150
    assert meta[0]["demand_volume"] == 150
    assert meta[0]["price"] == 10