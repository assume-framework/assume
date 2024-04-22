# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms.complex_clearing_dmas import (
    ComplexDmasClearingRole,
)

start = datetime(2020, 1, 1)
end = datetime(2020, 12, 2)

simple_dayahead_auction_config = MarketConfig(
    market_id="simple_dayahead_auction",
    market_products=[MarketProduct(rd(hours=+1), 2, rd(hours=1))],
    additional_fields=["exclusive_id", "link", "block_id"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)


def test_dmas_market_init():
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 2


def test_insufficient_generation():
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )

    start = products[0][0]
    end = products[-1][1]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 30,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 60,
            "agent_id": "gen1",
            "bid_id": "bid2",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -201,
            "price": 700,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    assert meta[0]["price"] == 60
    assert meta[0]["supply_volume"] == 200
    assert meta[0]["demand_volume"] == 201


def test_remaining_generation():
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )

    start = products[0][0]
    end = products[-1][1]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 30,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 60,
            "agent_id": "gen1",
            "bid_id": "bid2",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 50,
            "price": 40,
            "agent_id": "gen1",
            "bid_id": "bid3",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 50,
            "price": 45,
            "agent_id": "gen1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -200,
            "price": 700,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    assert meta[0]["price"] == 45
    assert meta[0]["supply_volume"] == 200
    assert meta[0]["demand_volume"] == 200


def test_link_order():
    # test not taking a linked order.
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )

    start = products[0][0]
    end = products[-1][1]

    # cheap bid2 can not be taken, as it would require taking expensive bid1 - which would raise the price to 60 instead of 40
    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 60,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": 0,
            "link": -1,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 0,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": 1,
            "link": 0,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 200,
            "price": 40,
            "agent_id": "gen1",
            "bid_id": "bid3",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -200,
            "price": 700,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    # should actually clear to 40, link orders are limited for same hour
    assert meta[0]["price"] == 60
    assert meta[0]["supply_volume"] == 200
    assert meta[0]["demand_volume"] == 200


def test_use_link_order():
    # test taking a linked order - use more expensive hour 0 to have cheaper overall dispatch.
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )

    start1 = products[0][0]
    end1 = products[0][1]
    start2 = products[1][0]
    end2 = products[1][1]

    orderbook: Orderbook = [
        {
            "start_time": start1,
            "end_time": end1,
            "volume": 100,
            "price": 60,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": 0,
            "link": -1,
        },
        {
            "start_time": start2,
            "end_time": end2,
            "volume": 100,
            "price": 0,
            "agent_id": "gen1",
            "bid_id": "bid2",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": 1,
            "link": 0,
        },
        {
            "start_time": start1,
            "end_time": end1,
            "volume": 100,
            "price": 40,
            "agent_id": "gen1",
            "bid_id": "bid3",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start2,
            "end_time": end2,
            "volume": 100,
            "price": 100,
            "agent_id": "gen1",
            "bid_id": "bid3",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start1,
            "end_time": end1,
            "volume": -100,
            "price": 700,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start2,
            "end_time": end2,
            "volume": -100,
            "price": 700,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    assert meta[1]["price"] == 0
    assert meta[0]["price"] == 60
    assert meta[0]["supply_volume"] == 100
    assert meta[0]["demand_volume"] == 100
    assert meta[1]["supply_volume"] == 100
    assert meta[1]["demand_volume"] == 100


def test_market():
    """
    For debugging, the following might help:
    from pyomo.environ import value
    sinks = [value(model.sink[key]) for key in model.sink]
    sources = [value(model.source[key]) for key in model.source]
    [model.use_hourly_ask[(block, hour, agent)].value for block, hour, agent in orders["single_ask"].keys()]
    """
    next_opening = simple_dayahead_auction_config.opening_hours.after(datetime.now())
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )

    print(products)
    start = products[0][0]
    end = products[-1][1]

    orderbook: Orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 180.0,
            "price": 58,
            "agent_id": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 10,
            "price": 90,
            "agent_id": "gen1",
            "bid_id": "bid2",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 10,
            "price": 19,
            "agent_id": "gen1",
            "bid_id": "bid5",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 10,
            "price": 53,
            "agent_id": "gen1",
            "bid_id": "bid3",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -500,
            "price": 700,
            "agent_id": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta = mr.clear(orderbook, products)
    assert meta[0]["price"] == 90
    assert meta[0]["supply_volume"] == 210
    assert meta[0]["demand_volume"] == 500


# [{'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 34.63333333333333, 'volume': 1000.0, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'Operator 7'), 'bid_id': 'Unit 7_1', 'unit_id': 'Unit 7'}, {'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 25.65, 'volume': 500.0, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'Operator 2'), 'bid_id': 'Unit 2_1', 'unit_id': 'Unit 2'}, {'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 53.50000000000001, 'volume': 1000.0, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'Operator 4'), 'bid_id': 'Unit 4_1', 'unit_id': 'Unit 4'}, {'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 3000.0, 'volume': -2082.7, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'eom_de'), 'bid_id': 'demand_EOM_1', 'unit_id': 'demand_EOM'}, {'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 45.05, 'volume': 1000.0, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'Operator 3'), 'bid_id': 'Unit 3_1', 'unit_id': 'Unit 3'}, {'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 13.633333333333335, 'volume': 500.0, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'Operator 1'), 'bid_id': 'Unit 1_1', 'unit_id': 'Unit 1'}, {'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 43.63333333333334, 'volume': 1000.0, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'Operator 1'), 'bid_id': 'Unit 5_1', 'unit_id': 'Unit 5'}, {'start_time': datetime.datetime(2019, 1, 1, 1, 0), 'end_time': datetime.datetime(2019, 1, 1, 2, 0), 'only_hours': None, 'price': 33.63333333333333, 'volume': 1000.0, 'link': None, 'block_id': None, 'exclusive_id': None, 'agent_id': ('world', 'Operator 6'), 'bid_id': 'Unit 6_1', 'unit_id': 'Unit 6'}]
