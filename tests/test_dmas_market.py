# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct, Orderbook
from assume.common.utils import get_available_products
from assume.markets.clearing_algorithms.complex_clearing_dmas import (
    ComplexDmasClearingRole,
)

start = datetime(2018, 1, 1)
end = datetime(2018, 1, 2)

simple_dayahead_auction_config = MarketConfig(
    market_id="simple_dayahead_auction",
    market_products=[MarketProduct(timedelta(hours=1), 2, timedelta(hours=1))],
    additional_fields=["exclusive_id", "link", "block_id"],
    opening_hours=rr.rrule(
        rr.HOURLY,
        dtstart=datetime(2005, 6, 1),
        until=datetime(2005, 6, 2),
        cache=True,
    ),
    opening_duration=timedelta(hours=1),
    volume_unit="MW",
    price_unit="€/MW",
    market_mechanism="pay_as_clear",
)


def test_dmas_market_init():
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
    products = get_available_products(
        simple_dayahead_auction_config.market_products, next_opening
    )
    assert len(products) == 2


def test_insufficient_generation():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["price"] == 60
    assert meta[0]["supply_volume"] == 200
    assert meta[0]["demand_volume"] == 201


def test_remaining_generation():
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["price"] == 45
    assert meta[0]["supply_volume"] == 200
    assert meta[0]["demand_volume"] == 200


def test_link_order():
    # test not taking a linked order.
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # should actually clear to 40, link orders are limited for same hour
    assert meta[0]["price"] == 60
    assert meta[0]["supply_volume"] == 200
    assert meta[0]["demand_volume"] == 200


def test_use_link_order():
    # test taking a linked order - use more expensive hour 0 to have cheaper overall dispatch.
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "dem1",
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
            "agent_addr": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert meta[1]["price"] == 0
    assert meta[0]["price"] == 60
    assert meta[0]["supply_volume"] == 100
    assert meta[0]["demand_volume"] == 100
    assert meta[1]["supply_volume"] == 100
    assert meta[1]["demand_volume"] == 100


def test_use_link_order2():
    # test taking a linked order - use more expensive hour 0 to have cheaper overall dispatch.
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
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
            "price": 40,
            "agent_addr": "gen1",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start2,
            "end_time": end2,
            "volume": 50,
            "price": 80,
            "agent_addr": "gen1",
            "bid_id": "bid2",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start1,
            "end_time": end1,
            "volume": 100,
            "price": 80,
            "agent_addr": "gen2",
            "bid_id": "bid1",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start2,
            "end_time": end2,
            "volume": 100,
            "price": 50,
            "agent_addr": "gen2",
            "bid_id": "bid2",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
        {
            "start_time": start2,
            "end_time": end2,
            "volume": 10,
            "price": 60,
            "agent_addr": "gen3",
            "bid_id": "bid1",
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
            "agent_addr": "dem1",
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
            "agent_addr": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert meta[1]["price"] == 50
    assert meta[0]["price"] == 40
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
    next_opening = simple_dayahead_auction_config.opening_hours.after(
        datetime(2005, 6, 1)
    )
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "dem1",
            "bid_id": "bid4",
            "only_hours": None,
            "exclusive_id": None,
            "block_id": None,
            "link": None,
        },
    ]
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["price"] == 90
    assert meta[0]["supply_volume"] == 210
    assert meta[0]["demand_volume"] == 500


def test_clearing():
    start = datetime(2018, 1, 1, 1)
    end = datetime(2018, 1, 2, 1)
    products = [(start, end, None)]
    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 0.2,
            "volume": 4900,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "renewablesDE1"),
            "bid_id": "renewablesDE1_wind_1",
            "unit_id": "renewablesDE1_wind",
        },
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 65,  # .000505,
            "volume": 81.0,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "conventionalDE1"),
            "bid_id": "conventionalDE1_gas_34_1",
            "unit_id": "conventionalDE1_gas_34",
        },
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 1000.0,
            "volume": -4832,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "demandDE1"),
            "bid_id": "demandDE11_1",
            "unit_id": "demandDE11",
        },
    ]

    simple_dayahead_auction_config.maximum_bid_price = 1e9
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["price"] == 0.2

    """
    the following shows a flaw in the usage of GLPK.
    it does not occur for highs or CBC
    maximum_bid_price should not be too high.. Some floating point issue in pyomo..?
    I don't know why this happens with GLPK
    """
    # simple_dayahead_auction_config.maximum_bid_price = 1e12
    # mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    # accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # assert meta[0]["price"] == 65


def test_clearing_multi_hours():
    start = datetime(2018, 1, 1, 1)
    end = datetime(2018, 1, 1, 2)
    start2 = datetime(2018, 1, 1, 2)
    end2 = datetime(2018, 1, 1, 3)
    products = [(start, end, None), (start2, end2, None)]
    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 0.2,
            "volume": 4900,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "renewablesDE1"),
            "bid_id": "renewablesDE1_wind_1",
            "unit_id": "renewablesDE1_wind",
        },
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 65,  # .000505,
            "volume": 81.0,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "conventionalDE1"),
            "bid_id": "conventionalDE1_gas_34_1",
            "unit_id": "conventionalDE1_gas_34",
        },
        {
            "start_time": start,
            "end_time": end,
            "only_hours": None,
            "price": 1000.0,
            "volume": -4832,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "demandDE1"),
            "bid_id": "demandDE11_1",
            "unit_id": "demandDE11",
        },
        {
            "start_time": start2,
            "end_time": end2,
            "only_hours": None,
            "price": 0.2,
            "volume": 4800,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "renewablesDE1"),
            "bid_id": "renewablesDE1_wind_2",
            "unit_id": "renewablesDE1_wind",
        },
        {
            "start_time": start2,
            "end_time": end2,
            "only_hours": None,
            "price": 65,  # .000505,
            "volume": 81.0,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "conventionalDE1"),
            "bid_id": "conventionalDE1_gas_34_2",
            "unit_id": "conventionalDE1_gas_34",
        },
        {
            "start_time": start2,
            "end_time": end2,
            "only_hours": None,
            "price": 1000.0,
            "volume": -4832,
            "node": "DE1",
            "block_id": None,
            "link": None,
            "exclusive_id": None,
            "agent_addr": ("world", "demandDE1"),
            "bid_id": "demandDE11_2",
            "unit_id": "demandDE11",
        },
    ]

    simple_dayahead_auction_config.maximum_bid_price = 1e9
    mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    assert meta[0]["price"] == 0.2
    assert meta[1]["price"] == 65
    assert meta[0]["demand_volume"] == 4832
    assert meta[1]["demand_volume"] == 4832

    """
    the following shows a flaw in the usage of GLPK.
    it does not occur for highs or CBC
    maximum_bid_price should not be too high.. Some floating point issue in pyomo..?
    I don't know why this happens with GLPK
    """
    # simple_dayahead_auction_config.maximum_bid_price = 1e12
    # mr = ComplexDmasClearingRole(simple_dayahead_auction_config)
    # accepted_orders, rejected_orders, meta, flows = mr.clear(orderbook, products)
    # assert meta[0]["price"] == 65
