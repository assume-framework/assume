# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import calendar
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.tz import tzlocal

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import (
    aggregate_step_amount,
    convert_to_rrule_freq,
    datetime2timestamp,
    get_available_products,
    get_products_index,
    initializer,
    plot_orderbook,
    separate_orders,
    timestamp2datetime,
    visualize_orderbook,
)
from assume.scenario.loader_csv import make_market_config

from .utils import create_orderbook


def test_convert_rrule():
    freq, interval = convert_to_rrule_freq("1h")
    assert freq == rr.HOURLY
    assert interval == 1

    with pytest.raises(ValueError):
        freq, interval = convert_to_rrule_freq("h")

    freq, interval = convert_to_rrule_freq("99d")
    assert freq == rr.DAILY
    assert interval == 99


def test_make_market_config():
    market_id = "Test"
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 2)
    mc = MarketConfig(
        market_id=market_id,
        opening_hours=rr.rrule(rr.HOURLY, dtstart=start, until=end),
        opening_duration=pd.Timedelta(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[
            MarketProduct(pd.Timedelta(hours=1), 1, pd.Timedelta(hours=1))
        ],
    )
    market_params = {
        "market_id": "EOM",
        "operator": "EOM_operator",
        "product_type": "energy",
        "products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
        "opening_frequency": "1h",
        "opening_duration": "1h",
        "volume_unit": "MW",
        "maximum_bid_volume": 2000,
        "maximum_bid_price": 3000,
        "minimum_bid_price": -500,
        "price_unit": "€/MWh",
        "market_mechanism": "pay_as_clear",
    }
    mconfig = make_market_config(market_id, market_params, start, end)
    assert str(mc.opening_hours) == str(mconfig.opening_hours)
    mc.opening_hours = mconfig.opening_hours
    assert mc == mconfig


def test_available_products():
    start = datetime(2020, 1, 1)
    market_products = [
        MarketProduct(timedelta(hours=1), 1, timedelta(hours=1)),
        MarketProduct(timedelta(hours=2), 1, timedelta(hours=1)),
    ]

    products = get_available_products(market_products, start)
    assert len(products) == 2

    assert products[0] == (start + timedelta(hours=1), start + timedelta(hours=2), None)
    assert products[1] == (start + timedelta(hours=1), start + timedelta(hours=3), None)

    market_products = [
        MarketProduct(timedelta(hours=1), 24, timedelta(hours=1)),
    ]

    products = get_available_products(market_products, start)

    i = 0
    for prod in products:
        assert prod[0] == start + timedelta(hours=(1 + i)), "start {i}"
        assert prod[1] == start + timedelta(hours=(2 + i)), "end {i}"
        assert prod[2] is None, "only_hour {i}"
        i += 1


def test_aggregate_step_amount():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 2)
    step_func = aggregate_step_amount(
        [],
        start,
        end,
    )
    assert step_func == []

    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": 120,
            "price": 120,
            "agent_id": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_id": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_id": "dem1",
            "only_hours": None,
            "accepted_volume": 0,
        },
    ]

    step_func = aggregate_step_amount(orderbook, start, end)
    assert step_func


def test_aggregate_step_amount_multi_hour():
    # when we have a single multi-hour bid
    orderbook = [
        {
            "start_time": datetime(2019, 1, 3, 9, 0),
            "end_time": datetime(2019, 1, 3, 13, 0),
            "only_hours": None,
            "price": 0,
            "volume": -2756.0,
            "agent_id": ("world", "crm_de"),
            "bid_id": "demand_CRM_pos_1",
            "unit_id": "demand_CRM_pos",
            "accepted_volume": -992.0,
            "accepted_price": 0.0,
            "market_id": "CRM_pos",
        },
    ]
    # and we calculate the amount inbetween this bid
    step_func = aggregate_step_amount(
        orderbook, datetime(2019, 1, 3, 12), datetime(2019, 1, 3, 13)
    )
    # it will be empty
    assert step_func == []

    # if we calculate the step function for the whole interval
    # we get a representation of this bid
    step_func = aggregate_step_amount(
        orderbook, datetime(2019, 1, 3, 7), datetime(2019, 1, 3, 17)
    )
    assert step_func == [
        [datetime(2019, 1, 3, 9), -992.0],
        [datetime(2019, 1, 3, 13), 0.0],
    ]

    # if we select the start only - the step function goes back to this value
    step_func = aggregate_step_amount(
        orderbook, datetime(2019, 1, 3, 9), datetime(2019, 1, 3, 10)
    )
    assert step_func == [[datetime(2019, 1, 3, 9), -992.0]]

    # if we select the end only - the step function goes back to zero
    step_func = aggregate_step_amount(
        orderbook, datetime(2019, 1, 3, 13), datetime(2019, 1, 3, 14)
    )
    assert step_func == [[datetime(2019, 1, 3, 13), 0.0]]


def test_aggregate_step_amount_long():
    # when we have two valid bids and additional empty bids, this should not change much:
    orderbook = [
        {
            "start_time": datetime(2019, 1, 3, 9, 0),
            "end_time": datetime(2019, 1, 3, 13, 0),
            "only_hours": None,
            "price": 0,
            "volume": -2756.0,
            "agent_id": ("world", "crm_de"),
            "bid_id": "demand_CRM_pos_1",
            "unit_id": "demand_CRM_pos",
            "accepted_volume": -992.0,
            "accepted_price": 0.0,
            "market_id": "CRM_pos",
        },
        {
            "start_time": datetime(2019, 1, 3, 9, 0),
            "end_time": datetime(2019, 1, 3, 13, 0),
            "only_hours": None,
            "price": 0,
            "volume": -2756.0,
            "agent_id": ("world", "crm_de"),
            "bid_id": "demand_CRM_pos_1",
            "unit_id": "demand_CRM_pos",
            "accepted_volume": 0.0,
            "accepted_price": 0.0,
            "market_id": "CRM_pos",
        },
        {
            "start_time": datetime(2019, 1, 3, 13, 0),
            "end_time": datetime(2019, 1, 3, 17, 0),
            "only_hours": None,
            "price": 0,
            "volume": -2756.0,
            "agent_id": ("world", "crm_de"),
            "bid_id": "demand_CRM_pos_1",
            "unit_id": "demand_CRM_pos",
            "accepted_volume": -992.0,
            "accepted_price": 0.0,
            "market_id": "CRM_pos",
        },
        {
            "start_time": datetime(2019, 1, 3, 13, 0),
            "end_time": datetime(2019, 1, 3, 17, 0),
            "only_hours": None,
            "price": 0,
            "volume": -2756.0,
            "agent_id": ("world", "crm_de"),
            "bid_id": "demand_CRM_pos_1",
            "unit_id": "demand_CRM_pos",
            "accepted_volume": 0.0,
            "accepted_price": 0.0,
            "market_id": "CRM_pos",
        },
    ]

    # calculating the step function for the whole series
    step_func = aggregate_step_amount(
        orderbook, datetime(2019, 1, 3, 7), datetime(2019, 1, 3, 18)
    )
    assert step_func == [
        [datetime(2019, 1, 3, 9, 0), -992.0],
        [datetime(2019, 1, 3, 13, 0), -992.0],
        [datetime(2019, 1, 3, 17, 0), 0.0],
    ]
    # this returns the bids in a minimal representation


def test_initializer():
    class Test:
        @initializer
        def __init__(self, test: str, test2: str = "test2str"):
            pass

    t = Test("teststr")
    assert t.test == "teststr"
    assert t.test2 == "test2str"


def test_sep_block_orders():
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 2)
    index = pd.date_range(start, end - pd.Timedelta("1h"), freq="1h")
    orderbook = [
        {
            "start_time": start,
            "end_time": end,
            "volume": {i: 50 for i in index},
            "price": 120,
            "agent_id": "block1",
            "only_hours": None,
            "accepted_volume": {i: 0 for i in index},
            "accepted_price": {i: None for i in index},
            "bid_type": "BB",
            "bid_id": "block1_1",
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 60,
            "price": {i: 110 for i in index},
            "agent_id": "block1",
            "only_hours": None,
            "accepted_volume": {i: 0 for i in index},
            "accepted_price": {i: None for i in index},
            "bid_type": "BB",
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
            "accepted_price": None,
            "bid_type": "SB",
            "bid_id": "gen1_1",
        },
    ]
    index = pd.date_range(start, end - pd.Timedelta("1h"), freq="4h")
    orderbook.append(
        {
            "start_time": start,
            "end_time": end,
            "volume": 50,
            "price": {i: 110 for i in index},
            "agent_id": "mpb1",
            "only_hours": None,
            "accepted_volume": {i: 0 for i in index},
            "accepted_price": {i: None for i in index},
            "bid_type": "MPB",
            "bid_id": "mpb1_1",
        },
    )
    assert len(orderbook) == 4
    orderbook = separate_orders(orderbook)
    assert len(orderbook) == 55
    for order in orderbook:
        assert True not in [isinstance(order[key], dict) for key in order.keys()]


def test_get_products_index():
    index_1 = pd.date_range(
        start=datetime(2020, 1, 1, 0), end=datetime(2020, 1, 1, 5), freq="1h"
    )
    index_2 = pd.date_range(
        start=datetime(2020, 1, 1, 0), end=datetime(2020, 1, 1, 7), freq="1h"
    )
    orderbook = [
        {
            "start_time": datetime(2020, 1, 1, 0),
            "end_time": datetime(2020, 1, 1, 5),
            "volume": {i: 50 for i in index_1},
            "price": 120,
            "agent_id": "block1",
            "only_hours": None,
            "accepted_volume": {i: 0 for i in index_1},
            "accepted_price": {i: None for i in index_1},
            "bid_type": "BB",
            "bid_id": "block1_1",
        },
        {
            "start_time": datetime(2020, 1, 1, 0),
            "end_time": datetime(2020, 1, 1, 7),
            "volume": 60,
            "price": {i: 110 for i in index_2},
            "agent_id": "block1",
            "only_hours": None,
            "accepted_volume": {i: 0 for i in index_2},
            "accepted_price": {i: None for i in index_2},
            "bid_type": "BB",
        },
        {
            "start_time": datetime(2020, 1, 1, 23),
            "end_time": datetime(2020, 1, 2, 0),
            "volume": 80,
            "price": 58,
            "agent_id": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
            "accepted_price": None,
            "bid_type": "SB",
            "bid_id": "gen1_1",
        },
    ]
    assert len(orderbook) == 3
    products_index = get_products_index(orderbook)
    assert len(products_index) == 24


@patch("matplotlib.pyplot.show")
def test_plot_function(mock_pyplot):
    plot_orderbook([], [])


@patch("matplotlib.pyplot.show")
def test_visualize_function(mock_pyplot):
    orderbook = create_orderbook()
    i = -1
    for o in orderbook:
        o["link"] = i
        o["block_id"] = i + 1
        i += 1
    visualize_orderbook(orderbook)


def test_broken_timestamps():
    # timestamp is not in UTC, but is timezone-independent
    # in general, we expect timestamps to be unix-epoch timestamps
    unix_start = datetime(1970, 1, 1)

    unix_epoch_start = datetime.fromtimestamp(0)
    true_unix_epoch_start = unix_start
    offset = tzlocal().utcoffset(datetime.fromtimestamp(0))
    # this should be 1970-01-01-00-00 but it isn't (when run in CET locale)
    # so we always have this offset
    assert true_unix_epoch_start + offset == unix_epoch_start
    # however, we want to have everything in UTC
    # so we need this approach to get a datetime
    assert unix_start == datetime.fromtimestamp(0, tz=timezone.utc).replace(tzinfo=None)
    # and for the utc timestamp
    assert 0 == calendar.timegm(unix_start.utctimetuple())

    # pandas fromtimestamp has this problem too:
    assert offset + unix_start == pd.Timestamp.fromtimestamp(0)

    # though this can work
    assert unix_start == pd.Timestamp(0)

    # the other way works with pandas
    assert pd.Timestamp(unix_start).timestamp() == 0


def test_timestamp2datetime():
    unix_start = datetime(1970, 1, 1)
    assert unix_start == timestamp2datetime(0)


def test_datetime2timestamp():
    unix_start = datetime(1970, 1, 1)
    assert 0 == datetime2timestamp(unix_start)


if __name__ == "__main__":
    test_convert_rrule()
    test_available_products()
    test_plot_function()
    test_make_market_config()
    test_initializer()
    test_sep_block_orders()
    test_aggregate_step_amount()
