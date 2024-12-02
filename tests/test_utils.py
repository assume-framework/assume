# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import calendar
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest
from dateutil import rrule as rr
from dateutil.tz import tzlocal

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import (
    aggregate_step_amount,
    convert_to_rrule_freq,
    datetime2timestamp,
    get_available_products,
    get_products_index,
    initializer,
    parse_duration,
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
            "agent_addr": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 80,
            "price": 58,
            "agent_addr": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": 100,
            "price": 53,
            "agent_addr": "gen1",
            "only_hours": None,
            "accepted_volume": 0,
        },
        {
            "start_time": start,
            "end_time": end,
            "volume": -180,
            "price": 70,
            "agent_addr": "dem1",
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
            "agent_addr": ("world", "crm_de"),
            "bid_id": "demand_CRM_pos_1",
            "unit_id": "demand_CRM_pos",
            "accepted_volume": -992.0,
            "accepted_price": 0.0,
            "market_id": "CRM_pos",
        },
    ]
    # and we calculate the amount between this bid
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
            "agent_addr": ("world", "crm_de"),
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
            "agent_addr": ("world", "crm_de"),
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
            "agent_addr": ("world", "crm_de"),
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
            "agent_addr": ("world", "crm_de"),
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
            "agent_addr": "block1",
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
            "agent_addr": "block1",
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
            "agent_addr": "gen1",
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
            "agent_addr": "mpb1",
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
            "agent_addr": "block1",
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
            "agent_addr": "block1",
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
            "agent_addr": "gen1",
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


def test_create_date_range():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 5)
    n = 1000
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)

    t = time.time()
    for i in range(n):
        index = FastIndex(start, end, freq="1h")
        fs = FastSeries(index)
    res = time.time() - t

    t = time.time()
    for i in range(n):
        q_pd = pd.date_range(start, end, freq="1h")
    res_pd = time.time() - t
    # this is sometimes faster, sometimes not
    # as a lot of objects are created
    assert res < res_pd + 0.1

    new_end = datetime(2020, 1, 1, 3)

    # check that slicing is faster
    t = time.time()
    for i in range(n):
        q_slice = fs.loc[start:new_end]
    res_slice = time.time() - t

    series = pd.Series(0, index=q_pd)

    t = time.time()
    for i in range(n):
        q_pd_slice = series.loc[start:new_end]
    res_slice_pd = time.time() - t
    # more than at least factor 5
    assert res_slice < res_slice_pd / 5

    # check that setting items is faster:
    t = time.time()
    for i in range(n):
        fs.at[start] = 1
    res_slice = time.time() - t

    series = pd.Series(0, index=q_pd)

    t = time.time()
    for i in range(n):
        series.at[start] = 1
    res_slice_pd = time.time() - t
    # more than at least factor 5
    assert res_slice < res_slice_pd / 5

    # check that setting slices is faster
    t = time.time()
    for i in range(n):
        fs.loc[start:new_end] = 17
    res_slice = time.time() - t

    series = pd.Series(0, index=q_pd)

    t = time.time()
    for i in range(n):
        series.loc[start:new_end] = 17
    res_slice_pd = time.time() - t
    # more than at least factor 5
    assert res_slice < res_slice_pd / 5

    se = pd.Series(0.0, index=fs.index.get_date_list())
    se.loc[start]

    series.loc[new_end] = 33

    fs[new_end] = 33
    new = series.loc[start:new_end][::-1]
    assert new.iloc[0] == 33
    new = fs.loc[start:new_end][::-1]
    assert new[0] == 33
    fs.data
    fs.index._get_idx_from_date(start)
    fs.index._get_idx_from_date(new_end)
    fs.data[0:4]


def test_convert_pd():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 1, 1, 5)
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)

    df = fs.as_df()
    assert isinstance(df, pd.DataFrame)


def test_set_list():
    start = datetime(2020, 1, 1, 0)
    end = datetime(2020, 2, 1, 5)
    n = 1000
    index = FastIndex(start, end, freq="1h")
    fs = FastSeries(index)

    dr = pd.date_range(start, end=datetime(2020, 1, 3, 5))
    dr = pd.Series(dr)

    series = pd.Series(0, index=pd.date_range(start, end, freq="1h"))

    t = time.time()
    for i in range(n):
        result_pd = fs[dr]
    res_fds = time.time() - t
    f_series = series > 1
    series[f_series] = 4
    fs.data[f_series] = 4

    # accessing lists or series elements is also faster
    # check getting list or series
    t = time.time()
    for i in range(n):
        result = series[dr]
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd

    # check setting list or series with single value
    t = time.time()
    for i in range(n):
        fs[dr] = 3
    res_fds = time.time() - t

    t = time.time()
    for i in range(n):
        series[dr] = 3
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd

    # check setting list or series with a series
    d_new = pd.Series(dr.index)

    t = time.time()
    for i in range(n):
        fs[dr] = d_new
    res_fds = time.time() - t

    t = time.time()
    for i in range(n):
        series[dr] = d_new
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd

    # check setting list or series with a list
    d_new = list(d_new)

    t = time.time()
    for i in range(n):
        fs[dr] = d_new
    res_fds = time.time() - t

    t = time.time()
    for i in range(n):
        series[dr] = d_new
    res_pd = time.time() - t
    print(res_fds)
    print(res_pd)
    assert res_fds < res_pd


def test_parse_duration():
    assert parse_duration("24h") == timedelta(days=1)
    assert parse_duration("1d") == timedelta(days=1)
    assert parse_duration("12h") == timedelta(hours=12)
    assert parse_duration("0.25h") == timedelta(minutes=15)
    assert parse_duration("15m") == timedelta(minutes=15)
    assert parse_duration("1m") == timedelta(minutes=1)
    assert parse_duration("10s") == timedelta(seconds=10)
    with pytest.raises(ValueError):
        parse_duration("1")
    with pytest.raises(ValueError):
        parse_duration("100ms")


if __name__ == "__main__":
    test_convert_rrule()
    test_available_products()
    test_plot_function()
    test_make_market_config()
    test_initializer()
    test_sep_block_orders()
    test_aggregate_step_amount()
