from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.scenario_loader import convert_to_rrule_freq, make_market_config
from assume.common.utils import (
    aggregate_step_amount,
    get_available_products,
    initializer,
    plot_orderbook,
    separate_block_orders,
    visualize_orderbook,
)

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
    market_name = "Test"
    start = datetime(2020, 1, 1)
    end = datetime(2020, 12, 2)
    mc = MarketConfig(
        market_name,
        rr.rrule(rr.HOURLY, dtstart=start, until=end),
        pd.Timedelta(hours=1),
        "pay_as_clear",
        [MarketProduct(pd.Timedelta(hours=1), 1, pd.Timedelta(hours=1))],
    )
    market_params = {
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
    mconfig = make_market_config(market_name, market_params, start, end)
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
    index = pd.date_range(start, end - pd.Timedelta("1H"), freq="1H")
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
    assert len(orderbook) == 2
    orderbook = separate_block_orders(orderbook)
    assert len(orderbook) == 25


@patch("matplotlib.pyplot.show")
def test_plot_function(mock_pyplot):
    plot_orderbook([], [])


@patch("matplotlib.pyplot.show")
def test_plot_function(mock_pyplot):
    orderbook = create_orderbook()
    i = -1
    for o in orderbook:
        o["link"] = i
        o["block_id"] = i + 1
        i += 1
    visualize_orderbook(orderbook)


if __name__ == "__main__":
    test_convert_rrule()
    test_available_products()
    test_plot_function()
    test_make_market_config()
    test_initializer()
    test_plot_function()
