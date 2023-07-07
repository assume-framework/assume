from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import NaiveStrategy
from assume.units.demand import Demand


def test_demand():
    strategies = {"energy": NaiveStrategy()}

    index = pd.date_range(
        start=datetime(2023, 7, 1),
        end=datetime(2023, 7, 2),
        freq="1h",
    )
    product_tuple = (
        datetime(2023, 7, 1, hour=1),
        datetime(2023, 7, 1, hour=2),
        None,
    )
    dem = Demand(
        "test01",
        "UO1",
        "energy",
        strategies,
        index,
        150,
        0,
        150,
        price=2000,
    )
    op_window = dem.calculate_operational_window("energy", product_tuple)

    assert op_window["ops"]["max_power"]["volume"] == -150
    assert op_window["ops"]["max_power"]["cost"] == 2000

    mc = MarketConfig(
        "Test",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, product_tuple)

    for bid in bids:
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1


def test_demand_series():
    strategies = {"energy": NaiveStrategy()}

    index = pd.date_range(
        start=datetime(2023, 7, 1),
        end=datetime(2023, 7, 2),
        freq="1h",
    )
    product_tuple = (
        datetime(2023, 7, 1, hour=1),
        datetime(2023, 7, 1, hour=2),
        None,
    )

    demand = pd.Series(100, index=index)
    demand[1] = 80
    price = pd.Series(1000, index=index)
    price[1] = 0
    dem = Demand(
        "test01",
        "UO1",
        "energy",
        strategies,
        index,
        150,
        0,
        demand,
        price=price,
    )
    op_window = dem.calculate_operational_window("energy", product_tuple)

    # power should be the highest demand which is used throughout the period
    # in our case 80 MW
    assert op_window["ops"]["max_power"]["volume"] == -80
    # price is (0 + 1000) / 2 for this period
    assert op_window["ops"]["max_power"]["cost"] == 500

    mc = MarketConfig(
        "Test",
        rr.rrule(rr.HOURLY),
        timedelta(hours=1),
        "not needed",
        [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, product_tuple)

    for bid in bids:
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1
    assert bids[0]["volume"] == op_window["ops"]["max_power"]["volume"]
    assert bids[0]["price"] == op_window["ops"]["max_power"]["cost"]


if __name__ == "__main__":
    test_demand()
