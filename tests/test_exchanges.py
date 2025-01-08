# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import NaiveSingleBidStrategy
from assume.units.exchanges import Exchanges


def test_exchanges_export():
    strategies = {"EOM": NaiveSingleBidStrategy()}

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
    forecaster = NaiveForecast(index, exp=100)
    exch = Exchanges(
        id="exp",
        direction="export",
        unit_operator="eom_de",
        technology="energy",
        bidding_strategies=strategies,
        max_power=1000000,
        min_power=0,
        forecaster=forecaster,
        price=2999,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    _, max_power = exch.calculate_min_max_power(start, end)

    assert max_power.max() == -100
    assert exch.calculate_marginal_cost(start, max_power.max()) == 2999

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = exch.calculate_bids(mc, [product_tuple])

    for bid in bids:
        assert "start_time" in bid.keys()
        assert "end_time" in bid.keys()
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1


def test_exchanges_import():
    strategies = {"EOM": NaiveSingleBidStrategy()}

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
    forecaster = NaiveForecast(index, imp=100)
    exch = Exchanges(
        id="imp",
        direction="import",
        unit_operator="eom_de",
        technology="energy",
        bidding_strategies=strategies,
        max_power=1000000,
        min_power=0,
        forecaster=forecaster,
        price=0,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    _, max_power = exch.calculate_min_max_power(start, end)

    assert max_power.max() == 100
    assert exch.calculate_marginal_cost(start, max_power.max()) == 0

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = exch.calculate_bids(mc, [product_tuple])

    for bid in bids:
        assert "start_time" in bid.keys()
        assert "end_time" in bid.keys()
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1


if __name__ == "__main__":
    test_exchanges_export()
    test_exchanges_import()
