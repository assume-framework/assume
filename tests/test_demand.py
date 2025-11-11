# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.forecaster import DemandForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import EnergyHeuristicElasticStrategy, EnergyNaiveStrategy
from assume.units.demand import Demand


def test_demand():
    strategies = {"EOM": EnergyNaiveStrategy()}

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
    forecaster = DemandForecaster(index, demand=-150)
    dem = Demand(
        id="demand",
        unit_operator="UO1",
        technology="energy",
        bidding_strategies=strategies,
        max_power=-150,
        min_power=0,
        forecaster=forecaster,
        price=2000,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    _, max_power = dem.calculate_min_max_power(start, end)

    assert max_power.max() == -150
    assert dem.calculate_marginal_cost(start, max_power.max()) == 2000

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, [product_tuple])

    for bid in bids:
        assert "start_time" in bid.keys()
        assert "end_time" in bid.keys()
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1


def test_demand_series():
    strategies = {"EOM": EnergyNaiveStrategy()}

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

    demand = pd.Series(-100, index=index)
    demand.iloc[1] = -80
    price = pd.Series(1000, index=index)
    price.iloc[1] = 0

    forecaster = DemandForecaster(index, demand=demand)
    dem = Demand(
        id="demand",
        unit_operator="UO1",
        technology="energy",
        bidding_strategies=strategies,
        max_power=-150,
        min_power=0,
        forecaster=forecaster,
        price=price,
    )
    start = product_tuple[0]
    end = product_tuple[1]
    _, max_power = dem.calculate_min_max_power(start, end)

    # power should be the highest demand which is used throughout the period
    # in our case 80 MW
    max_power = max_power.max()
    assert max_power == -80

    assert dem.calculate_marginal_cost(start, max_power) == 0
    assert dem.calculate_marginal_cost(end, max_power) == 1000

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, [product_tuple])

    for bid in bids:
        assert "price" in bid.keys()
        assert "volume" in bid.keys()
    assert len(bids) == 1
    assert bids[0]["volume"] == max_power
    assert bids[0]["price"] == price.iloc[1]


def test_demand_energy_heuristic_elastic_config_and_errors():
    strategies = {"EOM": EnergyHeuristicElasticStrategy()}

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
    forecaster = DemandForecaster(index, demand=-100)

    # Valid elastic demand (isoelastic model)
    dem = Demand(
        id="demand_energy_heuristic_elastic",
        unit_operator="UO1",
        technology="energy",
        bidding_strategies=strategies,
        max_power=-100,
        min_power=0,
        forecaster=forecaster,
        price=1000,
        elasticity=-0.5,
        elasticity_model="isoelastic",
        num_bids=4,
    )

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, [product_tuple])
    assert len(bids) == 4
    for bid in bids:
        assert "price" in bid and "volume" in bid and bid["price"] > 0

    # Valid elastic demand (linear model)
    dem = Demand(
        id="demand_energy_heuristic_elastic",
        unit_operator="UO1",
        technology="energy",
        bidding_strategies=strategies,
        max_power=-100,
        min_power=0,
        forecaster=forecaster,
        price=1000,
        elasticity_model="linear",
        num_bids=4,
    )

    mc = MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY),
        opening_duration=timedelta(hours=1),
        market_mechanism="not needed",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
    )

    bids = dem.calculate_bids(mc, [product_tuple])
    assert len(bids) == 4
    for bid in bids:
        assert "price" in bid and "volume" in bid and bid["price"] > 0

    with pytest.raises(
        ValueError,
        match="max_power must be < 0 but is 100 for unit bad_maxpower",
    ):
        Demand(
            id="bad_maxpower",
            unit_operator="UO1",
            technology="energy",
            bidding_strategies=strategies,
            max_power=100,
            min_power=0,
            forecaster=forecaster,
            price=1000,
            elasticity=0.3,
            elasticity_model="isoelastic",
            num_bids=3,
        )

    with pytest.raises(
        ValueError,
        match="min_power must be < 0 but is 100 for unit bad_minpower",
    ):
        Demand(
            id="bad_minpower",
            unit_operator="UO1",
            technology="energy",
            bidding_strategies=strategies,
            max_power=-100,
            min_power=100,
            forecaster=forecaster,
            price=1000,
            elasticity=0.3,
            elasticity_model="isoelastic",
            num_bids=3,
        )

    with pytest.raises(
        ValueError,
        match="max_power=-100 must be <= min_power=-200 for unit bad_power_diff",
    ):
        Demand(
            id="bad_power_diff",
            unit_operator="UO1",
            technology="energy",
            bidding_strategies=strategies,
            max_power=-100,
            min_power=-200,
            forecaster=forecaster,
            price=1000,
            elasticity=0.3,
            elasticity_model="isoelastic",
            num_bids=3,
        )

    # Invalid: elasticity is positive (isoelastic model)
    with pytest.raises(
        ValueError,
        match="'elasticity' parameter must be given and negative for isoelastic demand",
    ):
        Demand(
            id="bad_elasticity",
            unit_operator="UO1",
            technology="energy",
            bidding_strategies=strategies,
            max_power=-100,
            min_power=0,
            forecaster=forecaster,
            price=1000,
            elasticity=0.3,
            elasticity_model="isoelastic",
            num_bids=3,
        )

    # Invalid: slope of demand curve is positive (linear model)
    with pytest.raises(ValueError, match="Invalid slope of demand curve"):
        Demand(
            id="bad_slope",
            unit_operator="UO1",
            technology="energy",
            bidding_strategies=strategies,
            max_power=-100,
            min_power=0,
            forecaster=forecaster,
            price=-1000,
            elasticity_model="linear",
            num_bids=3,
        )

    # Invalid: num_bids < 1
    with pytest.raises(ValueError, match="'num_bids' parameter must be >= 1"):
        Demand(
            id="bad_num_bids",
            unit_operator="UO1",
            technology="energy",
            bidding_strategies=strategies,
            max_power=-100,
            min_power=0,
            forecaster=forecaster,
            price=1000,
            elasticity_model="linear",
            num_bids=0,
        )

    # Invalid: unsupported elasticity model
    with pytest.raises(ValueError, match="Invalid elasticity_model"):
        Demand(
            id="bad_model",
            unit_operator="UO1",
            technology="energy",
            bidding_strategies=strategies,
            max_power=-100,
            min_power=0,
            forecaster=forecaster,
            price=1000,
            elasticity=-1.0,
            elasticity_model="invalid_model",
            num_bids=3,
        )


if __name__ == "__main__":
    test_demand()
