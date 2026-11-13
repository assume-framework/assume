# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from datetime import datetime, timedelta

import pandas as pd
import pytest
from dateutil import rrule as rr

from assume.common.exceptions import ValidationError
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


def test_demand_has_no_costs():
    index = pd.date_range(start=datetime(2023, 7, 1), periods=3, freq="1h")
    dem = Demand(
        id="demand",
        unit_operator="UO1",
        technology="energy",
        bidding_strategies={"EOM": EnergyNaiveStrategy()},
        max_power=-150,
        min_power=0,
        forecaster=DemandForecaster(index, demand=-150),
        price=2000,
    )
    dem.outputs["energy"].loc[index[1] : index[2]] = -150

    dem.calculate_costs(index[0], index[2])

    # written as 0, as the dashboards calculate with this column
    assert "energy_generation_costs" in dem.outputs
    assert dem.outputs["energy_generation_costs"].at[index[1]] == 0
    assert dem.outputs["total_costs"].at[index[1]] == 0


if __name__ == "__main__":
    test_demand()


def _demand_kwargs(**overrides):
    index = pd.date_range(start=datetime(2023, 7, 1), periods=24, freq="1h")
    return {
        "id": "dem_test",
        "unit_operator": "operator",
        "technology": "energy",
        "bidding_strategies": {"EOM": EnergyNaiveStrategy()},
        "forecaster": DemandForecaster(index, demand=-100),
        "max_power": -200,
        "min_power": -100,
        "price": 1000,
    } | overrides


def test_invalid_demand_reports_all_violations():
    with pytest.raises(ValidationError) as exc:
        Demand(**_demand_kwargs(max_power=10, min_power=20))
    assert exc.value.id == "dem_test"
    assert exc.value.field == "max_power"
    message = str(exc.value)
    assert "max_power must be < 0 but is 10 for unit dem_test" in message
    assert "min_power must be < 0 but is 20 for unit dem_test" in message


def test_demand_zero_power_is_allowed_despite_the_message():
    dem = Demand(**_demand_kwargs(max_power=0, min_power=0))
    assert dem.max_power == 0


@pytest.mark.parametrize(
    "kwargs, field",
    [
        ({"elasticity_model": "quadratic", "num_bids": 5}, "elasticity_model"),
        ({"elasticity_model": "linear", "num_bids": 1}, "num_bids"),
        ({"elasticity_model": "isoelastic", "num_bids": 3}, "elasticity"),
        ({"elasticity_model": "linear", "num_bids": 3, "max_price": -5}, "max_price"),
    ],
)
def test_invalid_elastic_demand_is_a_validation_error(kwargs, field):
    with pytest.raises(ValidationError) as exc:
        Demand(**_demand_kwargs(**kwargs))
    assert exc.value.field == field
    assert isinstance(exc.value, ValueError)


def test_elastic_parameters_are_ignored_without_elasticity_model():
    dem = Demand(**_demand_kwargs(num_bids=1, elasticity=3))
    assert dem.elasticity_model == 0
    # max_price defaults to the price
    assert dem.max_price == 1000
    assert Demand(**_demand_kwargs(max_price=50)).max_price == 50
