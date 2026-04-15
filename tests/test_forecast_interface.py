# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from assume.common.fast_pandas import FastIndex
from assume.common.forecast_algorithms import (
    calculate_naive_congestion_signal,
    calculate_naive_price,
    calculate_naive_price_inelastic,
    calculate_naive_renewable_utilisation,
    calculate_naive_residual_load,
    get_forecast_registries,
)
from assume.common.forecaster import (
    DemandForecaster,
    DsmUnitForecaster,
    PowerplantForecaster,
)
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.strategies import EnergyHeuristicElasticStrategy, EnergyNaiveStrategy
from assume.units import Demand, PowerPlant

path = Path("./tests/fixtures/forecast_init")

parse_date = {"index_col": "datetime", "parse_dates": ["datetime"]}


@pytest.fixture
def market_setup():
    market_configs_dict = {
        "EOM": {
            "market_id": "EOM",
            "product_type": "energy",
            "market_products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
            "opening_duration": "1h",
            "volume_unit": "MWh",
            "maximum_bid_volume": 100000,
            "maximum_bid_price": 3000,
            "minimum_bid_price": -500,
            "price_unit": "EUR/MWh",
            "market_mechanism": "pay_as_clear",
            "param_dict": {
                "grid_data": None,
            },
        }
    }

    products = [
        MarketProduct(
            duration=pd.Timedelta(product["duration"]),
            count=product["count"],
            first_delivery=pd.Timedelta(product["first_delivery"]),
        )
        for product in market_configs_dict["EOM"]["market_products"]
    ]
    market_configs_dict["EOM"]["market_products"] = products

    lines = pd.read_csv(path / "lines.csv", index_col="line")
    buses = pd.read_csv(path / "buses.csv", index_col="name")

    market_configs_dict["EOM"]["param_dict"]["grid_data"] = {
        "buses": buses,
        "lines": lines,
    }

    empty_grid_market = MarketConfig(**market_configs_dict["EOM"])
    empty_grid_market.param_dict = {"grid_data": {}}

    market_configs = {"EOM": MarketConfig(**market_configs_dict["EOM"])}
    return {
        "market_configs": market_configs.values(),
        "empty_grid_markets": {"EOM": empty_grid_market}.values(),
    }


@pytest.fixture
def index():
    return pd.DatetimeIndex(
        pd.date_range("2019-01-01", periods=24, freq="h"),
    )


@pytest.fixture
def shared_FastIndex(index):
    return FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))


@pytest.fixture
def forecast_setup(index, shared_FastIndex):
    powerplants_units = pd.read_csv(path / "powerplant_units.csv", index_col="name")
    demand_units = pd.read_csv(path / "demand_units.csv", index_col="name")
    availability = pd.read_csv(path / "availability.csv", **parse_date)
    demand_df = pd.read_csv(path / "demand.csv", **parse_date)
    fuel_prices_df = pd.read_csv(path / "fuel_prices.csv", index_col="fuel")
    forecast_df = pd.read_csv(path / "forecasts.csv", **parse_date)

    demand_units["min_power"] = -abs(demand_units["min_power"])
    demand_units["max_power"] = -abs(demand_units["max_power"])

    fuel_prices_df.index = index[:1]
    fuel_prices_df = fuel_prices_df.reindex(index, method="ffill")

    units: dict = {}
    forecast_registries = get_forecast_registries()

    # create a mock dsm forecaster as it also calculates congestion_signal
    # and renewable_utilisation forecasts
    dsm_forecaster = DsmUnitForecaster(
        index=shared_FastIndex,
        forecast_registries=forecast_registries,
    )

    for id, plant in powerplants_units.iterrows():
        plant["forecaster"] = PowerplantForecaster(
            index=shared_FastIndex,
            availability=availability.get(id, pd.Series(1.0, index, name=id)),
            fuel_prices=fuel_prices_df,
            forecast_registries=forecast_registries,
        )
        plant["bidding_strategies"] = {"EOM": EnergyNaiveStrategy()}
        plant["id"] = id
        units[id] = PowerPlant(**plant)

    for id, demand in demand_units.iterrows():
        demand["forecaster"] = DemandForecaster(
            index=shared_FastIndex,
            availability=availability.get(id, pd.Series(1.0, index, name=id)),
            demand=-demand_df[id].abs(),
            forecast_registries=forecast_registries,
        )
        demand["bidding_strategies"] = {"EOM": EnergyNaiveStrategy()}
        demand["id"] = id
        units[id] = Demand(**demand)

    elastic_demand = demand.copy()
    elastic_demand["bidding_strategies"] = {"EOM": EnergyHeuristicElasticStrategy()}
    elastic_demand["elasticity_model"] = "linear"
    elastic_demand["num_bids"] = 3

    elastic_unit = Demand(**elastic_demand)

    units_elastic = {id: unit for id, unit in units.items()}
    units_elastic[elastic_unit.id] = elastic_unit

    return {
        "units": units.values(),
        "units_elastic": units_elastic.values(),
        "forecast_df": forecast_df,
        "mock_dsm_forecaster": dsm_forecaster,
    }


def test_forecast_interface__calc_and_update_forecasts(
    index, market_setup, forecast_setup
):
    # 1. Initialize forecasts (includes preprocess)
    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]
    mock_dsm_forecaster.initialize(
        forecast_setup["units"],
        market_setup["market_configs"],
        None,  # no forecast_df --> calculate on its own
        None,  # forecaster has no unit
    )

    market_forecast = mock_dsm_forecaster.price
    load_forecast = mock_dsm_forecaster.residual_load
    congestion_signal = mock_dsm_forecaster.congestion_signal
    rn_utilization = mock_dsm_forecaster.renewable_utilisation_signal

    # 2. Assert that results are generated like expected
    expected_price = pd.read_csv(path / "results/price.csv", **parse_date)
    expected_load = pd.read_csv(path / "results/load_forecast.csv", **parse_date)
    expected_cgn = pd.read_csv(path / "results/congestion_signal.csv", **parse_date)
    expected_uti = pd.read_csv(path / "results/renewable_utilization.csv", **parse_date)

    assert_series_equal(
        expected_load["load_forecast"],
        pd.Series(
            load_forecast["EOM"], index
        ),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )

    assert_series_equal(
        expected_price["price"],
        pd.Series(
            market_forecast["EOM"], index
        ),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )

    # Check congestion signal and renewable_utilization are as expected
    for key in congestion_signal:
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in expected_cgn:  # also test that all keys are present
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in rn_utilization:
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()

    for key in expected_uti:  # also test that all keys are present
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()

    # 3. Update all forecasts
    # Default update should do nothing on all forecasts!!!
    mock_dsm_forecaster.update()

    market_forecast = mock_dsm_forecaster.price
    load_forecast = mock_dsm_forecaster.residual_load
    congestion_signal = mock_dsm_forecaster.congestion_signal
    rn_utilization = mock_dsm_forecaster.renewable_utilisation_signal

    assert_series_equal(
        expected_load["load_forecast"],
        pd.Series(
            load_forecast["EOM"], index
        ),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )

    assert_series_equal(
        expected_price["price"],
        pd.Series(
            market_forecast["EOM"], index
        ),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )

    for key in congestion_signal:
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in expected_cgn:
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in rn_utilization:
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()

    for key in expected_uti:
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()


def test_forecast_interface__uses_given_forecast(index, market_setup, forecast_setup):
    forecasts = forecast_setup["forecast_df"]
    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]

    # Add trivial node-wise forecasts (all 1s) to the forecast_df
    # congestion_signal: lookup key is {node}_congestion_signal
    forecasts["north_1_congestion_signal"] = pd.Series(1.0, index=index)
    forecasts["north_2_congestion_signal"] = pd.Series(1.0, index=index)
    # renewable_utilisation: lookup key is {node}_renewable_utilisation
    forecasts["north_1_renewable_utilisation"] = pd.Series(1.0, index=index)
    forecasts["north_2_renewable_utilisation"] = pd.Series(1.0, index=index)
    forecasts["all_nodes_renewable_utilisation"] = pd.Series(1.0, index=index)

    mock_dsm_forecaster.initialize(
        forecast_setup["units"],
        market_setup["market_configs"],
        forecasts,
        None,
    )

    # Check price and residual_load are taken from the given forecast
    market_forecast = mock_dsm_forecaster.price
    load_forecast = mock_dsm_forecaster.residual_load
    assert_series_equal(
        pd.Series(market_forecast["EOM"], index),
        forecasts["price_EOM"],
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    assert_series_equal(
        pd.Series(load_forecast["EOM"], index),
        forecasts["residual_load_EOM"],
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )

    # Check congestion_signal uses given forecasts (stored under congestion_severity keys)
    congestion_signal = mock_dsm_forecaster.congestion_signal
    assert list(congestion_signal["north_1_congestion_severity"]) == [1.0] * 24
    assert list(congestion_signal["north_2_congestion_severity"]) == [1.0] * 24

    # Check renewable_utilisation uses given forecasts
    rn_utilization = mock_dsm_forecaster.renewable_utilisation_signal
    assert list(rn_utilization["north_1_renewable_utilisation"]) == [1.0] * 24
    assert list(rn_utilization["north_2_renewable_utilisation"]) == [1.0] * 24
    assert list(rn_utilization["all_nodes_renewable_utilisation"]) == [1.0] * 24


def test_forecast_interface__empty_grid(market_setup, forecast_setup):
    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]

    mock_dsm_forecaster.initialize(
        forecast_setup["units"],
        market_setup["empty_grid_markets"],
        None,
        None,
    )

    assert mock_dsm_forecaster.congestion_signal == {}
    assert mock_dsm_forecaster.renewable_utilisation_signal == {}


def test_forecast_interface__elastic_demand(index, market_setup, forecast_setup):
    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]

    mock_dsm_forecaster.initialize(
        forecast_setup["units_elastic"],
        market_setup["market_configs"],
        None,
        None,
    )

    market_forecast = mock_dsm_forecaster.price
    load_forecast = mock_dsm_forecaster.residual_load
    congestion_signal = mock_dsm_forecaster.congestion_signal
    rn_utilization = mock_dsm_forecaster.renewable_utilisation_signal

    # 2. Assert that results are generated like expected
    expected_load = pd.read_csv(path / "results/load_forecast.csv", **parse_date)
    expected_cgn = pd.read_csv(path / "results/congestion_signal.csv", **parse_date)
    expected_uti = pd.read_csv(path / "results/renewable_utilization.csv", **parse_date)

    assert_series_equal(
        expected_load["load_forecast"],
        pd.Series(
            load_forecast["EOM"], index
        ),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )

    assert np.isclose(list(market_forecast["EOM"]), [71 / 3] * 24).all()

    # Check congestion signal and renewable_utilization are as expected
    for key in congestion_signal:
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in expected_cgn:  # also test that all keys are present
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in rn_utilization:
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()

    for key in expected_uti:  # also test that all keys are present
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()


def test_forecast_interface__cache(market_setup, forecast_setup):
    # clear cache uses
    calculate_naive_price.cache_clear()
    calculate_naive_residual_load.cache_clear()
    calculate_naive_congestion_signal.cache_clear()
    calculate_naive_renewable_utilisation.cache_clear()
    calculate_naive_price_inelastic.cache_clear()

    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]
    n = 2
    for i in range(n):
        mock_dsm_forecaster.initialize(
            forecast_setup["units"],
            market_setup["market_configs"],
            None,  # no forecast_df --> calculate on its own
            None,  # forecaster has no unit
        )

        assert calculate_naive_price.cache_info().hits == i
        assert calculate_naive_price.cache_info().misses == 1

        assert calculate_naive_residual_load.cache_info().hits == i
        assert calculate_naive_residual_load.cache_info().misses == 1

        assert calculate_naive_congestion_signal.cache_info().hits == i
        assert calculate_naive_congestion_signal.cache_info().misses == 1

        assert calculate_naive_renewable_utilisation.cache_info().hits == i
        assert calculate_naive_renewable_utilisation.cache_info().misses == 1

    for i, unit in enumerate(forecast_setup["units"]):
        unit.forecaster.initialize(
            forecast_setup["units"], market_setup["market_configs"], None, unit
        )

        assert calculate_naive_price.cache_info().hits == i + n
        assert calculate_naive_price.cache_info().misses == 1

        assert calculate_naive_residual_load.cache_info().hits == i + n
        assert calculate_naive_residual_load.cache_info().misses == 1

        assert calculate_naive_congestion_signal.cache_info().hits == n - 1
        assert calculate_naive_congestion_signal.cache_info().misses == 1

        assert calculate_naive_renewable_utilisation.cache_info().hits == n - 1
        assert calculate_naive_renewable_utilisation.cache_info().misses == 1

    assert calculate_naive_price_inelastic.cache_info().hits == 0
    assert calculate_naive_price_inelastic.cache_info().misses == 1
