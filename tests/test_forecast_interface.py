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
        pd.date_range("2019-01-01 08:00", periods=7, freq="h"),
    )


@pytest.fixture
def shared_FastIndex(index):
    return FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))


@pytest.fixture
def forecast_setup(index, shared_FastIndex):
    #############################################################
    # 1. Read in csv inputs
    #############################################################
    powerplants_units = pd.read_csv(path / "powerplant_units.csv", index_col="name")
    demand_units = pd.read_csv(path / "demand_units.csv", index_col="name")
    availability = pd.read_csv(path / "availability.csv", **parse_date)
    demand_df = pd.read_csv(path / "demand_df.csv", **parse_date)
    fuel_prices_df = pd.read_csv(path / "fuel_prices.csv", index_col="fuel")
    forecast_df = pd.read_csv(path / "forecasts.csv", **parse_date)

    #############################################################
    # 2. Process inputs
    #############################################################
    demand_units["min_power"] = -abs(demand_units["min_power"])
    demand_units["max_power"] = -abs(demand_units["max_power"])

    fuel_prices_df.index = index[:1]
    fuel_prices_df = fuel_prices_df.reindex(index, method="ffill")

    #############################################################
    # 3. Build forecasts and units
    #############################################################
    all_units_inelastic_case: dict = {}
    all_units_elastic_case: dict = {}
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
        all_units_inelastic_case[id] = PowerPlant(**plant)
        all_units_elastic_case[id] = PowerPlant(**plant)

    for id, demand in demand_units.iterrows():
        demand["forecaster"] = DemandForecaster(
            index=shared_FastIndex,
            availability=availability.get(id, pd.Series(1.0, index, name=id)),
            demand=-demand_df[id].abs(),
            forecast_registries=forecast_registries,
        )
        demand["bidding_strategies"] = {"EOM": EnergyNaiveStrategy()}
        demand["id"] = id
        all_units_inelastic_case[id] = Demand(**demand)

    elastic_demand = demand.copy()
    elastic_demand["bidding_strategies"] = {"EOM": EnergyHeuristicElasticStrategy()}
    elastic_demand["elasticity_model"] = "linear"
    elastic_demand["num_bids"] = 300
    elastic_demand["max_power"] = -3000
    elastic_demand["max_price"] = 300

    elastic_unit = Demand(**elastic_demand)

    all_units_elastic_case[elastic_unit.id] = elastic_unit

    return {
        "units": all_units_inelastic_case.values(),
        "units_elastic_case": all_units_elastic_case.values(),
        "forecast_df": forecast_df,
        "mock_dsm_forecaster": dsm_forecaster,
    }


def test_forecast_interface__calc_and_update_forecasts(
    index, market_setup, forecast_setup
):
    #############################################################
    # 1. Arrange
    #############################################################
    expected_price = pd.read_csv(path / "results/price.csv", **parse_date)
    expected_load = pd.read_csv(path / "results/load_forecast.csv", **parse_date)
    expected_cgn = pd.read_csv(path / "results/congestion_signal.csv", **parse_date)
    expected_uti = pd.read_csv(path / "results/renewable_utilization.csv", **parse_date)
    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]

    #############################################################
    # 2. (Act) Initialize forecasts (includes preprocess)
    #############################################################

    mock_dsm_forecaster.initialize(
        forecast_setup["units"],
        market_setup["market_configs"],
        None,  # no forecast_df --> calculate on its own
        None,  # forecaster has no unit
    )

    #############################################################
    # 3. Assert that results are generated like expected
    #############################################################
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

    # Check congestion signal and renewable_utilization are as expected
    # NOTE: congestion forecast is negative as max available power > demand at the nodes
    for key in congestion_signal:
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in expected_cgn:  # also test that all keys are present
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in rn_utilization:
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()

    for key in expected_uti:  # also test that all keys are present
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()

    #############################################################
    # 4. (Act Again) Update all forecasts
    #############################################################

    mock_dsm_forecaster.update()

    #############################################################
    # 5. Assert (Again) that results are generated like expected
    #    Default update should do nothing on all forecasts!!!
    #############################################################

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
    assert list(congestion_signal["north_1_congestion_severity"]) == [1.0] * len(index)
    assert list(congestion_signal["north_2_congestion_severity"]) == [1.0] * len(index)

    # Check renewable_utilisation uses given forecasts
    rn_utilization = mock_dsm_forecaster.renewable_utilisation_signal
    assert list(rn_utilization["north_1_renewable_utilisation"]) == [1.0] * len(index)
    assert list(rn_utilization["north_2_renewable_utilisation"]) == [1.0] * len(index)
    assert list(rn_utilization["all_nodes_renewable_utilisation"]) == [1.0] * len(index)


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
    """
    TODO: make better test scenario for elastic demand
    """
    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]

    mock_dsm_forecaster.initialize(
        forecast_setup["units_elastic_case"],
        market_setup["market_configs"],
        None,
        None,
    )

    # 2. Assert that results are generated like expected
    market_forecast = mock_dsm_forecaster.price

    assert np.isclose(list(market_forecast["EOM"]), [8.0] * 7).all()


def test_forecast_interface__cache(market_setup, forecast_setup):
    # clear cache uses
    calculate_naive_price.cache_clear()
    calculate_naive_residual_load.cache_clear()
    calculate_naive_congestion_signal.cache_clear()
    calculate_naive_renewable_utilisation.cache_clear()
    calculate_naive_price_inelastic.cache_clear()

    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]

    # simulate multiple dsm units by rerunning initialization
    n = 2
    for _ in range(n):
        mock_dsm_forecaster.initialize(
            forecast_setup["units"],
            market_setup["market_configs"],
            None,  # no forecast_df --> calculate on its own
            None,  # forecaster has no unit
        )

    for unit in forecast_setup["units"]:
        unit.forecaster.initialize(
            forecast_setup["units"], market_setup["market_configs"], None, unit
        )

    # price and residual_load are called by all initializations
    assert (
        calculate_naive_price.cache_info().hits == len(forecast_setup["units"]) + n - 1
    )
    assert calculate_naive_price.cache_info().misses == 1

    assert (
        calculate_naive_residual_load.cache_info().hits
        == len(forecast_setup["units"]) + n - 1
    )
    assert calculate_naive_residual_load.cache_info().misses == 1

    # congestion_signal and renewable_utilisation are called only by dsm units (n times)
    assert calculate_naive_congestion_signal.cache_info().hits == n - 1
    assert calculate_naive_congestion_signal.cache_info().misses == 1

    assert calculate_naive_renewable_utilisation.cache_info().hits == n - 1
    assert calculate_naive_renewable_utilisation.cache_info().misses == 1

    # NOTE: only missed once & no hits due to lru_cache also on calculate_naive_price
    assert calculate_naive_price_inelastic.cache_info().hits == 0
    assert calculate_naive_price_inelastic.cache_info().misses == 1
