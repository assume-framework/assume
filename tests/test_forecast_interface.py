# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from assume.common.forecaster import (
    DemandForecaster,
    DsmUnitForecaster,
    PowerplantForecaster,
)
from assume.common.market_objects import MarketConfig
from assume.strategies import EnergyNaiveStrategy
from assume.units import Demand, PowerPlant

path = Path("./tests/fixtures/forecast_init")

parse_date = {"index_col": "datetime", "parse_dates": ["datetime"]}


@pytest.fixture
def forecast_setup():
    market_configs = {
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
    index = pd.DatetimeIndex(
        pd.date_range("2019-01-01", periods=24, freq="h"),
    )
    powerplants_units = pd.read_csv(path / "powerplant_units.csv", index_col="name")
    demand_units = pd.read_csv(path / "demand_units.csv", index_col="name")
    availability = pd.read_csv(path / "availability.csv", **parse_date)
    demand_df = pd.read_csv(path / "demand.csv", **parse_date)
    fuel_prices_df = pd.read_csv(path / "fuel_prices.csv", index_col="fuel")
    lines = pd.read_csv(path / "lines.csv", index_col="line")
    buses = pd.read_csv(path / "buses.csv", index_col="name")
    forecast_df = pd.read_csv(path / "forecasts.csv", **parse_date)

    market_configs["EOM"]["param_dict"]["grid_data"] = {
        "buses": buses,
        "lines": lines,
    }

    market_configs["EOM"] = MarketConfig(**market_configs["EOM"])

    demand_units["min_power"] = -abs(demand_units["min_power"])
    demand_units["max_power"] = -abs(demand_units["max_power"])

    fuel_prices_df.index = index[:1]
    fuel_prices_df = fuel_prices_df.reindex(index, method="ffill")

    units: dict = {}

    # create a mock dsm forecaster as it also calculates congestion_signal
    # and renewable_utilisation forecasts
    dsm_forecaster = DsmUnitForecaster(
        index=index,
    )

    for id, plant in powerplants_units.iterrows():
        plant["forecaster"] = PowerplantForecaster(
            index=index,
            availability=availability.get(id, pd.Series(1.0, index, name=id)),
            fuel_prices=fuel_prices_df,
        )
        plant["bidding_strategies"] = {"EOM": EnergyNaiveStrategy()}
        plant["id"] = id
        units[id] = PowerPlant(**plant)

    for id, demand in demand_units.iterrows():
        demand["forecaster"] = DemandForecaster(
            index=index,
            availability=availability.get(id, pd.Series(1.0, index, name=id)),
            demand=-demand_df[id].abs(),
        )
        demand["bidding_strategies"] = {"EOM": EnergyNaiveStrategy()}
        demand["id"] = id
        units[id] = Demand(**demand)

    return {
        "index": index,
        "units": units.values(),
        "market_configs": market_configs.values(),
        "forecast_df": forecast_df,
        "mock_dsm_forecaster": dsm_forecaster,
    }


def test_forecast_interface__calc_and_update_forecasts(forecast_setup):
    # 1. Initialize forecasts (includes preprocess)
    index = forecast_setup["index"]
    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]
    mock_dsm_forecaster.initialize(
        forecast_setup["units"],
        forecast_setup["market_configs"],
        None,  # no forecast_df --> calculate on its own
        None,  # forecaster has no unit
    )

    market_forecast = mock_dsm_forecaster.price
    load_forecast = mock_dsm_forecaster.residual_load
    congestion_signal = mock_dsm_forecaster.congestion_signal
    rn_utilization = mock_dsm_forecaster.renewable_utilisation_signal

    # 2. Assert that results are generated like expected
    expected = pd.read_csv(path / "results/load_forecast.csv", **parse_date)
    expected_cgn = pd.read_csv(path / "results/congestion_signal.csv", **parse_date)
    expected_uti = pd.read_csv(path / "results/renewable_utilization.csv", **parse_date)

    assert_series_equal(
        expected["load_forecast"],
        pd.Series(
            load_forecast["EOM"], forecast_setup["index"]
        ),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )

    assert list(market_forecast["EOM"]) == [1000] * 24

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
        expected["load_forecast"],
        pd.Series(
            load_forecast["EOM"], forecast_setup["index"]
        ),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    assert list(market_forecast["EOM"]) == [1000] * 24

    for key in congestion_signal:
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in expected_cgn:
        assert np.isclose(congestion_signal[key].data, expected_cgn[key].values).all()

    for key in rn_utilization:
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()

    for key in expected_uti:
        assert np.isclose(rn_utilization[key].data, expected_uti[key].values).all()


def test_forecast_interface__uses_given_forecast(forecast_setup):
    forecasts = forecast_setup["forecast_df"]
    index = forecast_setup["index"]
    for unit in forecast_setup["units"]:
        unit.forecaster.initialize(
            forecast_setup["units"],
            forecast_setup["market_configs"],
            forecasts,
        )
        break

    market_forecast = unit.forecaster.price
    load_forecast = unit.forecaster.residual_load
    assert_series_equal(
        pd.Series(
            market_forecast["EOM"], index
        ),  # convert FastSeries to pd.Series for assertion
        forecasts["price_EOM"],
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    assert_series_equal(
        pd.Series(
            load_forecast["EOM"], index
        ),  # convert FastSeries to pd.Series for assertion
        forecasts["residual_load_EOM"],
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
