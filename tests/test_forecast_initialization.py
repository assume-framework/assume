# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

from assume.common.forecast_initialisation import (
    DummyPriceForecastInitialisation,
    LoadAndNodeForecastInitialisation,
    price_forcast_initialisations,
)

path = Path("./tests/fixtures/forecast_init")

parse_date = {"index_col": "datetime", "parse_dates": ["datetime"]}


@pytest.fixture
def market_configs():
    return {
        "EOM": {
            "operator": "EOM_operator",
            "product_type": "energy",
            "products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
            "opening_frequency": "1h",
            "opening_duration": "1h",
            "volume_unit": "MWh",
            "maximum_bid_volume": 100000,
            "maximum_bid_price": 3000,
            "minimum_bid_price": -500,
            "price_unit": "EUR/MWh",
            "market_mechanism": "pay_as_clear",
        }
    }


@pytest.fixture
def forecast_inputs(market_configs):
    return {
        "market_configs": market_configs,
        "index": pd.date_range("2019-01-01", periods=24, freq="h"),
        "powerplants_units": pd.read_csv(
            path / "powerplant_units.csv", index_col="name"
        ),
        "demand_units": pd.read_csv(path / "demand_units.csv", index_col="name"),
        "availability": pd.read_csv(path / "availability.csv", **parse_date),
        "demand": pd.read_csv(path / "demand.csv", **parse_date),
        "fuel_prices": pd.read_csv(path / "fuel_prices.csv", index_col="fuel"),
    }


@pytest.fixture(params=["naive_forecast", "dummy_forecast"])  # TODO!!!!!!!!!!!!!
def price_forecast_init(request, forecast_inputs):
    price_forecast_class = price_forcast_initialisations[request.param]
    price_initialisation = price_forecast_class(**forecast_inputs)
    return price_initialisation


@pytest.fixture
def load_node_forecast_init(forecast_inputs):
    forecast_inputs["lines"] = pd.read_csv(path / "lines.csv", index_col="line")
    forecast_inputs["buses"] = pd.read_csv(path / "buses.csv", index_col="name")
    load_node_forecast_initialisation = LoadAndNodeForecastInitialisation(
        **forecast_inputs
    )
    return load_node_forecast_initialisation


def test_forecast_init__calc_market_forecasts(
    price_forecast_init, load_node_forecast_init
):
    price_forecast = price_forecast_init.calculate_market_forecasts()
    load_forecast = load_node_forecast_init.calculate_residual_load_forecast()
    # assert the passed forecast is generated
    expected = pd.read_csv(path / "results/load_forecast.csv", **parse_date)
    assert_series_equal(
        expected["load_forecast"], load_forecast["EOM"], check_names=False
    )
    if isinstance(price_forecast_init, DummyPriceForecastInitialisation):
        assert list(price_forecast["EOM"]) == [0] * 24
    else:  # NaivePriceForecastInitialisation
        assert list(price_forecast["EOM"]) == [1000] * 24


def test_forecast_init__calc_node_forecasts(load_node_forecast_init):
    congestion_signal, rn_utilization = load_node_forecast_init.calc_node_forecasts()
    expected_cgn = pd.read_csv(path / "results/congestion_signal.csv", **parse_date)
    expected_uti = pd.read_csv(path / "results/renewable_utilization.csv", **parse_date)
    assert_frame_equal(
        expected_cgn,
        congestion_signal,
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    assert_frame_equal(
        expected_uti,
        rn_utilization,
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )


def test_forecast_init__uses_given_forecast(
    price_forecast_init, load_node_forecast_init
):
    forecasts = pd.read_csv(path / "forecasts.csv", **parse_date)
    price_forecast_init._forecasts = forecasts
    load_node_forecast_init._forecasts = forecasts

    price_forecast = price_forecast_init.calculate_market_forecasts()
    load_forecast = load_node_forecast_init.calculate_residual_load_forecast()
    assert_series_equal(
        price_forecast["EOM"],
        forecasts["price_EOM"],
        check_names=False,
        check_freq=False,
    )
    assert_series_equal(
        load_forecast["EOM"],
        forecasts["residual_load_EOM"],
        check_names=False,
        check_freq=False,
    )
