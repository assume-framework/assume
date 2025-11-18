# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

from assume.common.forecast_initialisation import ForecastInitialisation

path = "./tests/fixtures/forecast_init"


@pytest.fixture
def forecast_init():
    market_configs = {
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
    forecast_init = ForecastInitialisation(
        market_configs=market_configs,
        index=pd.date_range("2019-01-01", periods=24, freq="h"),
        powerplants_units=pd.read_csv(f"{path}/powerplant_units.csv", index_col="name"),
        demand_units=pd.read_csv(f"{path}/demand_units.csv", index_col="name"),
        availability=pd.read_csv(
            f"{path}/availability.csv", index_col="datetime", parse_dates=["datetime"]
        ),
        demand=pd.read_csv(
            f"{path}/demand.csv", index_col="datetime", parse_dates=["datetime"]
        ),
        fuel_prices=pd.read_csv(f"{path}/fuel_prices.csv", index_col="fuel"),
        lines=pd.read_csv(f"{path}/lines.csv", index_col="line"),
        buses=pd.read_csv(f"{path}/buses.csv", index_col="name"),
        forecasts=pd.DataFrame(),
        exchanges=pd.DataFrame(),
    )
    return forecast_init


def test_forecast_init__calc_market_forecasts(forecast_init):
    market_forecast, load_forecast = forecast_init.calculate_market_forecasts()
    # assert the passed forecast is generated
    expected = pd.read_csv(
        f"{path}/results/load_forecast.csv",
        index_col="datetime",
        parse_dates=["datetime"],
    )
    assert_series_equal(
        expected["load_forecast"], load_forecast["EOM"], check_names=False
    )
    assert list(market_forecast["EOM"]) == [1000] * 24


def test_forecast_init__calc_node_forecasts(forecast_init):
    congestion_signal, rn_utilization = forecast_init.calc_node_forecasts()
    expected_cgn = pd.read_csv(
        f"{path}/results/congestion_signal.csv",
        index_col="datetime",
        parse_dates=["datetime"],
    )
    expected_uti = pd.read_csv(
        f"{path}/results/renewable_utilization.csv",
        index_col="datetime",
        parse_dates=["datetime"],
    )
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


def test_forecast_init__uses_given_forecast(forecast_init):
    forecasts = pd.read_csv(
        f"{path}/forecasts.csv", index_col="datetime", parse_dates=["datetime"]
    )
    forecast_init._forecasts = forecasts
    price_forecast, load_forecast = forecast_init.calculate_market_forecasts()
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
