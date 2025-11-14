# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pandas as pd
import pytest

from assume.common.forecast_initialisation import ForecastInitialisation

fixture_path = "./tests/fixtures/forecast_init"


@pytest.fixture
def forecast_init():
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    powerplants_units = pd.read_csv(
        f"{fixture_path}/powerplants_units.csv", index_col="name"
    )
    demand_units = pd.read_csv(f"{fixture_path}/demand_units.csv", index_col="name")
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
    forecasts = pd.read_csv(
        f"{fixture_path}/forecasts.csv", index_col="datetime", parse_dates=["datetime"]
    )
    fuel_prices = pd.DataFrame(index=index, data={"co2": [10]})
    forecast_init = ForecastInitialisation(
        index,
        powerplants_units,
        demand_units,
        market_configs,
        forecasts,
        availability=pd.DataFrame(),
        demand=pd.DataFrame(),
        exchanges=pd.DataFrame(),
        fuel_prices=fuel_prices,
    )
    return forecast_init


def test_forecast_init(forecast_init):
    forecast_init.calc_forecast_if_needed()

    market_forecast, load_forecast = forecast_init.calculate_market_forecasts()
    # assert the passed forecast is generated
    assert list(market_forecast["EOM"]) == [10.3] * 4
    assert list(load_forecast["EOM"]) == [100, 200, 300, 400]


def test_forecast_init_used_forecast(forecast_init):
    prices = pd.DataFrame(index=forecast_init.index, data={"price_EOM": [1, 2, 3, 4]})
    forecast_init.set_forecast(prices)
    forecast_init.calc_forecast_if_needed()

    market_forecast, load_forecast = forecast_init.calculate_market_forecasts()
    # assert the passed forecast is kept
    assert market_forecast["EOM"].equals(prices["price_EOM"])
    assert list(load_forecast["EOM"]) == [100, 200, 300, 400]
