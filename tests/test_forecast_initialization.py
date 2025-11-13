# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock

import pandas as pd
import pytest

from assume.common.forecast_initialisation import ForecastInitialisation


@pytest.fixture
def forecast_init():
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    powerplants_units = pd.DataFrame.from_records(
        [
            {
                "name": "Unit 1",
                "technology": "nuclear",
                "bidding_EOM": "naive_eom",
                "fuel_type": "uranium",
                "emission_factor": 0.0,
                "max_power": 1000.0,
                "min_power": 200.0,
                "efficiency": 0.3,
                "additional_cost": 10.3,
                "unit_operator": "Operator 1",
            }
        ],
        index="name",
    )
    demand_units = pd.DataFrame.from_records(
        [
            {
                "name": "demand_EOM",
                "technology": "inflex_demand",
                "bidding_EOM": "naive_eom",
                "max_power": 1000000,
                "min_power": 0,
                "unit_operator": "eom_de",
            }
        ],
        index="name",
    )
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
    demand_series = [
        {"datetime": "2022-01-01 00:00:00", "demand_EOM": 100},
        {"datetime": "2022-01-01 01:00:00", "demand_EOM": 200},
        {"datetime": "2022-01-01 02:00:00", "demand_EOM": 300},
        {"datetime": "2022-01-01 03:00:00", "demand_EOM": 400},
    ]
    forecasts = pd.DataFrame.from_records(demand_series)
    forecasts["datetime"] = pd.to_datetime(forecasts["datetime"])
    fuel_prices = pd.DataFrame()
    forecast_init = ForecastInitialisation(
        index,
        powerplants_units,
        demand_units,
        market_configs,
        forecasts,
        availability=pd.DataFrame(),
        demand=pd.DataFrame(),
        exchanges=pd.DataFrame(),
        fuel_prices=pd.DataFrame(),
    )
    return forecast_init


def test_forecast_init(forecast_init):
    forecast_init.calculate_market_price_forecast = MagicMock()

    forecast_init.calc_forecast_if_needed()
    assert forecast_init.calculate_market_price_forecast.called


def test_forecast_init_used_forecast(forecast_init):
    index = pd.date_range("2022-01-01", periods=4, freq="h")
    price_series = [
        {"datetime": "2022-01-01 00:00:00", "price_EOM": 1},
        {"datetime": "2022-01-01 01:00:00", "price_EOM": 2},
        {"datetime": "2022-01-01 02:00:00", "price_EOM": 3},
        {"datetime": "2022-01-01 03:00:00", "price_EOM": 4},
    ]
    prices = pd.DataFrame.from_records(price_series)
    prices["datetime"] = pd.to_datetime(prices["datetime"])

    forecast_init.set_forecast(prices)
    forecast_init.calculate_market_price_forecast = MagicMock()
    forecast_init.calc_forecast_if_needed()
    assert not forecast_init.calculate_market_price_forecast.called
