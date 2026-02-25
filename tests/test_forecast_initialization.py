# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

from assume.common.forecast_algorithms import (
    calculate_naive_congestion_forecast,
    calculate_naive_renewable_utilisation,
)
from assume.common.forecaster import (
    DemandForecaster,
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
    }


def test_forecast_init__calc_market_forecasts(forecast_setup):
    for unit in forecast_setup["units"]:
        unit.forecaster.initialize(
            forecast_setup["units"],
            forecast_setup["market_configs"],
            None,  # no forecast_df --> calculate on its own
            unit,
        )
        break

    market_forecast = unit.forecaster.price
    load_forecast = unit.forecaster.residual_load

    # assert the passed forecast is generated
    expected = pd.read_csv(path / "results/load_forecast.csv", **parse_date)
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


def test_forecast_init__calc_node_forecasts(forecast_setup):
    congestion_signal = calculate_naive_congestion_forecast(
        forecast_setup["index"],
        forecast_setup["units"],
        forecast_setup["market_configs"],
        forecast_setup["forecast_df"],
    )
    rn_utilization = calculate_naive_renewable_utilisation(
        forecast_setup["index"],
        forecast_setup["units"],
        forecast_setup["market_configs"],
        forecast_setup["forecast_df"],
    )

    expected_cgn = pd.read_csv(path / "results/congestion_signal.csv", **parse_date)
    expected_uti = pd.read_csv(path / "results/renewable_utilization.csv", **parse_date)

    # NOTE: instead of reindexing to sort columns one could probably use: check_like = True
    # But this would allow differently ordered index as well
    assert_frame_equal(
        expected_cgn,
        congestion_signal.reindex(
            sorted(congestion_signal.columns), axis=1
        ),  # otherwise order of columns **sometimes** is wrong!
        check_names=False,
        check_freq=False,
    )
    assert_frame_equal(
        expected_uti.reindex(
            sorted(expected_uti.columns), axis=1
        ),  # otherwise order of columns **sometimes** is wrong!
        rn_utilization.reindex(
            sorted(rn_utilization.columns), axis=1
        ),  # otherwise order of columns **sometimes** is wrong!
        check_names=False,
        check_freq=False,
    )


def test_forecast_init__uses_given_forecast(forecast_setup):
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
