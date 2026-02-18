# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from pathlib import Path

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal, assert_series_equal

from assume.common.fast_pandas import FastIndex, FastSeries

#from assume.common.forecast_initialisation import ForecastInitialisation
from assume.common.forecaster import (
    PowerplantForecaster,
    DemandForecaster,
    calculate_naive_congestion_forecast,
    calculate_naive_renewable_utilisation
)
from assume.units import PowerPlant, Demand
from assume.common.market_objects import MarketConfig
from assume.strategies import EnergyNaiveStrategy

path = Path("./tests/fixtures/forecast_init")

parse_date = {"index_col": "datetime", "parse_dates": ["datetime"]}


@pytest.fixture
def forecast_preprocess():
    market_configs = {
        "EOM": {
            "market_id": "EOM",
            #"operator": "EOM_operator",
            "product_type": "energy",
            "market_products": [{"duration": "1h", "count": 1, "first_delivery": "1h"}],
            # "opening_frequency": "1h",
            "opening_duration": "1h",
            "volume_unit": "MWh",
            "maximum_bid_volume": 100000,
            "maximum_bid_price": 3000,
            "minimum_bid_price": -500,
            "price_unit": "EUR/MWh",
            "market_mechanism": "pay_as_clear",
            "param_dict": {
                "grid_data": None,
            }
        }
    }
    index = pd.DatetimeIndex(pd.date_range("2019-01-01", periods=24, freq="h"),)
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
 
    #unit_forecasts: dict[str, UnitForecaster] = {}
    units: dict[str, BaseUnit] = {}
    for id, plant in powerplants_units.iterrows():

        # plant_bidding_strategies = {
        #     key.split("bidding_")[1]: unit_paplantrams[key]
        #     for key in plant.keys()
        #     if key.startswith("bidding_") and plant[key]
        # }
        # plant_strategies = {
        #     bidding_strategies[strategy](unit_id=unit_id, **bidding_params,)
        #     for strategy in plant_bidding_strategies
        # }

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

    #index = FastIndex(start=index[0], end=index[-1], freq=pd.infer_freq(index))
    # forecast_df = FastSeries(index, forecast_df)

    return {
        "index" : index,
        "units" : units.values(),
        "market_configs": market_configs.values(),
        "forecast_df": forecast_df,
    }
    
    

    # forecast_init = ForecastInitialisation(
    #     market_configs=market_configs,
    #     index=pd.date_range("2019-01-01", periods=24, freq="h"),
    #     powerplants_units=pd.read_csv(path / "powerplant_units.csv", index_col="name"),
    #     demand_units=pd.read_csv(path / "demand_units.csv", index_col="name"),
    #     availability=pd.read_csv(path / "availability.csv", **parse_date),
    #     demand=pd.read_csv(path / "demand.csv", **parse_date),
    #     fuel_prices=pd.read_csv(path / "fuel_prices.csv", index_col="fuel"),
    #     lines=pd.read_csv(path / "lines.csv", index_col="line"),
    #     buses=pd.read_csv(path / "buses.csv", index_col="name"),
    # )
    #return forecast_init


def test_forecast_init__calc_market_forecasts(forecast_preprocess):
    for unit in forecast_preprocess["units"]:
        unit.forecaster.initialize(
            forecast_preprocess["units"],
            forecast_preprocess["market_configs"],
            None,  # no forecast_df --> calculate on its own
        )
        break

    market_forecast = unit.forecaster.price
    load_forecast = unit.forecaster.residual_load

    # assert the passed forecast is generated
    expected = pd.read_csv(path / "results/load_forecast.csv", **parse_date)
    assert_series_equal(
        expected["load_forecast"],
        pd.Series(load_forecast["EOM"], forecast_preprocess["index"]),  # convert FastSeries to pd.Series for comparison
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    assert list(market_forecast["EOM"]) == [1000] * 24


def test_forecast_init__calc_node_forecasts(forecast_preprocess):
    congestion_signal = calculate_naive_congestion_forecast(
            forecast_preprocess["index"],
            forecast_preprocess["units"],
            forecast_preprocess["market_configs"],
            forecast_preprocess["forecast_df"],
    )
    rn_utilization = calculate_naive_renewable_utilisation(
            forecast_preprocess["index"],
            forecast_preprocess["units"],
            forecast_preprocess["market_configs"],
            forecast_preprocess["forecast_df"],
    )

    expected_cgn = pd.read_csv(path / "results/congestion_signal.csv", **parse_date)
    expected_uti = pd.read_csv(path / "results/renewable_utilization.csv", **parse_date)

    # NOTE: instead of reindexing to sort columns one could probably use: check_like = True
    # But this would allow differently ordered index aswell
    assert_frame_equal(
        expected_cgn,
        congestion_signal.reindex(sorted(congestion_signal.columns), axis=1),  # otherwise order of columns **sometimes** is wrong!
        check_names=False,
        check_freq=False,
    )
    assert_frame_equal(
        expected_uti.reindex(sorted(expected_uti.columns), axis=1), # otherwise order of columns **sometimes** is wrong!
        rn_utilization.reindex(sorted(rn_utilization.columns), axis=1),  # otherwise order of columns **sometimes** is wrong!
        check_names=False,
        check_freq=False,
    )


def test_forecast_init__uses_given_forecast(forecast_preprocess):
    forecasts = forecast_preprocess["forecast_df"]
    index = forecast_preprocess["index"]
    for unit in forecast_preprocess["units"]:
        unit.forecaster.initialize(
            forecast_preprocess["units"],
            forecast_preprocess["market_configs"],
            forecasts,
        )
        break

    market_forecast = unit.forecaster.price
    load_forecast = unit.forecaster.residual_load
    assert_series_equal(
        pd.Series(market_forecast["EOM"], index),  # convert FastSeries to pd.Series for assertion
        forecasts["price_EOM"],
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    assert_series_equal(
        pd.Series(load_forecast["EOM"], index),  # convert FastSeries to pd.Series for assertion
        forecasts["residual_load_EOM"],
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
