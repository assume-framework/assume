# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT


from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_series_equal

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecast_algorithms import (
    _merit_order_clearing,
    calculate_naive_congestion_signal,
    calculate_naive_price,
    calculate_naive_price_inelastic,
    calculate_naive_renewable_utilisation,
    calculate_naive_residual_load,
    calculate_zonal_merit_order_prices,
    get_forecast_registries,
)
from assume.common.forecaster import (
    DemandForecaster,
    DsmUnitForecaster,
    ExchangeForecaster,
    PowerplantForecaster,
    UnitsOperatorForecaster,
)
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.scenario.loader_csv import save_unique_forecasts
from assume.strategies import (
    EnergyHeuristicElasticStrategy,
    EnergyNaiveStrategy,
    ExchangeEnergyNaiveStrategy,
)
from assume.units import Demand, PowerPlant
from assume.units.exchange import Exchange

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


def test_forecast_interface__elastic_demand_complex_clearing(
    market_setup, forecast_setup
):
    """
    The elastic price forecast clears its own orderbook without validate_orderbook,
    so the generated orders must contain the additional fields of the market,
    e.g. min_acceptance_ratio for complex clearing.
    """
    market_config = next(iter(market_setup["empty_grid_markets"]))
    market_config.market_mechanism = "complex_clearing"
    market_config.additional_fields = [
        "bid_type",
        "min_acceptance_ratio",
        "parent_bid_id",
    ]
    market_config.param_dict = {"grid_data": {}, "solver_name": "appsi_highs"}

    mock_dsm_forecaster = forecast_setup["mock_dsm_forecaster"]
    mock_dsm_forecaster.initialize(
        forecast_setup["units_elastic_case"],
        {"EOM": market_config}.values(),
        None,
        None,
    )

    price_forecast = np.array(list(mock_dsm_forecaster.price["EOM"]))

    # supply is fully dispatched, so the price is set by the elastic demand bids
    assert len(price_forecast) == 7
    assert np.isfinite(price_forecast).all()
    assert (price_forecast >= 8.0).all()
    assert (price_forecast <= market_config.maximum_bid_price).all()


def test_forecast_interface__cache(market_setup, forecast_setup, shared_FastIndex):
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

    # an operator-level forecaster initializes against the same units/markets
    # objects, so it shares the price / residual_load cache too
    operator_forecaster = UnitsOperatorForecaster(
        index=shared_FastIndex,
        forecast_registries=get_forecast_registries(),
    )
    operator_forecaster.initialize(
        forecast_setup["units"], market_setup["market_configs"], None
    )

    for unit in forecast_setup["units"]:
        unit.forecaster.initialize(
            forecast_setup["units"], market_setup["market_configs"], None, unit
        )

    # price and residual_load are called by all initializations: the n dsm runs,
    # the operator run, and one per unit. Only the first call misses.
    assert calculate_naive_price.cache_info().hits == len(forecast_setup["units"]) + n
    assert calculate_naive_price.cache_info().misses == 1

    assert (
        calculate_naive_residual_load.cache_info().hits
        == len(forecast_setup["units"]) + n
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


def test_units_operator_forecaster__matches_unit_forecasts(
    index, market_setup, forecast_setup, shared_FastIndex
):
    """An operator-level forecaster computes the same market-wide price and
    residual load as a unit forecaster, since it initializes against all units."""
    expected_price = pd.read_csv(path / "results/price.csv", **parse_date)
    expected_load = pd.read_csv(path / "results/load_forecast.csv", **parse_date)

    operator_forecaster = UnitsOperatorForecaster(
        index=shared_FastIndex,
        forecast_registries=get_forecast_registries(),
    )

    # the operator has no single unit, so initialize without an initializing_unit
    operator_forecaster.initialize(
        forecast_setup["units"],
        market_setup["market_configs"],
        None,  # no forecast_df --> calculate on its own
    )

    assert_series_equal(
        expected_price["price"],
        pd.Series(operator_forecaster.price["EOM"], index),
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )
    assert_series_equal(
        expected_load["load_forecast"],
        pd.Series(operator_forecaster.residual_load["EOM"], index),
        check_names=False,
        check_dtype=False,
        check_freq=False,
    )


def test_units_operator_forecaster__extra_kwargs(index, shared_FastIndex):
    """Arbitrary operator-level forecasts passed via kwargs are stored, with
    pd.Series converted to FastSeries."""
    custom_series = pd.Series(2.0, index=index)
    operator_forecaster = UnitsOperatorForecaster(
        index=shared_FastIndex,
        custom_forecast=custom_series,
    )

    assert isinstance(operator_forecaster.custom_forecast, FastSeries)
    assert list(operator_forecaster.custom_forecast) == [2.0] * len(index)


ZONAL_ALGORITHMS = {
    "price": "price_zonal_merit_order",
    "preprocess_price": "price_unit_zone",
}
NORTH_1_PRICES = [2, 6, 6, 6, 6, 2, 2]
# north_2 (solar and 1000 MW nuclear) is short from 10:00 to 12:00
NORTH_2_PRICES = [8, 8, 3000, 3000, 3000, 8, 8]


def zonal_market_configs(market_setup, zones_identifier=None):
    """Market configs of the fixture with or without zones (``zone_id`` of the buses)."""
    configs = {}
    for config in market_setup["market_configs"]:
        param_dict = dict(config.param_dict)
        if zones_identifier:
            param_dict["zones_identifier"] = zones_identifier
        configs[config.market_id] = replace(config, param_dict=param_dict)
    return configs.values()


def initialize_zonal(forecast_setup, market_configs, forecast_df=None):
    units = forecast_setup["units"]
    for unit in units:
        unit.forecaster.forecast_algorithms = dict(ZONAL_ALGORITHMS)
        unit.forecaster.initialize(units, market_configs, forecast_df, unit)
    return {unit.id: list(unit.forecaster.price["EOM"]) for unit in units}


@pytest.mark.parametrize(
    "supply, demand, expected",
    [
        # the second supply bid is partially accepted
        ([(10, 100), (50, 100)], [(3000, 150)], 50),
        # scarcity: the demand bid is not fully served
        ([(10, 100), (50, 100)], [(3000, 250)], 3000),
        # demand and supply are equal: the last accepted supply bid sets the price
        ([(10, 100), (50, 100)], [(3000, 100)], 10),
        # price-sensitive demand is partially served and sets the price
        ([(10, 100), (50, 100)], [(3000, 80), (30, 50)], 30),
        # supply exhausted, price-sensitive demand is not served
        ([(10, 100)], [(3000, 100), (30, 50)], 30),
        # price-sensitive demand below all supply bids is not served
        ([(40, 100)], [(3000, 50), (30, 50)], 40),
        # no demand: the cheapest supply bid sets the price
        ([(40, 100), (20, 100)], [], 20),
        # no supply: the highest demand bid sets the price
        ([], [(3000, 50), (30, 50)], 3000),
    ],
)
def test_merit_order_clearing(supply, demand, expected):
    def arrays(bids):
        return (
            np.array([price for price, _ in bids], dtype=float),
            np.array([volume for _, volume in bids], dtype=float),
        )

    assert _merit_order_clearing(*arrays(supply), *arrays(demand)) == expected


def test_zonal_merit_order_forecast__single_zone_equals_naive(
    market_setup, forecast_setup
):
    """Both nodes of the fixture are in zone DE_1, so the zonal forecast is the naive one."""
    expected_price = pd.read_csv(path / "results/price.csv", **parse_date)
    market_configs = zonal_market_configs(market_setup, zones_identifier="zone_id")

    prices = initialize_zonal(forecast_setup, market_configs)

    for price in prices.values():
        assert price == list(expected_price["price"])


def test_zonal_merit_order_forecast__nodal(market_setup, forecast_setup):
    """Without zones every node has its own merit order."""
    market_configs = zonal_market_configs(market_setup)
    calculate_zonal_merit_order_prices.cache_clear()

    prices = initialize_zonal(forecast_setup, market_configs)

    for unit in forecast_setup["units"]:
        expected = NORTH_1_PRICES if unit.node == "north_1" else NORTH_2_PRICES
        assert prices[unit.id] == expected, unit.id
    # the prices of all zones are calculated once and shared by all units
    assert calculate_zonal_merit_order_prices.cache_info().misses == 1
    assert calculate_zonal_merit_order_prices.cache_info().hits == len(prices) - 1


def test_zonal_merit_order_forecast__given_zone_forecast(
    index, market_setup, forecast_setup
):
    """A forecast of a zone in forecast_df (price_{market}_{zone}) is used for its units."""
    market_configs = zonal_market_configs(market_setup)
    forecast_df = pd.DataFrame({"price_EOM_north_1": [1.0] * len(index)}, index=index)

    prices = initialize_zonal(forecast_setup, market_configs, forecast_df)

    for unit in forecast_setup["units"]:
        expected = [1.0] * len(index) if unit.node == "north_1" else NORTH_2_PRICES
        assert prices[unit.id] == expected, unit.id


def test_zonal_merit_order_forecast__operator_uses_naive(
    market_setup, forecast_setup, shared_FastIndex
):
    """Forecasters without a unit (unit operators) use the naive forecast of the market."""
    expected_price = pd.read_csv(path / "results/price.csv", **parse_date)
    operator_forecaster = UnitsOperatorForecaster(
        index=shared_FastIndex,
        forecast_algorithms=dict(ZONAL_ALGORITHMS),
        forecast_registries=get_forecast_registries(),
    )

    operator_forecaster.initialize(
        forecast_setup["units"], zonal_market_configs(market_setup), None
    )

    assert list(operator_forecaster.price["EOM"]) == list(expected_price["price"])


def test_zonal_merit_order_forecast__exchange_and_demand_price(index, shared_FastIndex):
    """Imports are supply and exports are demand of the zone, demand bids at its price."""
    market_config = MarketConfig(
        market_id="EOM",
        param_dict={
            "grid_data": {
                "buses": pd.DataFrame(index=pd.Index(["A", "B"], name="name")),
                "lines": pd.DataFrame(),
            }
        },
    )
    registries = get_forecast_registries()
    plant = PowerPlant(
        id="plant_A",
        unit_operator="op",
        technology="gas",
        bidding_strategies={"EOM": EnergyNaiveStrategy()},
        max_power=100,
        min_power=0,
        efficiency=1,
        additional_cost=20,
        fuel_type="renewable",
        node="A",
        forecaster=PowerplantForecaster(
            index=shared_FastIndex, forecast_registries=registries
        ),
    )
    demand = Demand(
        id="demand_A",
        unit_operator="op",
        technology="inflex_demand",
        bidding_strategies={"EOM": EnergyNaiveStrategy()},
        max_power=-1000,
        min_power=0,
        price=300,
        node="A",
        forecaster=DemandForecaster(
            index=shared_FastIndex,
            demand=pd.Series(-120.0, index=index),
            forecast_registries=registries,
        ),
    )
    exchange = Exchange(
        id="exchange_A",
        unit_operator="op",
        bidding_strategies={"EOM": ExchangeEnergyNaiveStrategy()},
        price_import=0,
        price_export=3000,
        node="A",
        forecaster=ExchangeForecaster(
            index=shared_FastIndex,
            volume_import=pd.Series([50.0] * 3 + [0.0] * 4, index=index),
            volume_export=pd.Series([0.0] * 3 + [40.0] * 4, index=index),
            forecast_registries=registries,
        ),
    )

    prices = calculate_zonal_merit_order_prices(
        shared_FastIndex, (plant, demand, exchange), market_config
    )

    # import of 50 MW: the plant covers 70 MW of the 120 MW demand (partially accepted)
    assert list(prices["A"][:3]) == [20] * 3
    # export of 40 MW: the plant (100 MW) does not cover the 160 MW demand, the demand
    # bid at 300 EUR/MWh is not fully served
    assert list(prices["A"][3:]) == [300] * 4
    # no bids in zone B
    assert "B" not in prices


def test_save_unique_forecasts__zonal(tmp_path, market_setup, forecast_setup):
    """Zonal price forecasts are saved per zone (one column per market and zone)."""
    initialize_zonal(forecast_setup, zonal_market_configs(market_setup))

    save_unique_forecasts(forecast_setup["units"], tmp_path / "forecasts.csv")

    saved = pd.read_csv(tmp_path / "forecasts.csv", index_col="datetime")
    assert list(saved["price_zonal_merit_order_EOM_north_1"]) == NORTH_1_PRICES
    assert list(saved["price_zonal_merit_order_EOM_north_2"]) == NORTH_2_PRICES
