# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch as th
from pandas._testing import assert_series_equal

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecast_algorithms import (
    ADAPTIVE_MERIT_ORDER_SETTINGS,
    calculate_adaptive_merit_order_forecast_inputs,
    calculate_naive_congestion_signal,
    calculate_naive_price,
    calculate_naive_price_inelastic,
    calculate_naive_renewable_utilisation,
    calculate_naive_residual_load,
    evaluate_adaptive_merit_order_forecasts,
    fit_adaptive_merit_order_feature_scaling,
    fit_online_regularized_regression,
    gaussian_residual_quantile,
    get_forecast_registries,
    initialize_adaptive_merit_order_feature_state,
    initialize_adaptive_merit_order_model,
    initialize_online_regularized_regression,
    issue_adaptive_merit_order_correction,
    johnson_su_residual_quantile,
    transform_adaptive_merit_order_features,
    update_adaptive_merit_order_correction,
    update_online_regularized_regression,
)
from assume.common.forecaster import (
    DemandForecaster,
    DsmUnitForecaster,
    PowerplantForecaster,
    UnitsOperatorForecaster,
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


def test_adaptive_merit_order_inputs_use_fast_series(
    market_setup, forecast_setup, shared_FastIndex
):
    """Adaptive inputs retain ASSUME's shared FastIndex and FastSeries types."""
    inputs = calculate_adaptive_merit_order_forecast_inputs(
        shared_FastIndex,
        tuple(forecast_setup["units"]),
        next(iter(market_setup["empty_grid_markets"])),
    )

    assert all(isinstance(series, FastSeries) for series in inputs.values())
    assert all(series.index is shared_FastIndex for series in inputs.values())


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


def _adaptive_merit_order_forecast_inputs(
    periods: int = 240,
    merit_order: float | np.ndarray = 55.0,
    residual_load: float | np.ndarray = 100.0,
) -> dict[str, pd.Series]:
    signal_index = pd.date_range("2025-01-01", periods=periods, freq="h")

    def series(value):
        values = np.full(periods, value) if np.isscalar(value) else value
        return pd.Series(values, index=signal_index, dtype=float)

    return {
        "merit_order_price": series(merit_order),
        "wind_availability_factor": series(0.4),
        "solar_availability_factor": series(0.2),
        "residual_load": series(residual_load),
    }


def _adaptive_merit_order_state(forecast_inputs, **settings):
    config = ADAPTIVE_MERIT_ORDER_SETTINGS | settings
    model = initialize_adaptive_merit_order_model("EOM", forecast_inputs, config)
    return {"markets": {"EOM": model}}, model


def _issue_adaptive_merit_order_forecast(state, forecast_inputs, position):
    product_start = forecast_inputs["merit_order_price"].index[position].to_pydatetime()
    product = (product_start, product_start + timedelta(hours=1), None)
    issued = issue_adaptive_merit_order_correction(
        state,
        "operator",
        "EOM",
        product_start - timedelta(hours=1),
        [product],
    )[0]
    return product_start, issued


def test_adaptive_residual_and_additive_correction_are_immutable():
    forecast_inputs = _adaptive_merit_order_forecast_inputs()
    state, _ = _adaptive_merit_order_state(
        forecast_inputs,
        minimum_training_samples=2,
        features=(),
        scale_features=(),
    )
    product_start, issued = _issue_adaptive_merit_order_forecast(
        state, forecast_inputs, 0
    )
    outcome = update_adaptive_merit_order_correction(
        state, "EOM", [{"product_start": product_start, "price": 49.0}]
    )[0]

    assert issued["corrected_price_mean_forecast"] == pytest.approx(
        issued["merit_order_price_forecast"] + issued["residual_mean_forecast"]
    )
    assert outcome["realised_residual"] == -6
    assert outcome["post_forecast_residual"] == -6
    assert outcome["forecast_id"] == issued["forecast_id"]
    issued["corrected_price_mean_forecast"] = 1.0
    assert outcome["corrected_price_mean_forecast"] == 55


def test_adaptive_merit_order_gaussian_quantiles_use_inverse_cdf():
    assert gaussian_residual_quantile(50, 10, 0.5) == pytest.approx(50)
    assert gaussian_residual_quantile(50, 10, 0.1) == pytest.approx(
        37.184484344, rel=1e-9
    )
    assert gaussian_residual_quantile(50, 10, 0.9) == pytest.approx(
        62.815515656, rel=1e-9
    )


def test_adaptive_features_are_separate_missing_safe_and_frozen():
    feature_state = initialize_adaptive_merit_order_feature_state(
        (
            "merit_order_price",
            "wind_availability_factor",
            "solar_availability_factor",
            "previous_day_same_hour_residual",
        )
    )
    initial = [
        {
            "delivery_time": datetime(2025, 1, 1, hour),
            "merit_order_price": 10 + hour,
            "wind_availability_factor": 0.7,
            "solar_availability_factor": 0.3,
            "residual_load": 100,
            "previous_day_same_hour_residual": None,
        }
        for hour in range(3)
    ]
    fit_adaptive_merit_order_feature_scaling(feature_state, initial)
    scaling = (
        feature_state["means"].clone(),
        feature_state["scales"].clone(),
    )
    transformed = transform_adaptive_merit_order_features(
        feature_state,
        {
            "delivery_time": datetime(2025, 1, 2),
            "merit_order_price": 10_000,
            "wind_availability_factor": 0.7,
            "solar_availability_factor": 0.3,
            "residual_load": 100,
            "previous_day_same_hour_residual": None,
        },
    )

    assert "wind_availability_factor" in feature_state["feature_names"]
    assert "solar_availability_factor" in feature_state["feature_names"]
    assert "previous_day_same_hour_residual_missing" in feature_state["feature_names"]
    assert "generator_availability" not in initial[0]
    assert isinstance(transformed, th.Tensor)
    assert transformed.dtype == th.float64
    assert th.all(th.isfinite(transformed))
    assert th.equal(feature_state["means"], scaling[0])
    assert th.equal(feature_state["scales"], scaling[1])


def test_adaptive_holiday_feature_is_optional():
    forecast_inputs = _adaptive_merit_order_forecast_inputs()
    _, model = _adaptive_merit_order_state(
        forecast_inputs,
        use_holiday_feature=True,
        holiday_dates=(date(2025, 1, 1),),
    )

    assert "holiday" in model["residual_mean_features"]["feature_names"]
    assert "holiday" in model["residual_scale_features"]["feature_names"]


def test_online_regularized_regression_sparsity_l2_and_forgetting():
    generator = th.Generator().manual_seed(7)
    relevant = th.linspace(-2, 2, 200, dtype=th.float64)
    irrelevant = th.randn(200, generator=generator, dtype=th.float64)
    features = th.column_stack((th.ones(200), relevant, irrelevant))
    target = 3 * relevant
    model = initialize_online_regularized_regression(3, 0.2, 0.1, 0.995, 1000, 1e-8)
    fit_online_regularized_regression(model, features, target)
    without_l2 = initialize_online_regularized_regression(3, 0.2, 0, 0.995, 1000, 1e-8)
    fit_online_regularized_regression(without_l2, features, target)
    gram_before = model["gram"].clone()
    update_online_regularized_regression(
        model, th.tensor([1.0, 0.0, 0.0], dtype=th.float64), 0.0
    )

    assert isinstance(model["coefficients"], th.Tensor)
    assert model["coefficients"].dtype == th.float64
    assert model["coefficients"][1].item() > 2
    assert abs(model["coefficients"][1]) < abs(without_l2["coefficients"][1])
    assert model["coefficients"][2].item() == pytest.approx(0, abs=1e-10)
    assert model["gram"][1, 1].item() == pytest.approx(
        model["forgetting_factor"] * gram_before[1, 1].item()
    )


def test_online_irls_reuses_pre_outcome_statistics():
    model = initialize_online_regularized_regression(1, 0, 0, 0.9, 100, 1e-10)
    features = th.ones((2, 1), dtype=th.float64)
    fit_online_regularized_regression(model, features, th.zeros(2))
    previous_statistics = (
        model["gram"].clone(),
        model["target_moment"].clone(),
        model["effective_weight"],
    )

    for _ in range(2):
        update_online_regularized_regression(
            model,
            th.ones(1, dtype=th.float64),
            10,
            weight=2,
            previous_statistics=previous_statistics,
        )

    assert model["gram"].item() == pytest.approx(0.9 * 2 + 2)
    assert model["target_moment"].item() == pytest.approx(20)
    assert model["effective_weight"] == pytest.approx(0.9 * 2 + 2)


def test_johnson_su_quantiles_allow_asymmetric_uncertainty():
    q10 = johnson_su_residual_quantile(50, 2, 1, 2, 0.1)
    q50 = johnson_su_residual_quantile(50, 2, 1, 2, 0.5)
    q90 = johnson_su_residual_quantile(50, 2, 1, 2, 0.9)

    assert q10 < q50 < q90
    assert q50 != pytest.approx(50)
    assert (50 - q10) != pytest.approx(q90 - 50)


def test_adaptive_default_features_match_lasso_selection():
    assert ADAPTIVE_MERIT_ORDER_SETTINGS["features"] == (
        "merit_order_price",
        "wind_availability_factor",
        "solar_availability_factor",
        "residual_load",
        "previous_day_same_hour_residual",
        "previous_day_same_hour_price",
        "weekday",
        "weekend",
    )


def test_adaptive_fallback_activation_and_no_lookahead():
    forecast_inputs = _adaptive_merit_order_forecast_inputs()
    state, model = _adaptive_merit_order_state(
        forecast_inputs,
        minimum_training_samples=2,
        features=(),
        scale_features=(),
        residual_mean_l1_regularization=0,
        residual_mean_l2_regularization=0,
    )
    first_start, first = _issue_adaptive_merit_order_forecast(state, forecast_inputs, 0)
    update_adaptive_merit_order_correction(
        state, "EOM", [{"product_start": first_start, "price": 49.0}]
    )
    second_start, second = _issue_adaptive_merit_order_forecast(
        state, forecast_inputs, 1
    )
    update_adaptive_merit_order_correction(
        state, "EOM", [{"product_start": second_start, "price": 49.0}]
    )
    third_start, third = _issue_adaptive_merit_order_forecast(state, forecast_inputs, 2)

    assert first["training_status"] == "fallback_no_uncertainty"
    assert second["corrected_price_mean_forecast"] == 55
    assert third["training_status"] == "trained"
    assert third["corrected_price_mean_forecast"] == pytest.approx(49)
    assert model["residual_scale_model"]["effective_weight"] == pytest.approx(4)
    assert third["residual_std_forecast"] > 0
    assert (
        update_adaptive_merit_order_correction(
            state, "EOM", [{"product_start": first_start, "price": 500.0}]
        )
        == []
    )
    assert first["corrected_price_mean_forecast"] == 55
    assert third_start not in model["price_by_product"]


def test_adaptive_empirical_fallback_and_time_varying_scale():
    residual_load = np.tile([0.0, 1.0], 13)
    forecast_inputs = _adaptive_merit_order_forecast_inputs(
        periods=26, merit_order=50.0, residual_load=residual_load
    )
    state, _ = _adaptive_merit_order_state(
        forecast_inputs,
        minimum_training_samples=24,
        features=(),
        scale_features=("residual_load",),
        residual_mean_l1_regularization=0,
        residual_mean_l2_regularization=0,
        residual_scale_l1_regularization=0,
        residual_scale_l2_regularization=0,
        sigma_floor=0.001,
    )
    for position in range(24):
        product_start, issued = _issue_adaptive_merit_order_forecast(
            state, forecast_inputs, position
        )
        if position == 2:
            assert issued["training_status"] == "fallback_empirical_uncertainty"
        magnitude = 1 if residual_load[position] == 0 else 10
        sign = -1 if (position // 2) % 2 else 1
        update_adaptive_merit_order_correction(
            state,
            "EOM",
            [{"product_start": product_start, "price": 50 + sign * magnitude}],
        )

    issue_time = forecast_inputs["merit_order_price"].index[23].to_pydatetime()
    products = []
    for position in (24, 25):
        product_start = (
            forecast_inputs["merit_order_price"].index[position].to_pydatetime()
        )
        products.append((product_start, product_start + timedelta(hours=1), None))
    low, high = issue_adaptive_merit_order_correction(
        state, "operator", "EOM", issue_time, products
    )
    assert low["residual_std_forecast"] < high["residual_std_forecast"]


def test_adaptive_evaluation_uses_only_earlier_issue_times():
    records = [
        {
            "issue_time": "2025-01-01",
            "product_start": "2025-01-02 00:00",
            "merit_order_price_forecast": 50,
            "corrected_price_mean_forecast": 51,
            "price_q10": 45,
            "price_q50": 51,
            "price_q90": 57,
            "realised_price": 52,
        },
        {
            "issue_time": "2025-01-01",
            "product_start": "2025-01-02 01:00",
            "merit_order_price_forecast": 50,
            "corrected_price_mean_forecast": 51,
            "price_q10": 45,
            "price_q50": 51,
            "price_q90": 57,
            "realised_price": 54,
        },
        {
            "issue_time": "2025-01-02",
            "product_start": "2025-01-03 00:00",
            "merit_order_price_forecast": 50,
            "corrected_price_mean_forecast": 53,
            "price_q10": 47,
            "price_q50": 53,
            "price_q90": 59,
            "realised_price": 53,
        },
    ]
    evaluation = evaluate_adaptive_merit_order_forecasts(records)
    assert list(evaluation["samples"]["constant_historical_bias_forecast"]) == [
        50,
        50,
        53,
    ]
    assert set(evaluation["summary"]["method"]) == {
        "merit_order_only",
        "constant_historical_bias",
        "adaptive_merit_order_correction",
    }


def test_adaptive_forecast_uses_requested_horizon_without_changing_price(
    market_setup, forecast_setup, shared_FastIndex
):
    forecaster = UnitsOperatorForecaster(
        index=shared_FastIndex, forecast_registries=get_forecast_registries()
    )
    forecaster.initialize(forecast_setup["units"], market_setup["empty_grid_markets"])
    original_price = forecaster.price["EOM"].data.copy()
    forecasts = forecaster.get_adaptive_merit_order_forecast(
        "EOM", shared_FastIndex.start, timedelta(hours=2)
    )

    assert len(forecasts) == 2
    assert forecasts[0]["product_start"] == shared_FastIndex.start + timedelta(hours=1)
    assert "EOM" in forecaster.adaptive_merit_order_state["markets"]
    assert np.array_equal(forecaster.price["EOM"].data, original_price)
