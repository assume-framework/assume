# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the rolling-horizon DSM engine and related components.

Covers:
- Rolling-horizon config loading and validation
- _parse_duration_to_steps
- _collect_series_attrs_for_window / _restore_series_attrs
- _solve_rolling_horizon_opt (result shape and properties)
- _check_and_reoptimize_rolling_window
- calculate_price_from_cleared_history adaptive algorithm
- SteelplantForecaster.update() price-sync behaviour
"""

import pandas as pd
import pytest

from assume.common.fast_pandas import FastIndex, FastSeries
from assume.common.forecast_algorithms import calculate_price_from_cleared_history
from assume.common.forecaster import SteelplantForecaster
from assume.strategies.naive_strategies import DsmEnergyOptimizationStrategy
from assume.units.steel_plant import SteelPlant

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_SHORT = 6  # Very short horizon keeps solver fast


@pytest.fixture
def dsm_components():
    return {
        "electrolyser": {
            "max_power": 100,
            "min_power": 0,
            "ramp_up": 100,
            "ramp_down": 100,
            "min_operating_time": 0,
            "min_down_time": 0,
            "efficiency": 1,
        },
        "dri_plant": {
            "specific_hydrogen_consumption": 1,
            "specific_natural_gas_consumption": 1,
            "specific_electricity_consumption": 1,
            "specific_iron_ore_consumption": 1,
            "max_power": 100,
            "min_power": 0,
            "fuel_type": "hydrogen",
            "ramp_up": 100,
            "ramp_down": 100,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
        "eaf": {
            "max_power": 100,
            "min_power": 0,
            "specific_electricity_consumption": 1,
            "specific_dri_demand": 1,
            "specific_lime_demand": 1,
            "lime_co2_factor": 0.1,
            "ramp_up": 100,
            "ramp_down": 100,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
    }


@pytest.fixture
def rh_config():
    """Minimal rolling-horizon config: 4h look-ahead, 2h commit, 2h step."""
    return {
        "horizon_mode": "rolling_horizon",
        "look_ahead_horizon": "4h",
        "commit_horizon": "2h",
        "rolling_step": "2h",
    }


def _make_rh_plant(dsm_components, rh_config, n=N_SHORT, demand=0):
    """Return a SteelPlant configured for rolling-horizon with n timesteps.

    demand=0 is used so that ``steel_output_association_constraint`` (sum==0)
    and ``demand_upper_bound`` (sum_power <= 0) are both trivially satisfied by
    the all-zeros solution.  This keeps the windows feasible and lets us test
    the rolling-horizon machinery without worrying about the steel-plant
    unit-of-measure mismatch between power and steel production.
    """
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    prices = [30 + 10 * (i % 5) for i in range(n)]
    forecaster = SteelplantForecaster(
        index,
        electricity_price=prices,
        fuel_prices={"natural_gas": [20] * n, "co2": [15] * n},
    )
    return SteelPlant(
        id="test_rh_plant",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=demand,
        technology="steel_plant",
        dsm_optimisation_config=rh_config,
    )


# ---------------------------------------------------------------------------
# 1. Rolling-horizon config validation
# ---------------------------------------------------------------------------


def test_rolling_horizon_default_mode_is_full_horizon(dsm_components):
    """Without dsm_optimisation_config, horizon mode defaults to 'full_horizon'."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
    )
    plant = SteelPlant(
        id="test_default",
        unit_operator="op",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=100,
        technology="steel_plant",
    )
    assert plant._horizon_mode == "full_horizon"


def test_rolling_horizon_mode_stored_from_config(dsm_components, rh_config):
    """dsm_optimisation_config is correctly parsed into rolling-horizon settings."""
    plant = _make_rh_plant(dsm_components, rh_config)
    assert plant._horizon_mode == "rolling_horizon"
    assert plant._rh_look_ahead == "4h"
    assert plant._rh_commit == "2h"
    assert plant._rh_step == "2h"


def test_rolling_horizon_missing_params_raises_value_error(dsm_components):
    """Specifying rolling_horizon mode without all three sub-params raises ValueError."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
    )
    with pytest.raises(ValueError, match="Rolling horizon mode requires"):
        SteelPlant(
            id="bad_rh",
            unit_operator="op",
            objective="min_variable_cost",
            flexibility_measure="cost_based_load_shift",
            bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
            node="south",
            components=dsm_components,
            forecaster=forecaster,
            demand=100,
            technology="steel_plant",
            dsm_optimisation_config={
                "horizon_mode": "rolling_horizon",
                "look_ahead_horizon": "24h",
                # missing commit_horizon and rolling_step
            },
        )


# ---------------------------------------------------------------------------
# 2. _parse_duration_to_steps
# ---------------------------------------------------------------------------


def test_parse_duration_to_steps_hourly_index(dsm_components, rh_config):
    plant = _make_rh_plant(dsm_components, rh_config)
    assert plant._parse_duration_to_steps("1h") == 1
    assert plant._parse_duration_to_steps("4h") == 4
    assert plant._parse_duration_to_steps("24h") == 24
    assert plant._parse_duration_to_steps("48h") == 48


# ---------------------------------------------------------------------------
# 3. _collect_series_attrs_for_window / _restore_series_attrs
# ---------------------------------------------------------------------------


def test_collect_and_restore_series_attrs(dsm_components, rh_config):
    """_collect replaces FastSeries with slices; _restore puts originals back."""
    plant = _make_rh_plant(dsm_components, rh_config)
    plant.setup_model(presolve=False)

    N = len(plant.index)
    # electricity_price is a FastSeries of length N on the plant
    original_price = plant.electricity_price

    saved = plant._collect_series_attrs_for_window(0, 4, N)

    # After collection the attr is replaced with a numpy array (window slice)
    assert len(plant.electricity_price) == 4

    plant._restore_series_attrs(saved)

    # After restore the original FastSeries is back
    assert len(plant.electricity_price) == N
    assert isinstance(plant.electricity_price, FastSeries)


# ---------------------------------------------------------------------------
# 4. Rolling-horizon full solve
# ---------------------------------------------------------------------------


def test_rolling_horizon_result_length_matches_full_horizon(dsm_components, rh_config):
    """opt_power_requirement covers every step of the full horizon."""
    plant = _make_rh_plant(dsm_components, rh_config, n=N_SHORT)
    plant.setup_model(presolve=False)
    plant.determine_optimal_operation_without_flex()

    assert plant.opt_power_requirement is not None
    assert isinstance(plant.opt_power_requirement, FastSeries)
    assert len(plant.opt_power_requirement) == N_SHORT


def test_rolling_horizon_opt_power_nonnegative(dsm_components, rh_config):
    """All committed power values from rolling horizon are non-negative."""
    plant = _make_rh_plant(dsm_components, rh_config, n=N_SHORT)
    plant.setup_model(presolve=False)
    plant.determine_optimal_operation_without_flex()

    assert all(v >= 0 for v in plant.opt_power_requirement.data)


def test_rolling_horizon_variable_cost_series_length(dsm_components, rh_config):
    """variable_cost_series has the same length as the full horizon."""
    plant = _make_rh_plant(dsm_components, rh_config, n=N_SHORT)
    plant.setup_model(presolve=False)
    plant.determine_optimal_operation_without_flex()

    assert len(plant.variable_cost_series) == N_SHORT


def test_rolling_horizon_total_cost_is_sum_of_var_cost(dsm_components, rh_config):
    """total_cost equals the sum of variable_cost_series."""
    plant = _make_rh_plant(dsm_components, rh_config, n=N_SHORT)
    plant.setup_model(presolve=False)
    plant.determine_optimal_operation_without_flex()

    assert abs(plant.total_cost - sum(plant.variable_cost_series.data)) < 1e-6


def test_rolling_horizon_optimisation_counter_starts_at_zero(dsm_components, rh_config):
    """optimisation_counter is initialised to 0 during setup_model.

    The counter is only incremented by the bidding strategy (naive_strategies.py)
    when it first generates bids, not by _solve_rolling_horizon_opt itself.
    Therefore it should still be 0 right after the rolling-horizon solve.
    """
    plant = _make_rh_plant(dsm_components, rh_config, n=N_SHORT)
    plant.setup_model(presolve=False)
    plant.determine_optimal_operation_without_flex()

    assert plant.optimisation_counter == 0


# ---------------------------------------------------------------------------
# 5. _check_and_reoptimize_rolling_window
# ---------------------------------------------------------------------------


def test_check_reoptimize_returns_false_for_full_horizon_mode(dsm_components):
    """In full_horizon mode, _check_and_reoptimize_rolling_window always returns False."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
    )
    plant = SteelPlant(
        id="test_full_horizon_check",
        unit_operator="op",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=100,
        technology="steel_plant",
    )
    # full_horizon mode → always False regardless of time
    result = plant._check_and_reoptimize_rolling_window(pd.Timestamp("2023-01-01"))
    assert result is False


def test_check_reoptimize_returns_false_before_threshold(dsm_components, rh_config):
    """Before reaching the next window boundary, no re-optimization is needed."""
    plant = _make_rh_plant(dsm_components, rh_config, n=N_SHORT)
    plant.setup_model(presolve=False)
    # Simulate that we've already optimized through step 4
    plant._rh_optimized_until_step = 4

    # Requesting step 2 (< optimized_until_step) should not trigger re-opt
    t2 = pd.Timestamp("2023-01-01 02:00")
    result = plant._check_and_reoptimize_rolling_window(t2)
    assert result is False


# ---------------------------------------------------------------------------
# 6. calculate_price_from_cleared_history
# ---------------------------------------------------------------------------


class _MockUnit:
    """Minimal unit stand-in for algorithm tests."""

    def __init__(self, cleared_prices):
        self.outputs = {"energy_accepted_price": cleared_prices}


def test_price_from_cleared_history_produces_updated_forecast():
    """With sufficient cleared history, the EOM forecast is updated."""
    # 24 non-zero cleared prices
    cleared = pd.Series(
        [50.0 + i for i in range(24)],
        index=pd.date_range("2023-01-01", periods=24, freq="h"),
    )
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )
    current_forecast = {"EOM": FastSeries(index=fast_index, value=40.0)}

    result = calculate_price_from_cleared_history(
        current_forecast=current_forecast,
        preprocess_information={},
        unit=_MockUnit(cleared),
        current_time=pd.Timestamp("2023-01-02"),
    )

    assert "EOM" in result
    # The forecast should have been modified (not all 40.0)
    assert not all(v == 40.0 for v in result["EOM"].data[:48])


def test_price_from_cleared_history_floor_price():
    """All forecasted prices must be >= 5.0 (the algorithm floor)."""
    # Very low cleared prices to trigger the floor
    cleared = pd.Series(
        [1.0] * 24,
        index=pd.date_range("2023-01-01", periods=24, freq="h"),
    )
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )
    current_forecast = {"EOM": FastSeries(index=fast_index, value=10.0)}

    result = calculate_price_from_cleared_history(
        current_forecast=current_forecast,
        preprocess_information={},
        unit=_MockUnit(cleared),
        current_time=pd.Timestamp("2023-01-02"),
    )

    assert all(v >= 5.0 for v in result["EOM"].data)


def test_price_from_cleared_history_insufficient_data_returns_current():
    """With fewer than 2 cleared-price data points, the current forecast is unchanged."""
    cleared = pd.Series([50.0])  # Only 1 data point
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )
    current_forecast = {"EOM": FastSeries(index=fast_index, value=40.0)}

    result = calculate_price_from_cleared_history(
        current_forecast=current_forecast,
        preprocess_information={},
        unit=_MockUnit(cleared),
        current_time=pd.Timestamp("2023-01-02"),
    )

    # Should be the same object / unchanged
    assert result is current_forecast


def test_price_from_cleared_history_no_unit_returns_current():
    """Without a unit kwarg, the current forecast is returned unchanged."""
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )
    current_forecast = {"EOM": FastSeries(index=fast_index, value=40.0)}

    result = calculate_price_from_cleared_history(
        current_forecast=current_forecast,
        preprocess_information={},
    )
    assert result is current_forecast


def test_price_from_cleared_history_empty_outputs_returns_current():
    """A unit with an empty outputs dict falls back to the current forecast."""

    class _EmptyUnit:
        outputs = {"energy_accepted_price": pd.Series([], dtype=float)}

    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )
    current_forecast = {"EOM": FastSeries(index=fast_index, value=40.0)}

    result = calculate_price_from_cleared_history(
        current_forecast=current_forecast,
        preprocess_information={},
        unit=_EmptyUnit(),
    )
    assert result is current_forecast


def test_price_from_cleared_history_all_zero_prices_returns_current():
    """Zero-only cleared prices (all filtered out) → falls back to current forecast."""
    cleared = pd.Series(
        [0.0] * 24,
        index=pd.date_range("2023-01-01", periods=24, freq="h"),
    )
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )
    current_forecast = {"EOM": FastSeries(index=fast_index, value=40.0)}

    result = calculate_price_from_cleared_history(
        current_forecast=current_forecast,
        preprocess_information={},
        unit=_MockUnit(cleared),
    )
    # All zeros filtered → < 2 actual prices → fallback
    assert result is current_forecast


# ---------------------------------------------------------------------------
# 6b. calculate_price_from_cleared_history – behavioural algorithm tests
# ---------------------------------------------------------------------------


def test_price_from_cleared_history_high_prices_produce_higher_forecast():
    """Cleared prices of 100 EUR/MWh must produce a higher forecast than 20 EUR/MWh."""
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )

    cleared_high = pd.Series(
        [100.0] * 24,
        index=pd.date_range("2023-01-01", periods=24, freq="h"),
    )
    cleared_low = pd.Series(
        [20.0] * 24,
        index=pd.date_range("2023-01-01", periods=24, freq="h"),
    )

    forecast_high = calculate_price_from_cleared_history(
        current_forecast={"EOM": FastSeries(index=fast_index, value=40.0)},
        preprocess_information={},
        unit=_MockUnit(cleared_high),
        current_time=pd.Timestamp("2023-01-02"),
    )
    forecast_low = calculate_price_from_cleared_history(
        current_forecast={"EOM": FastSeries(index=fast_index, value=40.0)},
        preprocess_information={},
        unit=_MockUnit(cleared_low),
        current_time=pd.Timestamp("2023-01-02"),
    )

    avg_high = sum(forecast_high["EOM"].data) / len(forecast_high["EOM"].data)
    avg_low = sum(forecast_low["EOM"].data) / len(forecast_low["EOM"].data)
    assert avg_high > avg_low, (
        f"Higher cleared prices should produce a higher forecast: "
        f"avg_high={avg_high:.2f} vs avg_low={avg_low:.2f}"
    )


def test_price_from_cleared_history_rising_trend_elevates_later_forecast_values():
    """When cleared prices rise over 24 h the forecasted average should exceed the
    simple average of the cleared prices, reflecting that the algorithm anchors
    around the most-recent (highest) prices and applies a positive trend."""
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )
    # Prices rise linearly from 20 to 100 EUR/MWh over 24 h; simple avg = 60
    rising_prices = pd.Series(
        [20.0 + 80.0 * i / 23 for i in range(24)],
        index=pd.date_range("2023-01-01", periods=24, freq="h"),
    )
    cleared_avg = sum(rising_prices) / len(rising_prices)  # ≈ 60

    result = calculate_price_from_cleared_history(
        current_forecast={"EOM": FastSeries(index=fast_index, value=50.0)},
        preprocess_information={},
        unit=_MockUnit(rising_prices),
        current_time=pd.Timestamp("2023-01-02"),
    )

    data = list(result["EOM"].data)
    assert len(data) == 48
    forecast_avg = sum(data) / len(data)
    # With rising prices ending at 100 EUR/MWh the forecast average must be
    # substantially above the historical simple average of ~60 EUR/MWh.
    assert forecast_avg > cleared_avg, (
        f"Forecast avg {forecast_avg:.2f} should exceed cleared avg "
        f"{cleared_avg:.2f} when prices are rising"
    )


# ---------------------------------------------------------------------------
# 4b. Rolling-window window-level optimisation
#     Uses _check_and_reoptimize_rolling_window (→ _solve_rolling_horizon_next_window)
#     with non-zero demand.  setup_model(presolve=True) uses the full-horizon path
#     which does NOT apply demand_upper_bound, so the initial solve is always
#     feasible.  The per-window constraint (window_demand_con) operates on
#     eaf.steel_output (steel units), not on total_power_input, so it is also
#     consistent with the equality constraint sum(steel_output)==remaining_demand.
# ---------------------------------------------------------------------------


def _make_rh_plant_with_demand(dsm_components, rh_config, prices, demand, n=N_SHORT):
    """Return a rolling-horizon SteelPlant pre-solved via setup_model(presolve=True).

    The initial solve uses the full-horizon path (switch_flex_off=False), so it
    is not subject to the demand_upper_bound unit-of-measure issue and will
    succeed with any positive demand value.
    """
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    forecaster = SteelplantForecaster(
        index,
        electricity_price=prices,
        fuel_prices={"natural_gas": [20] * n, "co2": [15] * n},
    )
    plant = SteelPlant(
        id="test_rh_window_plant",
        unit_operator="op",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=demand,
        technology="steel_plant",
        dsm_optimisation_config=rh_config,
    )
    plant.setup_model(presolve=True)
    return plant


def test_rolling_window_first_trigger_at_step_zero(dsm_components, rh_config):
    """_check_and_reoptimize_rolling_window returns True at t0 (step 0 >= threshold 0)."""
    prices = [50.0] * N_SHORT
    plant = _make_rh_plant_with_demand(dsm_components, rh_config, prices, demand=60)

    t0 = pd.Timestamp("2023-01-01 00:00")
    triggered = plant._check_and_reoptimize_rolling_window(t0)
    assert triggered is True
    # After committing window 0 (steps 0-1), optimized_until_step advances by rolling_steps
    assert plant._rh_optimized_until_step == 2


def test_rolling_window_concentrates_production_in_cheap_committed_hours(
    dsm_components, rh_config
):
    """Cost-minimising rolling-horizon optimizer defers committed production to cheap hours.

    Horizon layout (n=6, 4h look-ahead, 2h commit, 2h step):
      Window 0: look-ahead steps 0-3, commits steps 0-1.
        Prices: [200, 200, 200, 10]  → cheapest hour is step 3 (look-ahead only).
        The optimizer schedules all 60 units at step 3; committed steps 0-1 get 0.

      Window 1: look-ahead steps 2-5, commits steps 2-3.
        Prices: [200, 10, 10, 10]  → cheapest hour in look-ahead is step 3.
        Remaining demand = 60.  Optimizer places 60 units at step 3 (local t=1).
        Committed step 3 has production; committed step 2 is idle.
    """
    n = 6
    # First 3 hours very expensive, last 3 cheap
    prices = [200.0] * 3 + [10.0] * 3
    demand = 60
    plant = _make_rh_plant_with_demand(dsm_components, rh_config, prices, demand, n=n)

    # --- Window 0 ---
    t0 = pd.Timestamp("2023-01-01 00:00")
    triggered0 = plant._check_and_reoptimize_rolling_window(t0)
    assert triggered0 is True

    # Steps 0 and 1 are expensive; the optimizer puts production at step 3 (cheap,
    # look-ahead only) so committed production at steps 0-1 must be zero.
    prod0 = plant._rh_full_horizon_production[0]
    prod1 = plant._rh_full_horizon_production[1]
    assert prod0 == pytest.approx(0.0, abs=1e-3), (
        f"Expected zero committed production at expensive step 0, got {prod0:.4f}"
    )
    assert prod1 == pytest.approx(0.0, abs=1e-3), (
        f"Expected zero committed production at expensive step 1, got {prod1:.4f}"
    )

    # --- Window 1 ---
    t2 = pd.Timestamp("2023-01-01 02:00")
    triggered1 = plant._check_and_reoptimize_rolling_window(t2)
    assert triggered1 is True

    # Step 3 is cheap (price=10); optimizer commits production there
    prod3 = plant._rh_full_horizon_production[3]
    assert prod3 > 0.0, (
        f"Expected non-zero committed production at cheap step 3, got {prod3:.4f}"
    )

    # After windows 0 and 1 the total committed production must equal demand
    total_committed = sum(plant._rh_full_horizon_production[i] for i in range(4))
    assert total_committed == pytest.approx(demand, rel=1e-3), (
        f"Expected total committed = {demand}, got {total_committed:.4f}"
    )


def test_rolling_window_total_production_meets_demand_after_all_windows(
    dsm_components, rh_config
):
    """After all three windows complete, the cumulative committed production equals demand."""
    n = 6
    prices = [30.0 + 10.0 * (i % 3) for i in range(n)]  # modest variation
    demand = 60
    plant = _make_rh_plant_with_demand(dsm_components, rh_config, prices, demand, n=n)

    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    # Drive through all three rolling windows
    for step, ts in enumerate(timestamps[::2]):  # step=0, 2, 4
        plant._check_and_reoptimize_rolling_window(ts)

    total = sum(plant._rh_full_horizon_production[i] for i in range(n))
    assert total == pytest.approx(demand, rel=1e-3), (
        f"Expected total production == {demand} after all windows, got {total:.4f}"
    )


def test_rolling_window_second_window_triggered_at_commit_boundary(
    dsm_components, rh_config
):
    """After the first window commits steps 0-1, a request at step 2 triggers re-opt."""
    prices = [50.0] * N_SHORT
    plant = _make_rh_plant_with_demand(dsm_components, rh_config, prices, demand=60)

    t0 = pd.Timestamp("2023-01-01 00:00")
    plant._check_and_reoptimize_rolling_window(t0)

    # _rh_optimized_until_step is now 2; requesting step 2 must trigger a new window
    t2 = pd.Timestamp("2023-01-01 02:00")
    triggered = plant._check_and_reoptimize_rolling_window(t2)
    assert triggered is True
    assert plant._rh_optimized_until_step == 4


# ---------------------------------------------------------------------------
# 7. SteelplantForecaster.update() – price-sync behavioural test
# ---------------------------------------------------------------------------


def test_steelplant_forecaster_update_syncs_electricity_price_to_unit(dsm_components):
    """After update() the unit's electricity_price reflects the cleared-price history.

    The forecaster is configured with ``update_price='price_from_cleared_history'``.
    The unit carries cleared prices of 100 EUR/MWh in its outputs.  After calling
    forecaster.update(unit=plant, …) the plant's electricity_price must be higher
    than the initial flat forecast of 40 EUR/MWh.
    """
    from assume.common.forecast_algorithms import get_forecast_registries

    n = 24
    index = pd.date_range("2023-01-01", periods=n, freq="h")

    # Build a 48-h forecast window (the algorithm always writes 48 values)
    fast_index = FastIndex(
        start=pd.Timestamp("2023-01-02"),
        end=pd.Timestamp("2023-01-03 23:00"),
        freq="h",
    )

    forecaster = SteelplantForecaster(
        index=FastIndex(start=index[0], end=index[-1], freq="h"),
        electricity_price=[40.0] * n,
        fuel_prices={"natural_gas": [30] * n, "co2": [20] * n},
        forecast_algorithms={"update_price": "price_from_cleared_history"},
        forecast_registries=get_forecast_registries(),
    )
    # Provide the preprocess_information that the parent update() calls require
    forecaster.preprocess_information = {
        "price": {},
        "residual_load": {},
        "congestion_signal": {},
        "renewable_utilisation": {},
    }
    # Seed the price dict that the algorithm will update
    forecaster.price = {"EOM": FastSeries(index=fast_index, value=40.0)}

    plant = SteelPlant(
        id="test_update_sync",
        unit_operator="op",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=100,
        technology="steel_plant",
    )

    # 24 cleared prices of 100 EUR/MWh (much higher than the initial 40)
    plant.outputs = {
        "energy_accepted_price": pd.Series(
            [100.0] * n,
            index=pd.date_range("2023-01-01", periods=n, freq="h"),
        )
    }

    forecaster.update(unit=plant, current_time=pd.Timestamp("2023-01-02"))

    # After the update the plant's electricity_price must reflect the higher
    # cleared prices; at least some values must exceed the original 40 EUR/MWh.
    updated = list(plant.electricity_price.data)
    assert any(v > 40.0 for v in updated), (
        "electricity_price should be updated above 40 EUR/MWh after update() "
        f"with 100 EUR/MWh cleared prices; got max={max(updated):.2f}"
    )

if __name__ == "__main__":
    pytest.main(["-s", __file__])