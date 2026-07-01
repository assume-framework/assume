# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for the rolling-horizon DSM engine and related components.

Covers:
- Rolling-horizon config loading and validation
- _parse_duration_to_steps
- _collect_series_attrs_for_window / _restore_series_attrs
- _check_and_reoptimize_rolling_window / _solve_rolling_horizon_next_window
- end-to-end steel-plant scenario through World.run()
"""

import pandas as pd
import pytest

from assume.common.fast_pandas import FastSeries
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
    assert plant.horizon_mode == "full_horizon"


def test_rolling_horizon_mode_stored_from_config(dsm_components, rh_config):
    """dsm_optimisation_config is correctly parsed into rolling-horizon settings."""
    plant = _make_rh_plant(dsm_components, rh_config)
    assert plant.horizon_mode == "rolling_horizon"
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
# 7. State carryover across rolling windows
#     SoC / operational_status computed at the end of one committed window must
#     become the initial condition of the next window. Previously the update was
#     written to a discarded local dict, so every window restarted from t=0.
# ---------------------------------------------------------------------------


class _SolvedVariable:
    """A solved Pyomo variable: indexing by time step returns its optimised value."""

    def __init__(self, value):
        self._value = value

    def __getitem__(self, time_step):
        return self._value


class _SolvedBlock:
    """A solved component block exposing the state read back at a window boundary."""

    def __init__(self, soc=None, operational_status=None):
        if soc is not None:
            self.soc = _SolvedVariable(soc)
        if operational_status is not None:
            self.operational_status = _SolvedVariable(operational_status)


class _SolvedInstance:
    """A solved rolling-horizon window model, keyed by component block name."""

    def __init__(self, blocks: dict):
        self.dsm_blocks = blocks


def test_update_init_states_records_unseeded_blocks():
    """_update_init_states must record state for blocks the seeding step never touched."""
    plant = object.__new__(SteelPlant)
    init_states = {"dri_storage": {"soc": 0.5}}  # only storage pre-seeded
    instance = _SolvedInstance(
        {
            "dri_storage": _SolvedBlock(soc=0.9),
            "electrolyser": _SolvedBlock(operational_status=1),  # NOT pre-seeded
        }
    )

    plant._update_init_states(instance, commit_local=1, init_states=init_states)

    # Pre-seeded storage advances in place ...
    assert init_states["dri_storage"]["soc"] == 0.9
    # ... and the non-seeded electrolyser's status is now recorded for the next window.
    assert init_states["electrolyser"]["operational_status"] == 1


def test_rolling_horizon_state_persists_across_windows(dsm_components, rh_config):
    """Carried state is seeded once and advanced, not reset to t=0 each window."""
    import copy

    comps = copy.deepcopy(dsm_components)
    comps["electrolyser"]["min_power"] = 10  # forces an operational_status variable
    prices = [50.0] * N_SHORT
    plant = _make_rh_plant_with_demand(comps, rh_config, prices, demand=60)

    # Not seeded until the first window solves.
    assert plant._rh_init_states is None

    t0 = pd.Timestamp("2023-01-01 00:00")
    assert plant._check_and_reoptimize_rolling_window(t0) is True

    # Seed-once wiring populated the persistent store, and the broadened update
    # recorded on/off status for the electrolyser (never seeded from config).
    assert plant._rh_init_states is not None
    assert "operational_status" in plant._rh_init_states.get("electrolyser", {})

    seeded = plant._rh_init_states
    t2 = pd.Timestamp("2023-01-01 02:00")
    assert plant._check_and_reoptimize_rolling_window(t2) is True
    # Same dict object reused across windows → state was seeded once, not re-collected.
    assert plant._rh_init_states is seeded


# ---------------------------------------------------------------------------
# 8. End-to-end: a rolling-horizon steel-plant scenario through World.run()
# ---------------------------------------------------------------------------
#
# Every test above builds the SteelPlant / forecaster directly and therefore
# bypasses two real code paths:
#   * the CSV scenario loader (load_config_and_create_forecaster), and
#   * the market -> bidding-strategy -> unit dispatch loop inside World.run().
#
# Both paths have hidden load-time / first-bid crashes that green unit tests
# did not catch. This integration test exercises the full pipeline on the
# example_03 steel-plant scenario (configured for rolling horizon via the
# per-plant columns in industrial_dsm_units.csv) and asserts it runs to
# completion and produces a dispatch schedule.


def test_fixture_loader_builds_steelplant_forecaster():
    """Fast loader-path smoke test for the self-contained ``rolling_horizon_dsm`` fixture.

    Builds the scenario's forecasters (no market run) and checks the three-component
    steel plant comes up with a ``SteelplantForecaster``.
    """
    from assume.scenario.loader_csv import load_config_and_create_forecaster

    scenario_data = load_config_and_create_forecaster(
        inputs_path="tests/fixtures",
        scenario="rolling_horizon_dsm",
        study_case="base",
    )
    assert "test_steel_plant_rh" in scenario_data["unit_forecasts"]


def test_fixture_rolling_horizon_runs_end_to_end():
    """The self-contained ``rolling_horizon_dsm`` fixture runs the full pipeline.

    Exercises market clearing -> ``DsmEnergyOptimizationStrategy`` -> per-window
    solve across the whole 24h horizon on a real three-component steel plant
    (electrolyser -> dri_plant -> eaf). Asserts the rolling window advanced over
    the full horizon and that the seeded carried-state store tracked the
    electrolyser's on/off status across windows (Bug 2 fix, end-to-end).
    """
    from assume import World
    from assume.scenario.loader_csv import load_scenario_folder

    world = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world,
        inputs_path="tests/fixtures",
        scenario="rolling_horizon_dsm",
        study_case="base",
    )

    plant = world.units["test_steel_plant_rh"]
    assert plant.horizon_mode == "rolling_horizon"
    assert {"electrolyser", "dri_plant", "eaf"}.issubset(plant.components.keys())

    world.run()

    # The rolling window advanced across the entire horizon (not a one-shot solve).
    assert plant._rh_optimized_until_step == len(plant.index)
    # Seed-once carried-state store populated, and on/off status carried across
    # windows for the electrolyser (min_power > 0 gives it an operational_status).
    assert plant._rh_init_states is not None
    assert "operational_status" in plant._rh_init_states.get("electrolyser", {})


def test_steelplant_rolling_horizon_scenario_runs_end_to_end():
    """A rolling-horizon steel-plant scenario loads and runs through World.run().

    Regression guard for two bugs that bypass-the-loader unit tests cannot see:
    an UnboundLocalError while the loader builds the SteelplantForecaster, and a
    crash in the DSM bidding strategy on the first market round.
    """
    from assume import World
    from assume.scenario.loader_csv import load_scenario_folder

    world = World(database_uri=None, export_csv_path=None)
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario="example_03",
        study_case="base_case_2019_with_DSM",
    )

    # Loader path: the steel plant must come up with its rolling-horizon config
    # read from the per-plant CSV columns (not the removed config.yaml block).
    plant = world.units["A360"]
    assert plant.horizon_mode == "rolling_horizon"
    assert plant._rh_look_ahead == "72h"

    # Full pipeline: market clearing -> DSM strategy -> rolling-horizon solve.
    world.run()

    # NOTE: World.run() logs and swallows exceptions raised inside a unit's
    # bidding strategy, so "world.run() returned" is NOT proof the strategy
    # succeeded. opt_power_requirement is also populated by the setup-time
    # presolve, so it is not a reliable signal either.
    #
    # _rh_optimized_until_step starts at 0 and is advanced ONLY by
    # _solve_rolling_horizon_next_window, which the bidding strategy reaches
    # *after* the forecaster.update() call. A non-zero value therefore proves
    # the market-time rolling-horizon path executed without crashing.
    assert plant.opt_power_requirement is not None
    assert plant._rh_optimized_until_step > 0


if __name__ == "__main__":
    pytest.main(["-s", __file__])
