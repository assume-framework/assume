# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import SteelplantForecaster
from assume.strategies.naive_strategies import (
    DsmEnergyNaiveRedispatchStrategy,
    DsmEnergyOptimizationStrategy,
)
from assume.units.steel_plant import SteelPlant


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


def create_steel_plant(dsm_components, flexibility_measure):
    """Helper function to create a SteelPlant with a specific flexibility measure."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
        renewable_utilisation_signal=[0.1 * i for i in range(24)],
    )

    bidding_strategies = {
        "EOM": DsmEnergyOptimizationStrategy(),
        "RD": DsmEnergyNaiveRedispatchStrategy(),
    }

    plant = SteelPlant(
        id=f"test_steel_plant_{flexibility_measure}",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure=flexibility_measure,
        bidding_strategies=bidding_strategies,
        node="south",
        components=dsm_components,
        forecaster=forecast,
        demand=1000,
        technology="steel_plant",
        cost_tolerance=10,
        congestion_threshold=0.8,
        peak_load_cap=95,
    )
    plant.setup_model()
    return plant


@pytest.fixture
def steel_plant_cost_based(dsm_components):
    """Fixture for cost-based load shifting."""
    return create_steel_plant(dsm_components, "cost_based_load_shift")


@pytest.fixture
def steel_plant_congestion(dsm_components):
    """Fixture for congestion management flexibility."""
    return create_steel_plant(dsm_components, "congestion_management_flexibility")


@pytest.fixture
def steel_plant_peak_shifting(dsm_components):
    """Fixture for peak load shifting."""
    return create_steel_plant(dsm_components, "peak_load_shifting")


@pytest.fixture
def steel_plant_renewable_utilisation(dsm_components):
    """Fixture for renewable utilisation flexibility."""
    return create_steel_plant(dsm_components, "renewable_utilisation")


# Test cases
def test_initialize_components(steel_plant_cost_based):
    """Verify components are properly initialized."""
    assert "electrolyser" in steel_plant_cost_based.model.dsm_blocks.keys()
    assert "dri_plant" in steel_plant_cost_based.model.dsm_blocks.keys()
    assert "eaf" in steel_plant_cost_based.model.dsm_blocks.keys()


def test_determine_optimal_operation_without_flex(steel_plant_cost_based):
    """Test optimal operation without flexibility for cost-based load shifting."""
    steel_plant_cost_based.determine_optimal_operation_without_flex()
    assert steel_plant_cost_based.opt_power_requirement is not None
    assert isinstance(steel_plant_cost_based.opt_power_requirement, FastSeries)


def test_congestion_management_flexibility(steel_plant_congestion):
    """
    Test congestion management flexibility measure.
    """
    steel_plant_congestion.determine_optimal_operation_with_flex()

    # Calculate the congestion indicator
    congestion_indicator = {
        t: int(
            steel_plant_congestion.congestion_signal.iloc[t]
            > steel_plant_congestion.congestion_threshold
        )
        for t in range(len(steel_plant_congestion.index))
    }

    instance = steel_plant_congestion.model.create_instance()
    instance = steel_plant_congestion.switch_to_flex(instance)

    # Set the congestion_indicator in the instance
    # Delete the old component if it exists to avoid Pyomo warnings
    if hasattr(instance, "congestion_indicator"):
        instance.del_component("congestion_indicator")
    instance.congestion_indicator = pyo.Param(
        instance.time_steps,
        initialize=congestion_indicator,
        within=pyo.Binary,
    )

    # Solve the instance
    steel_plant_congestion.solver.solve(instance, tee=False)

    # Calculate adjusted total power input
    adjusted_total_power_input = [
        instance.total_power_input[t].value
        + instance.load_shift_pos[t].value
        - instance.load_shift_neg[t].value
        for t in instance.time_steps
    ]

    # Assert load shifting respects congestion indicator
    for t in instance.time_steps:
        if instance.congestion_indicator[t] == 1:  # Congestion period
            assert (
                adjusted_total_power_input[t] <= instance.total_power_input[t].value
            ), f"Load shift not aligned with congestion signal at time {t}"


def test_peak_load_shifting(steel_plant_peak_shifting):
    """
    Test peak load shifting flexibility measure.
    """
    steel_plant_peak_shifting.determine_optimal_operation_with_flex()

    # Calculate the peak load cap value
    max_load = steel_plant_peak_shifting.opt_power_requirement.max()
    peak_load_cap_value = max_load * (steel_plant_peak_shifting.peak_load_cap / 100)
    peak_indicator = {
        t: int(
            steel_plant_peak_shifting.opt_power_requirement.iloc[t]
            > peak_load_cap_value
        )
        for t in range(len(steel_plant_peak_shifting.opt_power_requirement))
    }

    instance = steel_plant_peak_shifting.model.create_instance()
    instance = steel_plant_peak_shifting.switch_to_flex(instance)

    # Set the peak_load_cap_value and peak_indicator in the instance
    # Delete old components if they exist to avoid Pyomo warnings
    if hasattr(instance, "peak_load_cap_value"):
        instance.del_component("peak_load_cap_value")
    if hasattr(instance, "peak_indicator"):
        instance.del_component("peak_indicator")

    instance.peak_load_cap_value = pyo.Param(
        initialize=peak_load_cap_value,
        within=pyo.NonNegativeReals,
    )
    instance.peak_indicator = pyo.Param(
        instance.time_steps,
        initialize=peak_indicator,
        within=pyo.Binary,
    )

    # Solve the instance
    steel_plant_peak_shifting.solver.solve(instance, tee=False)

    # Calculate adjusted total power input
    adjusted_total_power_input = [
        instance.total_power_input[t].value
        + instance.load_shift_pos[t].value
        - instance.load_shift_neg[t].value
        for t in instance.time_steps
    ]

    # Assert load shifting respects peak load cap
    for t in instance.time_steps:
        if instance.peak_indicator[t] == 1:
            assert adjusted_total_power_input[t] <= instance.peak_load_cap_value, (
                f"Peak load exceeded at time {t}"
            )


def test_renewable_utilisation(steel_plant_renewable_utilisation):
    """
    Tests the renewable utilisation flexibility measure by ensuring that the load increase aligns
    with the renewable signal intensity and does not exceed allowable thresholds.
    """
    # Set the flexibility measure to renewable utilisation
    steel_plant_renewable_utilisation.flexibility_measure = "renewable_utilisation"
    steel_plant_renewable_utilisation.determine_optimal_operation_with_flex()

    # Normalization of renewable_utilisation_signal
    min_signal = steel_plant_renewable_utilisation.renewable_utilisation_signal.min()
    max_signal = steel_plant_renewable_utilisation.renewable_utilisation_signal.max()

    if max_signal - min_signal > 0:
        renewable_signal_normalised = (
            steel_plant_renewable_utilisation.renewable_utilisation_signal - min_signal
        ) / (max_signal - min_signal)
    else:
        renewable_signal_normalised = FastSeries(
            index=steel_plant_renewable_utilisation.renewable_utilisation_signal.index,
            value=1,
        )

    # Map normalized renewable signals to a dictionary for Pyomo parameters
    renewable_signal_dict = {
        t: renewable_signal_normalised.iloc[t]
        for t in range(len(renewable_signal_normalised))
    }

    instance = steel_plant_renewable_utilisation.model.create_instance()
    instance = steel_plant_renewable_utilisation.switch_to_flex(instance)

    # Set the normalized renewable signal in the instance
    # Delete the old component if it exists to avoid Pyomo warnings
    if hasattr(instance, "renewable_signal"):
        instance.del_component("renewable_signal")
    instance.renewable_signal = pyo.Param(
        instance.time_steps,
        initialize=renewable_signal_dict,
        within=pyo.NonNegativeReals,
    )

    # Solve the instance
    steel_plant_renewable_utilisation.solver.solve(instance, tee=False)

    # Calculate adjusted total power input
    adjusted_total_power_input = [
        instance.total_power_input[t].value
        + instance.load_shift_pos[t].value
        - instance.load_shift_neg[t].value
        for t in instance.time_steps
    ]

    # Assert load shifting respects renewable signal intensity
    for t in instance.time_steps:
        assert adjusted_total_power_input[1] <= instance.total_power_input[1].value, (
            f"Load shift exceeds renewable intensity signal at time {t}"
        )


@pytest.fixture
def steel_plant_without_electrolyser(dsm_components) -> SteelPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
    )

    dsm_components.pop("electrolyser", None)
    plant = SteelPlant(
        id="test_steel_plant_no_electrolyser",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={
            "EOM": DsmEnergyOptimizationStrategy(),
            "RD": DsmEnergyNaiveRedispatchStrategy(),
        },
        node="south",
        components=dsm_components,
        forecaster=forecast,
        demand=1000,
        technology="steel_plant",
    )
    plant.setup_model()
    return plant


# --- Initialization Tests ---
def test_handle_missing_components():
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = SteelplantForecaster(
        index,
        market_prices={"EOM": [50] * 24},
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
    )
    with pytest.raises(
        ValueError, match="Component dri_plant is required for the steel plant unit."
    ):
        _ = SteelPlant(
            id="test_steel_plant",
            unit_operator="test_operator",
            objective="min_variable_cost",
            flexibility_measure="cost_based_load_shift",
            bidding_strategies={},
            node="south",
            components={},
            forecaster=forecast,
            demand=1000,
            technology="steel_plant",
        )


def test_handle_missing_electrolyser(steel_plant_without_electrolyser):
    steel_plant_without_electrolyser.determine_optimal_operation_without_flex()
    instance = steel_plant_without_electrolyser.model.create_instance()
    instance = steel_plant_without_electrolyser.switch_to_opt(instance)
    steel_plant_without_electrolyser.solver.solve(instance, tee=False)

    # Verify no electrolyser-related constraints
    for t in instance.time_steps:
        assert "electrolyser" not in instance.dsm_blocks


# --- Objective Handling ---
@pytest.fixture
def reset_objectives(create_steel):
    """
    Helper to reset objectives in the model.
    """

    def _reset(instance):
        if hasattr(instance, "obj_rule_opt"):
            instance.obj_rule_opt.deactivate()
        if hasattr(instance, "obj_rule_flex"):
            instance.obj_rule_flex.deactivate()

    return _reset


def _make_plant(dsm_components, forecaster, plant_id="test_plant", demand=1000):
    """Create a SteelPlant with the given forecaster (without calling setup_model)."""
    return SteelPlant(
        id=plant_id,
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=demand,
        technology="steel_plant",
    )


def test_steel_plant_reads_normalized_profile_from_forecaster(dsm_components):
    """SteelPlant.__init__ copies normalized_load_profile from the forecaster."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    profile = [i / 23.0 for i in range(24)]
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
        normalized_load_profile=profile,
    )
    plant = _make_plant(dsm_components, forecaster)
    assert plant.normalized_load_profile is not None
    assert len(plant.normalized_load_profile) == 24


def test_steel_plant_reads_steel_demand_from_forecaster(dsm_components):
    """SteelPlant.__init__ copies steel_demand_per_timestep from the forecaster."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    hourly_demand = [50.0] * 24
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
        steel_demand=hourly_demand,
    )
    plant = _make_plant(dsm_components, forecaster)
    assert plant.steel_demand_per_timestep is not None
    assert len(plant.steel_demand_per_timestep) == 24


def test_steel_plant_no_profile_no_demand_attrs_are_none(dsm_components):
    """Without profile/demand in forecaster the corresponding plant attrs are None."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
    )
    plant = _make_plant(dsm_components, forecaster)
    assert plant.normalized_load_profile is None
    assert plant.steel_demand_per_timestep is None


def test_load_profile_deviation_stored_as_attribute(dsm_components):
    """Custom load_profile_deviation is stored on the plant."""
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * 24,
        fuel_prices={"natural_gas": [30] * 24, "co2": [20] * 24},
        normalized_load_profile=[0.5] * 24,
    )
    plant = SteelPlant(
        id="test_deviation",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=1000,
        technology="steel_plant",
        load_profile_deviation=0.05,
    )
    assert plant.load_profile_deviation == 0.05


def test_profile_guided_optimization_uses_correct_param_values(dsm_components):
    """
    Profile-guided strategy: the Pyomo model stores the normalized_load_profile and
    load_profile_deviation parameters with the exact values from the forecaster, and
    the full-horizon optimization meets the total steel demand.
    """
    n = 8
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    # Linearly increasing profile: 0.0, 1/7, 2/7, … 1.0
    profile = [round(i / (n - 1), 6) for i in range(n)]
    forecaster = SteelplantForecaster(
        index,
        electricity_price=[50] * n,
        fuel_prices={"natural_gas": [30] * n, "co2": [20] * n},
        normalized_load_profile=profile,
    )
    plant = SteelPlant(
        id="test_profile_param_values",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=80,
        technology="steel_plant",
        load_profile_deviation=0.10,
    )
    plant.setup_model()

    # Pyomo model must carry the profile and deviation as parameters
    assert hasattr(plant.model, "normalized_load_profile")
    assert hasattr(plant.model, "load_profile_deviation")

    # Each Pyomo param value must match the input profile exactly
    for t in range(n):
        assert pyo.value(plant.model.normalized_load_profile[t]) == pytest.approx(
            profile[t], abs=1e-6
        ), f"Profile mismatch at timestep {t}"

    assert pyo.value(plant.model.load_profile_deviation) == pytest.approx(
        0.10, abs=1e-6
    )

    # Run the full-horizon optimization and verify total demand is met.
    # The profile only constrains the rolling-horizon path; in the full-horizon
    # solve the equality constraint sum(eaf.steel_output) == 80 is always active.
    # With 3 components each consuming 1 MWh per unit of steel:
    #   sum(total_power_input) == 3 * 80 == 240 MWh.
    plant.determine_optimal_operation_without_flex()
    assert plant.opt_power_requirement is not None
    total_power = sum(plant.opt_power_requirement.data)
    assert total_power == pytest.approx(3 * 80, rel=1e-3)


def test_min_demand_strategy_skips_global_equality_constraint(dsm_components):
    """
    Min-demand strategy omits the global ``steel_output_association_constraint``
    (sum == demand) from the Pyomo model, while the cost-optimised strategy
    (no per-timestep minimums) keeps it.

    The global equality constraint is only skipped when per-timestep minimums are
    supplied via the forecaster's ``steel_demand`` kwarg.  Without the equality
    constraint the solver is free to satisfy costs without producing a fixed total;
    with it the solver must exactly meet the declared steel demand.

    Two separate copies of the components dict are used because ``setup_model()``
    converts the dict entries into component objects in-place.
    """
    import copy

    n = 8
    index = pd.date_range("2023-01-01", periods=n, freq="h")

    # --- Min-demand plant (per-timestep minimums supplied) ---
    forecaster_min = SteelplantForecaster(
        index,
        electricity_price=[50] * n,
        fuel_prices={"natural_gas": [30] * n, "co2": [20] * n},
        steel_demand=[50.0] * n,  # triggers min_demand strategy
    )
    plant_min = SteelPlant(
        id="test_min_demand_con",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=copy.deepcopy(dsm_components),
        forecaster=forecaster_min,
        demand=200,
        technology="steel_plant",
    )
    plant_min.setup_model()
    assert not hasattr(plant_min.model, "steel_output_association_constraint"), (
        "steel_output_association_constraint must be absent for min_demand strategy"
    )
    # The per-timestep demand param is present
    assert hasattr(plant_min.model, "steel_demand_per_timestep")

    # --- Cost-optimised plant (no per-timestep minimums) ---
    forecaster_cost = SteelplantForecaster(
        index,
        electricity_price=[50] * n,
        fuel_prices={"natural_gas": [30] * n, "co2": [20] * n},
    )
    plant_cost = SteelPlant(
        id="test_cost_opt_con",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=copy.deepcopy(dsm_components),
        forecaster=forecaster_cost,
        demand=200,
        technology="steel_plant",
    )
    plant_cost.setup_model()
    plant_cost.determine_optimal_operation_without_flex()

    # Global equality constraint is present for cost-optimised strategy
    assert hasattr(plant_cost.model, "steel_output_association_constraint"), (
        "steel_output_association_constraint must be present for cost_optimized strategy"
    )
    # The optimizer must have met the demand: total power = 3 × demand
    total_power = sum(plant_cost.opt_power_requirement.data)
    assert total_power == pytest.approx(3 * 200, rel=1e-3)


def test_cost_optimized_concentrates_production_in_cheap_hours(dsm_components):
    """
    Cost-optimised strategy: with electricity prices much higher in the first
    4 hours than in the last 4, the optimizer schedules all production in the
    cheap hours.

    The demand constraint forces sum(eaf.steel_output) == 100.  With max_power=100
    per component per hour and a demand of 100 units, 4 cheap hours are more than
    sufficient.  The optimizer should therefore leave the expensive hours idle and
    concentrate production in hours 4-7.
    """
    n = 8
    index = pd.date_range("2023-01-01", periods=n, freq="h")
    # First 4 hours very expensive, last 4 very cheap
    prices = [200.0] * 4 + [10.0] * 4
    forecaster = SteelplantForecaster(
        index,
        electricity_price=prices,
        fuel_prices={"natural_gas": [30] * n, "co2": [20] * n},
    )
    plant = SteelPlant(
        id="test_price_shift_opt",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        node="south",
        components=dsm_components,
        forecaster=forecaster,
        demand=100,
        technology="steel_plant",
    )
    plant.setup_model()
    plant.determine_optimal_operation_without_flex()

    opt_power = plant.opt_power_requirement
    assert opt_power is not None

    # Power in cheap hours (4-7) must exceed power in expensive hours (0-3)
    cheap_power = sum(opt_power.data[4:])
    expensive_power = sum(opt_power.data[:4])
    assert cheap_power > expensive_power, (
        f"Optimizer should prefer cheap hours: "
        f"cheap={cheap_power:.2f} MWh vs expensive={expensive_power:.2f} MWh"
    )

    # Total production must satisfy the steel demand constraint:
    # 3 components × 100 units × 1 MWh/unit = 300 MWh
    total_power = sum(opt_power.data)
    assert total_power == pytest.approx(3 * 100, rel=1e-3)

    # The model must NOT have profile or per-timestep demand params
    assert not hasattr(plant.model, "normalized_load_profile")
    assert not hasattr(plant.model, "steel_demand_per_timestep")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
