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

    return SteelPlant(
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
    return SteelPlant(
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


if __name__ == "__main__":
    pytest.main(["-s", __file__])
