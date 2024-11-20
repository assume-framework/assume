# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies.naive_strategies import (
    NaiveDSMStrategy,
    NaiveRedispatchSteelplantStrategy,
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
    forecast = NaiveForecast(
        index,
        price_EOM=[50] * 24,
        fuel_price_natural_gas=[30] * 24,
        co2_price=[20] * 24,
    )

    bidding_strategies = {
        "EOM": NaiveDSMStrategy(),
        "RD": NaiveRedispatchSteelplantStrategy(),
    }

    return SteelPlant(
        id=f"test_steel_plant_{flexibility_measure}",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure=flexibility_measure,
        bidding_strategies=bidding_strategies,
        index=index,
        node="south",
        components=dsm_components,
        forecaster=forecast,
        demand=1000,
        technology="steel_plant",
        cost_tolerance=10,
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
    assert isinstance(steel_plant_cost_based.opt_power_requirement, pd.Series)


@pytest.fixture
def steel_plant_without_electrolyser(dsm_components) -> SteelPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        price_EOM=[50] * 24,
        fuel_price_natural_gas=[30] * 24,
        co2_price=[20] * 24,
    )

    dsm_components.pop("electrolyser", None)
    return SteelPlant(
        id="test_steel_plant_no_electrolyser",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies={
            "EOM": NaiveDSMStrategy(),
            "RD": NaiveRedispatchSteelplantStrategy(),
        },
        index=index,
        node="south",
        components=dsm_components,
        forecaster=forecast,
        demand=1000,
        technology="steel_plant",
    )


# --- Initialization Tests ---
def test_handle_missing_components():
    with pytest.raises(
        ValueError, match="Component dri_plant is required for the steel plant unit."
    ):
        _ = SteelPlant(
            id="test_steel_plant",
            unit_operator="test_operator",
            objective="min_variable_cost",
            flexibility_measure="cost_based_load_shift",
            bidding_strategies={},
            index=pd.date_range("2023-01-01", periods=24, freq="h"),
            node="south",
            components={},
            forecaster=None,
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
def reset_objectives(steel_plant):
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
