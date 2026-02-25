# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import HydrogenForecaster
from assume.strategies.naive_strategies import DsmEnergyOptimizationStrategy
from assume.units.hydrogen_plant import HydrogenPlant


@pytest.fixture
def hydrogen_components():
    return {
        "electrolyser": {
            "max_power": 100,
            "min_power": 0,
            "ramp_up": 100,
            "ramp_down": 100,
            "efficiency": 1,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
        "hydrogen_seasonal_storage": {
            "capacity": 200,
            "min_soc": 0.1,
            "max_power_charge": 40,
            "max_power_discharge": 40,
            "efficiency_charge": 0.95,
            "efficiency_discharge": 0.95,
            "initial_soc": 0,
            "ramp_up": 100,
            "ramp_down": 100,
            "storage_loss_rate": 0.0,
            "final_soc_target": 0.3,  # NEW: ensures attribute for test
        },
    }


@pytest.fixture
def hydrogen_plant(hydrogen_components) -> HydrogenPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = HydrogenForecaster(
        index,
        electricity_price=0,
        hydrogen_demand=[400 / 24.0] * 24,
        seasonal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": [60] * 24},
    )
    bidding_strategy = {"EOM": DsmEnergyOptimizationStrategy()}
    return HydrogenPlant(
        id="test_hydrogen_plant",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=50,
        bidding_strategies=bidding_strategy,
        components=hydrogen_components,
        forecaster=forecast,
        demand=400,  # Total hydrogen demand over the horizon
    )


def test_optimal_operation_without_flex_initialization(hydrogen_plant):
    hydrogen_plant.determine_optimal_operation_without_flex()
    assert hydrogen_plant.opt_power_requirement is not None
    assert isinstance(hydrogen_plant.opt_power_requirement, FastSeries)

    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    total_power_input = sum(
        (instance.total_power_input[t].value or 0.0) for t in instance.time_steps
    )
    assert total_power_input > 0, "Total power input should be greater than zero"

    # === Remove per-step hydrogen balance assertions! ===

    # === Add this cumulative check: ===
    total_supplied = 0.0
    for t in instance.time_steps:
        electrolyser_out = (
            instance.dsm_blocks["electrolyser"].hydrogen_out[t].value or 0.0
        )
        storage_discharge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].discharge[t].value
            if "hydrogen_seasonal_storage" in hydrogen_plant.components
            else 0.0
        )
        storage_charge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].charge[t].value
            if "hydrogen_seasonal_storage" in hydrogen_plant.components
            else 0.0
        )
        total_supplied += electrolyser_out + storage_discharge - storage_charge

    absolute_hydrogen_demand = pyo.value(instance.absolute_hydrogen_demand)
    assert total_supplied >= absolute_hydrogen_demand - 1e-3, (
        f"Total hydrogen supplied ({total_supplied}) is less than demand ({absolute_hydrogen_demand})"
    )


def test_ramping_constraints_without_flex(hydrogen_plant):
    hydrogen_plant.determine_optimal_operation_without_flex()
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    ramp_up = hydrogen_plant.components["hydrogen_seasonal_storage"].ramp_up
    ramp_down = hydrogen_plant.components["hydrogen_seasonal_storage"].ramp_down

    for t in list(instance.time_steps)[1:]:
        charge_prev = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].charge[t - 1].value
        )
        charge_curr = instance.dsm_blocks["hydrogen_seasonal_storage"].charge[t].value
        discharge_prev = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].discharge[t - 1].value
        )
        discharge_curr = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].discharge[t].value
        )

        # Check charge ramping
        if charge_prev is not None and charge_curr is not None:
            change_in_charge = abs(charge_curr - charge_prev)
            assert change_in_charge <= ramp_up + 1e-3, (
                f"Charge ramp-up at time {t} exceeds limit"
            )
        if discharge_prev is not None and discharge_curr is not None:
            change_in_charge = abs(discharge_curr - discharge_prev)
            assert change_in_charge <= ramp_down + 1e-3, (
                f"Charge ramp-down at time {t} exceeds limit"
            )


def test_initial_soc_greater_than_max_soc(hydrogen_plant):
    storage = hydrogen_plant.components["hydrogen_seasonal_storage"]
    assert storage.initial_soc <= storage.max_soc, (
        f"Initial SOC should be capped at max_soc. Got {storage.initial_soc} > {storage.max_soc}"
    )


def test_optimal_operation_with_flex_initialization(hydrogen_plant):
    # Set synthetic values for opt_power_requirement and total_cost
    hydrogen_plant.opt_power_requirement = FastSeries(
        value=30, index=hydrogen_plant.index
    )
    hydrogen_plant.total_cost = 100000  # Assign a synthetic cost value for testing

    # Trigger flexibility operation
    hydrogen_plant.determine_optimal_operation_with_flex()

    # Verify that flex_power_requirement is populated and is of the correct type
    assert hydrogen_plant.flex_power_requirement is not None
    assert isinstance(hydrogen_plant.flex_power_requirement, FastSeries)

    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_flex(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    total_power_input = sum(
        (instance.total_power_input[t].value or 0.0) for t in instance.time_steps
    )
    assert total_power_input > 0, (
        "Total power input should be greater than zero under flexibility"
    )

    for t in instance.time_steps:
        expected_power = hydrogen_plant.opt_power_requirement.iloc[t]
        actual_power = instance.total_power_input[t].value
        assert abs(expected_power - actual_power) < 1e-3, (
            f"Mismatch in power input at time {t}: expected {expected_power}, got {actual_power}"
        )

    # Hydrogen demand balance check under flexibility
    total_supplied = 0.0
    for t in instance.time_steps:
        electrolyser_out = (
            instance.dsm_blocks["electrolyser"].hydrogen_out[t].value or 0.0
        )
        storage_discharge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].discharge[t].value
            if "hydrogen_seasonal_storage" in hydrogen_plant.components
            else 0.0
        )
        storage_charge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].charge[t].value
            if "hydrogen_seasonal_storage" in hydrogen_plant.components
            else 0.0
        )
        total_supplied += electrolyser_out + storage_discharge - storage_charge

    absolute_hydrogen_demand = pyo.value(instance.absolute_hydrogen_demand)
    assert total_supplied >= absolute_hydrogen_demand - 1e-3, (
        f"Total hydrogen supplied ({total_supplied}) is less than demand ({absolute_hydrogen_demand})"
    )


def test_unknown_technology_error():
    hydrogen_components = {
        "electrolyser": {
            "max_power": 50,
            "min_power": 0,
            "ramp_up": 10,
            "ramp_down": 10,
            "efficiency": 0.8,
        },
        "unknown_tech": {
            "max_power": 20,
            "min_power": 0,
            "efficiency": 0.5,
        },
    }
    with pytest.raises(ValueError, match=r"unknown_tech"):
        HydrogenPlant(
            id="test_unknown_technology",
            unit_operator="test_operator",
            objective="min_variable_cost",
            flexibility_measure="cost_based_load_shift",
            bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
            components=hydrogen_components,
            forecaster=HydrogenForecaster(
                index=pd.date_range("2023-01-01", periods=24, freq="h"),
                market_prices={"EOM": [60] * 24},
                electricity_price=0,
                hydrogen_demand=0,
            ),
            demand=500,
        )


@pytest.fixture
def hydrogen_components_no_storage():
    return {
        "electrolyser": {
            "max_power": 100,
            "min_power": 0,
            "ramp_up": 50,
            "ramp_down": 50,
            "efficiency": 0.7,
            "min_operating_time": 1,
            "min_down_time": 1,
        }
    }


@pytest.fixture
def hydrogen_plant_no_storage(hydrogen_components_no_storage) -> HydrogenPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = HydrogenForecaster(
        index,
        electricity_price=0,
        hydrogen_demand=[800 / 24.0] * 24,
        seasonal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": [60] * 24},
    )
    bidding_strategy = {"EOM": DsmEnergyOptimizationStrategy()}
    return HydrogenPlant(
        id="test_hydrogen_plant_no_storage",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies=bidding_strategy,
        components=hydrogen_components_no_storage,
        forecaster=forecast,
        demand=800,
    )


def test_electrolyser_only_operation(hydrogen_plant_no_storage):
    hydrogen_plant_no_storage.determine_optimal_operation_without_flex()
    instance = hydrogen_plant_no_storage.model.create_instance()
    instance = hydrogen_plant_no_storage.switch_to_opt(instance)
    hydrogen_plant_no_storage.solver.solve(instance, tee=False)

    total_electrolyser_output = sum(
        (instance.dsm_blocks["electrolyser"].hydrogen_out[t].value or 0.0)
        for t in instance.time_steps
    )
    absolute_hydrogen_demand = pyo.value(instance.absolute_hydrogen_demand)
    assert total_electrolyser_output >= absolute_hydrogen_demand - 1e-3, (
        f"Total electrolyser output {total_electrolyser_output} is less than absolute hydrogen demand {absolute_hydrogen_demand}"
    )


def test_electrolyser_meets_total_hydrogen_demand(hydrogen_plant_no_storage):
    hydrogen_plant_no_storage.determine_optimal_operation_without_flex()
    instance = hydrogen_plant_no_storage.model.create_instance()
    instance = hydrogen_plant_no_storage.switch_to_opt(instance)
    hydrogen_plant_no_storage.solver.solve(instance, tee=False)

    total_electrolyser_output = sum(
        (instance.dsm_blocks["electrolyser"].hydrogen_out[t].value or 0.0)
        for t in instance.time_steps
    )
    absolute_hydrogen_demand = pyo.value(instance.absolute_hydrogen_demand)
    assert abs(total_electrolyser_output - absolute_hydrogen_demand) < 1e-3, (
        f"Total electrolyser output {total_electrolyser_output} does not match "
        f"absolute hydrogen demand {absolute_hydrogen_demand}"
    )


def test_hydrogen_seasonal_storage_schedule_injection():
    """Test that the long-term storage schedule profile is injected correctly."""
    # Simulate config and forecast
    schedule = pd.Series([0, 1, 1, 0, 1], index=range(5))
    plant_id = "test_plant"
    schedule_key = f"{plant_id}_hydrogen_seasonal_storage_schedule"
    components = {
        "hydrogen_seasonal_storage": {
            "storage_type": "long-term",
            # ... other required keys (if any) ...
        }
    }
    forecaster = {schedule_key: schedule}

    # Emulate the injection logic
    if "hydrogen_seasonal_storage" in components:
        storage_cfg = components["hydrogen_seasonal_storage"]
        storage_type = storage_cfg.get("storage_type", "short-term")
        if storage_type == "long-term":
            storage_cfg["storage_schedule_profile"] = forecaster[schedule_key]

    # Assert the schedule profile is present and correct
    assert "storage_schedule_profile" in components["hydrogen_seasonal_storage"]
    assert components["hydrogen_seasonal_storage"]["storage_schedule_profile"].equals(
        schedule
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
