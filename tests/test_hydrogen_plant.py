# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecasts import NaiveForecast
from assume.strategies.naive_strategies import NaiveDADSMStrategy
from assume.units.hydrogen_plant import HydrogenPlant


@pytest.fixture
def hydrogen_components():
    # Define full component configuration for hydrogen plant components including electrolyser and seasonal storage
    return {
        "electrolyser": {
            "max_power": 100,  # Maximum power input in MW
            "min_power": 0,  # Minimum power input in MW
            "ramp_up": 100,  # Ramp-up rate in MW per time step
            "ramp_down": 100,  # Ramp-down rate in MW per time step
            "efficiency": 1,  # Efficiency of the electrolyser
            "min_operating_time": 0,  # Minimum number of operating steps
            "min_down_time": 0,  # Minimum downtime steps
        },
        "hydrogen_seasonal_storage": {
            "max_capacity": 500,  # Maximum storage capacity in MWh
            "min_capacity": 50,  # Minimum storage capacity in MWh
            "max_power_charge": 30,  # Maximum charging power in MW
            "max_power_discharge": 30,  # Maximum discharging power in MW
            "efficiency_charge": 0.9,  # Efficiency for charging
            "efficiency_discharge": 0.9,  # Efficiency for discharging
            "initial_soc": 1.0,  # Initial state of charge (SOC) as fraction of max capacity
            "final_soc_target": 0.5,  # Target SOC at end of horizon, as a fraction of max capacity
            "ramp_up": 10,  # Maximum increase in charge/discharge per step
            "ramp_down": 10,
            "horizon": 23,  # Maximum decrease in charge/discharge per step
            "storage_loss_rate": 0.01,  # 1% storage loss per time step
            "off_season_start": "0",  # Off-season start times, as comma-separated values
            "off_season_end": "12",  # Off-season end times, as comma-separated values
            "on_season_start": "13",  # On-season start times, as comma-separated values
            "on_season_end": "23",  # On-season end times, as comma-separated values
        },
    }


@pytest.fixture
def hydrogen_plant(hydrogen_components) -> HydrogenPlant:
    # Define the time index and forecast data
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        price_EOM=[60] * 24,  # Forecast electricity prices
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": NaiveDADSMStrategy(),
    }

    # Initialize the HydrogenPlant with the specified components, forecast, and strategy
    return HydrogenPlant(
        id="test_hydrogen_plant",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="max_load_shift",
        cost_tolerance=50,
        bidding_strategies=bidding_strategy,
        components=hydrogen_components,
        forecaster=forecast,
        demand=500,  # Total hydrogen demand over the horizon
    )


def test_optimal_operation_without_flex_initialization(hydrogen_plant):
    # Run the initial without-flexibility operation to populate opt_power_requirement
    hydrogen_plant.determine_optimal_operation_without_flex()

    # Check that opt_power_requirement is populated and of the correct type
    assert (
        hydrogen_plant.opt_power_requirement is not None
    ), "opt_power_requirement should be populated"
    assert isinstance(hydrogen_plant.opt_power_requirement, FastSeries)

    # Create an instance of the model and switch to optimization mode
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    # Check that the total power input is greater than zero, indicating operation
    total_power_input = sum(
        instance.total_power_input[t].value
        for t in instance.time_steps
        if instance.total_power_input[t].value is not None
    )
    assert total_power_input > 0, "Total power input should be greater than zero"

    # Check hydrogen balance after solving
    for t in instance.time_steps:
        electrolyser_out = instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
        hydrogen_demand = (
            instance.hydrogen_demand[t].value
            if hasattr(instance.hydrogen_demand[t], "value")
            else instance.hydrogen_demand[t]
        )
        storage_charge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].charge[t].value
        )
        storage_discharge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].discharge[t].value
        )

        # Check that the balance holds within a small tolerance
        balance_difference = abs(
            electrolyser_out - (hydrogen_demand + storage_charge - storage_discharge)
        )
        assert (
            balance_difference < 1e-3
        ), f"Hydrogen balance mismatch at time {t}, difference: {balance_difference}"

    # Now test the with-flexibility operation to verify interaction with opt_power_requirement
    hydrogen_plant.determine_optimal_operation_with_flex()

    # Re-solve the instance to verify the operation with flexibility
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_flex(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    # Verify that opt_power_requirement values are used correctly
    for t in instance.time_steps:
        assert (
            instance.total_power_input[t].value is not None
        ), f"Total power input at time {t} should be set"
        expected_power = hydrogen_plant.opt_power_requirement.iloc[t]
        actual_power = instance.total_power_input[t].value
        assert (
            abs(expected_power - actual_power) < 1e-3
        ), f"Mismatch in power input at time {t}: expected {expected_power}, got {actual_power}"


def test_ramping_constraints_without_flex(hydrogen_plant):
    # Test that ramping constraints are respected in operation without flexibility
    hydrogen_plant.determine_optimal_operation_without_flex()
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    # Access ramp_up and ramp_down as attributes
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
            assert (
                change_in_charge <= ramp_up + 1e3
            ), f"Charge ramp-up at time {t} exceeds limit"
        if discharge_prev is not None and discharge_curr is not None:
            change_in_charge = abs(discharge_curr - discharge_prev)
            assert (
                change_in_charge <= ramp_down + 1e3
            ), f"Charge ramp-down at time {t} exceeds limit"


def test_final_soc_target_without_flex(hydrogen_plant):
    hydrogen_plant.determine_optimal_operation_without_flex()
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    # Access final SOC using integer index
    final_step_index = instance.time_steps[-1]
    final_soc = (
        instance.dsm_blocks["hydrogen_seasonal_storage"].soc[final_step_index].value
    )
    final_soc_target = (
        hydrogen_plant.components["hydrogen_seasonal_storage"].final_soc_target
        * hydrogen_plant.components["hydrogen_seasonal_storage"].max_capacity
    )

    assert final_soc is not None, "Final SOC should not be None"
    assert final_soc >= final_soc_target, "Final SOC does not meet target"


def test_initial_soc_greater_than_capacity(hydrogen_plant):
    # After initialization, check if initial SOC > max capacity was adjusted
    storage = hydrogen_plant.components["hydrogen_seasonal_storage"]
    adjusted_soc = (
        storage.initial_soc * storage.max_capacity
        if storage.initial_soc > 1
        else storage.initial_soc
    )
    assert (
        adjusted_soc <= storage.max_capacity
    ), "Initial SOC should be adjusted if set above max capacity"


def test_optimal_operation_with_flex_initialization(hydrogen_plant):
    # Set synthetic values for opt_power_requirement and total_cost
    hydrogen_plant.opt_power_requirement = FastSeries(
        value=30, index=hydrogen_plant.index
    )
    hydrogen_plant.total_cost = 100000  # Assign a synthetic cost value for testing

    # Trigger flexibility operation
    hydrogen_plant.determine_optimal_operation_with_flex()

    # Verify that flex_power_requirement is populated and is of the correct type
    assert (
        hydrogen_plant.flex_power_requirement is not None
    ), "flex_power_requirement should be populated"
    assert isinstance(hydrogen_plant.flex_power_requirement, FastSeries)

    # Create an instance of the model and switch to flexibility mode
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_flex(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    # Check that the total power input reflects flexible operation
    total_power_input = sum(
        instance.total_power_input[t].value
        for t in instance.time_steps
        if instance.total_power_input[t].value is not None
    )
    assert (
        total_power_input > 0
    ), "Total power input should be greater than zero under flexibility"

    # Verify that the synthetic opt_power_requirement values are being used
    for t in instance.time_steps:
        expected_power = hydrogen_plant.opt_power_requirement.iloc[t]
        actual_power = instance.total_power_input[t].value
        assert (
            abs(expected_power - actual_power) < 1e-3
        ), f"Mismatch in power input at time {t}: expected {expected_power}, got {actual_power}"

    # Additional check to ensure the hydrogen demand is met
    for t in instance.time_steps:
        electrolyser_out = instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
        hydrogen_demand = (
            instance.hydrogen_demand[t].value
            if hasattr(instance.hydrogen_demand[t], "value")
            else instance.hydrogen_demand[t]
        )
        storage_charge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].charge[t].value
        )
        storage_discharge = (
            instance.dsm_blocks["hydrogen_seasonal_storage"].discharge[t].value
        )

        # Check that the hydrogen demand balance holds within a small tolerance
        balance_difference = abs(
            electrolyser_out - (hydrogen_demand + storage_charge - storage_discharge)
        )
        assert (
            balance_difference < 1e-3
        ), f"Hydrogen balance mismatch at time {t}, difference: {balance_difference}"


def test_unknown_technology_error():
    # Test for unknown technology error with required components present
    hydrogen_components = {
        "electrolyser": {  # Required component
            "max_power": 50,
            "min_power": 0,
            "ramp_up": 10,
            "ramp_down": 10,
            "efficiency": 0.8,
        },
        "unknown_tech": {  # Unknown component to trigger the error
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
            flexibility_measure="max_load_shift",
            bidding_strategies={"EOM": NaiveDADSMStrategy()},
            components=hydrogen_components,
            forecaster=NaiveForecast(
                index=pd.date_range("2023-01-01", periods=24, freq="h"),
                price_EOM=[60] * 24,
            ),
            demand=500,
        )


@pytest.fixture
def hydrogen_components_no_storage():
    # Define component configuration for hydrogen plant with only an electrolyser (no storage)
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
    # Define the time index and forecast data
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        price_EOM=[60] * 24,  # Forecast electricity prices
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": NaiveDADSMStrategy(),
    }

    # Initialize the HydrogenPlant with only an electrolyser component (no storage)
    return HydrogenPlant(
        id="test_hydrogen_plant_no_storage",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="max_load_shift",
        bidding_strategies=bidding_strategy,
        components=hydrogen_components_no_storage,
        forecaster=forecast,
        demand=800,  # Total hydrogen demand over the horizon
    )


def test_electrolyser_only_operation(hydrogen_plant_no_storage):
    # Test that hydrogen demand is met solely by electrolyser output when there's no storage
    hydrogen_plant_no_storage.determine_optimal_operation_without_flex()
    instance = hydrogen_plant_no_storage.model.create_instance()
    instance = hydrogen_plant_no_storage.switch_to_opt(instance)
    hydrogen_plant_no_storage.solver.solve(instance, tee=False)

    for t in instance.time_steps:
        electrolyser_out = instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
        hydrogen_demand = (
            instance.hydrogen_demand[t].value
            if hasattr(instance.hydrogen_demand[t], "value")
            else instance.hydrogen_demand[t]
        )

        # Verify that electrolyser output meets hydrogen demand without storage
        assert (
            electrolyser_out == hydrogen_demand
        ), f"Hydrogen demand not met solely by electrolyser at time {t}"


def test_electrolyser_meets_total_hydrogen_demand(hydrogen_plant_no_storage):
    # Test that the sum of electrolyser output alone meets total hydrogen demand when there is no storage
    hydrogen_plant_no_storage.determine_optimal_operation_without_flex()
    instance = hydrogen_plant_no_storage.model.create_instance()
    instance = hydrogen_plant_no_storage.switch_to_opt(instance)
    hydrogen_plant_no_storage.solver.solve(instance, tee=False)

    # Calculate the total electrolyser output across all time steps
    total_electrolyser_output = sum(
        instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
        for t in instance.time_steps
    )

    # Access absolute hydrogen demand from the model
    absolute_hydrogen_demand = pyo.value(instance.absolute_hydrogen_demand)

    # Assert that the total electrolyser output matches the absolute hydrogen demand
    assert total_electrolyser_output == absolute_hydrogen_demand, (
        f"Total electrolyser output {total_electrolyser_output} does not match "
        f"absolute hydrogen demand {absolute_hydrogen_demand}"
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
