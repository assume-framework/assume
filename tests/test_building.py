# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import BuildingForecaster
from assume.common.market_objects import MarketConfig
from assume.strategies.naive_strategies import DsmEnergyOptimizationStrategy
from assume.units.building import Building


# Fixtures for Component Configurations
@pytest.fixture
def generic_storage_config():
    return {
        "capacity": 100,  # Maximum energy capacity in MWh
        "min_soc": 0,  # Minimum SOC
        "max_soc": 1,  # Maximum SOC
        "max_power_charge": 100,  # Maximum charging power in MW
        "max_power_discharge": 100,  # Maximum discharging power in MW
        "efficiency_charge": 0.9,  # Charging efficiency
        "efficiency_discharge": 0.9,  # Discharging efficiency
        "initial_soc": 0,  # Initial SOC
        "ramp_up": 10,  # Maximum ramp-up rate in MW
        "ramp_down": 10,  # Maximum ramp-down rate in MW
        "storage_loss_rate": 0.01,  # 1% storage loss per time step
    }


@pytest.fixture
def thermal_storage_config(generic_storage_config):
    return generic_storage_config.copy()


@pytest.fixture
def ev_config():
    return {
        "capacity": 10.0,  # EV battery capacity in MWh
        "min_soc": 0,
        "max_soc": 1,
        "max_power_charge": 3,  # Charge values will reflect a fraction of the capacity
        "max_power_discharge": 2,  # Discharge values will also be a fraction of the capacity
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.9,
        "initial_soc": 0,  # initial SOC
    }


@pytest.fixture
def electric_boiler_config():
    return {
        "max_power": 100,
        "efficiency": 0.85,
        "fuel_type": "electricity",  # Electric fuel type supports operational constraints
        "min_power": 0,
        "ramp_up": 100,
        "ramp_down": 100,
        "min_operating_steps": 2,
        "min_down_steps": 1,
        "initial_operational_status": 1,
    }


@pytest.fixture
def heat_pump_config():
    return {
        "max_power": 80,
        "cop": 3.5,
        "min_power": 10,
        "ramp_up": 20,
        "ramp_down": 20,
        "min_operating_steps": 2,
        "min_down_steps": 2,
        "initial_operational_status": 1,  # Assuming it starts as operational
    }


@pytest.fixture
def pv_plant_config():
    return {
        "max_power": 50,
    }


# Fixtures for Default Objective and Flexibility Measure
@pytest.fixture
def default_objective():
    return "min_variable_cost"


@pytest.fixture
def default_flexibility_measure():
    return "cost_based_load_shift"


# Fixtures for Availability Profiles
@pytest.fixture
def ev_availability_profile():
    # Create an availability profile as a pandas Series (1 = available, 0 = unavailable)
    return pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        index=pd.date_range("2023-01-01", periods=10, freq="h"),
    )


# Fixtures for Price and Forecast Data
@pytest.fixture
def price_profile():
    return pd.Series(
        [50, 45, 55, 40, 1000, 55, 1000, 65, 45, 70],
        index=pd.date_range("2023-01-01", periods=10, freq="h"),
    )


@pytest.fixture
def index():
    return pd.date_range("2023-01-01", periods=10, freq="h")


@pytest.fixture
def forecaster(price_profile):
    index = pd.date_range("2023-01-01", periods=10, freq="h")
    forecaster = BuildingForecaster(
        index=index,
        fuel_prices={"natural_gas": 30},
        heat_demand=50,
        ev_load_profile=5,
        battery_load_profile=3,
        pv_profile=10,
        availability=0.25,
        electricity_price=price_profile,
        market_prices={},
        load_profile=20,
    )
    # If the Building class expects specific keys for EV availability, add them here
    # forecaster.forecasts["electric_vehicle_availability"] = ev_availability_profile
    return forecaster


# Fixtures for Building Components
@pytest.fixture
def building_components_heatpump(
    generic_storage_config,
    thermal_storage_config,
    ev_config,
    heat_pump_config,
    pv_plant_config,
    ev_availability_profile,
):
    return {
        "heat_pump": heat_pump_config,
        "generic_storage": generic_storage_config,
        "electric_vehicle": {
            **ev_config,
            "availability_profile": ev_availability_profile,
        },
        "pv_plant": {
            **pv_plant_config,
        },
        "thermal_storage": thermal_storage_config,
    }


@pytest.fixture
def building_components_boiler(
    generic_storage_config,
    thermal_storage_config,
    ev_config,
    electric_boiler_config,
    pv_plant_config,
    ev_availability_profile,
):
    return {
        "boiler": electric_boiler_config,
        "generic_storage": generic_storage_config,
        "electric_vehicle": {
            **ev_config,
            "availability_profile": ev_availability_profile,
        },
        "pv_plant": {
            **pv_plant_config,
        },
        "thermal_storage": thermal_storage_config,
    }


# Test Cases
def test_building_initialization_heatpump(
    forecaster,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecaster,
    )

    assert building.id == "building"
    assert building.unit_operator == "operator_hp"
    assert building.components == building_components_heatpump
    assert building.has_heatpump is True
    assert building.has_boiler is False
    assert building.has_thermal_storage is True
    assert building.has_ev is True
    assert building.has_battery_storage is True
    assert building.has_pv is True


def test_building_initialization_boiler(
    forecaster,
    building_components_boiler,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_boiler",
        bidding_strategies={},
        components=building_components_boiler,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecaster,  # Passed via **kwargs
    )

    assert building.unit_operator == "operator_boiler"
    assert building.components == building_components_boiler
    assert building.has_heatpump is False
    assert building.has_boiler is True
    assert building.has_thermal_storage is True
    assert building.has_ev is True
    assert building.has_battery_storage is True
    assert building.has_pv is True


def test_building_initialization_invalid_component(
    forecaster, default_objective, default_flexibility_measure
):
    invalid_components = {"invalid_component": {"some_param": 123}}

    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building",
            unit_operator="operator_invalid",
            bidding_strategies={},
            components=invalid_components,
            objective=default_objective,
            flexibility_measure=default_flexibility_measure,
            forecaster=forecaster,
        )

    # Match the actual error message
    assert (
        "Components invalid_component is not a valid component for the building unit."
        in str(exc_info.value)
    )


def test_building_optimization_heatpump(
    forecaster,
    index,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_heatpump,
        forecaster=forecaster,  # Passed via **kwargs
    )

    # Perform optimization
    building.determine_optimal_operation_without_flex()

    # Check if optimal power requirement is calculated
    assert building.opt_power_requirement is not None
    assert len(building.opt_power_requirement) == len(index)
    assert isinstance(building.opt_power_requirement, FastSeries)

    # Check if variable cost series is calculated
    assert building.variable_cost_series is not None
    assert len(building.variable_cost_series) == len(index)
    assert isinstance(building.variable_cost_series, FastSeries)


def test_building_optimization_boiler(
    forecaster,
    index,
    building_components_boiler,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_boiler",
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_boiler,
        forecaster=forecaster,
    )

    # Perform optimization
    building.determine_optimal_operation_without_flex()

    # Check if optimal power requirement is calculated
    assert building.opt_power_requirement is not None
    assert len(building.opt_power_requirement) == len(index)
    assert isinstance(building.opt_power_requirement, FastSeries)

    # Check if variable cost series is calculated
    assert building.variable_cost_series is not None
    assert len(building.variable_cost_series) == len(index)
    assert isinstance(building.variable_cost_series, FastSeries)


def test_building_marginal_cost_calculation_heatpump(
    forecaster,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_heatpump,
        forecaster=forecaster,  # Passed via **kwargs
    )

    building.determine_optimal_operation_without_flex()

    # Select a timestamp to test
    test_time = building.index[0]
    power = building.opt_power_requirement.at[test_time]
    variable_cost = building.variable_cost_series.at[test_time]

    if power != 0:
        expected_marginal_cost = abs(variable_cost / power)
    else:
        expected_marginal_cost = 0

    calculated_marginal_cost = building.calculate_marginal_cost(test_time, power)

    assert calculated_marginal_cost == expected_marginal_cost


def test_building_marginal_cost_calculation_boiler(
    forecaster,
    building_components_boiler,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_boiler",
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_boiler,
        forecaster=forecaster,  # Passed via **kwargs
    )

    building.determine_optimal_operation_without_flex()

    # Select a timestamp to test
    test_time = building.index[0]
    power = building.opt_power_requirement.at[test_time]
    variable_cost = building.variable_cost_series.at[test_time]

    if power != 0:
        expected_marginal_cost = abs(variable_cost / power)
    else:
        expected_marginal_cost = 0

    calculated_marginal_cost = building.calculate_marginal_cost(test_time, power)

    assert calculated_marginal_cost == expected_marginal_cost


def test_building_objective_function_heatpump(
    forecaster,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_heatpump,
        forecaster=forecaster,  # Passed via **kwargs
    )

    # Access the objective function
    objective = building.model.obj_rule_opt

    assert isinstance(objective, pyo.Objective)
    assert objective.sense == pyo.minimize


def test_building_objective_function_invalid(
    forecaster,
    building_components_heatpump,
):
    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building",
            unit_operator="operator_invalid",
            bidding_strategies={},
            components=building_components_heatpump,
            objective="unknown_objective",
            forecaster=forecaster,  # Passed via **kwargs
        )

    assert "Unknown objective: unknown_objective" in str(exc_info.value)


def test_building_define_constraints_heatpump(
    forecaster,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building",
        unit_operator="operator_constraints_hp",
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        bidding_strategies={},
        components=building_components_heatpump,
        forecaster=forecaster,  # Passed via **kwargs
    )

    # Check if constraints are defined
    constraints = list(building.model.component_map(pyo.Constraint).keys())
    assert "total_power_input_constraint" in constraints
    if building.has_heatpump:
        assert "heating_demand_balance_constraint" in constraints


def test_building_missing_required_component(
    forecaster,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    """
    Test that the Building class raises a ValueError if a required component is missing.
    """
    # Set the required technologies for the test
    Building.required_technologies = ["boiler"]

    # Remove a required component from the configuration
    incomplete_components = building_components_heatpump.copy()
    incomplete_components.pop("boiler", None)  # Remove "boiler" to trigger the error

    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building",
            unit_operator="operator_hp",
            bidding_strategies={},
            components=incomplete_components,
            objective=default_objective,
            flexibility_measure=default_flexibility_measure,
            forecaster=forecaster,
        )

    # Assert the correct error message
    assert "Component boiler is required for the building plant unit." in str(
        exc_info.value
    )

    # Reset required technologies to avoid affecting other tests
    Building.required_technologies = []


def test_building_solver_infeasibility_logging(
    forecaster,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    """
    Test that the Building class logs the correct messages when the solver reports infeasibility or other statuses.
    """
    # Create a Building instance
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecaster,
    )

    # Mock the solver to simulate infeasibility
    class MockResults:
        class Solver:
            status = "mock_status"
            termination_condition = "infeasible"

        solver = Solver()

    # Populate model variables with dummy values
    for t in building.model.time_steps:
        building.model.total_power_input[t].value = 0
        building.model.variable_cost[t].value = 0

    building.solver.solve = lambda instance, options: MockResults()

    # Call the method to ensure the log messages are triggered
    building.determine_optimal_operation_without_flex()


def test_building_bidding_strategy_execution(
    forecaster,
    index,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    """
    Test that the DsmEnergyOptimizationStrategy's calculate_bids method is executed correctly,
    and unit.determine_optimal_operation_without_flex() is called.
    """
    # Create the Building instance with a DsmEnergyOptimizationStrategy
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={"EOM": DsmEnergyOptimizationStrategy()},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecaster,
    )

    # Create dummy market configuration and product tuples
    market_config = MarketConfig(
        product_type="electricity",
        market_id="EOM",
    )
    product_tuples = [
        (index[0], index[1], "hour_1"),
        (index[1], index[2], "hour_2"),
        (index[2], index[3], "hour_3"),
    ]

    # Call the bidding strategy
    bids = building.bidding_strategies["EOM"].calculate_bids(
        unit=building,
        market_config=market_config,
        product_tuples=product_tuples,
    )

    # Verify that bids are generated correctly
    assert len(bids) == len(product_tuples)
    for bid, product in zip(bids, product_tuples):
        assert bid["start_time"] == product[0]
        assert bid["end_time"] == product[1]
        assert bid["volume"] <= 0  # Demand-side bids have non-positive volume
        assert bid["price"] >= 0  # Marginal price should be non-negative


def test_building_unknown_flexibility_measure(
    forecaster,
    building_components_heatpump,
    default_objective,
):
    """
    Test that the Building class raises a ValueError for an unknown flexibility measure.
    """
    invalid_flexibility_measure = "invalid_flex_measure"

    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building",
            unit_operator="operator_hp",
            bidding_strategies={},
            components=building_components_heatpump,
            objective=default_objective,
            flexibility_measure=invalid_flexibility_measure,
            forecaster=forecaster,
        )

    # Assert the correct error message
    assert f"Unknown flexibility measure: {invalid_flexibility_measure}" in str(
        exc_info.value
    )


def test_building_prosumer_constraint(forecaster, building_components_heatpump):
    """
    Test that the `grid_export_constraint` is correctly applied when the building is not a prosumer.
    """
    # Create a building instance with is_prosumer set to "No"
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        components=building_components_heatpump,
        forecaster=forecaster,
        is_prosumer="No",
    )

    constraints = list(building.model.component_map(pyo.Constraint).keys())
    assert "grid_export_constraint" in constraints, (
        "Non-prosumer should have grid export constraint."
    )


def test_building_prosumer_no_constraint(forecaster, building_components_heatpump):
    """
    Test that the `grid_export_constraint` is NOT applied when the building is a prosumer.
    """
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        components=building_components_heatpump,
        forecaster=forecaster,
        is_prosumer="Yes",
    )

    constraints = list(building.model.component_map(pyo.Constraint).keys())
    assert "grid_export_constraint" not in constraints, (
        "Prosumer should not have grid export constraint."
    )


def test_prosumer_energy_export(forecaster, building_components_heatpump):
    """
    Ensure that a prosumer building can export excess energy to the grid when applicable.
    """
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        components=building_components_heatpump,
        forecaster=forecaster,
        is_prosumer="Yes",
    )

    # Run optimization
    building.determine_optimal_operation_without_flex()

    # Verify that some power can be negative (exported to the grid)
    export_possible = any(building.opt_power_requirement < 0)
    assert export_possible, "Prosumer should be able to export power to the grid."


def test_non_prosumer_no_energy_export(forecaster, building_components_heatpump):
    """
    Ensure that a non-prosumer building does not export energy to the grid.
    """
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        components=building_components_heatpump,
        forecaster=forecaster,
        is_prosumer="No",
    )

    # Run optimization
    building.determine_optimal_operation_without_flex()

    # Verify that power input is never negative (no export to the grid)
    assert all(building.opt_power_requirement >= 0), (
        "Non-prosumer should not be able to export power."
    )

    # check that power is zero when price is 1000
    for idx in building.index:
        if building.forecaster.electricity_price.at[idx] == 1000:
            assert building.opt_power_requirement.at[idx] >= 0, (
                "Prosumer should be able to export power to the grid."
            )


def test_building_constraint_enforcement(forecaster, building_components_heatpump):
    """
    Test that all relevant constraints are being applied in the Pyomo model.
    """
    building = Building(
        id="building",
        unit_operator="operator_hp",
        bidding_strategies={},
        components=building_components_heatpump,
        forecaster=forecaster,
    )

    constraints = list(building.model.component_map(pyo.Constraint).keys())
    assert "total_power_input_constraint" in constraints, (
        "Total power input constraint should be enforced."
    )
    assert "variable_cost_constraint" in constraints, (
        "Variable cost constraint should be enforced."
    )
    if building.has_heatpump:
        assert "heating_demand_balance_constraint" in constraints, (
            "Heating demand constraint should be enforced."
        )


def test_invalid_prosumer_value(forecaster, building_components_heatpump):
    """
    Test that an invalid prosumer value raises a ValueError.
    """
    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building",
            unit_operator="operator_invalid",
            bidding_strategies={},
            components=building_components_heatpump,
            forecaster=forecaster,
            is_prosumer="maybe",  # Invalid boolean string
        )
    assert "Invalid truth value" in str(exc_info.value), (
        "Invalid is_prosumer value should raise an error."
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
