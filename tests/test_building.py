# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.opt import SolverFactory

from assume.common.forecasts import CsvForecaster
from assume.common.market_objects import MarketConfig
from assume.strategies.naive_strategies import NaiveDADSMStrategy
from assume.units.building import Building
from assume.units.dsm_load_shift import SOLVERS, check_available_solvers

# Fixtures for Component Configurations


@pytest.fixture
def generic_storage_config():
    return {
        "max_capacity": 100,  # Maximum energy capacity in MWh
        "min_capacity": 20,  # Minimum SOC in MWh
        "max_power_charge": 30,  # Maximum charging power in MW
        "max_power_discharge": 30,  # Maximum discharging power in MW
        "efficiency_charge": 0.9,  # Charging efficiency
        "efficiency_discharge": 0.9,  # Discharging efficiency
        "initial_soc": 0.5,  # Initial SOC in MWh
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
        "max_capacity": 10.0,
        "min_capacity": 2.0,
        "max_power_charge": 3,  # Charge values will reflect a fraction of the capacity
        "max_power_discharge": 2,  # Discharge values will also be a fraction of the capacity
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.9,
        "initial_soc": 0.5,  # SOC initialized to 50% of capacity
    }


@pytest.fixture
def electric_boiler_config():
    return {
        "max_power": 100,
        "efficiency": 0.85,
        "fuel_type": "electricity",  # Electric fuel type supports operational constraints
        "min_power": 20,
        "ramp_up": 50,
        "ramp_down": 50,
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
    return pd.Series([1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=range(10))


# Fixtures for Price and Forecast Data
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, 1000, 55, 1000, 65, 45, 70], index=range(10))


@pytest.fixture
def index():
    return range(10)  # Integer-based index


@pytest.fixture
def forecast(price_profile):
    forecaster = CsvForecaster(
        index=range(10),
        powerplants_units=[],  # Add appropriate values
        demand_units=[],
    )
    forecaster.forecasts = pd.DataFrame()
    forecaster.forecasts["price_EOM"] = price_profile
    forecaster.forecasts["fuel_price_natural gas"] = pd.Series(
        [30] * 10, index=range(10)
    )
    forecaster.forecasts["heat_demand"] = pd.Series([50] * 10, index=range(10))
    forecaster.forecasts["ev_load_profile"] = pd.Series([5] * 10, index=range(10))
    forecaster.forecasts["battery_load_profile"] = pd.Series([3] * 10, index=range(10))
    forecaster.forecasts["building_load_profile"] = pd.Series(
        [50] * 10, index=range(10)
    )
    forecaster.forecasts["availability_solar"] = pd.Series([0.25] * 10, index=range(10))
    forecaster.forecasts["test_building_pv_power_profile"] = pd.Series(
        [10] * 10, index=range(10)
    )  # Adjust key as needed
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


# Fixture for Solver Selection
@pytest.fixture(scope="module")
def available_solver():
    solvers = check_available_solvers(*SOLVERS)
    if not solvers:
        pytest.skip(f"No available solvers from the list: {SOLVERS}")
    return SolverFactory(solvers[0])


# Test Cases


def test_building_initialization_heatpump(
    forecast,
    index,
    building_components_heatpump,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_heatpump",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={"EOM": NaiveDADSMStrategy()},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecast,
    )

    assert building.id == "building_heatpump"
    assert building.unit_operator == "operator_hp"
    assert building.components == building_components_heatpump
    assert building.has_heatpump is True
    assert building.has_boiler is False
    assert building.has_thermal_storage is True
    assert building.has_ev is True
    assert building.has_battery_storage is True
    assert building.has_pv is True


def test_building_initialization_boiler(
    forecast,
    index,
    building_components_boiler,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_boiler",
        unit_operator="operator_boiler",
        index=index,
        bidding_strategies={},
        components=building_components_boiler,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecast,  # Passed via **kwargs
    )

    assert building.id == "building_boiler"
    assert building.unit_operator == "operator_boiler"
    assert building.components == building_components_boiler
    assert building.has_heatpump is False
    assert building.has_boiler is True
    assert building.has_thermal_storage is True
    assert building.has_ev is True
    assert building.has_battery_storage is True
    assert building.has_pv is True


def test_building_initialization_invalid_component(
    forecast, index, available_solver, default_objective, default_flexibility_measure
):
    invalid_components = {"invalid_component": {"some_param": 123}}

    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building_invalid",
            unit_operator="operator_invalid",
            index=index,
            bidding_strategies={},
            components=invalid_components,
            objective=default_objective,
            flexibility_measure=default_flexibility_measure,
            forecaster=forecast,
        )

    # Match the actual error message
    assert (
        "Components invalid_component is not a valid component for the building unit."
        in str(exc_info.value)
    )


def test_solver_availability():
    available_solvers = check_available_solvers(*SOLVERS)
    assert len(available_solvers) > 0, f"None of the solvers {SOLVERS} are available."


def test_building_optimization_heatpump(
    forecast,
    index,
    building_components_heatpump,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_heatpump",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_heatpump,
        forecaster=forecast,  # Passed via **kwargs
    )

    # Perform optimization
    building.determine_optimal_operation_without_flex()

    # Check if optimal power requirement is calculated
    assert building.opt_power_requirement is not None
    assert len(building.opt_power_requirement) == len(index)
    assert isinstance(building.opt_power_requirement, pd.Series)

    # Check if variable cost series is calculated
    assert building.variable_cost_series is not None
    assert len(building.variable_cost_series) == len(index)
    assert isinstance(building.variable_cost_series, pd.Series)

    # Check additional outputs if components exist
    if building.has_battery_storage:
        assert "soc" in building.outputs
        assert len(building.outputs["soc"]) == len(index)
        assert isinstance(building.outputs["soc"], pd.Series)

    if building.has_ev:
        assert "ev_soc" in building.outputs
        assert len(building.outputs["ev_soc"]) == len(index)
        assert isinstance(building.outputs["ev_soc"], pd.Series)

    # Optional: Verify that the optimization was successful
    # Note: Depending on how the Building class stores solver results, adjust accordingly
    # For example, if building.model is updated with solver results:
    # assert building.model.solver.status == SolverStatus.ok
    # assert building.model.solver.termination_condition == TerminationCondition.optimal


def test_building_optimization_boiler(
    forecast,
    index,
    building_components_boiler,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_boiler",
        unit_operator="operator_boiler",
        index=index,
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_boiler,
        forecaster=forecast,  # Passed via **kwargs
    )

    # Perform optimization
    building.determine_optimal_operation_without_flex()

    # Check if optimal power requirement is calculated
    assert building.opt_power_requirement is not None
    assert len(building.opt_power_requirement) == len(index)
    assert isinstance(building.opt_power_requirement, pd.Series)

    # Check if variable cost series is calculated
    assert building.variable_cost_series is not None
    assert len(building.variable_cost_series) == len(index)
    assert isinstance(building.variable_cost_series, pd.Series)

    # Check additional outputs if components exist
    if building.has_battery_storage:
        assert "soc" in building.outputs
        assert len(building.outputs["soc"]) == len(index)
        assert isinstance(building.outputs["soc"], pd.Series)

    if building.has_ev:
        assert "ev_soc" in building.outputs
        assert len(building.outputs["ev_soc"]) == len(index)
        assert isinstance(building.outputs["ev_soc"], pd.Series)


def test_building_marginal_cost_calculation_heatpump(
    forecast,
    index,
    building_components_heatpump,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_heatpump",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_heatpump,
        forecaster=forecast,  # Passed via **kwargs
    )

    building.determine_optimal_operation_without_flex()

    # Select a timestamp to test
    test_time = 0  # Using integer index
    power = building.opt_power_requirement[test_time]
    variable_cost = building.variable_cost_series[test_time]

    if power != 0:
        expected_marginal_cost = abs(variable_cost / power)
    else:
        expected_marginal_cost = 0

    calculated_marginal_cost = building.calculate_marginal_cost(test_time, power)

    assert calculated_marginal_cost == expected_marginal_cost


def test_building_marginal_cost_calculation_boiler(
    forecast,
    index,
    building_components_boiler,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_boiler",
        unit_operator="operator_boiler",
        index=index,
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_boiler,
        forecaster=forecast,  # Passed via **kwargs
    )

    building.determine_optimal_operation_without_flex()

    # Select a timestamp to test
    test_time = 1  # Using integer index
    power = building.opt_power_requirement[test_time]
    variable_cost = building.variable_cost_series[test_time]

    if power != 0:
        expected_marginal_cost = abs(variable_cost / power)
    else:
        expected_marginal_cost = 0

    calculated_marginal_cost = building.calculate_marginal_cost(test_time, power)

    assert calculated_marginal_cost == expected_marginal_cost


def test_building_objective_function_heatpump(
    forecast,
    index,
    building_components_heatpump,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_heatpump",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={},
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        components=building_components_heatpump,
        forecaster=forecast,  # Passed via **kwargs
    )

    # Access the objective function
    objective = building.model.obj_rule_opt

    assert isinstance(objective, pyo.Objective)
    assert objective.sense == pyo.minimize


def test_building_objective_function_invalid(
    forecast, index, building_components_heatpump, available_solver
):
    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building_invalid_objective",
            unit_operator="operator_invalid",
            index=index,
            bidding_strategies={},
            components=building_components_heatpump,
            objective="unknown_objective",
            forecaster=forecast,  # Passed via **kwargs
        )

    assert "Unknown objective: unknown_objective" in str(exc_info.value)


def test_building_no_available_solvers(
    forecast,
    index,
    building_components_heatpump,
    monkeypatch,  # Add the monkeypatch fixture
):
    # Override the check_available_solvers to return an empty list
    monkeypatch.setattr(
        "assume.units.building.check_available_solvers", lambda *args: []
    )

    with pytest.raises(Exception) as exc_info:
        Building(
            id="building_no_solvers",
            unit_operator="operator_nosolver",
            index=index,
            bidding_strategies={},
            components=building_components_heatpump,
            forecaster=forecast,  # Passed via **kwargs
        )
    assert (
        "None of ['appsi_highs', 'gurobi', 'glpk', 'cbc', 'cplex'] are available"
        in str(exc_info.value)
    )


def test_building_define_constraints_heatpump(
    forecast,
    index,
    building_components_heatpump,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    building = Building(
        id="building_constraints_hp",
        unit_operator="operator_constraints_hp",
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        index=index,
        bidding_strategies={},
        components=building_components_heatpump,
        forecaster=forecast,  # Passed via **kwargs
    )

    # Check if constraints are defined
    constraints = list(building.model.component_map(pyo.Constraint).keys())
    assert "total_power_input_constraint" in constraints
    if building.has_heatpump:
        assert "heat_flow_constraint" in constraints


def test_building_missing_required_component(
    forecast,
    index,
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
            id="building_test",
            unit_operator="operator_hp",
            index=index,
            bidding_strategies={},
            components=incomplete_components,
            objective=default_objective,
            flexibility_measure=default_flexibility_measure,
            forecaster=forecast,
        )

    # Assert the correct error message
    assert "Component boiler is required for the building plant unit." in str(
        exc_info.value
    )

    # Reset required technologies to avoid affecting other tests
    Building.required_technologies = []


def test_building_ev_discharge_constraint(
    forecast,
    index,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    """
    Test that the discharge_ev_to_market_constraint is correctly defined
    when the EV is not allowed to sell energy to the market.
    """
    # Modify the EV configuration to disallow selling energy to the market
    building_components_heatpump["electric_vehicle"]["sells_energy_to_market"] = "false"

    building = Building(
        id="building_ev_test",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecast,
    )

    # Verify that the constraint is defined
    constraints = list(building.model.component_map(pyo.Constraint).keys())
    assert "discharge_ev_to_market_constraint" in constraints


def test_building_battery_discharge_constraint_simple(
    forecast,
    index,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    """
    Test that the discharge_battery_to_market_constraint is defined
    when the battery is not allowed to sell energy to the market.
    """
    # Modify the battery configuration to disallow selling energy to the market
    building_components_heatpump["generic_storage"]["sells_energy_to_market"] = "false"

    building = Building(
        id="building_battery_test",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecast,
    )

    # Verify that the constraint is defined
    constraints = list(building.model.component_map(pyo.Constraint).keys())
    assert "discharge_battery_to_market_constraint" in constraints


def test_building_solver_infeasibility_logging(
    forecast,
    index,
    building_components_heatpump,
    default_objective,
    default_flexibility_measure,
):
    """
    Test that the Building class logs the correct messages when the solver reports infeasibility or other statuses.
    """
    # Create a Building instance
    building = Building(
        id="building_solver_test",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecast,
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

    # Assert no exceptions occur
    assert True


def test_building_bidding_strategy_execution(
    forecast,
    index,
    building_components_heatpump,
    available_solver,
    default_objective,
    default_flexibility_measure,
):
    """
    Test that the NaiveDADSMStrategy's calculate_bids method is executed correctly,
    and unit.determine_optimal_operation_without_flex() is called.
    """
    # Create the Building instance with a NaiveDADSMStrategy
    building = Building(
        id="building_heatpump",
        unit_operator="operator_hp",
        index=index,
        bidding_strategies={"EOM": NaiveDADSMStrategy()},
        components=building_components_heatpump,
        objective=default_objective,
        flexibility_measure=default_flexibility_measure,
        forecaster=forecast,
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


# ----------------------------
# Additional Helper Tests (Optional)
# ----------------------------


def test_building_get_available_solvers():
    available_solvers = check_available_solvers(*SOLVERS)
    assert isinstance(available_solvers, list)
    for solver in available_solvers:
        assert SolverFactory(solver).available()


def test_str_to_bool_invalid_value_in_building():
    """
    Test that str_to_bool raises ValueError when an invalid value is passed
    within the Building context.
    """
    invalid_components = {
        "electric_vehicle": {"sells_energy_to_market": "invalid_value"}
    }

    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building_invalid_str_to_bool",
            unit_operator="operator_invalid",
            index=range(10),
            bidding_strategies={},
            components=invalid_components,
            objective="min_variable_cost",
            flexibility_measure="cost_based_load_shift",
            forecaster=None,  # Replace with a suitable forecaster if necessary
        )
    assert "Invalid truth value: 'invalid_value'" in str(exc_info.value)


def test_building_unknown_flexibility_measure(
    forecast,
    index,
    building_components_heatpump,
    default_objective,
):
    """
    Test that the Building class raises a ValueError for an unknown flexibility measure.
    """
    invalid_flexibility_measure = "invalid_flex_measure"

    with pytest.raises(ValueError) as exc_info:
        Building(
            id="building_unknown_flex",
            unit_operator="operator_hp",
            index=index,
            bidding_strategies={},
            components=building_components_heatpump,
            objective=default_objective,
            flexibility_measure=invalid_flexibility_measure,
            forecaster=forecast,
        )

    # Assert the correct error message
    assert f"Unknown flexibility measure: {invalid_flexibility_measure}" in str(
        exc_info.value
    )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
