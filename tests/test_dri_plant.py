# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import DRIPlant

# Define the solver to use
use_solver = "appsi_highs"  # Replace with the appropriate solver


# Fixture for creating an electricity price profile, including negative prices
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, 1000, 55, 1000, 65, 45, 70], index=range(10))


# Fixture for DRI Plant configuration
@pytest.fixture
def dri_plant_config():
    return {
        "specific_hydrogen_consumption": 1,
        "specific_natural_gas_consumption": 1,
        "specific_electricity_consumption": 1,
        "specific_iron_ore_consumption": 1,
        "max_power": 50,
        "min_power": 10,  # Changed from 0 to 10 to test min_power constraints
        "fuel_type": "natural_gas",
        "ramp_up": 50,
        "ramp_down": 50,
        "min_operating_steps": 2,
        "min_down_steps": 2,
        "initial_operational_status": 1,
        "natural_gas_co2_factor": 0.5,
    }


# Fixture for creating and solving the DRI Plant model
@pytest.fixture
def dri_plant_model(dri_plant_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    # Add the price profiles to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )
    model.natural_gas_price = pyo.Param(model.time_steps, initialize=3)
    model.iron_ore_price = pyo.Param(model.time_steps, initialize=10)
    model.co2_price = pyo.Param(model.time_steps, initialize=30)

    # Initialize the DRIPlant
    dri_plant = DRIPlant(**dri_plant_config, time_steps=model.time_steps)
    model.dri_plant = pyo.Block()
    dri_plant.add_to_model(model, model.dri_plant)

    # Constraint for total DRI production over all time steps
    total_dri_production = 300
    model.total_dri_constraint = pyo.Constraint(
        expr=sum(model.dri_plant.dri_output[t] for t in model.time_steps)
        == total_dri_production
    )

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(model.dri_plant.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)

    return model, results


def test_model_solves_successfully(dri_plant_model):
    """
    Test that the optimization model solves successfully.
    """
    model, results = dri_plant_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution. Status: {results.solver.status}, "
        f"Termination Condition: {results.solver.termination_condition}"
    )


def test_total_dri_production(dri_plant_model):
    """
    Test that the total DRI production constraint is met.
    """
    model, _ = dri_plant_model
    total_dri = sum(pyo.value(model.dri_plant.dri_output[t]) for t in model.time_steps)
    assert total_dri == 300, f"Total DRI production is {total_dri}, expected 300."


def test_dri_plant_ramping_constraints(dri_plant_model, dri_plant_config):
    """
    Test that ramp-up and ramp-down constraints are respected.
    """
    model, _ = dri_plant_model
    ramp_up_limit = dri_plant_config["ramp_up"]
    ramp_down_limit = dri_plant_config["ramp_down"]

    for t in model.time_steps:
        if t > model.time_steps.at(1):
            power_current = pyo.value(model.dri_plant.power_in[t])
            power_prev = pyo.value(model.dri_plant.power_in[t - 1])
            ramp_up = power_current - power_prev
            ramp_down = power_prev - power_current

            assert ramp_up <= ramp_up_limit + 1e-5, (
                f"Ramp-up at time {t} is {ramp_up}, exceeds limit of {ramp_up_limit}."
            )
            assert ramp_down <= ramp_down_limit + 1e-5, (
                f"Ramp-down at time {t} is {ramp_down}, exceeds limit of {ramp_down_limit}."
            )


def test_dri_plant_power_bounds(dri_plant_model, dri_plant_config):
    """
    Test that power input respects the minimum and maximum power bounds.
    """
    model, _ = dri_plant_model
    min_power = dri_plant_config["min_power"]
    max_power = dri_plant_config["max_power"]

    for t in model.time_steps:
        power_in = pyo.value(model.dri_plant.power_in[t])
        operational_status = pyo.value(model.dri_plant.operational_status[t])
        assert (
            min_power * operational_status
            <= power_in
            <= max_power * operational_status + 1e-5
        ), (
            f"Power input at time {t} is {power_in}, which is outside the bounds "
            f"{min_power * operational_status} - {max_power * operational_status}."
        )


def test_dri_plant_off_when_high_price(dri_plant_model, price_profile):
    """
    Test that the DRI plant is turned off when the electricity price high.
    """
    model, _ = dri_plant_model

    for t in model.time_steps:
        price = price_profile[t]
        power_in = pyo.value(model.dri_plant.power_in[t])
        if price == 1000:
            assert power_in == 0, (
                f"DRI plant is on at time {t} despite high price {price}."
            )


def test_min_operating_steps(dri_plant_model, dri_plant_config):
    """
    Test that the DRI plant operates for at least the minimum number of operating steps after startup.
    """
    model, _ = dri_plant_model
    min_steps = dri_plant_config["min_operating_steps"]
    operational_status = model.dri_plant.operational_status

    startup_times = [
        t for t in model.time_steps if pyo.value(model.dri_plant.start_up[t]) == 1
    ]

    for t in startup_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 1, (
                    f"Operational status at time {future_t} should be 1 after startup at {t}, but is {status}."
                )


def test_min_downtime_steps(dri_plant_model, dri_plant_config):
    """
    Test that the DRI plant remains off for at least the minimum number of downtime steps after shutdown.
    """
    model, _ = dri_plant_model
    min_steps = dri_plant_config["min_down_steps"]
    operational_status = model.dri_plant.operational_status

    shutdown_times = [
        t for t in model.time_steps if pyo.value(model.dri_plant.shut_down[t]) == 1
    ]

    for t in shutdown_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 0, (
                    f"Operational status at time {future_t} should be 0 after shutdown at {t}, but is {status}."
                )


def test_initial_operational_status(dri_plant_model, dri_plant_config):
    """
    Test that the initial operational status of the DRI plant is correctly set.
    """
    model, _ = dri_plant_model
    initial_status = dri_plant_config["initial_operational_status"]
    first_time_step = model.time_steps.at(1)
    actual_status = pyo.value(model.dri_plant.operational_status[first_time_step])
    assert actual_status == initial_status, (
        f"Initial operational status is {actual_status}, expected {initial_status}."
    )


def test_operating_cost(dri_plant_model, price_profile):
    """
    Test that the operating cost is calculated correctly based on fuel and electricity consumption.
    """
    model, _ = dri_plant_model
    total_calculated_cost = sum(
        pyo.value(model.dri_plant.operating_cost[t]) for t in model.time_steps
    )
    total_model_cost = pyo.value(model.total_cost)
    assert abs(total_calculated_cost - total_model_cost) < 1e-5, (
        f"Calculated total operating cost {total_calculated_cost} does not match model's total cost {total_model_cost}."
    )

    # Additionally, verify individual operating costs
    for t in model.time_steps:
        natural_gas_consumption = pyo.value(model.dri_plant.natural_gas_in[t])
        electricity_consumption = pyo.value(model.dri_plant.power_in[t])
        iron_ore_consumption = pyo.value(model.dri_plant.iron_ore_in[t])
        co2_emission = pyo.value(model.dri_plant.co2_emission[t])
        price_ng = pyo.value(model.natural_gas_price[t])
        price_elec = pyo.value(model.electricity_price[t])
        price_iron = pyo.value(model.iron_ore_price[t])
        price_co2 = pyo.value(model.co2_price[t])

        expected_cost = (
            natural_gas_consumption * price_ng
            + electricity_consumption * price_elec
            + iron_ore_consumption * price_iron
            + co2_emission * price_co2
        )
        actual_cost = pyo.value(model.dri_plant.operating_cost[t])
        assert abs(actual_cost - expected_cost) < 1e-5, (
            f"Operating cost at time {t} is {actual_cost}, expected {expected_cost}."
        )


def test_fuel_consumption_constraints(dri_plant_model, dri_plant_config):
    """
    Test that fuel consumption aligns with the specified fuel type.
    """
    model, _ = dri_plant_model
    fuel_type = dri_plant_config["fuel_type"]

    for t in model.time_steps:
        dri_output = pyo.value(model.dri_plant.dri_output[t])
        hydrogen_in = pyo.value(model.dri_plant.hydrogen_in[t])
        natural_gas_in = pyo.value(model.dri_plant.natural_gas_in[t])

        if fuel_type == "hydrogen":
            expected_dri_output = (
                hydrogen_in / model.dri_plant.specific_hydrogen_consumption
            )
            assert abs(dri_output - expected_dri_output) < 1e-5, (
                f"At time {t}, DRI output {dri_output} does not match hydrogen consumption {hydrogen_in}."
            )
            # Ensure natural_gas_in is zero
            assert natural_gas_in == 0, (
                f"At time {t}, natural_gas_in is {natural_gas_in} but fuel_type is hydrogen."
            )
        elif fuel_type == "natural_gas":
            expected_dri_output = (
                natural_gas_in / model.dri_plant.specific_natural_gas_consumption
            )
            assert abs(dri_output - expected_dri_output) < 1e-5, (
                f"At time {t}, DRI output {dri_output} does not match natural gas consumption {natural_gas_in}."
            )
            # Ensure hydrogen_in is zero
            assert pyo.value(model.dri_plant.hydrogen_in[t]) == 0, (
                f"At time {t}, hydrogen_in is {pyo.value(model.dri_plant.hydrogen_in[t])} but fuel_type is natural_gas."
            )
        elif fuel_type == "both":
            expected_dri_output = (
                hydrogen_in / model.dri_plant.specific_hydrogen_consumption
                + natural_gas_in / model.dri_plant.specific_natural_gas_consumption
            )
            assert abs(dri_output - expected_dri_output) < 1e-5, (
                f"At time {t}, DRI output {dri_output} does not match combined fuel consumption."
            )
        else:
            pytest.fail(f"Unknown fuel_type '{fuel_type}' specified in configuration.")


def test_iron_ore_consumption(dri_plant_model, dri_plant_config):
    """
    Test that iron ore consumption is correctly linked to DRI output.
    """
    model, _ = dri_plant_model

    for t in model.time_steps:
        dri_output = pyo.value(model.dri_plant.dri_output[t])
        iron_ore_in = pyo.value(model.dri_plant.iron_ore_in[t])
        specific_iron_ore = pyo.value(model.dri_plant.specific_iron_ore_consumption)
        expected_iron_ore = dri_output * specific_iron_ore
        assert abs(iron_ore_in - expected_iron_ore) < 1e-5, (
            f"Iron ore consumption at time {t} is {iron_ore_in}, expected {expected_iron_ore}."
        )
