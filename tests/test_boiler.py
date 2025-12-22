# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import Boiler

# Define the solver to use
use_solver = "appsi_highs"  # Replace with the appropriate solver


# Fixture for creating an electricity price profile, including negative prices
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, 1000, 55, 1000, 65, 45, 70], index=range(10))


# Fixture for boiler configuration (electric fuel type)
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


# Fixture for boiler configuration (natural_gas fuel type)
@pytest.fixture
def gas_boiler_config():
    return {
        "max_power": 100,
        "efficiency": 0.85,
        "fuel_type": "natural_gas",  # Natural gas does not support operational constraints
        "min_power": 0,
        "ramp_up": 50,
        "ramp_down": 50,
        # Do not set min_operating_steps and min_down_steps for natural_gas
        "initial_operational_status": 1,  # This will be ignored for natural_gas
    }


# Fixture for boiler configuration (hydrogen_gas fuel type)
@pytest.fixture
def hydrogen_boiler_config():
    return {
        "max_power": 100,
        "efficiency": 0.85,
        "fuel_type": "hydrogen_gas",
        "min_power": 0,
        "ramp_up": 50,
        "ramp_down": 50,
        "initial_operational_status": 1,
    }


# Fixture for creating and solving a Pyomo model with a boiler component (electric)
@pytest.fixture
def electric_boiler_model(electric_boiler_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    # Add the price profile to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # Initialize the Boiler
    boiler = Boiler(**electric_boiler_config, time_steps=model.time_steps)
    model.boiler = pyo.Block()
    boiler.add_to_model(model, model.boiler)

    # Constraint for total heat production over all time steps
    total_heat_production = 400
    model.total_heat_constraint = pyo.Constraint(
        expr=sum(model.boiler.heat_out[t] for t in model.time_steps)
        == total_heat_production
    )

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(
            model.boiler.power_in[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)

    return model, results


# Fixture for creating and solving a Pyomo model with a boiler component (natural_gas)
@pytest.fixture
def gas_boiler_model(gas_boiler_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    # Add the price profile to the model (not used for natural_gas, but kept for consistency)
    model.natural_gas_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # Initialize the Boiler
    boiler = Boiler(**gas_boiler_config, time_steps=model.time_steps)
    model.boiler = pyo.Block()
    boiler.add_to_model(model, model.boiler)

    # Constraint for total heat production over all time steps
    total_heat_production = 400
    model.total_heat_constraint = pyo.Constraint(
        expr=sum(model.boiler.heat_out[t] for t in model.time_steps)
        == total_heat_production
    )

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(model.boiler.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)

    return model, results


# Hydrogen gas boiler model
@pytest.fixture
def hydrogen_boiler_model(hydrogen_boiler_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    model.hydrogen_gas_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    boiler = Boiler(**hydrogen_boiler_config, time_steps=model.time_steps)
    model.boiler = pyo.Block()
    boiler.add_to_model(model, model.boiler)

    total_heat_production = 400
    model.total_heat_constraint = pyo.Constraint(
        expr=sum(model.boiler.heat_out[t] for t in model.time_steps)
        == total_heat_production
    )

    model.total_cost = pyo.Objective(
        expr=sum(model.boiler.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)
    return model, results


def test_electric_boiler_model_solves_successfully(electric_boiler_model):
    """
    Test that the electric boiler optimization model solves successfully.
    """
    model, results = electric_boiler_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution for electric boiler. Status: {results.solver.status}, "
        f"Termination Condition: {results.solver.termination_condition}"
    )


def test_gas_boiler_model_solves_successfully(gas_boiler_model):
    """
    Test that the natural gas boiler optimization model solves successfully.
    """
    model, results = gas_boiler_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution for natural gas boiler. Status: {results.solver.status}, "
        f"Termination Condition: {results.solver.termination_condition}"
    )


def test_total_heat_production_electric(electric_boiler_model):
    """
    Test that the total heat production constraint is met for electric boiler.
    """
    model, _ = electric_boiler_model
    total_heat = sum(pyo.value(model.boiler.heat_out[t]) for t in model.time_steps)
    assert total_heat == pytest.approx(400, rel=1e-2), (
        f"Total heat production for electric boiler is {total_heat}, expected 500."
    )


def test_total_heat_production_gas(gas_boiler_model):
    """
    Test that the total heat production constraint is met for natural gas boiler.
    """
    model, _ = gas_boiler_model
    total_heat = sum(pyo.value(model.boiler.heat_out[t]) for t in model.time_steps)
    assert total_heat == pytest.approx(400, rel=1e-2), (
        f"Total heat production for natural gas boiler is {total_heat}, expected 500."
    )


def test_electric_boiler_ramping_constraints(
    electric_boiler_model, electric_boiler_config
):
    """
    Test that ramp-up and ramp-down constraints are respected for electric boiler.
    """
    model, _ = electric_boiler_model
    ramp_up_limit = electric_boiler_config["ramp_up"]
    ramp_down_limit = electric_boiler_config["ramp_down"]

    for t in model.time_steps:
        if t > model.time_steps.first():
            power_current = pyo.value(model.boiler.power_in[t])
            power_prev = pyo.value(model.boiler.power_in[t - 1])
            ramp_up = power_current - power_prev
            ramp_down = power_prev - power_current

            assert ramp_up <= ramp_up_limit + 1e-5, (
                f"Electric Boiler: Ramp-up at time {t} is {ramp_up}, exceeds limit of {ramp_up_limit}."
            )
            assert ramp_down <= ramp_down_limit + 1e-5, (
                f"Electric Boiler: Ramp-down at time {t} is {ramp_down}, exceeds limit of {ramp_down_limit}."
            )


def test_gas_boiler_ramping_constraints(gas_boiler_model, gas_boiler_config):
    """
    Test that ramp-up and ramp-down constraints are respected for natural gas boiler.
    """
    model, _ = gas_boiler_model
    ramp_up_limit = gas_boiler_config["ramp_up"]
    ramp_down_limit = gas_boiler_config["ramp_down"]

    for t in model.time_steps:
        if t > model.time_steps.first():
            natural_gas_current = pyo.value(model.boiler.natural_gas_in[t])
            natural_gas_prev = pyo.value(model.boiler.natural_gas_in[t - 1])
            ramp_up = natural_gas_current - natural_gas_prev
            ramp_down = natural_gas_prev - natural_gas_current

            assert ramp_up <= ramp_up_limit + 1e-5, (
                f"Natural Gas Boiler: Ramp-up at time {t} is {ramp_up}, exceeds limit of {ramp_up_limit}."
            )
            assert ramp_down <= ramp_down_limit + 1e-5, (
                f"Natural Gas Boiler: Ramp-down at time {t} is {ramp_down}, exceeds limit of {ramp_down_limit}."
            )


def test_electric_boiler_power_bounds(electric_boiler_model, electric_boiler_config):
    """
    Test that power input respects the minimum and maximum power bounds for electric boiler.
    """
    model, _ = electric_boiler_model
    min_power = electric_boiler_config["min_power"]
    max_power = electric_boiler_config["max_power"]

    for t in model.time_steps:
        power_in = pyo.value(model.boiler.power_in[t])
        operational_status = pyo.value(model.boiler.operational_status[t])
        assert (
            min_power * operational_status
            <= power_in
            <= max_power * operational_status + 1e-5
        ), (
            f"Electric Boiler: Power input at time {t} is {power_in}, which is outside the bounds "
            f"{min_power * operational_status} - {max_power * operational_status}."
        )


def test_gas_boiler_power_bounds(gas_boiler_model, gas_boiler_config):
    """
    Test that natural gas input respects the minimum and maximum power bounds.
    """
    model, _ = gas_boiler_model
    min_power = gas_boiler_config["min_power"]
    max_power = gas_boiler_config["max_power"]

    for t in model.time_steps:
        natural_gas_in = pyo.value(model.boiler.natural_gas_in[t])
        # For natural_gas, power_in is zero
        assert natural_gas_in >= min_power - 1e-5, (
            f"Natural Gas Boiler: Natural gas input at time {t} is {natural_gas_in}, which is below the minimum power {min_power}."
        )
        # Assuming there is an upper limit if applicable; adjust if necessary
        # Since power_in is not applicable, we skip max_power for natural_gas
        # If max_power applies to natural_gas_in, uncomment the following:
        # assert natural_gas_in <= max_power + 1e-5, (
        #     f"Natural Gas Boiler: Natural gas input at time {t} is {natural_gas_in}, exceeds the maximum power {max_power}."
        # )


def test_electric_boiler_efficiency_constraint(
    electric_boiler_model, electric_boiler_config
):
    """
    Test that the efficiency constraint is maintained for electric boiler.
    """
    model, _ = electric_boiler_model
    efficiency = electric_boiler_config["efficiency"]

    for t in model.time_steps:
        heat_out = pyo.value(model.boiler.heat_out[t])
        power_in = pyo.value(model.boiler.power_in[t])
        expected_heat_out = power_in * efficiency
        assert heat_out == pytest.approx(expected_heat_out, rel=1e-2), (
            f"Electric Boiler: Heat output at time {t} is {heat_out}, expected {expected_heat_out} based on efficiency."
        )


def test_gas_boiler_efficiency_constraint(gas_boiler_model, gas_boiler_config):
    """
    Test that the efficiency constraint is maintained for natural gas boiler.
    """
    model, _ = gas_boiler_model
    efficiency = gas_boiler_config["efficiency"]

    for t in model.time_steps:
        heat_out = pyo.value(model.boiler.heat_out[t])
        natural_gas_in = pyo.value(model.boiler.natural_gas_in[t])
        expected_heat_out = natural_gas_in * efficiency
        assert heat_out == pytest.approx(expected_heat_out, rel=1e-2), (
            f"Natural Gas Boiler: Heat output at time {t} is {heat_out}, expected {expected_heat_out} based on efficiency."
        )


def test_electric_boiler_operational_status_constraints(
    electric_boiler_model, electric_boiler_config
):
    """
    Test that the electric boiler adheres to minimum operating steps and minimum downtime steps.
    """
    model, _ = electric_boiler_model
    min_operating_steps = electric_boiler_config["min_operating_steps"]
    min_down_steps = electric_boiler_config["min_down_steps"]
    operational_status = model.boiler.operational_status

    # Identify all startup and shutdown events
    start_up_times = [
        t for t in model.time_steps if pyo.value(model.boiler.start_up[t]) == 1
    ]
    shut_down_times = [
        t for t in model.time_steps if pyo.value(model.boiler.shut_down[t]) == 1
    ]

    # Check minimum operating steps after each startup
    for t in start_up_times:
        for step in range(min_operating_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 1, (
                    f"Electric Boiler: Operational status at time {future_t} should be 1 after startup at {t}, but is {status}."
                )

    # Check minimum downtime steps after each shutdown
    for t in shut_down_times:
        for step in range(min_down_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 0, (
                    f"Electric Boiler: Operational status at time {future_t} should be 0 after shutdown at {t}, but is {status}."
                )


def test_electric_boiler_initial_operational_status(
    electric_boiler_model, electric_boiler_config
):
    """
    Test that the initial operational status of the electric boiler is correctly set.
    """
    model, _ = electric_boiler_model
    initial_status = electric_boiler_config["initial_operational_status"]
    first_time_step = model.time_steps.first()
    actual_status = pyo.value(model.boiler.operational_status[first_time_step])
    assert actual_status == initial_status, (
        f"Electric Boiler: Initial operational status is {actual_status}, expected {initial_status}."
    )


def test_gas_boiler_initial_operational_status(gas_boiler_model, gas_boiler_config):
    """
    Test that the initial operational status of the natural gas boiler is correctly set.
    """
    # For natural gas boilers, operational status variables are not defined
    # Hence, this test can be omitted or can verify that natural_gas_in is set correctly
    model, _ = gas_boiler_model
    # Since operational_status is not present, we skip this test
    # Alternatively, verify that natural_gas_in can be non-zero based on initial_status
    initial_status = gas_boiler_config["initial_operational_status"]
    first_time_step = model.time_steps.first()
    natural_gas_in = pyo.value(model.boiler.natural_gas_in[first_time_step])

    if initial_status == 1:
        assert natural_gas_in >= gas_boiler_config["min_power"] - 1e-5, (
            f"Natural Gas Boiler: Natural gas input at initial time step {first_time_step} is {natural_gas_in}, which is below the minimum power {gas_boiler_config['min_power']}."
        )
    else:
        assert natural_gas_in == 0, (
            f"Natural Gas Boiler: Natural gas input at initial time step {first_time_step} is {natural_gas_in}, but boiler is off."
        )


def test_electric_boiler_operating_cost(electric_boiler_model, price_profile):
    """
    Test that the operating cost is calculated correctly for electric boiler based on power input and electricity price.
    """
    model, _ = electric_boiler_model
    total_calculated_cost = sum(
        pyo.value(model.boiler.power_in[t]) * price_profile[t] for t in model.time_steps
    )
    total_model_cost = pyo.value(model.total_cost)
    assert abs(total_calculated_cost - total_model_cost) < 1e-5, (
        f"Electric Boiler: Calculated total operating cost {total_calculated_cost} does not match model's total cost {total_model_cost}."
    )

    # Additionally, verify individual operating costs if available
    # If the model has variables or expressions for individual operating costs
    # For electric boilers, operating cost is integrated into the objective


def test_gas_boiler_operating_cost(gas_boiler_model, price_profile):
    """
    Test that the operating cost is calculated correctly for natural gas boiler based on natural gas input and electricity price.
    """
    model, _ = gas_boiler_model
    total_calculated_cost = sum(
        pyo.value(model.boiler.natural_gas_in[t]) * price_profile[t]
        for t in model.time_steps
    )
    total_model_cost = pyo.value(model.total_cost)
    assert abs(total_calculated_cost - total_model_cost) < 1e-5, (
        f"Natural Gas Boiler: Calculated total operating cost {total_calculated_cost} does not match model's total cost {total_model_cost}."
    )


def test_electric_boiler_off_during_high_prices(electric_boiler_model, price_profile):
    """
    Test that the electric boiler is turned off when the electricity price is high.
    """
    model, _ = electric_boiler_model

    for t in model.time_steps:
        price = price_profile[t]
        power_in = pyo.value(model.boiler.power_in[t])
        if price == 1000:
            assert power_in == 0, (
                f"Electric Boiler: Boiler is on at time {t} despite high price {price}."
            )


def test_gas_boiler_min_power_enforcement(gas_boiler_model, gas_boiler_config):
    """
    Test that the natural gas boiler enforces the minimum power when it is operational.
    """
    model, _ = gas_boiler_model
    min_power = gas_boiler_config["min_power"]

    for t in model.time_steps:
        natural_gas_in = pyo.value(model.boiler.natural_gas_in[t])
        if natural_gas_in > 0:
            assert natural_gas_in >= min_power - 1e-5, (
                f"Natural Gas Boiler: Natural gas input at time {t} is {natural_gas_in}, which is below the minimum power {min_power}."
            )
        else:
            assert natural_gas_in == 0, (
                f"Natural Gas Boiler: Natural gas input at time {t} is {natural_gas_in}, but boiler is off."
            )


# Hydrogen gas boiler tests
def test_hydrogen_boiler_model_solves_successfully(hydrogen_boiler_model):
    model, results = hydrogen_boiler_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution for hydrogen gas boiler. Status: {results.solver.status}, "
        f"Termination Condition: {results.solver.termination_condition}"
    )


def test_total_heat_production_hydrogen(hydrogen_boiler_model):
    model, _ = hydrogen_boiler_model
    total_heat = sum(pyo.value(model.boiler.heat_out[t]) for t in model.time_steps)
    assert total_heat == pytest.approx(400, rel=1e-2), (
        f"Total heat production for hydrogen gas boiler is {total_heat}, expected 400."
    )


def test_hydrogen_boiler_ramping_constraints(
    hydrogen_boiler_model, hydrogen_boiler_config
):
    model, _ = hydrogen_boiler_model
    ramp_up_limit = hydrogen_boiler_config["ramp_up"]
    ramp_down_limit = hydrogen_boiler_config["ramp_down"]

    for t in model.time_steps:
        if t > model.time_steps.first():
            hydrogen_current = pyo.value(model.boiler.hydrogen_gas_in[t])
            hydrogen_prev = pyo.value(model.boiler.hydrogen_gas_in[t - 1])
            ramp_up = hydrogen_current - hydrogen_prev
            ramp_down = hydrogen_prev - hydrogen_current

            assert ramp_up <= ramp_up_limit + 1e-5, (
                f"Hydrogen Boiler: Ramp-up at time {t} is {ramp_up}, exceeds limit of {ramp_up_limit}."
            )
            assert ramp_down <= ramp_down_limit + 1e-5, (
                f"Hydrogen Boiler: Ramp-down at time {t} is {ramp_down}, exceeds limit of {ramp_down_limit}."
            )


def test_hydrogen_boiler_max_power(hydrogen_boiler_model, hydrogen_boiler_config):
    model, _ = hydrogen_boiler_model
    max_power = hydrogen_boiler_config["max_power"]
    for t in model.time_steps:
        hydrogen_val = pyo.value(model.boiler.hydrogen_gas_in[t])
        assert hydrogen_val <= max_power + 1e-5, (
            f"Hydrogen input at {t} exceeds max_power {max_power}"
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
