# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import math

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import HeatPump

# Define the solver to use
use_solver = "appsi_highs"  # Replace with the appropriate solver


# Fixture for creating an electricity price profile, including high prices to test logic
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, 1000, 55, 1000, 65, 45, 70], index=range(10))


# Fixture for heat pump configuration
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


# Fixture for creating and solving a Pyomo model with a heat pump component
@pytest.fixture
def heat_pump_model(heat_pump_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    # Add the price profile to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # Initialize the HeatPump
    heat_pump = HeatPump(time_steps=model.time_steps, **heat_pump_config)
    model.heat_pump = pyo.Block()
    heat_pump.add_to_model(model, model.heat_pump)

    # Constraint for total heat production over all time steps
    total_heat_production = 400
    model.total_heat_constraint = pyo.Constraint(
        expr=sum(model.heat_pump.heat_out[t] for t in model.time_steps)
        == total_heat_production
    )

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(model.heat_pump.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)

    return model, results


def test_model_solves_successfully(heat_pump_model):
    """
    Test that the optimization model solves successfully.
    """
    model, results = heat_pump_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution. Status: {results.solver.status}, Termination Condition: {results.solver.termination_condition}"
    )


def test_total_heat_production(heat_pump_model):
    """
    Test that the total heat production constraint is met.
    """
    model, _ = heat_pump_model
    total_heat = sum(pyo.value(model.heat_pump.heat_out[t]) for t in model.time_steps)
    assert abs(total_heat - 400) < 1e-5, (
        f"Total heat production is {total_heat}, expected 400."
    )


def test_heat_pump_ramping_constraints(heat_pump_model, heat_pump_config):
    """
    Test that ramp-up and ramp-down constraints are respected.
    """
    model, _ = heat_pump_model
    ramp_up_limit = heat_pump_config["ramp_up"]
    ramp_down_limit = heat_pump_config["ramp_down"]

    for t in model.time_steps:
        if t > model.time_steps.first():
            power_current = pyo.value(model.heat_pump.power_in[t])
            power_prev = pyo.value(model.heat_pump.power_in[t - 1])
            ramp_up = power_current - power_prev
            ramp_down = power_prev - power_current

            assert ramp_up <= ramp_up_limit + 1e-5, (
                f"Ramp-up at time {t} is {ramp_up}, exceeds limit of {ramp_up_limit}."
            )
            assert ramp_down <= ramp_down_limit + 1e-5, (
                f"Ramp-down at time {t} is {ramp_down}, exceeds limit of {ramp_down_limit}."
            )


def test_heat_pump_power_bounds(heat_pump_model, heat_pump_config):
    """
    Test that power input respects the minimum and maximum power bounds.
    """
    model, _ = heat_pump_model
    min_power = heat_pump_config["min_power"]
    max_power = heat_pump_config["max_power"]

    for t in model.time_steps:
        power_in = pyo.value(model.heat_pump.power_in[t])
        operational_status = (
            pyo.value(model.heat_pump.operational_status[t])
            if hasattr(model.heat_pump, "operational_status")
            else 1
        )
        assert (
            min_power * operational_status - 1e-5
            <= power_in
            <= max_power * operational_status + 1e-5
        ), (
            f"Power input at time {t} is {power_in}, which is outside the bounds "
            f"{min_power * operational_status} - {max_power * operational_status}."
        )


def test_heat_pump_consumption_behavior(
    heat_pump_model, price_profile, heat_pump_config
):
    """
    Test that the heat pump consumes power when electricity prices are low and remains off when prices are high.
    """
    model, _ = heat_pump_model
    high_price_threshold = 500  # Define a threshold for high prices; adjust as needed

    for t in model.time_steps:
        price = price_profile[t]
        power_in = pyo.value(model.heat_pump.power_in[t])
        if price >= high_price_threshold:
            # When price is high, heat pump should not consume power
            assert power_in == 0, (
                f"Heat pump is consuming power at time {t} despite high price {price}."
            )
        else:
            # When price is low, heat pump should consume at least min_power
            operational_status = round(
                pyo.value(model.heat_pump.operational_status[t]), 0
            )
            if operational_status:
                # Adjust the assert statement to use math.isclose for comparison
                assert (
                    math.isclose(power_in, heat_pump_config["min_power"], rel_tol=1e-6)
                    or power_in > heat_pump_config["min_power"]
                ), (
                    f"Heat pump power at time {t} is {power_in}, which is below the minimum power {heat_pump_config['min_power']}."
                )
            else:
                assert power_in == 0, (
                    f"Heat pump should be off at time {t}, but power_in is {power_in}."
                )


def test_min_operating_steps(heat_pump_model, heat_pump_config):
    """
    Test that the heat pump operates for at least the minimum number of operating steps after startup.
    """
    model, _ = heat_pump_model
    min_steps = heat_pump_config["min_operating_steps"]
    operational_status = model.heat_pump.operational_status

    startup_times = [
        t for t in model.time_steps if pyo.value(model.heat_pump.start_up[t]) == 1
    ]

    for t in startup_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert math.isclose(status, 1, rel_tol=1e-6), (
                    f"Operational status at time {future_t} should be 1 after startup at {t}, but is {status}."
                )


def test_min_downtime_steps(heat_pump_model, heat_pump_config):
    """
    Test that the heat pump remains off for at least the minimum number of downtime steps after shutdown.
    """
    model, _ = heat_pump_model
    min_steps = heat_pump_config["min_down_steps"]
    operational_status = model.heat_pump.operational_status

    shutdown_times = [
        t for t in model.time_steps if pyo.value(model.heat_pump.shut_down[t]) == 1
    ]

    for t in shutdown_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 0, (
                    f"Operational status at time {future_t} should be 0 after shutdown at {t}, but is {status}."
                )


def test_initial_operational_status(heat_pump_model, heat_pump_config):
    """
    Test that the initial operational status of the heat pump is correctly set.
    """
    import pytest

    model, _ = heat_pump_model
    initial_status = heat_pump_config["initial_operational_status"]
    first_time_step = model.time_steps.first()
    actual_status = pyo.value(model.heat_pump.operational_status[first_time_step])
    assert actual_status == pytest.approx(initial_status, abs=1e-8), (
        f"Initial operational status is {actual_status}, expected {initial_status}."
    )


def test_operating_cost(heat_pump_model, price_profile, heat_pump_config):
    """
    Test that the operating cost is calculated correctly based on power input and electricity price.
    """
    model, _ = heat_pump_model
    total_calculated_cost = sum(
        pyo.value(model.heat_pump.power_in[t]) * price_profile[t]
        for t in model.time_steps
    )
    total_model_cost = pyo.value(model.total_cost)
    assert abs(total_calculated_cost - total_model_cost) < 1e-5, (
        f"Calculated total operating cost {total_calculated_cost} does not match model's total cost {total_model_cost}."
    )

    for t in model.time_steps:
        expected_cost = pyo.value(model.heat_pump.power_in[t]) * price_profile[t]
        actual_cost = pyo.value(model.heat_pump.operating_cost[t])
        assert abs(actual_cost - expected_cost) < 1e-5, (
            f"Operating cost at time {t} is {actual_cost}, expected {expected_cost}."
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
