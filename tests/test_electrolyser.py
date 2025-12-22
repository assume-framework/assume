# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import Electrolyser

# Define the solver to use
use_solver = "appsi_highs"  # Replace with the appropriate solver


# Fixture for creating an electricity price profile, including negative prices
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, 1000, 55, 1000, 65, 45, 70], index=range(10))


# Fixture for electrolyser configuration
@pytest.fixture
def electrolyser_config():
    return {
        "max_power": 50,
        "min_power": 10,
        "efficiency": 1,
        "ramp_up": 30,
        "ramp_down": 30,
        "min_operating_steps": 2,
        "min_down_steps": 2,
        "initial_status": 1,
    }


# Fixture for creating and solving the electrolyser model
@pytest.fixture
def electrolyser_model(electrolyser_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    # Add the price profile to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # Initialize the Electrolyser
    electrolyser = Electrolyser(**electrolyser_config, time_steps=model.time_steps)
    model.electrolyser = pyo.Block()
    electrolyser.add_to_model(model, model.electrolyser)

    # Constraint for total hydrogen production over all time steps
    total_hydrogen_production = 200
    model.total_hydrogen_constraint = pyo.Constraint(
        expr=sum(model.electrolyser.hydrogen_out[t] for t in model.time_steps)
        == total_hydrogen_production
    )

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(model.electrolyser.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)

    return model, results


def test_model_solves_successfully(electrolyser_model):
    """
    Test that the optimization model solves successfully.
    """
    model, results = electrolyser_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution. Status: {results.solver.status}, Termination Condition: {results.solver.termination_condition}"
    )


def test_total_hydrogen_production(electrolyser_model):
    """
    Test that the total hydrogen production constraint is met.
    """
    model, _ = electrolyser_model
    total_hydrogen = sum(
        pyo.value(model.electrolyser.hydrogen_out[t]) for t in model.time_steps
    )
    assert total_hydrogen == 200, (
        f"Total hydrogen production is {total_hydrogen}, expected 300."
    )


def test_electrolyser_ramping_constraints(electrolyser_model, electrolyser_config):
    """
    Test that ramp-up and ramp-down constraints are respected.
    """
    model, _ = electrolyser_model
    ramp_up_limit = electrolyser_config["ramp_up"]
    ramp_down_limit = electrolyser_config["ramp_down"]

    for t in model.time_steps:
        if t > model.time_steps.at(1):
            power_current = pyo.value(model.electrolyser.power_in[t])
            power_prev = pyo.value(model.electrolyser.power_in[t - 1])
            ramp_up = power_current - power_prev
            ramp_down = power_prev - power_current

            assert ramp_up <= ramp_up_limit + 1e-5, (
                f"Ramp-up at time {t} is {ramp_up}, exceeds limit of {ramp_up_limit}."
            )
            assert ramp_down <= ramp_down_limit + 1e-5, (
                f"Ramp-down at time {t} is {ramp_down}, exceeds limit of {ramp_down_limit}."
            )


def test_electrolyser_power_bounds(electrolyser_model, electrolyser_config):
    """
    Test that power input respects the minimum and maximum power bounds.
    """
    import pytest

    model, _ = electrolyser_model
    min_power = electrolyser_config["min_power"]
    max_power = electrolyser_config["max_power"]
    tol = 1e-8

    for t in model.time_steps:
        power_in = pyo.value(model.electrolyser.power_in[t])
        operational_status = pyo.value(model.electrolyser.operational_status[t])
        if operational_status < 0.5:  # OFF
            assert power_in == pytest.approx(0, abs=tol), (
                f"Power input at time {t} is {power_in} but electrolyser is off."
            )
        else:  # ON
            assert min_power - tol <= power_in <= max_power + tol, (
                f"Power input at time {t} is {power_in}, which is outside the bounds "
                f"{min_power} - {max_power} (with tolerance {tol})."
            )


def test_electrolyser_off_when_high_price(electrolyser_model, price_profile):
    """
    Test that the electrolyser is turned off when the electricity price is high.
    """
    model, _ = electrolyser_model

    for t in model.time_steps:
        price = price_profile[t]
        power_in = pyo.value(model.electrolyser.power_in[t])
        if price > 500:
            assert power_in == 0, (
                f"Electrolyser is on at time {t} despite high price {price}."
            )


def test_min_operating_steps(electrolyser_model, electrolyser_config):
    """
    Test that the electrolyser operates for at least the minimum number of operating steps after startup.
    """
    model, _ = electrolyser_model
    min_steps = electrolyser_config["min_operating_steps"]
    operational_status = model.electrolyser.operational_status

    startup_times = [
        t for t in model.time_steps if pyo.value(model.electrolyser.start_up[t]) == 1
    ]

    for t in startup_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 1, (
                    f"Operational status at time {future_t} should be 1 after startup at {t}, but is {status}."
                )


def test_min_downtime_steps(electrolyser_model, electrolyser_config):
    """
    Test that the electrolyser remains off for at least the minimum number of downtime steps after shutdown.
    """
    model, _ = electrolyser_model
    min_steps = electrolyser_config["min_down_steps"]
    operational_status = model.electrolyser.operational_status

    shutdown_times = [
        t for t in model.time_steps if pyo.value(model.electrolyser.shut_down[t]) == 1
    ]

    for t in shutdown_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 0, (
                    f"Operational status at time {future_t} should be 0 after shutdown at {t}, but is {status}."
                )


def test_initial_operational_status(electrolyser_model, electrolyser_config):
    """
    Test that the initial operational status of the electrolyser is correctly set.
    """
    model, _ = electrolyser_model
    initial_status = electrolyser_config["initial_status"]
    first_time_step = model.time_steps.at(1)
    actual_status = pyo.value(model.electrolyser.operational_status[first_time_step])
    assert actual_status == initial_status, (
        f"Initial operational status is {actual_status}, expected {initial_status}."
    )


def test_operating_cost(electrolyser_model, price_profile):
    """
    Test that the operating cost is calculated correctly based on power input and electricity price.
    """
    model, _ = electrolyser_model
    total_calculated_cost = sum(
        pyo.value(model.electrolyser.operating_cost[t]) for t in model.time_steps
    )
    total_model_cost = pyo.value(model.total_cost)
    assert abs(total_calculated_cost - total_model_cost) < 1e-5, (
        f"Calculated total operating cost {total_calculated_cost} does not match model's total cost {total_model_cost}."
    )

    # Additionally, verify individual operating costs
    for t in model.time_steps:
        expected_cost = pyo.value(model.electrolyser.power_in[t]) * price_profile[t]
        actual_cost = pyo.value(model.electrolyser.operating_cost[t])
        assert abs(actual_cost - expected_cost) < 1e-5, (
            f"Operating cost at time {t} is {actual_cost}, expected {expected_cost}."
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
