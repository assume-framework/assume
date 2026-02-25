# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import ElectricArcFurnace

# Define the solver to use
use_solver = "appsi_highs"  # Replace with the appropriate solver


# Fixture for creating an electricity price profile, including negative prices
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, 1000, 55, 1000, 65, 45, 70], index=range(10))


# Fixture for EAF configuration
@pytest.fixture
def eaf_config():
    return {
        "max_power": 100,
        "min_power": 20,
        "specific_electricity_consumption": 1,
        "specific_dri_demand": 1,
        "specific_lime_demand": 0.05,
        "lime_co2_factor": 0.1,
        "ramp_up": 50,
        "ramp_down": 50,
        "min_operating_steps": 2,
        "min_down_steps": 2,
        "initial_operational_status": 1,
    }


# Fixture for creating and solving the EAF model
@pytest.fixture
def eaf_model(eaf_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    # Add the price profile to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    model.co2_price = pyo.Param(model.time_steps, initialize=30)
    model.lime_price = pyo.Param(model.time_steps, initialize=20)

    # Initialize the Electric Arc Furnace
    eaf = ElectricArcFurnace(**eaf_config, time_steps=model.time_steps)
    model.eaf = pyo.Block()
    eaf.add_to_model(model, model.eaf)

    # Constraint for total steel production over all time steps
    total_steel_production = 500
    model.total_steel_constraint = pyo.Constraint(
        expr=sum(model.eaf.steel_output[t] for t in model.time_steps)
        == total_steel_production
    )

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(model.eaf.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)

    return model, results


def test_model_solves_successfully(eaf_model):
    """
    Test that the optimization model solves successfully.
    """
    model, results = eaf_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution. Status: {results.solver.status}, "
        f"Termination Condition: {results.solver.termination_condition}"
    )


def test_total_steel_production(eaf_model):
    """
    Test that the total steel production constraint is met.
    """
    model, _ = eaf_model
    total_steel = sum(pyo.value(model.eaf.steel_output[t]) for t in model.time_steps)
    assert abs(total_steel - 500) < 1e-5, (
        f"Total steel production is {total_steel}, expected 800."
    )


def test_eaf_ramping_constraints(eaf_model, eaf_config):
    """
    Test that ramp-up and ramp-down constraints are respected.
    """
    model, _ = eaf_model
    ramp_up_limit = eaf_config["ramp_up"]
    ramp_down_limit = eaf_config["ramp_down"]

    for t in model.time_steps:
        if t > model.time_steps.at(1):
            power_current = pyo.value(model.eaf.power_in[t])
            power_prev = pyo.value(model.eaf.power_in[t - 1])
            ramp_up = power_current - power_prev
            ramp_down = power_prev - power_current

            assert ramp_up <= ramp_up_limit + 1e-5, (
                f"Ramp-up at time {t} is {ramp_up}, exceeds limit of {ramp_up_limit}."
            )
            assert ramp_down <= ramp_down_limit + 1e-5, (
                f"Ramp-down at time {t} is {ramp_down}, exceeds limit of {ramp_down_limit}."
            )


def test_eaf_power_bounds(eaf_model, eaf_config):
    """
    Test that power input respects the minimum and maximum power bounds.
    """
    model, _ = eaf_model
    min_power = eaf_config["min_power"]
    max_power = eaf_config["max_power"]

    for t in model.time_steps:
        power_in = pyo.value(model.eaf.power_in[t])
        operational_status = pyo.value(model.eaf.operational_status[t])
        lower_bound = min_power * operational_status
        upper_bound = max_power * operational_status + 1e-5  # Adding tolerance
        assert lower_bound <= power_in <= upper_bound, (
            f"Power input at time {t} is {power_in}, which is outside the bounds "
            f"{lower_bound} - {upper_bound}."
        )


def test_eaf_off_when_high_price(eaf_model, price_profile):
    """
    Test that the EAF is turned off when the electricity price is high.
    """
    model, _ = eaf_model

    for t in model.time_steps:
        price = price_profile[t]
        power_in = pyo.value(model.eaf.power_in[t])
        if price == 1000:
            assert power_in == 0, f"EAF is on at time {t} despite high price {price}."


def test_eaf_min_operating_steps(eaf_model, eaf_config):
    """
    Test that the EAF operates for at least the minimum number of operating steps after startup.
    """
    model, _ = eaf_model
    min_steps = eaf_config["min_operating_steps"]
    operational_status = model.eaf.operational_status

    startup_times = [
        t for t in model.time_steps if pyo.value(model.eaf.start_up[t]) == 1
    ]

    for t in startup_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 1, (
                    f"Operational status at time {future_t} should be 1 after startup at {t}, but is {status}."
                )


def test_eaf_min_downtime_steps(eaf_model, eaf_config):
    """
    Test that the EAF remains off for at least the minimum number of downtime steps after shutdown.
    """
    model, _ = eaf_model
    min_steps = eaf_config["min_down_steps"]
    operational_status = model.eaf.operational_status

    shutdown_times = [
        t for t in model.time_steps if pyo.value(model.eaf.shut_down[t]) == 1
    ]

    for t in shutdown_times:
        for step in range(min_steps):
            future_t = t + step
            if future_t in model.time_steps:
                status = pyo.value(operational_status[future_t])
                assert status == 0, (
                    f"Operational status at time {future_t} should be 0 after shutdown at {t}, but is {status}."
                )


def test_initial_operational_status(eaf_model, eaf_config):
    """
    Test that the initial operational status of the EAF is correctly set.
    """
    model, _ = eaf_model
    initial_status = eaf_config["initial_operational_status"]
    first_time_step = model.time_steps.at(1)
    actual_status = pyo.value(model.eaf.operational_status[first_time_step])
    assert actual_status == initial_status, (
        f"Initial operational status is {actual_status}, expected {initial_status}."
    )


def test_eaf_dri_demand_relation(eaf_model, eaf_config):
    """
    Test that DRI input relates correctly to steel output.
    """
    model, _ = eaf_model
    specific_dri_demand = eaf_config["specific_dri_demand"]

    for t in model.time_steps:
        steel_output = pyo.value(model.eaf.steel_output[t])
        dri_input = pyo.value(model.eaf.dri_input[t])
        expected_dri = steel_output * specific_dri_demand
        assert abs(dri_input - expected_dri) < 1e-5, (
            f"DRI input at time {t} is {dri_input}, expected {expected_dri} based on steel output."
        )


def test_eaf_lime_demand_relation(eaf_model, eaf_config):
    """
    Test that lime demand relates correctly to steel output.
    """
    model, _ = eaf_model
    specific_lime_demand = eaf_config["specific_lime_demand"]

    for t in model.time_steps:
        steel_output = pyo.value(model.eaf.steel_output[t])
        lime_demand = pyo.value(model.eaf.lime_demand[t])
        expected_lime = steel_output * specific_lime_demand
        assert abs(lime_demand - expected_lime) < 1e-5, (
            f"Lime demand at time {t} is {lime_demand}, expected {expected_lime} based on steel output."
        )


def test_eaf_co2_emission_relation(eaf_model):
    """
    Test that CO2 emissions are calculated correctly based on lime demand.
    """
    model, _ = eaf_model

    for t in model.time_steps:
        lime_demand = pyo.value(model.eaf.lime_demand[t])
        co2_emission = pyo.value(model.eaf.co2_emission[t])
        expected_co2 = lime_demand * model.eaf.lime_co2_factor
        assert abs(co2_emission - expected_co2) < 1e-5, (
            f"CO2 emission at time {t} is {co2_emission}, expected {expected_co2} based on lime demand."
        )


def test_eaf_operating_cost(eaf_model, price_profile):
    """
    Test that the operating cost is calculated correctly based on power input, CO2 emissions, and lime consumption.
    """
    model, _ = eaf_model
    co2_price = model.co2_price
    lime_price = model.lime_price

    total_calculated_cost = sum(
        pyo.value(model.eaf.operating_cost[t]) for t in model.time_steps
    )
    total_model_cost = pyo.value(model.total_cost)
    assert abs(total_calculated_cost - total_model_cost) < 1e-5, (
        f"Calculated total operating cost {total_calculated_cost} does not match model's total cost {total_model_cost}."
    )

    # Additionally, verify individual operating costs
    for t in model.time_steps:
        power_in = pyo.value(model.eaf.power_in[t])
        co2_emission = pyo.value(model.eaf.co2_emission[t])
        lime_demand = pyo.value(model.eaf.lime_demand[t])
        expected_cost = (
            power_in * price_profile[t]
            + co2_emission * co2_price[t]
            + lime_demand * lime_price[t]
        )
        actual_cost = pyo.value(model.eaf.operating_cost[t])
        assert abs(actual_cost - expected_cost) < 1e-5, (
            f"Operating cost at time {t} is {actual_cost}, expected {expected_cost}."
        )


def test_eaf_steel_output_relation(eaf_model):
    """
    Test that steel output is correctly related to power input and specific electricity consumption.
    """
    model, _ = eaf_model
    specific_electricity_consumption = model.eaf.specific_electricity_consumption

    for t in model.time_steps:
        power_in = pyo.value(model.eaf.power_in[t])
        steel_output = pyo.value(model.eaf.steel_output[t])
        expected_steel_output = power_in / specific_electricity_consumption
        assert abs(steel_output - expected_steel_output) < 1e-5, (
            f"Steel output at time {t} is {steel_output}, expected {expected_steel_output} based on power input."
        )
