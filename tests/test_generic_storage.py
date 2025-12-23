# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import GenericStorage

# Define the solver to use
use_solver = "appsi_highs"  # Replace with the appropriate solver


# Fixture for creating an electricity price profile, including negative prices
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, -10, 55, -5, 65, 45, 70], index=range(10))


# Fixture for generic storage configuration
@pytest.fixture
def generic_storage_config():
    return {
        "capacity": 100,  # energy capacity in MWh
        "min_soc": 0,  # Minimum SOC
        "max_soc": 1,  # Maximum SOC
        "max_power_charge": 0,  # Maximum charging power in MW
        "max_power_discharge": 0,  # Maximum discharging power in MW
        "efficiency_charge": 0.9,  # Charging efficiency
        "efficiency_discharge": 0.9,  # Discharging efficiency
        "initial_soc": 0,  # Initial SOC
        "ramp_up": 50,  # Maximum ramp-up rate in MW
        "ramp_down": 50,  # Maximum ramp-down rate in MW
        "storage_loss_rate": 0.01,  # 1% storage loss per time step
    }


# Fixture for creating and solving the generic storage model
@pytest.fixture
def generic_storage_model(generic_storage_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(10))

    # Add the price profile to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # Initialize the GenericStorage
    storage = GenericStorage(**generic_storage_config, time_steps=model.time_steps)
    model.storage = pyo.Block()
    storage.add_to_model(model, model.storage)

    # Define the objective function
    # For demonstration, let's assume the objective is to minimize the cost of purchasing electricity,
    # considering charging (buying) and discharging (selling) based on prices.
    # Charging incurs a cost, discharging can generate revenue.
    # Total Cost = sum(charge * price - discharge * price)
    model.total_cost = pyo.Objective(
        expr=sum(
            model.storage.charge[t] * model.electricity_price[t]
            - model.storage.discharge[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )

    # SOC balance constraints are already included in the GenericStorage class

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=False)

    return model, results


def test_model_solves_successfully(generic_storage_model):
    """
    Test that the optimization model solves successfully.
    """
    model, results = generic_storage_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), (
        f"Solver did not find an optimal solution. Status: {results.solver.status}, "
        f"Termination Condition: {results.solver.termination_condition}"
    )


def test_state_of_charge_constraints(generic_storage_model, generic_storage_config):
    """
    Test that the state of charge (SOC) remains within the specified bounds
    and evolves correctly based on charging, discharging, and storage losses.
    """
    model, _ = generic_storage_model
    min_soc = generic_storage_config["min_soc"]
    max_soc = generic_storage_config["max_soc"]
    efficiency_charge = generic_storage_config["efficiency_charge"]
    efficiency_discharge = generic_storage_config["efficiency_discharge"]
    storage_loss_rate = generic_storage_config["storage_loss_rate"]
    initial_soc = generic_storage_config["initial_soc"]

    time_steps = sorted(model.time_steps)
    previous_soc = initial_soc

    for i, t in enumerate(time_steps):
        current_soc = pyo.value(model.storage.soc[t])

        # Check SOC bounds
        assert min_soc <= current_soc <= max_soc, (
            f"SOC at time {t} is {current_soc}, which is outside the bounds "
            f"{min_soc} - {max_soc}."
        )

        if i == 0:
            # Initial SOC at first time step
            assert abs(current_soc - previous_soc) < 1e-4, (
                f"Initial SOC at time {t} is {current_soc}, expected {previous_soc}."
            )
        else:
            prev_t = time_steps[i - 1]
            prev_charge = pyo.value(model.storage.charge[prev_t])
            prev_discharge = pyo.value(model.storage.discharge[prev_t])
            expected_soc = (
                previous_soc
                + (efficiency_charge * prev_charge)
                - (prev_discharge / efficiency_discharge)
                - (storage_loss_rate * previous_soc)
            )
            assert abs(current_soc - expected_soc) < 1e-4, (
                f"SOC at time {t} is {current_soc}, but expected {expected_soc} based on charging, discharging, and losses."
            )

        previous_soc = current_soc


def test_power_bounds(generic_storage_model, generic_storage_config):
    """
    Test that charging and discharging powers respect their respective maximum limits.
    """
    model, _ = generic_storage_model
    max_charge = generic_storage_config["max_power_charge"]
    max_discharge = generic_storage_config["max_power_discharge"]

    for t in model.time_steps:
        charge = pyo.value(model.storage.charge[t])
        discharge = pyo.value(model.storage.discharge[t])

        assert 0 <= charge <= max_charge + 1e-5, (
            f"Charging power at time {t} is {charge}, which exceeds the maximum limit of {max_charge}."
        )
        assert 0 <= discharge <= max_discharge + 1e-5, (
            f"Discharging power at time {t} is {discharge}, which exceeds the maximum limit of {max_discharge}."
        )


def test_ramping_constraints(generic_storage_model, generic_storage_config):
    """
    Test that ramp-up and ramp-down constraints for charging and discharging are respected.
    """
    model, _ = generic_storage_model
    ramp_up = generic_storage_config["ramp_up"]
    ramp_down = generic_storage_config["ramp_down"]

    for var in ["charge", "discharge"]:
        previous_power = 0
        for t in model.time_steps:
            current_power = pyo.value(getattr(model.storage, var)[t])
            if t == model.time_steps.at(1):
                # Assuming initial power is zero
                expected_ramp_up = current_power <= ramp_up + 1e-5
                expected_ramp_down = (
                    previous_power <= ramp_down + 1e-5
                )  # Since previous_power is zero
            else:
                ramp_up_diff = current_power - pyo.value(
                    getattr(model.storage, var)[t - 1]
                )
                ramp_down_diff = (
                    pyo.value(getattr(model.storage, var)[t - 1]) - current_power
                )
                expected_ramp_up = ramp_up_diff <= ramp_up + 1e-5
                expected_ramp_down = ramp_down_diff <= ramp_down + 1e-5
            assert expected_ramp_up, (
                f"Ramp-up for {var} at time {t} is {current_power - previous_power}, "
                f"which exceeds the limit of {ramp_up}."
            )
            assert expected_ramp_down, (
                f"Ramp-down for {var} at time {t} is {pyo.value(getattr(model.storage, var)[t - 1]) - current_power}, "
                f"which exceeds the limit of {ramp_down}."
            )
            previous_power = current_power


def test_storage_loss_rate(generic_storage_model, generic_storage_config):
    """
    Test that storage losses are correctly applied to the state of charge.
    """
    model, _ = generic_storage_model
    storage_loss_rate = generic_storage_config["storage_loss_rate"]
    initial_soc = generic_storage_config["initial_soc"]
    efficiency_charge = generic_storage_config["efficiency_charge"]
    efficiency_discharge = generic_storage_config["efficiency_discharge"]

    time_steps = sorted(model.time_steps)
    previous_soc = initial_soc

    for i, t in enumerate(time_steps):
        actual_soc = pyo.value(model.storage.soc[t])
        if i == 0:
            # Initial SOC at first time step
            assert abs(actual_soc - previous_soc) < 1e-4, (
                f"Initial SOC at time {t} is {actual_soc}, expected {previous_soc}."
            )
        else:
            prev_t = time_steps[i - 1]
            prev_charge = pyo.value(model.storage.charge[prev_t])
            prev_discharge = pyo.value(model.storage.discharge[prev_t])
            expected_soc = (
                previous_soc
                + (efficiency_charge * prev_charge)
                - (prev_discharge / efficiency_discharge)
                - (storage_loss_rate * previous_soc)
            )
            assert abs(actual_soc - expected_soc) < 1e-4, (
                f"SOC at time {t} is {actual_soc}, but expected {expected_soc} based on storage losses."
            )
        previous_soc = actual_soc


def test_min_soc_enforcement(generic_storage_model, generic_storage_config):
    """
    Test that the state of charge does not go below the minimum soc.
    """
    model, _ = generic_storage_model
    min_soc = generic_storage_config["min_soc"]

    for t in model.time_steps:
        soc = pyo.value(model.storage.soc[t])
        assert soc >= min_soc - 1e-5, (
            f"SOC at time {t} is {soc}, which is below the minimum soc of {min_soc}."
        )


def test_max_soc_enforcement(generic_storage_model, generic_storage_config):
    """
    Test that the state of charge does not exceed the maximum soc.
    """
    model, _ = generic_storage_model
    max_soc = generic_storage_config["max_soc"]

    for t in model.time_steps:
        soc = pyo.value(model.storage.soc[t])
        assert soc <= max_soc + 1e-5, (
            f"SOC at time {t} is {soc}, which exceeds the maximum soc of {max_soc}."
        )


if __name__ == "__main__":
    pytest.main(["-s", __file__])
