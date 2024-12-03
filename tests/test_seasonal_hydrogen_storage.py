# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import SeasonalHydrogenStorage

# Define solver name to be used for testing
USE_SOLVER = "appsi_highs"  # Update to your specific solver if necessary


# Fixture to create a price profile over a 100-step horizon
@pytest.fixture
def price_profile():
    # A price profile with varying values to simulate different cost conditions
    return pd.Series([50, 45, 55, 40, 60, 70, 20, 65, 50, 80] * 10, index=range(100))


# Fixture to define seasonal storage configuration for tests
@pytest.fixture
def seasonal_storage_config():
    return {
        "max_capacity": 500,  # Maximum energy capacity in MWh
        "min_capacity": 50,  # Minimum energy capacity in MWh
        "max_power_charge": 30,  # Maximum charging power in MW
        "max_power_discharge": 30,  # Maximum discharging power in MW
        "efficiency_charge": 0.9,  # Charging efficiency
        "efficiency_discharge": 0.9,  # Discharging efficiency
        "initial_soc": 0.5,  # Initial SOC as a fraction of max capacity
        "final_soc_target": 0.6,  # SOC target as a fraction of max capacity at the end of the horizon
        "ramp_up": 10,  # Maximum increase in charge/discharge power per step
        "ramp_down": 10,  # Maximum decrease in charge/discharge power per step
        "storage_loss_rate": 0.01,  # 1% storage loss rate per step
        # Multiple seasonal periods specified as comma-separated ranges
        "off_season_start": "0,50",  # Off-season start steps
        "off_season_end": "10,60",  # Off-season end steps
        "on_season_start": "11,61",  # On-season start steps
        "on_season_end": "49,99",  # On-season end steps
    }


# Fixture to create and solve a model using SeasonalHydrogenStorage
@pytest.fixture
def seasonal_storage_model(seasonal_storage_config, price_profile):
    # Define the Pyomo model
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(100))  # 100 time steps

    # Add electricity price parameter to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # Initialize the SeasonalHydrogenStorage unit
    storage = SeasonalHydrogenStorage(
        **seasonal_storage_config, time_steps=model.time_steps, horizon=100
    )
    model.storage = pyo.Block()
    storage.add_to_model(model, model.storage)

    # Define an objective to minimize the cost of charging and maximize the benefit from discharging
    model.total_cost = pyo.Objective(
        expr=sum(
            model.storage.charge[t] * model.electricity_price[t]
            - model.storage.discharge[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )

    # Solve the model
    solver = pyo.SolverFactory(USE_SOLVER)
    results = solver.solve(model, tee=False)

    return model, results


# Test to check if the model solves successfully
def test_model_solves_successfully(seasonal_storage_model):
    model, results = seasonal_storage_model
    assert (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ), "Solver did not find an optimal solution."


# Test to verify that SOC stays within the defined bounds
def test_state_of_charge_constraints(seasonal_storage_model, seasonal_storage_config):
    model, _ = seasonal_storage_model
    min_soc = seasonal_storage_config["min_capacity"]
    max_soc = seasonal_storage_config["max_capacity"]

    for t in model.time_steps:
        soc = pyo.value(model.storage.soc[t])
        assert (
            min_soc <= soc <= max_soc
        ), f"SOC at time {t} is {soc}, outside bounds {min_soc} - {max_soc}."


# Test to verify that charging and discharging powers do not exceed defined limits
def test_power_bounds(seasonal_storage_model, seasonal_storage_config):
    model, _ = seasonal_storage_model
    max_charge = seasonal_storage_config["max_power_charge"]
    max_discharge = seasonal_storage_config["max_power_discharge"]

    for t in model.time_steps:
        charge = pyo.value(model.storage.charge[t])
        discharge = pyo.value(model.storage.discharge[t])
        assert (
            0 <= charge <= max_charge
        ), f"Charge at time {t} exceeds max charge {max_charge}."
        assert (
            0 <= discharge <= max_discharge
        ), f"Discharge at time {t} exceeds max discharge {max_discharge}."


# Test to ensure no discharging occurs during off-season
def test_off_season_no_discharge(seasonal_storage_model):
    model, _ = seasonal_storage_model
    off_season = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
    ]

    for t in off_season:
        discharge = pyo.value(model.storage.discharge[t])
        assert discharge == 0, f"Discharge at time {t} during off-season is not zero."


# Test to ensure no charging occurs during on-season
def test_on_season_no_charge(seasonal_storage_model):
    model, _ = seasonal_storage_model
    on_season = list(range(11, 50)) + list(range(61, 99))

    for t in on_season:
        charge = pyo.value(model.storage.charge[t])
        assert charge == 0, f"Charge at time {t} during on-season is not zero."


# Test to ensure the SOC at the end of the horizon meets the final SOC target
def test_final_soc_target(seasonal_storage_model, seasonal_storage_config):
    model, _ = seasonal_storage_model
    final_soc_target = seasonal_storage_config["final_soc_target"]
    max_capacity = seasonal_storage_config["max_capacity"]
    final_soc = pyo.value(model.storage.soc[99])

    expected_soc = final_soc_target * max_capacity
    assert (
        final_soc >= expected_soc
    ), f"Final SOC is {final_soc}, below target {expected_soc}."


# Test to verify that the storage loss rate affects the SOC over time
def test_storage_loss_rate(seasonal_storage_model, seasonal_storage_config):
    model, _ = seasonal_storage_model
    storage_loss_rate = seasonal_storage_config["storage_loss_rate"]

    # Check that SOC reflects the loss rate over time
    previous_soc = (
        seasonal_storage_config["initial_soc"] * seasonal_storage_config["max_capacity"]
    )
    for t in model.time_steps:
        soc = pyo.value(model.storage.soc[t])
        charge = pyo.value(model.storage.charge[t])
        discharge = pyo.value(model.storage.discharge[t])

        expected_soc = (
            previous_soc
            + seasonal_storage_config["efficiency_charge"] * charge
            - discharge / seasonal_storage_config["efficiency_discharge"]
            - storage_loss_rate * previous_soc
        )
        assert (
            abs(soc - expected_soc) < 1e-4
        ), f"SOC at time {t} deviates from expected due to storage loss."
        previous_soc = soc
