# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import ElectricVehicle

use_solver = "appsi_highs"


# Fixture for EV configuration (without profiles)
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


# Fixture for creating a model with an EV component using an availability profile
@pytest.fixture
def ev_model_with_availability(ev_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])

    # Create an availability profile as a pandas Series (1 = available, 0 = unavailable)
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=model.time_steps
    )

    # Provide an empty charging profile (since we are testing availability)
    charging_profile = None

    ev = ElectricVehicle(
        **ev_config,
        time_steps=model.time_steps,
        availability_profile=availability_profile,
        charging_profile=charging_profile,
    )
    model.ev = pyo.Block()
    ev.add_to_model(model, model.ev)

    # Objective function (dummy for testing)
    model.total_charge = pyo.Objective(
        expr=sum(model.ev.charge[t] for t in model.time_steps), sense=pyo.maximize
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=True)

    return model, results


# Fixture for creating a model with an EV component using a charging profile
@pytest.fixture
def ev_model_with_charging_profile(ev_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])

    # Provide an availability profile where the EV is always available (1 for all time steps)
    availability_profile = pd.Series([1] * 10, index=model.time_steps)

    # Create a charging profile as a pandas Series (predefined charging schedule)
    charging_profile = pd.Series(
        [0.1, 0.05, 0.15, 0.2, 0.0, 0.1, 0.05, 0.1, 0.0, 0.15], index=model.time_steps
    )

    ev = ElectricVehicle(
        **ev_config,
        time_steps=model.time_steps,
        availability_profile=availability_profile,
        charging_profile=charging_profile,
    )
    model.ev = pyo.Block()
    ev.add_to_model(model, model.ev)

    # Objective function (dummy for testing)
    model.total_charge = pyo.Objective(
        expr=sum(model.ev.charge[t] for t in model.time_steps), sense=pyo.maximize
    )

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=True)

    return model, results


# Test for EV with availability profile
def test_ev_availability_profile(ev_model_with_availability):
    model, results = ev_model_with_availability

    # Check if charging and discharging respect the availability profile
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=model.time_steps
    )

    for t in model.time_steps:
        availability = availability_profile[t]
        charge = pyo.value(model.ev.charge[t])
        discharge = pyo.value(model.ev.discharge[t])

        # When availability is 0, charge and discharge should be zero
        if availability == 0:
            assert charge == 0
            assert discharge == 0
        # When available, check if charge and discharge are within allowable limits
        else:
            assert charge <= model.ev.max_power_charge
            assert discharge <= model.ev.max_power_discharge


# Test for EV with charging profile
def test_ev_charging_profile(ev_model_with_charging_profile, ev_config):
    model, results = ev_model_with_charging_profile

    # Check if charging follows the predefined charging profile
    charging_profile = pd.Series(
        [0.1, 0.05, 0.15, 0.2, 0.0, 0.1, 0.05, 0.1, 0.0, 0.15], index=model.time_steps
    )

    for t in model.time_steps:
        charge = pyo.value(model.ev.charge[t])
        expected_charge = charging_profile[t]

        # Check if the charge exactly matches the charging profile
        assert charge == pytest.approx(expected_charge, rel=1e-2)

    # Check if SOC stays within 0 and 1
    for t in model.time_steps:
        soc = pyo.value(model.ev.soc[t])
        assert ev_config["min_capacity"] <= soc <= ev_config["max_capacity"]
