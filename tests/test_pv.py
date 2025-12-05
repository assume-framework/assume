# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import PVPlant

use_solver = "appsi_highs"


# Fixture for creating an electricity price profile, including negative prices
@pytest.fixture
def price_profile():
    return pd.Series([50, 45, 55, 40, -10, 55, -5, 65, 45, 70], index=range(10))


# Fixture for PV plant configuration (without profiles)
@pytest.fixture
def pv_plant_config():
    return {
        "max_power": 50,
    }


# Fixture for creating a model with a PV plant component, availability profile, and price profile
@pytest.fixture
def pv_plant_model_with_availability_and_price(pv_plant_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])

    # Create an availability profile as a pandas Series
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=model.time_steps
    )

    # Add the price profile to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # PV plant setup
    pv_plant = PVPlant(
        **pv_plant_config,
        time_steps=list(model.time_steps),
        availability_profile=availability_profile,
    )
    model.pv_plant = pyo.Block()
    pv_plant.add_to_model(model, model.pv_plant)

    # Objective: Maximize revenue (electricity price * power output)
    def objective_rule(model):
        return sum(model.pv_plant.operating_cost[t] for t in model.time_steps)

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=True)

    return model, results


# Fixture for creating a model with a PV plant component, power profile, and price profile
@pytest.fixture
def pv_plant_model_with_power_profile_and_price(pv_plant_config, price_profile):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])

    # Create a power profile as a pandas Series
    power_profile = pd.Series(
        [10, 20, 30, 40, 50, 30, 20, 10, 0, 15], index=model.time_steps
    )

    # Add the price profile to the model
    model.electricity_price = pyo.Param(
        model.time_steps, initialize=price_profile.to_dict()
    )

    # PV plant setup
    pv_plant = PVPlant(
        **pv_plant_config,
        time_steps=list(model.time_steps),
        power_profile=power_profile,
    )
    model.pv_plant = pyo.Block()
    pv_plant.add_to_model(model, model.pv_plant)

    # Objective: Maximize revenue (electricity price * power output)
    def objective_rule(model):
        return sum(model.pv_plant.operating_cost[t] for t in model.time_steps)

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # Solve the model
    solver = pyo.SolverFactory(use_solver)
    results = solver.solve(model, tee=True)

    return model, results


# Test for PV plant with availability and price profile
def test_pv_plant_availability_profile_with_price(
    pv_plant_model_with_availability_and_price, price_profile
):
    model, results = pv_plant_model_with_availability_and_price

    # Create an availability profile as a pandas Series
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=model.time_steps
    )

    # Check if power output respects the availability and price profile
    for t in model.time_steps:
        availability = availability_profile[t]
        price = price_profile[t]
        power_output = pyo.value(model.pv_plant.power[t])

        # No production if price is negative or availability is zero
        if price < 0 or availability == 0:
            assert power_output == 0, (
                f"Power output should be zero at time {t} due to negative price or zero availability"
            )

        # Production allowed if price is positive and availability is 1
        else:
            assert power_output <= model.pv_plant.max_power * availability, (
                f"Power output should not exceed max power at time {t}"
            )


# Test for PV plant with power profile, ensuring it follows the profile regardless of price
def test_pv_plant_power_profile_with_price(
    pv_plant_model_with_power_profile_and_price, price_profile
):
    model, results = pv_plant_model_with_power_profile_and_price

    # Create a power profile as a pandas Series
    power_profile = pd.Series(
        [10, 20, 30, 40, 50, 30, 20, 10, 0, 15], index=model.time_steps
    )

    # Check if power output follows the power profile regardless of price
    for t in model.time_steps:
        expected_power_output = power_profile[t]
        power_output = pyo.value(model.pv_plant.power[t])

        # Ensure the power output follows the power profile exactly
        assert power_output == pytest.approx(expected_power_output, rel=1e-2), (
            f"Power output should follow the power profile at time {t} regardless of price"
        )
