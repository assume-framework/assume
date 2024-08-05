# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pyomo.environ as pyo
import pytest

from assume.units.dst_components import (
    create_driplant,
    create_dristorage,
    create_electric_arc_furnance,
    create_electrolyser,
    create_hydrogen_storage,
)


@pytest.fixture
def electrolyser_config():
    return {
        "rated_power": 50,
        "min_power": 0,
        "ramp_up": 30,
        "ramp_down": 30,
        "min_operating_time": 1,
        "min_down_time": 1,
        "efficiency": 1,
    }


@pytest.fixture
def electrolyser_model(electrolyser_config):
    model = pyo.ConcreteModel()
    time_steps = range(10)
    model.time_steps = pyo.Set(initialize=time_steps)
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)
    model_part = create_electrolyser(
        model, time_steps=model.time_steps, **electrolyser_config
    )
    model.electrolyser = model_part
    # Objective function to minimize operating cost (Just for testing purpose)
    model.total_cost = pyo.Objective(
        expr=sum(
            model.electrolyser.electrolyser_operating_cost[t] for t in model.time_steps
        ),
        sense=pyo.minimize,
    )
    # Constraint for total hydrogen production over all time steps (Just for testing purpose)
    total_hydrogen_production = 300
    model.total_hydrogen_constraint = pyo.Constraint(
        expr=sum(model.electrolyser.hydrogen_out[t] for t in model.time_steps)
        == total_hydrogen_production
    )

    # Solve the model once in the fixture
    instance = model.create_instance()
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(instance, tee=False)

    return instance, results


def test_electrolyser_ramping_and_power_bounds(electrolyser_model):
    instance, results = electrolyser_model

    # Check ramp-up constraints
    for t in range(1, len(instance.time_steps)):
        ramp_up_diff = pyo.value(
            instance.electrolyser.power_in[t] - instance.electrolyser.power_in[t - 1]
        )
        assert ramp_up_diff <= instance.electrolyser.ramp_up

    # Check ramp-down constraints
    for t in range(1, len(instance.time_steps)):
        ramp_down_diff = pyo.value(
            instance.electrolyser.power_in[t - 1] - instance.electrolyser.power_in[t]
        )
        assert ramp_down_diff <= instance.electrolyser.ramp_down

    # Check power bounds
    for t in instance.time_steps:
        power_in = pyo.value(instance.electrolyser.power_in[t])
        assert (
            instance.electrolyser.min_power
            <= power_in
            <= instance.electrolyser.rated_power
        )

    # Equality checks for specific values
    assert pyo.value(instance.electrolyser.power_in[2]) == 50  # Expected power at t=2
    assert pyo.value(instance.electrolyser.power_in[5]) == 0  # Expected power at t=5


# Test for DRI Plant
@pytest.fixture
def driplant_config():
    return {
        "specific_hydrogen_consumption": 1,
        "specific_natural_gas_consumption": 1,
        "specific_electricity_consumption": 1,
        "specific_iron_ore_consumption": 1,
        "rated_power": 50,
        "min_power": 0,
        "fuel_type": "hydrogen",
        "ramp_up": 50,
        "ramp_down": 50,
        "min_operating_time": 0,
        "min_down_time": 0,
    }


# Fixture for DRI Plant model
@pytest.fixture
def driplant_model(driplant_config):
    model = pyo.ConcreteModel()
    time_steps = range(10)
    model.time_steps = pyo.Set(initialize=time_steps)
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)
    model.natural_gas_price = pyo.Param(model.time_steps, initialize=3, mutable=True)
    model.iron_ore_price = 4
    model_part = create_driplant(model, time_steps=model.time_steps, **driplant_config)
    model.driplant = model_part
    # Objective function to minimize operating cost (Just for testing purpose)
    model.total_cost = pyo.Objective(
        expr=sum(model.driplant.dri_operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )
    # Constraint for total DRI production over all time steps (Just for testing purpose)
    total_dri_production = 300
    model.total_dri_constraint = pyo.Constraint(
        expr=sum(model.driplant.dri_output[t] for t in model.time_steps)
        == total_dri_production
    )

    # Solve the model once in the fixture
    instance = model.create_instance()
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(instance, tee=True)

    return instance, results


def test_driplant_ramping_and_power_bounds(driplant_model):
    instance, results = driplant_model

    # Check ramp-up constraints
    for t in range(1, len(instance.time_steps)):
        ramp_up_diff = pyo.value(
            instance.driplant.power_dri[t] - instance.driplant.power_dri[t - 1]
        )
        assert ramp_up_diff <= instance.driplant.ramp_up_dri

    # Check ramp-down constraints
    for t in range(1, len(instance.time_steps)):
        ramp_down_diff = pyo.value(
            instance.driplant.power_dri[t - 1] - instance.driplant.power_dri[t]
        )
        assert ramp_down_diff <= instance.driplant.ramp_down_dri

    # Check power bounds
    for t in instance.time_steps:
        power_dri = pyo.value(instance.driplant.power_dri[t])
        assert (
            instance.driplant.min_power_dri
            <= power_dri
            <= instance.driplant.rated_power_dri
        )

    # Equality checks for specific values
    assert pyo.value(instance.driplant.power_dri[1]) == 50  # Expected power at t=1
    assert pyo.value(instance.driplant.power_dri[5]) == 50  # Expected power at t=5


# Test for Electric Arc Furnace
@pytest.fixture
def eaf_config():
    return {
        "rated_power": 100,
        "min_power": 20,
        "specific_electricity_consumption": 1,
        "specific_dri_demand": 1,
        "specific_lime_demand": 0.05,
        "ramp_up": 50,
        "ramp_down": 50,
        "min_operating_time": 1,
        "min_down_time": 1,
    }


@pytest.fixture
def eaf_model(eaf_config):
    model = pyo.ConcreteModel()
    time_steps = range(10)
    model.time_steps = pyo.Set(initialize=time_steps)
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)
    model.lime_co2_factor = 0.03
    model.co2_price = 30
    model.lime_price = 20
    model_part = create_electric_arc_furnance(
        model, time_steps=model.time_steps, **eaf_config
    )
    model.eaf = model_part
    # Objective function to minimize operating cost (Just for testing purposes)
    model.total_cost = pyo.Objective(
        expr=sum(model.eaf.eaf_operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )
    # Constraint for total steel production over all time steps (Just for testing purposes)
    total_steel_production = 800
    model.total_steel_constraint = pyo.Constraint(
        expr=sum(model.eaf.steel_output[t] for t in model.time_steps)
        == total_steel_production
    )

    # Solve the model once in the fixture
    instance = model.create_instance()
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(instance, tee=True)

    return instance, results


def test_eaf_ramping_and_power_bounds(eaf_model):
    instance, results = eaf_model

    # Check ramp-up constraints
    for t in range(1, len(instance.time_steps)):
        ramp_up_diff = pyo.value(
            instance.eaf.power_eaf[t] - instance.eaf.power_eaf[t - 1]
        )
        assert ramp_up_diff <= instance.eaf.ramp_up_eaf

    # Check ramp-down constraints
    for t in range(1, len(instance.time_steps)):
        ramp_down_diff = pyo.value(
            instance.eaf.power_eaf[t - 1] - instance.eaf.power_eaf[t]
        )
        assert ramp_down_diff <= instance.eaf.ramp_down_eaf

    # Check power bounds
    for t in instance.time_steps:
        power_eaf = pyo.value(instance.eaf.power_eaf[t])
        assert instance.eaf.min_power_eaf <= power_eaf <= instance.eaf.rated_power_eaf

    # Equality checks for specific values
    assert pyo.value(instance.eaf.power_eaf[5]) == 40  # Expected power at t=5


# Test for hydrogen storage
@pytest.fixture
def h2storage_config():
    return {
        "max_capacity": 1000,  # Maximum storage capacity in kg
        "min_capacity": 0,  # Maximum storage capacity in kg
        "initial_soc": 500,  # Initial state of charge in kg
        "storage_loss_rate": 0,  # Maximum charging rate in kg/h
        "charge_loss_rate": 0,  # Maximum discharging rate in kg/h
        "discharge_loss_rate": 0,  # Charging efficiency
    }


@pytest.fixture
def h2storage_model(h2storage_config):
    model = pyo.ConcreteModel()
    time_steps = range(10)
    model.time_steps = pyo.Set(initialize=time_steps)
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)
    model_part = create_hydrogen_storage(
        model, time_steps=model.time_steps, **h2storage_config
    )
    model.hydrogen_storage = model_part
    # Objective function to minimize operating cost (Just for testing purposes)
    model.total_cost = pyo.Objective(
        expr=sum(
            model.hydrogen_storage.charge[t] * model.electricity_price[t]
            - model.hydrogen_storage.discharge[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )
    # Constraint for total hydrogen throughput over all time steps (Just for testing purposes)
    total_throughput = 500  # Arbitrary value for testing
    model.total_throughput_constraint = pyo.Constraint(
        expr=sum(
            model.hydrogen_storage.charge[t] - model.hydrogen_storage.discharge[t]
            for t in model.time_steps
        )
        == total_throughput
    )

    # Solve the model once in the fixture
    instance = model.create_instance()
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(instance, tee=True)

    return instance, results


def test_storage_charge_discharge_soc(h2storage_model, h2storage_config):
    instance, results = h2storage_model

    # Check charge, discharge, and SOC constraints
    for t in instance.time_steps:
        charge = pyo.value(instance.hydrogen_storage.charge[t])
        discharge = pyo.value(instance.hydrogen_storage.discharge[t])
        soc = pyo.value(instance.hydrogen_storage.soc[t])

        # Ensure SOC remains within bounds
        assert (
            h2storage_config["min_capacity"] <= soc <= h2storage_config["max_capacity"]
        )

        # Ensure charge rate does not exceed maximum
        assert charge <= h2storage_config["max_capacity"]

        # Ensure discharge rate does not exceed maximum
        assert discharge <= h2storage_config["max_capacity"]

    # Equality checks for specific values
    # Replace with actual expected values
    assert pyo.value(instance.hydrogen_storage.soc[5]) == 1000  # Example SOC at t=5


# Test for Dri storage
@pytest.fixture
def dristorage_config():
    return {
        "max_capacity": 1000,  # Maximum storage capacity in tons
        "min_capacity": 100,  # Minimum storage capacity in tons
        "initial_soc": 500,  # Initial state of charge in tons
        "storage_loss_rate": 0,  # Storage loss rate per time step
        "charge_loss_rate": 0,  # Charge loss rate per time step
        "discharge_loss_rate": 0,  # Discharge loss rate per time step
    }


@pytest.fixture
def dristorage_model(dristorage_config):
    model = pyo.ConcreteModel()
    time_steps = range(10)
    model.time_steps = pyo.Set(initialize=time_steps)
    model.iron_ore_price = pyo.Param(model.time_steps, initialize=1, mutable=True)
    model_part = create_dristorage(
        model, time_steps=model.time_steps, **dristorage_config
    )
    model.dristorage = model_part
    # Objective function to minimize operating cost (Just for testing purposes)
    model.total_cost = pyo.Objective(
        expr=sum(
            model.dristorage.charge_dri[t] * model.iron_ore_price[t]
            - model.dristorage.discharge_dri[t] * model.iron_ore_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )
    # Constraint for total iron throughput over all time steps (Just for testing purposes)
    total_throughput = 300  # Arbitrary value for testing
    model.total_throughput_constraint = pyo.Constraint(
        expr=sum(
            model.dristorage.charge_dri[t] - model.dristorage.discharge_dri[t]
            for t in model.time_steps
        )
        == total_throughput
    )

    # Solve the model once in the fixture
    instance = model.create_instance()
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(instance, tee=True)

    return instance, results


def test_dristorage_charge_discharge_soc(dristorage_model, dristorage_config):
    instance, results = dristorage_model

    # Check charge, discharge, and SOC constraints
    for t in instance.time_steps:
        charge = pyo.value(instance.dristorage.charge_dri[t])
        discharge = pyo.value(instance.dristorage.discharge_dri[t])
        soc = pyo.value(instance.dristorage.soc_dri[t])

        # Ensure SOC remains within bounds
        assert (
            dristorage_config["min_capacity"]
            <= soc
            <= dristorage_config["max_capacity"]
        )

        # Ensure charge rate does not exceed maximum
        assert charge <= dristorage_config["max_capacity"]

        # Ensure discharge rate does not exceed maximum
        assert discharge <= dristorage_config["max_capacity"]

    # Equality checks for specific values
    assert pyo.value(instance.dristorage.soc_dri[5]) == 500  # Example SOC at t=5


if __name__ == "__main__":
    pytest.main(["-s", __file__])
