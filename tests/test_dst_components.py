# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import (
    Boiler,
    DRIPlant,
    DRIStorage,
    ElectricArcFurnace,
    ElectricVehicle,
    Electrolyser,
    HeatPump,
    HydrogenStorage,
    PVPlant,
)


@pytest.fixture
def electrolyser_config():
    return {
        "max_power": 50,
        "min_power": 0,
        "ramp_up": 30,
        "ramp_down": 30,
        "min_operating_steps": 2,
        "min_down_steps": 2,
        "efficiency": 1,
    }


@pytest.fixture
def electrolyser_model(electrolyser_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)

    electrolyser = Electrolyser(**electrolyser_config, time_steps=model.time_steps)
    model.electrolyser = pyo.Block()
    electrolyser.add_to_model(model, model.electrolyser)

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(
            model.electrolyser.electrolyser_operating_cost[t] for t in model.time_steps
        ),
        sense=pyo.minimize,
    )
    # Constraint for total hydrogen production over all time steps
    total_hydrogen_production = 300
    model.total_hydrogen_constraint = pyo.Constraint(
        expr=sum(model.electrolyser.hydrogen_out[t] for t in model.time_steps)
        == total_hydrogen_production
    )

    # Solve the model once in the fixture
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=False)

    return model, results


def test_electrolyser_ramping_and_power_bounds(electrolyser_model):
    model, results = electrolyser_model

    # Check ramp-up constraints
    for t in range(1, len(model.time_steps)):
        ramp_up_diff = pyo.value(
            model.electrolyser.power_in[t] - model.electrolyser.power_in[t - 1]
        )
        assert ramp_up_diff <= model.electrolyser.ramp_up

    # Check ramp-down constraints
    for t in range(1, len(model.time_steps)):
        ramp_down_diff = pyo.value(
            model.electrolyser.power_in[t - 1] - model.electrolyser.power_in[t]
        )
        assert ramp_down_diff <= model.electrolyser.ramp_down

    # Check power bounds
    for t in model.time_steps:
        power_in = pyo.value(model.electrolyser.power_in[t])
        assert model.electrolyser.min_power <= power_in <= model.electrolyser.max_power

    # Equality checks for specific values
    assert pyo.value(model.electrolyser.power_in[5]) == 40  # Expected power at t=6
    assert pyo.value(model.electrolyser.power_in[6]) == 10  # Expected power at t=7
    assert pyo.value(model.electrolyser.power_in[7]) == 0  # Expected power at t=8


# Test for DRI Plant
@pytest.fixture
def dri_plant_config():
    return {
        "specific_hydrogen_consumption": 1,
        "specific_natural_gas_consumption": 1,
        "specific_electricity_consumption": 1,
        "specific_iron_ore_consumption": 1,
        "max_power": 50,
        "min_power": 0,
        "fuel_type": "hydrogen",
        "ramp_up": 50,
        "ramp_down": 50,
        "min_operating_steps": 0,
        "min_down_steps": 0,
    }


@pytest.fixture
def dri_plant_model(dri_plant_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)
    model.natural_gas_price = pyo.Param(model.time_steps, initialize=3, mutable=True)
    model.iron_ore_price = 4

    dri_plant = DRIPlant(**dri_plant_config, time_steps=model.time_steps)
    model.dri_plant = pyo.Block()
    dri_plant.add_to_model(model, model.dri_plant)

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(model.dri_plant.dri_operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )
    # Constraint for total DRI production over all time steps
    total_dri_production = 300
    model.total_dri_constraint = pyo.Constraint(
        expr=sum(model.dri_plant.dri_output[t] for t in model.time_steps)
        == total_dri_production
    )

    # Solve the model once in the fixture
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


def test_dri_plant_ramping_and_power_bounds(dri_plant_model):
    model, results = dri_plant_model

    # Check ramp-up constraints
    for t in range(1, len(model.time_steps)):
        ramp_up_diff = pyo.value(
            model.dri_plant.power_dri[t] - model.dri_plant.power_dri[t - 1]
        )
        assert ramp_up_diff <= model.dri_plant.ramp_up

    # Check ramp-down constraints
    for t in range(1, len(model.time_steps)):
        ramp_down_diff = pyo.value(
            model.dri_plant.power_dri[t - 1] - model.dri_plant.power_dri[t]
        )
        assert ramp_down_diff <= model.dri_plant.ramp_down

    # Check power bounds
    for t in model.time_steps:
        power_dri = pyo.value(model.dri_plant.power_dri[t])
        assert model.dri_plant.min_power <= power_dri <= model.dri_plant.max_power


# Test for Electric Arc Furnace
@pytest.fixture
def eaf_config():
    return {
        "max_power": 100,
        "min_power": 20,
        "specific_electricity_consumption": 1,
        "specific_dri_demand": 1,
        "specific_lime_demand": 0.05,
        "ramp_up": 50,
        "ramp_down": 50,
        "min_operating_steps": 1,
        "min_down_steps": 1,
    }


@pytest.fixture
def eaf_model(eaf_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)
    model.lime_co2_factor = 0.03
    model.co2_price = 30
    model.lime_price = 20

    eaf = ElectricArcFurnace(**eaf_config, time_steps=model.time_steps)
    model.eaf = pyo.Block()
    eaf.add_to_model(model, model.eaf)

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(model.eaf.eaf_operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )
    # Constraint for total steel production over all time steps
    total_steel_production = 800
    model.total_steel_constraint = pyo.Constraint(
        expr=sum(model.eaf.steel_output[t] for t in model.time_steps)
        == total_steel_production
    )

    # Solve the model once in the fixture
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


def test_eaf_ramping_and_power_bounds(eaf_model):
    model, results = eaf_model

    # Check ramp-up constraints
    for t in range(1, len(model.time_steps)):
        ramp_up_diff = pyo.value(model.eaf.power_eaf[t] - model.eaf.power_eaf[t - 1])
        assert ramp_up_diff <= model.eaf.ramp_up

    # Check ramp-down constraints
    for t in range(1, len(model.time_steps)):
        ramp_down_diff = pyo.value(model.eaf.power_eaf[t - 1] - model.eaf.power_eaf[t])
        assert ramp_down_diff <= model.eaf.ramp_down

    min_power = pyo.value(model.eaf.min_power)
    max_power = pyo.value(model.eaf.max_power)
    # Check power bounds
    for t in model.time_steps:
        power_eaf = pyo.value(model.eaf.power_eaf[t])
        status = pyo.value(model.eaf.operational_status[t])
        assert min_power * status <= power_eaf * status <= max_power * status


# Test for hydrogen storage
@pytest.fixture
def h2storage_config():
    return {
        "max_capacity": 1000,
        "min_capacity": 0,
        "initial_soc": 0.5,
        "efficiency_charge": 0.9,
        "efficiency_discharge": 0.95,
        "storage_loss_rate": 0,
    }


@pytest.fixture
def h2storage_model(h2storage_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)

    h2_storage = HydrogenStorage(**h2storage_config, time_steps=model.time_steps)
    model.hydrogen_storage = pyo.Block()
    h2_storage.add_to_model(model, model.hydrogen_storage)

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(
            model.hydrogen_storage.charge[t] * model.electricity_price[t]
            - model.hydrogen_storage.discharge[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )
    # Constraint for total hydrogen throughput over all time steps
    total_throughput = 500
    model.total_throughput_constraint = pyo.Constraint(
        expr=sum(
            model.hydrogen_storage.charge[t] - model.hydrogen_storage.discharge[t]
            for t in model.time_steps
        )
        == total_throughput
    )

    # Solve the model once in the fixture
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


def test_storage_charge_discharge_soc(h2storage_model, h2storage_config):
    model, results = h2storage_model

    # Check charge, discharge, and SOC constraints
    for t in model.time_steps:
        charge = pyo.value(model.hydrogen_storage.charge[t])
        discharge = pyo.value(model.hydrogen_storage.discharge[t])
        soc = pyo.value(model.hydrogen_storage.soc[t])

        # Ensure SOC remains within bounds
        assert (
            h2storage_config["min_capacity"] <= soc <= h2storage_config["max_capacity"]
        )

        # Ensure charge and discharge rates do not exceed capacity
        assert charge <= h2storage_config["max_capacity"]
        assert discharge <= h2storage_config["max_capacity"]


# Test for Dri storage
@pytest.fixture
def dri_storage_config():
    return {
        "max_capacity": 1000,
        "min_capacity": 100,
        "initial_soc": 0.5,
        "storage_loss_rate": 0,
    }


@pytest.fixture
def dri_storage_model(dri_storage_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])
    model.iron_ore_price = pyo.Param(model.time_steps, initialize=1, mutable=True)

    dri_storage = DRIStorage(**dri_storage_config, time_steps=model.time_steps)
    model.dri_storage = pyo.Block()
    dri_storage.add_to_model(model, model.dri_storage)

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(
            model.dri_storage.charge[t] * model.iron_ore_price[t]
            - model.dri_storage.discharge[t] * model.iron_ore_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )
    # Constraint for total iron throughput over all time steps
    total_throughput = 300
    model.total_throughput_constraint = pyo.Constraint(
        expr=sum(
            model.dri_storage.charge[t] - model.dri_storage.discharge[t]
            for t in model.time_steps
        )
        == total_throughput
    )

    # Solve the model once in the fixture
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


def test_dri_storage_charge_discharge_soc(dri_storage_model, dri_storage_config):
    model, results = dri_storage_model

    # Check charge, discharge, and SOC constraints
    for t in model.time_steps:
        charge = pyo.value(model.dri_storage.charge[t])
        discharge = pyo.value(model.dri_storage.discharge[t])
        soc = pyo.value(model.dri_storage.soc[t])

        # Ensure SOC remains within bounds
        assert (
            dri_storage_config["min_capacity"]
            <= soc
            <= dri_storage_config["max_capacity"]
        )

        # Ensure charge and discharge rates do not exceed capacity
        assert charge <= dri_storage_config["max_capacity"]
        assert discharge <= dri_storage_config["max_capacity"]


# Fixture for boiler configuration (updated to electric fuel type)
@pytest.fixture
def boiler_config():
    return {
        "max_power": 100,
        "efficiency": 0.85,
        "fuel_type": "electricity",  # Updated to electric to support operational status
        "min_power": 20,
        "ramp_up": 50,
        "ramp_down": 50,
        "min_operating_steps": 2,
        "min_down_steps": 1,
    }


# Fixture for creating and solving a Pyomo model with a boiler component
@pytest.fixture
def boiler_model(boiler_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)

    boiler = Boiler(**boiler_config, time_steps=model.time_steps)
    model.boiler = pyo.Block()
    boiler.add_to_model(model, model.boiler)

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(
            model.boiler.power_in[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )

    # Constraint for total heat production over all time steps
    total_heat_production = 500
    model.total_heat_constraint = pyo.Constraint(
        expr=sum(model.boiler.heat_out[t] for t in model.time_steps)
        == total_heat_production
    )

    # Solve the model once in the fixture
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


# Test for boiler ramping constraints, power bounds, and efficiency
def test_boiler_ramping_and_power_bounds(boiler_model, boiler_config):
    model, results = boiler_model

    # Check ramp-up constraints
    for t in range(1, len(model.time_steps)):
        ramp_up_diff = pyo.value(
            model.boiler.power_in[t] - model.boiler.power_in[t - 1]
        )
        assert ramp_up_diff <= model.boiler.ramp_up

    # Check ramp-down constraints
    for t in range(1, len(model.time_steps)):
        ramp_down_diff = pyo.value(
            model.boiler.power_in[t - 1] - model.boiler.power_in[t]
        )
        assert ramp_down_diff <= model.boiler.ramp_down

    # Check power bounds
    for t in model.time_steps:
        power_in = pyo.value(model.boiler.power_in[t])
        assert model.boiler.min_power <= power_in <= model.boiler.max_power

    # Check efficiency constraint
    for t in model.time_steps:
        heat_out = pyo.value(model.boiler.heat_out[t])
        power_in = pyo.value(model.boiler.power_in[t])
        assert heat_out == pytest.approx(
            power_in * boiler_config["efficiency"], rel=1e-2
        )


# Fixture for heat pump configuration
@pytest.fixture
def heat_pump_config():
    return {
        "max_power": 80,
        "cop": 3.5,
        "min_power": 10,
        "ramp_up": 20,
        "ramp_down": 20,
        "min_operating_steps": 1,
        "min_down_steps": 1,
    }


# Fixture for creating and solving a Pyomo model with a heat pump component
@pytest.fixture
def heat_pump_model(heat_pump_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])
    model.electricity_price = pyo.Param(model.time_steps, initialize=1, mutable=True)

    heat_pump = HeatPump(time_steps=model.time_steps, **heat_pump_config)
    model.heat_pump = pyo.Block()
    heat_pump.add_to_model(model, model.heat_pump)

    # Objective function to minimize operating cost
    model.total_cost = pyo.Objective(
        expr=sum(
            model.heat_pump.power_in[t] * model.electricity_price[t]
            for t in model.time_steps
        ),
        sense=pyo.minimize,
    )

    # Constraint for total heat production over all time steps
    total_heat_production = 400
    model.total_heat_constraint = pyo.Constraint(
        expr=sum(model.heat_pump.heat_out[t] for t in model.time_steps)
        == total_heat_production
    )

    # Solve the model once in the fixture
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


# Test for heat pump ramping constraints, power bounds, and COP
def test_heat_pump_ramping_and_power_bounds(heat_pump_model, heat_pump_config):
    model, results = heat_pump_model

    # Check ramp-up constraints
    for t in range(1, len(model.time_steps)):
        ramp_up_diff = pyo.value(
            model.heat_pump.power_in[t] - model.heat_pump.power_in[t - 1]
        )
        assert ramp_up_diff <= model.heat_pump.ramp_up

    # Check ramp-down constraints
    for t in range(1, len(model.time_steps)):
        ramp_down_diff = pyo.value(
            model.heat_pump.power_in[t - 1] - model.heat_pump.power_in[t]
        )
        assert ramp_down_diff <= model.heat_pump.ramp_down

    # Check power bounds
    for t in model.time_steps:
        power_in = pyo.value(model.heat_pump.power_in[t])
        operational_status = (
            pyo.value(model.heat_pump.operational_status[t])
            if hasattr(model.heat_pump, "operational_status")
            else 1
        )
        assert (
            model.heat_pump.min_power * operational_status
            <= power_in
            <= model.heat_pump.max_power * operational_status
        )

    # Check COP constraint
    for t in model.time_steps:
        heat_out = pyo.value(model.heat_pump.heat_out[t])
        power_in = pyo.value(model.heat_pump.power_in[t])
        assert heat_out == pytest.approx(power_in * heat_pump_config["cop"], rel=1e-2)

    # Check operational status if applicable
    if hasattr(model.heat_pump, "operational_status"):
        for t in model.time_steps:
            status = pyo.value(model.heat_pump.operational_status[t])
            assert status in [0, 1]  # Binary variable


# Fixture for PV plant configuration (without profiles)
@pytest.fixture
def pv_plant_config():
    return {
        "max_power": 50,
    }


# Fixture for creating a model with a PV plant component, using an availability profile
@pytest.fixture
def pv_plant_model_with_availability(pv_plant_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])

    # Create an availability profile as a pandas Series
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=model.time_steps
    )

    pv_plant = PVPlant(
        **pv_plant_config,
        time_steps=model.time_steps,
        availability_profile=availability_profile,
    )
    model.pv_plant = pyo.Block()
    pv_plant.add_to_model(model, model.pv_plant)

    # Solve the model (assuming an objective and constraints)
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


# Fixture for creating a model with a PV plant component, using a power profile
@pytest.fixture
def pv_plant_model_with_power_profile(pv_plant_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=[idx for idx, _ in enumerate(range(10))])

    # Create a power profile as a pandas Series
    power_profile = pd.Series(
        [10, 20, 30, 40, 50, 30, 20, 10, 0, 15], index=model.time_steps
    )

    pv_plant = PVPlant(
        **pv_plant_config,
        time_steps=list(model.time_steps),
        power_profile=power_profile,
    )
    model.pv_plant = pyo.Block()
    pv_plant.add_to_model(model, model.pv_plant)

    # Solve the model (assuming an objective and constraints)
    solver = pyo.SolverFactory("glpk")
    results = solver.solve(model, tee=True)

    return model, results


# Test for PV plant with availability profile
def test_pv_plant_availability_profile(pv_plant_model_with_availability):
    model, results = pv_plant_model_with_availability

    # Create an availability profile as a pandas Series
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=model.time_steps
    )

    # Check if power output respects the availability profile
    for t in model.time_steps:
        availability = availability_profile[t]  # Get availability value (0 or 1)
        power_output = pyo.value(model.pv_plant.power[t])
        assert power_output <= model.pv_plant.max_power * availability
        if availability == 0:
            assert power_output == 0  # When not available, power output should be zero


# Test for PV plant with power profile
def test_pv_plant_power_profile(pv_plant_model_with_power_profile):
    model, results = pv_plant_model_with_power_profile

    # Create a power profile as a pandas Series
    power_profile = pd.Series(
        [10, 20, 30, 40, 50, 30, 20, 10, 0, 15], index=model.time_steps
    )

    # Check if power output follows the power profile
    for t in model.time_steps:
        expected_power_output = power_profile[t]
        power_output = pyo.value(model.pv_plant.power[t])
        assert power_output == pytest.approx(expected_power_output, rel=1e-2)


# Fixture for EV configuration (without profiles)
@pytest.fixture
def ev_config():
    return {
        "max_capacity": 1.0,  # SOC between 0 and 1
        "min_capacity": 0.0,
        "max_power_charge": 0.2,  # Charge values will reflect a fraction of the capacity
        "max_power_discharge": 0.2,  # Discharge values will also be a fraction of the capacity
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
    solver = pyo.SolverFactory("glpk")
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
    solver = pyo.SolverFactory("glpk")
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
def test_ev_charging_profile(ev_model_with_charging_profile):
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
        assert 0 <= soc <= 1
