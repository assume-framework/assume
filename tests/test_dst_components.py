# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import (
    DriPlant,
    DRIStorage,
    ElectricArcFurnace,
    Electrolyser,
    GenericStorage,
)


# Test Electrolyser class
@pytest.fixture
def electrolyser_unit() -> Electrolyser:
    return Electrolyser(
        model=pyo.ConcreteModel(),
        id="Test_Electrolyser",
        rated_power=100,
        min_power=10,
        ramp_up=20,
        ramp_down=20,
        min_operating_time=1,
        min_down_time=1,
        efficiency=0.8,
        fuel_type="Electricity",
        index=pd.date_range("2022-01-01", periods=4, freq="h"),
    )


def test_init_function(electrolyser_unit):
    assert electrolyser_unit.id == "Test_Electrolyser"
    assert electrolyser_unit.rated_power == 100
    assert electrolyser_unit.min_power == 10
    assert electrolyser_unit.ramp_up == 20
    assert electrolyser_unit.ramp_down == 20
    assert electrolyser_unit.min_operating_time == 1
    assert electrolyser_unit.min_down_time == 1
    assert electrolyser_unit.efficiency == 0.8
    assert electrolyser_unit.fuel_type == "Electricity"


def test_add_to_model(electrolyser_unit):
    electrolyser_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electrolyser_unit, "b")


def test_parameters(electrolyser_unit):
    electrolyser_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electrolyser_unit.b, "rated_power")
    assert hasattr(electrolyser_unit.b, "min_power")
    assert hasattr(electrolyser_unit.b, "ramp_up")
    assert hasattr(electrolyser_unit.b, "ramp_down")
    assert hasattr(electrolyser_unit.b, "min_operating_time")
    assert hasattr(electrolyser_unit.b, "min_down_time")
    assert hasattr(electrolyser_unit.b, "efficiency")


def test_variables(electrolyser_unit):
    electrolyser_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electrolyser_unit.b, "power_in")
    assert hasattr(electrolyser_unit.b, "hydrogen_out")
    assert hasattr(electrolyser_unit.b, "electricity_cost")


def test_constraints(electrolyser_unit):
    electrolyser_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electrolyser_unit.b, "power_upper_bound")
    assert hasattr(electrolyser_unit.b, "power_lower_bound")
    assert hasattr(electrolyser_unit.b, "ramp_up_constraint")
    assert hasattr(electrolyser_unit.b, "ramp_down_constraint")
    assert hasattr(electrolyser_unit.b, "min_operating_time_electrolyser_constraint")
    assert hasattr(electrolyser_unit.b, "min_downtime_electrolyser_constraint")
    assert hasattr(electrolyser_unit.b, "power_in_equation")
    assert hasattr(electrolyser_unit.b, "operating_cost_with_el_price")


# Test driplant class
@pytest.fixture
def driplant_unit() -> DriPlant:
    return DriPlant(
        model=pyo.ConcreteModel(),
        id="Test_DriPlant",
        specific_hydrogen_consumption=0.5,
        specific_natural_gas_consumption=0.8,
        specific_electricity_consumption=0.2,
        specific_iron_ore_consumption=0.1,
        rated_power=200,
        min_power=20,
        fuel_type="natural_gas",
        ramp_up=30,
        ramp_down=30,
        min_operating_time=2,
        min_down_time=1,
        index=pd.date_range("2022-01-01", periods=4, freq="h"),
    )


def test_init_function(driplant_unit):
    assert driplant_unit.id == "Test_DriPlant"
    assert driplant_unit.specific_hydrogen_consumption == 0.5
    assert driplant_unit.specific_natural_gas_consumption == 0.8
    assert driplant_unit.specific_electricity_consumption == 0.2
    assert driplant_unit.specific_iron_ore_consumption == 0.1
    assert driplant_unit.rated_power == 200
    assert driplant_unit.min_power == 20
    assert driplant_unit.fuel_type == "natural_gas"


def test_add_to_model(driplant_unit):
    driplant_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(driplant_unit, "b")


def test_parameters(driplant_unit):
    driplant_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(driplant_unit.b, "specific_hydrogen_consumption")
    assert hasattr(driplant_unit.b, "specific_natural_gas_consumption")
    assert hasattr(driplant_unit.b, "specific_electricity_consumption_dri")
    assert hasattr(driplant_unit.b, "specific_iron_ore_consumption")
    assert hasattr(driplant_unit.b, "min_power_dri")
    assert hasattr(driplant_unit.b, "rated_power_dri")
    assert hasattr(driplant_unit.b, "ramp_up_dri")
    assert hasattr(driplant_unit.b, "ramp_down_dri")
    assert hasattr(driplant_unit.b, "min_operating_time_dri")
    assert hasattr(driplant_unit.b, "min_down_time_dri")


def test_variables(driplant_unit):
    driplant_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(driplant_unit.b, "iron_ore_in")
    assert hasattr(driplant_unit.b, "natural_gas_in")
    assert hasattr(driplant_unit.b, "dri_operating_cost")
    assert hasattr(driplant_unit.b, "hydrogen_in")
    assert hasattr(driplant_unit.b, "operational_status")
    assert hasattr(driplant_unit.b, "dri_output")
    assert hasattr(driplant_unit.b, "power_dri")


def test_constraints(driplant_unit):
    driplant_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(driplant_unit.b, "dri_power_lower_bound")
    assert hasattr(driplant_unit.b, "dri_power_upper_bound")
    assert hasattr(driplant_unit.b, "dri_output_constraint")
    assert hasattr(driplant_unit.b, "dri_output_electricity_constraint")
    assert hasattr(driplant_unit.b, "iron_ore_constraint")
    assert hasattr(driplant_unit.b, "ramp_up_dri_constraint")
    assert hasattr(driplant_unit.b, "ramp_down_dri_constraint")
    assert hasattr(driplant_unit.b, "min_operating_time_dri__constraint")
    assert hasattr(driplant_unit.b, "min_down_time_dri_constraint")
    assert hasattr(driplant_unit.b, "dri_operating_cost_constraint")


# Test ElectricArcFurnace class
@pytest.fixture
def electric_arc_furnace_unit() -> ElectricArcFurnace:
    return ElectricArcFurnace(
        model=pyo.ConcreteModel(),
        id="Test_EAF",
        rated_power=200,
        min_power=50,
        specific_electricity_consumption=0.5,
        specific_dri_demand=0.7,
        specific_lime_demand=0.2,
        ramp_up=30,
        ramp_down=30,
        min_operating_time=3,
        min_down_time=2,
    )


def test_init_function_electric_arc_furnace(electric_arc_furnace_unit):
    assert electric_arc_furnace_unit.id == "Test_EAF"
    assert electric_arc_furnace_unit.rated_power == 200
    assert electric_arc_furnace_unit.min_power == 50
    assert electric_arc_furnace_unit.specific_electricity_consumption == 0.5
    assert electric_arc_furnace_unit.specific_dri_demand == 0.7
    assert electric_arc_furnace_unit.specific_lime_demand == 0.2
    assert electric_arc_furnace_unit.ramp_up == 30
    assert electric_arc_furnace_unit.ramp_down == 30
    assert electric_arc_furnace_unit.min_operating_time == 3
    assert electric_arc_furnace_unit.min_down_time == 2


def test_add_to_model_electric_arc_furnace(electric_arc_furnace_unit):
    electric_arc_furnace_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electric_arc_furnace_unit, "b")


def test_parameters_electric_arc_furnace(electric_arc_furnace_unit):
    electric_arc_furnace_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electric_arc_furnace_unit.b, "rated_power_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "min_power_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "specific_electricity_consumption_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "specific_dri_demand")
    assert hasattr(electric_arc_furnace_unit.b, "specific_lime_demand")
    assert hasattr(electric_arc_furnace_unit.b, "ramp_up_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "ramp_down_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "min_operating_time_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "min_down_time_eaf")


def test_variables_electric_arc_furnace(electric_arc_furnace_unit):
    electric_arc_furnace_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electric_arc_furnace_unit.b, "power_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "dri_input")
    assert hasattr(electric_arc_furnace_unit.b, "steel_output")
    assert hasattr(electric_arc_furnace_unit.b, "eaf_operating_cost")
    assert hasattr(electric_arc_furnace_unit.b, "emission_eaf")
    assert hasattr(electric_arc_furnace_unit.b, "lime_demand")


def test_constraints_electric_arc_furnace(electric_arc_furnace_unit):
    electric_arc_furnace_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(electric_arc_furnace_unit.b, "electricity_input_upper_bound")
    assert hasattr(electric_arc_furnace_unit.b, "electricity_input_lower_bound")
    assert hasattr(electric_arc_furnace_unit.b, "steel_output_dri_relation")
    assert hasattr(electric_arc_furnace_unit.b, "steel_output_power_relation")
    assert hasattr(electric_arc_furnace_unit.b, "eaf_lime_demand")
    assert hasattr(electric_arc_furnace_unit.b, "eaf_co2_emission")
    assert hasattr(electric_arc_furnace_unit.b, "ramp_up_eaf_constraint")
    assert hasattr(electric_arc_furnace_unit.b, "ramp_down_eaf_constraint")
    assert hasattr(electric_arc_furnace_unit.b, "min_operating_time_eaf_constraint")
    assert hasattr(electric_arc_furnace_unit.b, "min_down_time_eaf_constraint")
    assert hasattr(electric_arc_furnace_unit.b, "eaf_operating_cost_cosntraint")


@pytest.fixture
def dri_storage_unit() -> DRIStorage:
    return DRIStorage(
        model=pyo.ConcreteModel(),
        id="Test_DRI_Storage",
        max_capacity=1000,
        min_capacity=100,
        initial_soc=500,
        storage_loss_rate=0.1,
        charge_loss_rate=0.05,
        discharge_loss_rate=0.07,
        index=pd.date_range("2022-01-01", periods=4, freq="h"),
    )


def test_init_function(dri_storage_unit):
    assert dri_storage_unit.id == "Test_DRI_Storage"
    assert dri_storage_unit.max_capacity == 1000
    assert dri_storage_unit.min_capacity == 100
    assert dri_storage_unit.initial_soc == 500
    assert dri_storage_unit.storage_loss_rate == 0.1
    assert dri_storage_unit.charge_loss_rate == 0.05
    assert dri_storage_unit.discharge_loss_rate == 0.07


def test_add_to_model(dri_storage_unit):
    dri_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(dri_storage_unit, "b")


def test_parameters(dri_storage_unit):
    dri_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(dri_storage_unit.b, "max_capacity_dri")
    assert hasattr(dri_storage_unit.b, "min_capacity_dri")
    assert hasattr(dri_storage_unit.b, "initial_soc_dri")
    assert hasattr(dri_storage_unit.b, "storage_loss_rate_dri")
    assert hasattr(dri_storage_unit.b, "charge_loss_rate_dri")
    assert hasattr(dri_storage_unit.b, "discharge_loss_rate_dri")


def test_variables(dri_storage_unit):
    dri_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(dri_storage_unit.b, "soc_dri")
    assert hasattr(dri_storage_unit.b, "uniformity_indicator_dri")
    assert hasattr(dri_storage_unit.b, "charge_dri")
    assert hasattr(dri_storage_unit.b, "discharge_dri")


def test_constraints(dri_storage_unit):
    dri_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(dri_storage_unit.b, "storage_min_capacity_dri_constraint")
    assert hasattr(dri_storage_unit.b, "storage_max_capacity_dri_constraint")
    assert hasattr(dri_storage_unit.b, "energy_in_max_capacity_dri_constraint")
    assert hasattr(dri_storage_unit.b, "energy_out_max_capacity_dri_constraint")
    assert hasattr(dri_storage_unit.b, "energy_in_uniformity_dri_constraint")
    assert hasattr(dri_storage_unit.b, "energy_out_uniformity_dri_constraint")
    assert hasattr(dri_storage_unit.b, "storage_capacity_change_dri_constraint")


@pytest.fixture
def generic_storage_unit() -> GenericStorage:
    return GenericStorage(
        model=pyo.ConcreteModel(),
        id="Test_Generic_Storage",
        max_capacity=1000,
        min_capacity=100,
        initial_soc=500,
        storage_loss_rate=0.1,
        charge_loss_rate=0.05,
        discharge_loss_rate=0.07,
    )


def test_init_function(generic_storage_unit):
    assert generic_storage_unit.id == "Test_Generic_Storage"
    assert generic_storage_unit.max_capacity == 1000
    assert generic_storage_unit.min_capacity == 100
    assert generic_storage_unit.initial_soc == 500
    assert generic_storage_unit.storage_loss_rate == 0.1
    assert generic_storage_unit.charge_loss_rate == 0.05
    assert generic_storage_unit.discharge_loss_rate == 0.07


def test_add_to_model(generic_storage_unit):
    generic_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(generic_storage_unit, "b")


def test_parameters(generic_storage_unit):
    generic_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(generic_storage_unit.b, "max_capacity")
    assert hasattr(generic_storage_unit.b, "min_capacity")
    assert hasattr(generic_storage_unit.b, "initial_soc")
    assert hasattr(generic_storage_unit.b, "storage_loss_rate")
    assert hasattr(generic_storage_unit.b, "charge_loss_rate")
    assert hasattr(generic_storage_unit.b, "discharge_loss_rate")


def test_variables(generic_storage_unit):
    generic_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(generic_storage_unit.b, "soc")
    assert hasattr(generic_storage_unit.b, "uniformity_indicator")
    assert hasattr(generic_storage_unit.b, "charge")
    assert hasattr(generic_storage_unit.b, "discharge")


def test_constraints(generic_storage_unit):
    generic_storage_unit.add_to_model(unit_block=pyo.Block(), time_steps=[0, 1, 2])
    assert hasattr(generic_storage_unit.b, "storage_min_capacity_constraint")
    assert hasattr(generic_storage_unit.b, "storage_max_capacity_constraint")
    assert hasattr(generic_storage_unit.b, "energy_in_max_capacity_constraint")
    assert hasattr(generic_storage_unit.b, "energy_out_max_capacity_constraint")
    assert hasattr(generic_storage_unit.b, "energy_in_uniformity_constraint")
    assert hasattr(generic_storage_unit.b, "energy_out_uniformity_constraint")
    assert hasattr(generic_storage_unit.b, "storage_capacity_change_constraint")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
