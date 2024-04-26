# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.core.base import Constraint
from pyomo.environ import Boolean

from assume.units.dst_components import DriPlant, Electrolyser


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


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
