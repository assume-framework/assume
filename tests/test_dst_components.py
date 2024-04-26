# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest
from pyomo.core.base import Constraint
from pyomo.environ import Boolean

from assume.units.dst_components import Electrolyser


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


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
