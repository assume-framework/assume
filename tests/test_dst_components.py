# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pyomo.environ as pyo
import pytest

from assume.units.dst_components import create_electrolyser


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


if __name__ == "__main__":
    pytest.main(["-s", __file__])
