# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Tests for the FuelCell component.
"""

import pyomo.environ as pyo
import pytest

from assume.units.dst_components import FuelCell, demand_side_technologies

SOLVER = "appsi_highs"
TOL = 1e-5


def val(x):
    return pyo.value(x)


def assert_optimal(results):
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal


def solve(model):
    solver = pyo.SolverFactory(SOLVER)
    results = solver.solve(model)
    assert_optimal(results)
    return results


def make_fuel_cell_model(
    *,
    n_steps=2,
    max_power=8.0,
    efficiency=0.50,
    variable_cost=5.0,
    ramp_up=None,
    ramp_down=None,
):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(n_steps))

    component = FuelCell(
        time_steps=model.time_steps,
        max_power=max_power,
        efficiency=efficiency,
        variable_cost=variable_cost,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
    )

    model.fuel_cell = pyo.Block()
    component.add_to_model(model, model.fuel_cell)

    model.objective = pyo.Objective(
        expr=sum(model.fuel_cell.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    return model


# TEST 1: Fuel cell registration


def test_fuel_cell_registered():
    assert demand_side_technologies["fuel_cell"] is FuelCell


# TEST 2: Hydrogen-to-electricity conversion and cost


def test_fuel_cell_conversion_and_cost():
    model = make_fuel_cell_model()

    for t in model.time_steps:
        model.fuel_cell.power_out[t].fix(8.0)

    solve(model)

    for t in model.time_steps:
        fc = model.fuel_cell

        assert val(fc.power_out[t]) == pytest.approx(8.0, abs=TOL)
        # 8 MW_el / 0.5 = 16 MW_H2
        assert val(fc.hydrogen_in[t]) == pytest.approx(16.0, abs=TOL)
        # Hydrogen is not priced here, only variable cost: 8 * 5 = 40 EUR
        assert val(fc.operating_cost[t]) == pytest.approx(40.0, abs=TOL)


# TEST 3: Fuel cell electrical capacity limit


def test_fuel_cell_capacity_limit():
    model = make_fuel_cell_model(max_power=4.0)

    assert model.fuel_cell.power_out[0].lb == pytest.approx(0.0)
    assert model.fuel_cell.power_out[0].ub == pytest.approx(4.0)


# TEST 4: Fuel cell parameter validation


@pytest.mark.parametrize(
    "changes",
    [
        {"max_power": -1.0},
        {"min_power": -1.0},
        {"min_power": 20.0},
        {"efficiency": 0.0},
        {"efficiency": 1.2},
    ],
)
def test_fuel_cell_invalid_parameters(changes):
    config = {
        "time_steps": [0, 1],
        "max_power": 8.0,
        "efficiency": 0.5,
    }
    config.update(changes)

    with pytest.raises(ValueError):
        FuelCell(**config)


# TEST 5: Fuel cell ramp-up limit


def test_fuel_cell_ramp_up_limit():
    model = make_fuel_cell_model(n_steps=3, max_power=10.0, ramp_up=3.0)

    # Maximise output instead of minimising cost
    model.objective.deactivate()
    model.max_output = pyo.Objective(
        expr=sum(model.fuel_cell.power_out[t] for t in model.time_steps),
        sense=pyo.maximize,
    )

    solve(model)

    # First step starts from zero, then +3 MW per step
    for t, expected in zip(model.time_steps, [3.0, 6.0, 9.0]):
        assert val(model.fuel_cell.power_out[t]) == pytest.approx(expected, abs=TOL)


# TEST 6: Fuel cell energy conservation


def test_fuel_cell_energy_conservation():
    model = make_fuel_cell_model()

    for t in model.time_steps:
        model.fuel_cell.power_out[t].fix(8.0)

    solve(model)

    for t in model.time_steps:
        fc = model.fuel_cell

        hydrogen = val(fc.hydrogen_in[t])
        electricity = val(fc.power_out[t])

        assert electricity <= hydrogen + TOL

        assert electricity == pytest.approx(hydrogen * 0.50, abs=TOL)
