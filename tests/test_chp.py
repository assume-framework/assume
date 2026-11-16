# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT


import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.units.dst_components import CHP, demand_side_technologies

SOLVER = "appsi_highs"
TOL = 1e-5


def val(x):
    return pyo.value(x)


def assert_optimal(results):
    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal


@pytest.fixture
def time_index():
    return pd.date_range("2023-01-01", periods=4, freq="h")


def constant_series(value, index):
    return pd.Series(
        [value] * len(index),
        index=index,
        dtype=float,
    )


def solve(model):
    solver = pyo.SolverFactory(SOLVER)
    results = solver.solve(model)
    assert_optimal(results)
    return results


def make_chp_model(
    *,
    fuel_type="natural_gas",
    max_power=8.0,
    electrical_efficiency=0.40,
    thermal_efficiency=0.45,
    additional_cost=5.0,
    natural_gas_price=20.0,
):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=range(2))

    if fuel_type == "natural_gas":
        model.natural_gas_price = pyo.Param(
            model.time_steps,
            initialize={0: natural_gas_price, 1: natural_gas_price},
        )

    component = CHP(
        time_steps=model.time_steps,
        max_power=max_power,
        electrical_efficiency=electrical_efficiency,
        thermal_efficiency=thermal_efficiency,
        fuel_type=fuel_type,
        additional_cost=additional_cost,
    )

    model.chp = pyo.Block()
    component.add_to_model(model, model.chp)

    model.objective = pyo.Objective(
        expr=sum(model.chp.operating_cost[t] for t in model.time_steps),
        sense=pyo.minimize,
    )

    return model


# TEST 1: CHP registration


def test_chp_registered():
    assert demand_side_technologies["chp"] is CHP


# TEST 2: Natural-gas CHP conversion and cost


def test_natural_gas_chp():
    model = make_chp_model()

    for t in model.time_steps:
        model.chp.power_out[t].fix(8.0)

    solve(model)

    for t in model.time_steps:
        chp = model.chp

        assert val(chp.power_out[t]) == pytest.approx(8.0, abs=TOL)
        assert val(chp.fuel_in[t]) == pytest.approx(20.0, abs=TOL)
        assert val(chp.heat_out[t]) == pytest.approx(9.0, abs=TOL)
        assert val(chp.natural_gas_in[t]) == pytest.approx(20.0, abs=TOL)
        assert val(chp.hydrogen_in[t]) == pytest.approx(0.0, abs=TOL)

        # 20 * 20 + 8 * 5 = 440 EUR
        assert val(chp.operating_cost[t]) == pytest.approx(440.0, abs=TOL)


# TEST 3: Hydrogen CHP, internally balanced fuel


def test_hydrogen_chp():
    model = make_chp_model(
        fuel_type="hydrogen_gas",
        max_power=4.0,
        electrical_efficiency=0.40,
        thermal_efficiency=0.50,
        additional_cost=2.0,
    )

    # No hydrogen_gas_price should be necessary.
    assert not hasattr(model, "hydrogen_gas_price")

    for t in model.time_steps:
        model.chp.power_out[t].fix(4.0)

    solve(model)

    for t in model.time_steps:
        chp = model.chp

        assert val(chp.fuel_in[t]) == pytest.approx(10.0, abs=TOL)
        assert val(chp.hydrogen_in[t]) == pytest.approx(10.0, abs=TOL)
        assert val(chp.natural_gas_in[t]) == pytest.approx(0.0, abs=TOL)
        assert val(chp.heat_out[t]) == pytest.approx(5.0, abs=TOL)
        assert val(chp.operating_cost[t]) == pytest.approx(8.0, abs=TOL)


# TEST 4: CHP electrical capacity limit


def test_chp_capacity_limit():
    model = make_chp_model(max_power=4.0)

    assert model.chp.power_out[0].lb == pytest.approx(0.0)
    assert model.chp.power_out[0].ub == pytest.approx(4.0)


# TEST 5: CHP parameter validation


@pytest.mark.parametrize(
    "changes",
    [
        {"max_power": -1.0},
        {"electrical_efficiency": 0.0},
        {"thermal_efficiency": -0.1},
        {
            "electrical_efficiency": 0.7,
            "thermal_efficiency": 0.5,
        },
        {"fuel_type": "coal"},
        {"min_power": 20.0},
    ],
)
def test_chp_invalid_parameters(changes):
    config = {
        "time_steps": [0, 1],
        "max_power": 8.0,
        "electrical_efficiency": 0.4,
        "thermal_efficiency": 0.45,
    }
    config.update(changes)

    with pytest.raises(ValueError):
        CHP(**config)


# TEST 6: CHP-only natural gas price creation


# TEST 7: CHP heat production and electrical export in SCS


# TEST 8: CHP energy conservation


def test_chp_energy_conservation():
    model = make_chp_model()

    for t in model.time_steps:
        model.chp.power_out[t].fix(8.0)

    solve(model)

    for t in model.time_steps:
        chp = model.chp

        fuel = val(chp.fuel_in[t])
        electricity = val(chp.power_out[t])
        heat = val(chp.heat_out[t])

        assert electricity + heat <= fuel + TOL

        assert electricity == pytest.approx(fuel * 0.40, abs=TOL)

        assert heat == pytest.approx(fuel * 0.45, abs=TOL)
