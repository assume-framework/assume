# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

"""
Tests for CHP and its integration into SectorCoupledSystem.

Run:
    pytest tests/test_chp.py -v
"""

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.forecaster import SectorCoupledForecaster
from assume.units.dst_components import CHP, demand_side_technologies
from assume.units.sector_coupled_system import SectorCoupledSystem

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


def make_sector_system(
    time_index,
    *,
    heat_demand=5.0,
    electricity_price=100.0,
    natural_gas_price=10.0,
    components=None,
):
    forecaster = SectorCoupledForecaster(
        index=time_index,
        carrier_demands={
            "heat": constant_series(heat_demand, time_index),
        },
        fuel_prices={
            "natural gas": constant_series(natural_gas_price, time_index),
        },
        market_prices={
            "EOM": constant_series(electricity_price, time_index),
        },
    )

    if components is None:
        components = {
            "heat_pump": {
                "max_power": 5.0,
                "min_power": 0.0,
                "cop": 2.0,
                "ramp_up": 5.0,
                "ramp_down": 5.0,
            },
            "chp_gas": {
                "max_power": 4.0,
                "min_power": 0.0,
                "electrical_efficiency": 0.40,
                "thermal_efficiency": 0.50,
                "fuel_type": "natural_gas",
                "cycle_type": "combined_cycle",
                "ramp_up": 4.0,
                "ramp_down": 4.0,
            },
        }

    return SectorCoupledSystem(
        id="chp_test_system",
        unit_operator="test_operator",
        bidding_strategies={},
        forecaster=forecaster,
        components=components,
        system_type="heat",
        market_representation="aggregate",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=10,
        node="north",
    )


def solve_sector_system(system):
    system.setup_model(presolve=True)
    instance = system.model.create_instance()
    instance = system.switch_to_opt(instance)
    solve(instance)
    return instance


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


def test_scs_creates_gas_price_for_chp(time_index):
    system = make_sector_system(
        time_index,
        natural_gas_price=25.0,
    )

    system.setup_model(presolve=False)

    assert hasattr(system.model, "natural_gas_price")

    for t in system.model.time_steps:
        assert val(system.model.natural_gas_price[t]) == pytest.approx(25.0, abs=TOL)


# TEST 7: CHP heat production and electrical export in SCS


def test_scs_chp_heat_and_electricity(time_index):
    system = make_sector_system(
        time_index,
        heat_demand=5.0,
        electricity_price=100.0,
        natural_gas_price=10.0,
    )

    instance = solve_sector_system(system)

    hp = instance.dsm_blocks["heat_pump"]
    chp = instance.dsm_blocks["chp_gas"]

    for t in instance.time_steps:
        assert val(hp.heat_out[t]) == pytest.approx(0.0, abs=TOL)

        assert val(chp.heat_out[t]) == pytest.approx(5.0, abs=TOL)

        assert val(chp.fuel_in[t]) == pytest.approx(10.0, abs=TOL)

        assert val(chp.power_out[t]) == pytest.approx(4.0, abs=TOL)

        # SCS physical exchange:
        # HP import - CHP generation = 0 - 4
        assert val(instance.total_power_input[t]) == pytest.approx(-4.0, abs=TOL)

        heat_supply = val(hp.heat_out[t]) + val(chp.heat_out[t])

        assert heat_supply == pytest.approx(
            val(instance.heat_demand[t]),
            abs=TOL,
        )


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
