# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.forecaster import SectorCoupledForecaster
from assume.strategies.naive_strategies import (
    SectorCoupledRedispatchNaiveStrategy,
)
from assume.units.sector_coupled_system import SectorCoupledSystem


USE_SOLVER = "appsi_highs"
TOL = 1e-5


def _val(value):
    return pyo.value(value)


def _series(value, index):
    return pd.Series([value] * len(index), index=index, dtype=float)


def _make_forecaster(
    time_index,
    *,
    heat_demand=4.0,
    electricity_price=40.0,
    natural_gas_price=50.0,
):
    return SectorCoupledForecaster(
        index=time_index,
        carrier_demands={
            "heat": _series(heat_demand, time_index),
        },
        fuel_prices={
            "natural gas": _series(natural_gas_price, time_index),
        },
        market_prices={
            "EOM": _series(electricity_price, time_index),
        },
    )


def _make_components(
    *,
    hp_max_power=2.0,
    cop=2.0,
    boiler_max_power=5.0,
    boiler_efficiency=1.0,
):
    return {
        "heat_pump": {
            "max_power": hp_max_power,
            "min_power": 0.0,
            "cop": cop,
            "ramp_up": hp_max_power,
            "ramp_down": hp_max_power,
        },
        "boiler_gas": {
            "max_power": boiler_max_power,
            "min_power": 0.0,
            "efficiency": boiler_efficiency,
            "fuel_type": "natural_gas",
            "ramp_up": boiler_max_power,
            "ramp_down": boiler_max_power,
        },
    }


def _make_components_with_tes(
    *,
    hp_max_power=3.0,
    cop=2.0,
    boiler_max_power=5.0,
    boiler_efficiency=1.0,
    storage_capacity=2.0,
    storage_initial_soc=0.0,
):
    components = _make_components(
        hp_max_power=hp_max_power,
        cop=cop,
        boiler_max_power=boiler_max_power,
        boiler_efficiency=boiler_efficiency,
    )

    components["thermal_storage"] = {
        "storage_type": "short-term",
        "capacity": storage_capacity,
        "min_soc": 0.0,
        "max_soc": 1.0,
        "max_power_charge": 2.0,
        "max_power_discharge": 2.0,
        "efficiency_charge": 1.0,
        "efficiency_discharge": 1.0,
        "initial_soc": storage_initial_soc,
        "ramp_up": 2.0,
        "ramp_down": 2.0,
        "storage_loss_rate": 0.0,
    }

    return components


def _make_sector_system(
    forecaster,
    components,
    *,
    unit_id="hs_north",
    node="north",
):
    return SectorCoupledSystem(
        id=unit_id,
        unit_operator="test_operator",
        bidding_strategies={},
        forecaster=forecaster,
        components=components,
        system_type="heat",
        market_representation="aggregate",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=10,
        node=node,
    )


def _solve_sector_system(sector_system):
    sector_system.setup_model(presolve=True)
    instance = sector_system.model.create_instance()
    instance = sector_system.switch_to_opt(instance)

    solver = pyo.SolverFactory(USE_SOLVER)
    results = solver.solve(instance)

    return sector_system, instance, results


@pytest.fixture
def time_index():
    return pd.date_range("2023-01-01 00:00:00", periods=4, freq="h")


@pytest.fixture
def base_sector_system(time_index):
    return _make_sector_system(
        _make_forecaster(
            time_index,
            heat_demand=4.0,
            electricity_price=40.0,
            natural_gas_price=50.0,
        ),
        _make_components(
            hp_max_power=2.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
        ),
    )


@pytest.fixture
def solved_base_sector_system(base_sector_system):
    return _solve_sector_system(base_sector_system)


def test_sector_coupled_forecaster_exposes_heat_demand(time_index):
    forecaster = _make_forecaster(time_index, heat_demand=4.0)

    assert hasattr(forecaster, "heat_demand")
    assert list(forecaster.get_demand("heat")) == pytest.approx([4.0] * 4)
    assert list(forecaster.heat_demand) == pytest.approx([4.0] * 4)


def test_sector_coupled_model_solves_successfully(solved_base_sector_system):
    _, _, results = solved_base_sector_system

    assert results.solver.status == pyo.SolverStatus.ok
    assert (
        results.solver.termination_condition
        == pyo.TerminationCondition.optimal
    )


def test_sector_coupled_base_case_uses_heat_pump(solved_base_sector_system):
    _, instance, _ = solved_base_sector_system

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        assert _val(hp.power_in[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(hp.heat_out[t]) == pytest.approx(4.0, abs=TOL)
        assert _val(boiler.natural_gas_in[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(instance.total_power_input[t]) == pytest.approx(
            2.0, abs=TOL
        )


def test_sector_coupled_heat_balance(solved_base_sector_system):
    _, instance, _ = solved_base_sector_system

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        heat_supply = _val(hp.heat_out[t]) + _val(boiler.heat_out[t])
        heat_demand = _val(instance.heat_demand[t])

        assert heat_supply == pytest.approx(heat_demand, abs=TOL)


def test_sector_coupled_uses_natural_gas_price_key(solved_base_sector_system):
    _, instance, _ = solved_base_sector_system

    for t in instance.time_steps:
        assert _val(instance.natural_gas_price[t]) == pytest.approx(
            50.0, abs=TOL
        )


def test_sector_coupled_uses_boiler_when_electricity_is_more_expensive(
    time_index,
):
    sector_system = _make_sector_system(
        _make_forecaster(
            time_index,
            heat_demand=4.0,
            electricity_price=120.0,
            natural_gas_price=50.0,
        ),
        _make_components(
            hp_max_power=2.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
        ),
    )

    _, instance, _ = _solve_sector_system(sector_system)

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        assert _val(hp.power_in[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(hp.heat_out[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(boiler.natural_gas_in[t]) == pytest.approx(4.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(4.0, abs=TOL)
        assert _val(instance.total_power_input[t]) == pytest.approx(
            0.0, abs=TOL
        )


def test_sector_coupled_uses_boiler_when_heat_pump_capacity_is_binding(
    time_index,
):
    sector_system = _make_sector_system(
        _make_forecaster(
            time_index,
            heat_demand=6.0,
            electricity_price=40.0,
            natural_gas_price=50.0,
        ),
        _make_components(
            hp_max_power=2.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
        ),
    )

    _, instance, _ = _solve_sector_system(sector_system)

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        assert _val(hp.power_in[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(hp.heat_out[t]) == pytest.approx(4.0, abs=TOL)
        assert _val(boiler.natural_gas_in[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(instance.total_power_input[t]) == pytest.approx(
            2.0, abs=TOL
        )

        heat_supply = _val(hp.heat_out[t]) + _val(boiler.heat_out[t])
        assert heat_supply == pytest.approx(
            _val(instance.heat_demand[t]), abs=TOL
        )


def test_sector_coupled_with_tes_shifts_heat_generation_in_day_ahead(
    time_index,
):
    electricity_price = pd.Series(
        [10.0, 100.0, 50.0, 10.0],
        index=time_index,
        dtype=float,
    )
    heat_demand = pd.Series(
        [4.0, 4.0, 4.0, 4.0],
        index=time_index,
        dtype=float,
    )
    natural_gas_price = pd.Series(
        [200.0, 200.0, 200.0, 200.0],
        index=time_index,
        dtype=float,
    )

    forecaster = SectorCoupledForecaster(
        index=time_index,
        carrier_demands={"heat": heat_demand},
        fuel_prices={"natural gas": natural_gas_price},
        market_prices={"EOM": electricity_price},
    )

    sector_system = _make_sector_system(
        forecaster,
        _make_components_with_tes(
            hp_max_power=3.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
            storage_capacity=2.0,
            storage_initial_soc=0.0,
        ),
    )

    _, instance, results = _solve_sector_system(sector_system)

    assert results.solver.status == pyo.SolverStatus.ok
    assert (
        results.solver.termination_condition
        == pyo.TerminationCondition.optimal
    )

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]
    tes = instance.dsm_blocks["thermal_storage"]

    expected_power = [3.0, 1.0, 2.0, 2.0]
    expected_charge = [2.0, 0.0, 0.0, 0.0]
    expected_discharge = [0.0, 2.0, 0.0, 0.0]
    expected_soc = [1.0, 0.0, 0.0, 0.0]

    for t in instance.time_steps:
        assert _val(instance.total_power_input[t]) == pytest.approx(
            expected_power[t], abs=TOL
        )
        assert _val(tes.charge[t]) == pytest.approx(
            expected_charge[t], abs=TOL
        )
        assert _val(tes.discharge[t]) == pytest.approx(
            expected_discharge[t], abs=TOL
        )
        assert _val(tes.soc[t]) == pytest.approx(
            expected_soc[t], abs=TOL
        )
        assert _val(tes.power_in[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(0.0, abs=TOL)

        heat_supply = (
            _val(hp.heat_out[t])
            + _val(boiler.heat_out[t])
            + _val(tes.discharge[t])
        )
        heat_required = (
            _val(instance.heat_demand[t])
            + _val(tes.charge[t])
        )
        assert heat_supply == pytest.approx(heat_required, abs=TOL)


def test_sector_coupled_redispatch_bounds_full_boiler_replacement(
    solved_base_sector_system,
    time_index,
):
    sector_system, _, _ = solved_base_sector_system

    min_power, max_power = sector_system.get_redispatch_power_bounds(
        time_index[0]
    )

    assert min_power == pytest.approx(0.0, abs=TOL)
    assert max_power == pytest.approx(2.0, abs=TOL)


def test_sector_coupled_redispatch_bounds_boiler_capacity_binding(
    time_index,
):
    sector_system = _make_sector_system(
        _make_forecaster(
            time_index,
            heat_demand=6.0,
            electricity_price=40.0,
            natural_gas_price=50.0,
        ),
        _make_components(
            hp_max_power=2.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
        ),
    )

    sector_system, _, _ = _solve_sector_system(sector_system)

    min_power, max_power = sector_system.get_redispatch_power_bounds(
        time_index[0]
    )

    assert min_power == pytest.approx(0.5, abs=TOL)
    assert max_power == pytest.approx(2.0, abs=TOL)


def test_sector_coupled_eom_position_uses_aggregate_demand_sign(
    solved_base_sector_system,
    time_index,
):
    sector_system, _, _ = solved_base_sector_system

    position = sector_system.get_eom_market_positions(time_index[0])[0]

    assert position["unit_id"] == "hs_north"
    assert position["parent_unit_id"] == "hs_north"
    assert position["component_type"] == "aggregate"
    assert position["node"] == "north"
    assert position["physical_power"] == pytest.approx(2.0, abs=TOL)
    assert position["volume"] == pytest.approx(-2.0, abs=TOL)


def test_sector_coupled_redispatch_strategy_full_replacement(
    solved_base_sector_system,
    time_index,
):
    sector_system, _, _ = solved_base_sector_system
    start = time_index[0]
    end = time_index[1]

    strategy = SectorCoupledRedispatchNaiveStrategy()
    bids = strategy.calculate_bids(
        unit=sector_system,
        market_config=None,
        product_tuples=[(start, end, None)],
    )

    assert len(bids) == 1

    bid = bids[0]

    assert bid["unit_id"] == "hs_north"
    assert bid["parent_unit_id"] == "hs_north"
    assert bid["component_type"] == "aggregate"
    assert bid["volume"] == pytest.approx(-2.0, abs=TOL)
    assert bid["min_power"] == pytest.approx(-2.0, abs=TOL)
    assert bid["max_power"] == pytest.approx(0.0, abs=TOL)
    assert bid["p_nom"] == pytest.approx(2.0, abs=TOL)
    assert bid["node"] == "north"
    assert bid["upward_flexibility"] == pytest.approx(2.0, abs=TOL)
    assert bid["downward_flexibility"] == pytest.approx(0.0, abs=TOL)


def test_sector_coupled_redispatch_strategy_respects_heat_requirement(
    time_index,
):
    sector_system = _make_sector_system(
        _make_forecaster(
            time_index,
            heat_demand=6.0,
            electricity_price=40.0,
            natural_gas_price=50.0,
        ),
        _make_components(
            hp_max_power=2.0,
            cop=2.0,
            boiler_max_power=5.0,
            boiler_efficiency=1.0,
        ),
    )

    sector_system, _, _ = _solve_sector_system(sector_system)

    start = time_index[0]
    end = time_index[1]

    strategy = SectorCoupledRedispatchNaiveStrategy()
    bid = strategy.calculate_bids(
        unit=sector_system,
        market_config=None,
        product_tuples=[(start, end, None)],
    )[0]

    assert bid["volume"] == pytest.approx(-2.0, abs=TOL)
    assert bid["min_power"] == pytest.approx(-2.0, abs=TOL)
    assert bid["max_power"] == pytest.approx(-0.5, abs=TOL)
    assert bid["p_nom"] == pytest.approx(2.0, abs=TOL)
    assert bid["upward_flexibility"] == pytest.approx(1.5, abs=TOL)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
