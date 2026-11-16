# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

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
        objective="min_variable_cost",
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
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal


def test_sector_coupled_base_case_uses_heat_pump(solved_base_sector_system):
    _, instance, _ = solved_base_sector_system

    hp = instance.dsm_blocks["heat_pump"]
    boiler = instance.dsm_blocks["boiler_gas"]

    for t in instance.time_steps:
        assert _val(hp.power_in[t]) == pytest.approx(2.0, abs=TOL)
        assert _val(hp.heat_out[t]) == pytest.approx(4.0, abs=TOL)
        assert _val(boiler.natural_gas_in[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(instance.total_power_input[t]) == pytest.approx(2.0, abs=TOL)


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
        assert _val(instance.natural_gas_price[t]) == pytest.approx(50.0, abs=TOL)


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
        assert _val(instance.total_power_input[t]) == pytest.approx(0.0, abs=TOL)


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
        assert _val(instance.total_power_input[t]) == pytest.approx(2.0, abs=TOL)

        heat_supply = _val(hp.heat_out[t]) + _val(boiler.heat_out[t])
        assert heat_supply == pytest.approx(_val(instance.heat_demand[t]), abs=TOL)


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
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

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
        assert _val(tes.charge[t]) == pytest.approx(expected_charge[t], abs=TOL)
        assert _val(tes.discharge[t]) == pytest.approx(expected_discharge[t], abs=TOL)
        assert _val(tes.soc[t]) == pytest.approx(expected_soc[t], abs=TOL)
        assert _val(tes.power_in[t]) == pytest.approx(0.0, abs=TOL)
        assert _val(boiler.heat_out[t]) == pytest.approx(0.0, abs=TOL)

        heat_supply = (
            _val(hp.heat_out[t]) + _val(boiler.heat_out[t]) + _val(tes.discharge[t])
        )
        heat_required = _val(instance.heat_demand[t]) + _val(tes.charge[t])
        assert heat_supply == pytest.approx(heat_required, abs=TOL)


def test_sector_coupled_redispatch_bounds_full_boiler_replacement(
    solved_base_sector_system,
    time_index,
):
    sector_system, _, _ = solved_base_sector_system

    min_power, max_power = sector_system.get_redispatch_power_bounds(time_index[0])

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

    min_power, max_power = sector_system.get_redispatch_power_bounds(time_index[0])

    assert min_power == pytest.approx(0.5, abs=TOL)
    assert max_power == pytest.approx(2.0, abs=TOL)


def test_sector_coupled_eom_position_uses_aggregate_demand_sign(
    solved_base_sector_system,
    time_index,
):
    sector_system, _, _ = solved_base_sector_system

    position = sector_system.get_eom_market_position(time_index[0])

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

    assert bid["volume"] == pytest.approx(-2.0, abs=TOL)
    assert bid["min_power"] == pytest.approx(-2.0, abs=TOL)
    assert bid["max_power"] == pytest.approx(0.0, abs=TOL)
    assert bid["p_nom"] == pytest.approx(2.0, abs=TOL)
    assert bid["node"] == "north"
    # Market sign: up to 2 MW less consumption, no more consumption possible.
    assert bid["max_power"] - bid["volume"] == pytest.approx(2.0, abs=TOL)
    assert bid["volume"] - bid["min_power"] == pytest.approx(0.0, abs=TOL)


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
    assert bid["max_power"] - bid["volume"] == pytest.approx(1.5, abs=TOL)


def test_heat_hydrogen_h2_chp_couples_heat_hydrogen_and_electricity(time_index):
    """Validate the integrated electrolyser -> H2 CHP -> heat chain."""
    heat_demand = 5.0
    hydrogen_demand = 0.0

    forecaster = SectorCoupledForecaster(
        index=time_index,
        carrier_demands={
            "heat": _series(heat_demand, time_index),
            "hydrogen": _series(hydrogen_demand, time_index),
        },
        fuel_prices={},
        market_prices={
            "EOM": _series(40.0, time_index),
        },
    )

    components = {
        "electrolyser": {
            "max_power": 20.0,
            "min_power": 0.0,
            "efficiency": 0.8,
            "ramp_up": 20.0,
            "ramp_down": 20.0,
        },
        "chp_h2": {
            "max_power": 4.0,
            "min_power": 0.0,
            "electrical_efficiency": 0.4,
            "thermal_efficiency": 0.5,
            "fuel_type": "hydrogen_gas",
            "ramp_up": 4.0,
            "ramp_down": 4.0,
            "additional_cost": 0.0,
        },
    }

    sector_system = SectorCoupledSystem(
        id="hh_north",
        unit_operator="test_operator",
        bidding_strategies={},
        forecaster=forecaster,
        components=components,
        system_type="heat_hydrogen",
        objective="min_variable_cost",
        node="north",
    )

    sector_system, instance, results = _solve_sector_system(sector_system)

    assert results.solver.status == pyo.SolverStatus.ok
    assert results.solver.termination_condition == pyo.TerminationCondition.optimal

    electrolyser = instance.dsm_blocks["electrolyser"]
    chp = instance.dsm_blocks["chp_h2"]

    # 5 MW_th heat demand with eta_th=0.5 requires 10 MW_H2.
    # The CHP simultaneously produces 4 MW_el at eta_el=0.4.
    # Producing 10 MW_H2 at eta_ely=0.8 requires 12.5 MW_el.
    # Parent net grid exchange = 12.5 - 4.0 = +8.5 MW import.
    for t in instance.time_steps:
        assert _val(chp.heat_out[t]) == pytest.approx(5.0, abs=TOL)
        assert _val(chp.fuel_in[t]) == pytest.approx(10.0, abs=TOL)
        assert _val(chp.hydrogen_in[t]) == pytest.approx(10.0, abs=TOL)
        assert _val(chp.power_out[t]) == pytest.approx(4.0, abs=TOL)

        assert _val(electrolyser.hydrogen_out[t]) == pytest.approx(10.0, abs=TOL)
        assert _val(electrolyser.power_in[t]) == pytest.approx(12.5, abs=TOL)

        heat_supply = _val(sector_system._carrier_supply_expr(instance, "heat", t))
        hydrogen_supply = _val(
            sector_system._carrier_supply_expr(instance, "hydrogen", t)
        )

        assert heat_supply == pytest.approx(_val(instance.heat_demand[t]), abs=TOL)
        assert hydrogen_supply == pytest.approx(
            _val(instance.hydrogen_demand[t]), abs=TOL
        )
        assert _val(instance.total_power_input[t]) == pytest.approx(8.5, abs=TOL)

    # H2 for the CHP is internal, so no external hydrogen price is required.
    assert not hasattr(instance, "hydrogen_gas_price")

    # Market sign convention: +8.5 MW physical import -> -8.5 MW EOM demand bid.
    position = sector_system.get_eom_market_position(time_index[0])
    assert position["node"] == "north"
    assert position["physical_power"] == pytest.approx(8.5, abs=TOL)
    assert position["volume"] == pytest.approx(-8.5, abs=TOL)


if __name__ == "__main__":
    pytest.main(["-s", __file__])


# Joint redispatch bounds and following accepted redispatch


def _make_storage_heat_system(heat_pump_max_power=4.0):
    """
    Heat pump (COP 1) plus a full 2 MWh thermal storage, 2 MW heat demand.

    The DA plan runs the heat pump at 2 MW in all four hours. Each single hour
    could drop to 0 MW by emptying the storage, but two hours together can only
    drop to 1 MW each, because the storage holds 2 MWh and must be refilled.
    """
    index = pd.date_range("2023-01-01", periods=4, freq="h")
    forecaster = SectorCoupledForecaster(
        index=index,
        carrier_demands={"heat": _series(2.0, index)},
        fuel_prices={"natural gas": _series(50.0, index)},
        market_prices={"EOM": pd.Series([10.0, 11.0, 12.0, 13.0], index=index)},
    )
    components = {
        "heat_pump": {
            "max_power": heat_pump_max_power,
            "min_power": 0.0,
            "cop": 1.0,
            "ramp_up": 4.0,
            "ramp_down": 4.0,
        },
        "thermal_storage": {
            "storage_type": "short-term",
            "capacity": 2.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 4.0,
            "max_power_discharge": 4.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "initial_soc": 1.0,
            "ramp_up": 4.0,
            "ramp_down": 4.0,
            "storage_loss_rate": 0.0,
        },
    }
    unit = _make_sector_system(forecaster, components)
    unit.setup_model(presolve=True)
    unit.prepare_market_state()
    return unit


def _can_deliver(unit, consumption):
    """Solve the unit with its consumption fixed in the given hours."""
    starts = sorted(consumption)
    instance, _, offset = unit._redispatch_instance(starts[0])
    for start in starts:
        t = unit._index_position(start) - offset
        instance.total_power_input[t].fix(consumption[start])
    results = pyo.SolverFactory(USE_SOLVER).solve(instance, load_solutions=False)
    return results.solver.termination_condition == pyo.TerminationCondition.optimal


def test_single_hour_bounds_overstate_joint_flexibility():
    unit = _make_storage_heat_system()
    starts = list(unit.index[:2])

    single = {start: unit.get_redispatch_power_bounds(start) for start in starts}
    for start in starts:
        assert single[start][0] == pytest.approx(0.0, abs=TOL)

    # Using the single-hour minimum in both hours is not deliverable.
    assert not _can_deliver(unit, {start: single[start][0] for start in starts})


def test_joint_redispatch_bounds_hold_for_all_hours_together():
    unit = _make_storage_heat_system()
    starts = list(unit.index[:2])

    bounds = unit.get_joint_redispatch_power_bounds(starts)

    for start in starts:
        min_consumption, max_consumption = bounds[start]
        assert min_consumption == pytest.approx(1.0, abs=1e-4)
        # The storage is full, so the heat pump cannot consume more.
        assert max_consumption == pytest.approx(2.0, abs=TOL)

    assert _can_deliver(unit, {start: bounds[start][0] for start in starts})
    # Mixed choices inside the bounds are deliverable as well.
    assert _can_deliver(unit, {starts[0]: bounds[starts[0]][0], starts[1]: 2.0})


def test_single_hour_room_keeps_other_product_hours_at_da():
    """
    With a 3 MW heat pump, cutting hour 0 empties up to 2 MWh of storage that
    must be refilled at 1 MW/h. If hours 1 and 2 keep their DA consumption
    (the market moved hour 0 only), just hour 3 is left to refill: 1 MW room.
    """
    unit = _make_storage_heat_system(heat_pump_max_power=3.0)
    products = tuple(unit.index[:3])

    free_min, _ = unit.get_redispatch_power_bounds(unit.index[0])
    fixed_min, _ = unit.get_redispatch_power_bounds(unit.index[0], products)

    assert free_min == pytest.approx(0.0, abs=TOL)
    assert fixed_min == pytest.approx(1.0, abs=TOL)

    # Moving only hour 0 to the offered bound is deliverable.
    bounds = unit.get_joint_redispatch_power_bounds(list(products))
    alone = {start: float(unit.da_power.at[start]) for start in products}
    alone[products[0]] = bounds[products[0]][0]
    assert _can_deliver(unit, alone)


def test_redispatch_strategy_offers_joint_bounds():
    unit = _make_storage_heat_system()
    starts = list(unit.index[:2])
    products = [(start, start + pd.Timedelta(hours=1), None) for start in starts]

    bids = SectorCoupledRedispatchNaiveStrategy().calculate_bids(
        unit=unit, market_config=None, product_tuples=products
    )

    assert len(bids) == 2
    for bid in bids:
        assert bid["price"] == bid["price_up"] == pytest.approx(12.0, abs=1e-4)
        assert bid["volume"] == pytest.approx(-2.0, abs=TOL)
        # Market sign: less consumption is upward, up to -1 MW.
        assert bid["max_power"] == pytest.approx(-1.0, abs=1e-4)
        assert bid["min_power"] == pytest.approx(-2.0, abs=TOL)


def _redispatch_config():
    from assume.common.market_objects import MarketConfig

    return MarketConfig(market_id="redispatch", market_mechanism="redispatch")


def _redispatch_orderbook(unit, accepted):
    return [
        {
            "start_time": start,
            "end_time": start + pd.Timedelta(hours=1),
            "only_hours": None,
            "price": 0.0,
            "volume": -float(unit.da_power.at[start]),
            "accepted_volume": volume,
            "accepted_price": 0.0,
        }
        for start, volume in accepted.items()
    ]


def test_accepted_redispatch_updates_the_reference():
    unit = _make_storage_heat_system()
    starts = list(unit.index[:2])

    # Accept 1 MW less consumption in both hours.
    unit.set_dispatch_plan(
        _redispatch_config(), _redispatch_orderbook(unit, {s: 1.0 for s in starts})
    )

    assert list(unit.da_power.iloc[:2]) == pytest.approx([1.0, 1.0], abs=TOL)
    # The storage gave its 2 MWh and is refilled in the later hours.
    assert unit.da_power.iloc[2:].sum() == pytest.approx(6.0, abs=TOL)
    soc = unit.da_component_state["thermal_storage"]["soc"]  # share of capacity
    assert soc.iloc[1] == pytest.approx(0.0, abs=TOL)
    assert soc.iloc[3] == pytest.approx(1.0, abs=TOL)


def test_redispatch_that_cannot_be_followed_keeps_reference(caplog):
    unit = _make_storage_heat_system()
    starts = list(unit.index[:2])
    before = unit.da_power.copy()

    # 2 MW less in both hours would need 4 MWh from a 2 MWh storage.
    with caplog.at_level("WARNING"):
        unit.set_dispatch_plan(
            _redispatch_config(), _redispatch_orderbook(unit, {s: 2.0 for s in starts})
        )

    assert "cannot follow the accepted redispatch" in caplog.text
    pd.testing.assert_series_equal(unit.da_power, before)


def test_eom_orders_do_not_change_the_reference():
    from assume.common.market_objects import MarketConfig

    unit = _make_storage_heat_system()
    before = unit.da_power.copy()

    unit.set_dispatch_plan(
        MarketConfig(market_id="EOM"),
        _redispatch_orderbook(unit, {unit.index[0]: -2.0}),
    )

    pd.testing.assert_series_equal(unit.da_power, before)


# Electricity value and redispatch prices


def _make_hydrogen_export_system(prices):
    """Electrolyser + hydrogen storage + fuel cell, no hydrogen demand."""
    index = pd.date_range("2023-01-01", periods=len(prices), freq="h")
    forecaster = SectorCoupledForecaster(
        index=index,
        carrier_demands={"hydrogen": _series(0.0, index)},
        fuel_prices={"natural gas": _series(50.0, index)},
        market_prices={"EOM": pd.Series(prices, index=index, dtype=float)},
    )
    components = {
        "electrolyser": {
            "max_power": 4.0,
            "min_power": 0.0,
            "efficiency": 0.8,
            "ramp_up": 4.0,
            "ramp_down": 4.0,
        },
        "hydrogen_seasonal_storage": {
            "storage_type": "short-term",
            "capacity": 10.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 4.0,
            "max_power_discharge": 4.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "initial_soc": 0.0,
            "storage_loss_rate": 0.0,
        },
        "fuel_cell": {"max_power": 2.0, "efficiency": 0.5},
    }
    unit = SectorCoupledSystem(
        id="h2",
        unit_operator="op",
        bidding_strategies={},
        forecaster=forecaster,
        components=components,
        system_type="hydrogen",
        node="north",
    )
    unit.setup_model(presolve=True)
    unit.prepare_market_state()
    return unit


def test_generated_electricity_is_valued_at_the_price():
    """
    Cheap hours first, expensive hours later: the system stores hydrogen and
    exports through the fuel cell, because its cost is price x net exchange.
    """
    unit = _make_hydrogen_export_system([10.0, 10.0, 500.0, 500.0])

    assert list(unit.da_power) == pytest.approx([4.0, 4.0, -1.2, -2.0], abs=TOL)

    instance = unit._redispatch_instance(unit.index[0])[0]
    unit._solve_ok(instance)
    for t in instance.time_steps:
        assert pyo.value(instance.variable_cost[t]) == pytest.approx(
            pyo.value(instance.electricity_price[t] * instance.total_power_input[t]),
            abs=1e-4,
        )


def test_redispatch_prices_come_from_the_unit_costs():
    """
    Cutting hour 0 means refilling the storage later. With hour 1 kept at DA,
    the cheapest refill is hour 2 at 12 EUR/MWh. There is no room for more
    consumption, so the down price is the hour's own price.
    """
    unit = _make_storage_heat_system()
    starts = list(unit.index[:2])

    bounds = unit.get_joint_redispatch_power_bounds(starts)
    prices = unit.get_redispatch_prices(bounds)

    assert prices[starts[0]] == pytest.approx((12.0, 10.0), abs=1e-4)
    assert prices[starts[1]] == pytest.approx((12.0, 11.0), abs=1e-4)


def test_export_hours_are_not_offered_for_free():
    """
    Exporting less at hour 2 keeps 4 MWh of hydrogen for hour 4 (400 EUR/MWh,
    2 MWh of electricity). Averaged over the offered range, the unit pays
    2 x 400 EUR per offered MWh.
    """
    unit = _make_hydrogen_export_system([10.0, 10.0, 500.0, 500.0, 400.0])
    starts = list(unit.index[2:4])
    base = float(unit.da_power.at[starts[0]])
    assert base == pytest.approx(-2.0, abs=TOL)

    bounds = unit.get_joint_redispatch_power_bounds(starts)
    price_up, price_down = unit.get_redispatch_prices(bounds)[starts[0]]

    # No room for more export (the hydrogen is used up): the hour's price.
    assert bounds[starts[0]][0] == pytest.approx(base, abs=1e-4)
    assert price_up == pytest.approx(500.0, abs=1e-4)
    room = bounds[starts[0]][1] - base
    assert price_down == pytest.approx(2 * 400.0 / room, abs=1e-3)
    assert price_down > 0


def test_old_flexibility_columns_are_accepted_and_ignored():
    """CSV files may still carry flexibility_measure / cost_tolerance."""
    index = pd.date_range("2023-01-01", periods=4, freq="h")
    forecaster = SectorCoupledForecaster(
        index=index,
        carrier_demands={"heat": _series(2.0, index)},
        fuel_prices={"natural gas": _series(50.0, index)},
        market_prices={"EOM": _series(10.0, index)},
    )
    unit = SectorCoupledSystem(
        id="hs",
        unit_operator="op",
        bidding_strategies={},
        forecaster=forecaster,
        components={"heat_pump": {"max_power": 4.0, "min_power": 0.0, "cop": 1.0}},
        system_type="heat",
        node="north",
        flexibility_measure="electricity_price_signal",
        cost_tolerance=50,
    )

    assert unit.flexibility_measure == "cost_based_load_shift"
