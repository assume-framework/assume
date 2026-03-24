# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.forecaster import BuildingForecaster
from assume.units.building import Building

USE_SOLVER = "appsi_highs"


def _val(x):
    return pyo.value(x)


def _series(values, idx):
    return pd.Series(values, index=idx)


def _solve(instance):
    solver = pyo.SolverFactory(USE_SOLVER)
    return solver.solve(instance)


def _make_building(forecaster, components, prosumer="No"):
    return Building(
        id="A360",
        unit_operator="test_operator",
        bidding_strategies={},
        forecaster=forecaster,
        components=components,
        objective="min_variable_cost",
        flexibility_measure="electricity_price_signal",
        is_prosumer=prosumer,
    )


def _solve_building_opt(building):
    building.setup_model(presolve=True)
    instance = building.model.create_instance()
    instance = building.switch_to_opt(instance)
    results = _solve(instance)
    return building, instance, results


# ---------------------------------------------------------------------
# Shared time horizon
# ---------------------------------------------------------------------
@pytest.fixture
def time_index():
    return pd.date_range("2024-01-01 00:00:00", periods=8, freq="h")


# ---------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------
@pytest.fixture
def base_building_forecaster(time_index):
    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    return BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series([45, 35, 25, 15, 15, 25, 35, 45], time_index),
        load_profile=_series([5, 5, 5, 5, 5, 5, 5, 5], time_index),
        heat_demand=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        pv_profile=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        battery_load_profile=_series([0] * 8, time_index),
        ev_load_profile=_series([0] * 8, time_index),
    )


@pytest.fixture
def building_forecaster_with_ev_cs(time_index):
    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    return BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series([45, 35, 25, 15, 15, 25, 35, 45], time_index),
        load_profile=_series([5, 5, 5, 5, 5, 5, 5, 5], time_index),
        heat_demand=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        pv_profile=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        battery_load_profile=_series([0] * 8, time_index),
        ev_load_profile=_series([0] * 8, time_index),
        A360_electric_vehicle_1_availability_profile=_series(
            [1, 1, 0, 0, 1, 1, 1, 1], time_index
        ),
        A360_electric_vehicle_1_range=_series([0, 0, 5, 5, 0, 0, 0, 0], time_index),
        A360_electric_vehicle_2_availability_profile=_series(
            [1, 1, 1, 1, 1, 0, 0, 1], time_index
        ),
        A360_electric_vehicle_2_range=_series([0, 0, 0, 0, 0, 4, 4, 0], time_index),
        A360_charging_station_1_availability_profile=_series(
            [1, 1, 1, 1, 1, 1, 1, 1], time_index
        ),
        A360_charging_station_2_availability_profile=_series(
            [1, 1, 1, 1, 1, 1, 1, 1], time_index
        ),
    )


# ---------------------------------------------------------------------
# Component fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def heat_pump_components():
    return {
        "heat_pump": {
            "max_power": 10.0,
            "cop": 3.0,
            "min_power": 0.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
        }
    }


@pytest.fixture
def boiler_components():
    return {
        "boiler": {
            "max_power": 10.0,
            "efficiency": 0.9,
            "fuel_type": "electricity",
            "min_power": 0.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
        }
    }


@pytest.fixture
def thermal_storage_components():
    return {
        "thermal_storage": {
            "capacity": 20.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "initial_soc": 0.5,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
            "storage_loss_rate": 0.0,
        }
    }


@pytest.fixture
def pv_components():
    return {
        "pv_plant": {
            "max_power": 10.0,
            "uses_power_profile": "false",
        }
    }


@pytest.fixture
def battery_components():
    return {
        "generic_storage": {
            "capacity": 20.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "initial_soc": 0.5,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
            "storage_loss_rate": 0.0,
        }
    }


@pytest.fixture
def building_components_with_cs():
    return {
        "electric_vehicle_1": {
            "capacity": 50.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "initial_soc": 0.5,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
            "mileage": 1.0,
            "power_flow_directionality": "bidirectional",
        },
        "electric_vehicle_2": {
            "capacity": 40.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "max_power_charge": 8.0,
            "max_power_discharge": 0.0,
            "initial_soc": 0.6,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "ramp_up": 8.0,
            "ramp_down": 8.0,
            "mileage": 1.0,
            "power_flow_directionality": "unidirectional",
        },
        "charging_station_1": {
            "max_power": 10.0,
            "min_power": 0.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
            "power_flow_directionality": "bidirectional",
        },
        "charging_station_2": {
            "max_power": 8.0,
            "min_power": 0.0,
            "ramp_up": 8.0,
            "ramp_down": 8.0,
            "power_flow_directionality": "unidirectional",
        },
    }


@pytest.fixture
def building_components_without_cs():
    return {
        "electric_vehicle_1": {
            "capacity": 50.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "initial_soc": 0.5,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "ramp_up": 10.0,
            "ramp_down": 10.0,
            "mileage": 1.0,
            "power_flow_directionality": "bidirectional",
        },
        "electric_vehicle_2": {
            "capacity": 40.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "max_power_charge": 8.0,
            "max_power_discharge": 0.0,
            "initial_soc": 0.6,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "ramp_up": 8.0,
            "ramp_down": 8.0,
            "mileage": 1.0,
            "power_flow_directionality": "unidirectional",
        },
    }


# ---------------------------------------------------------------------
# Core building functionality
# ---------------------------------------------------------------------
def test_heat_pump_cop_relation(base_building_forecaster, heat_pump_components):
    forecaster = base_building_forecaster
    forecaster.heat_demand = _series([9] * 8, forecaster.index.as_datetimeindex())
    building = _make_building(forecaster, heat_pump_components)
    _, instance, _ = _solve_building_opt(building)

    for t in instance.time_steps:
        hp = instance.dsm_blocks["heat_pump"]
        assert abs(_val(hp.heat_out[t]) - 3.0 * _val(hp.power_in[t])) <= 1e-5


def test_boiler_efficiency_relation(base_building_forecaster, boiler_components):
    forecaster = base_building_forecaster
    forecaster.heat_demand = _series([9] * 8, forecaster.index.as_datetimeindex())
    building = _make_building(forecaster, boiler_components)
    _, instance, _ = _solve_building_opt(building)

    for t in instance.time_steps:
        boiler = instance.dsm_blocks["boiler"]
        assert abs(_val(boiler.heat_out[t]) - 0.9 * _val(boiler.power_in[t])) <= 1e-5


def test_heating_balance_with_heat_pump_and_storage(
    base_building_forecaster, heat_pump_components, thermal_storage_components
):
    forecaster = base_building_forecaster
    forecaster.heat_demand = _series([6] * 8, forecaster.index.as_datetimeindex())

    components = {}
    components.update(heat_pump_components)
    components.update(thermal_storage_components)

    building = _make_building(forecaster, components)
    _, instance, _ = _solve_building_opt(building)

    for t in instance.time_steps:
        hp_heat = _val(instance.dsm_blocks["heat_pump"].heat_out[t])
        ts_dis = _val(instance.dsm_blocks["thermal_storage"].discharge[t])
        ts_ch = _val(instance.dsm_blocks["thermal_storage"].charge[t])
        heat_demand = _val(instance.heat_demand[t])

        assert abs((hp_heat + ts_dis) - (heat_demand + ts_ch)) <= 1e-5


def test_generic_storage_soc_balance(base_building_forecaster, battery_components):
    building = _make_building(base_building_forecaster, battery_components)
    _, instance, _ = _solve_building_opt(building)

    storage = instance.dsm_blocks["generic_storage"]
    ts = list(instance.time_steps)

    eff_c = _val(storage.efficiency_charge)
    eff_d = _val(storage.efficiency_discharge)
    cap = _val(storage.capacity)
    init_soc = _val(storage.initial_soc)
    loss = _val(storage.storage_loss_rate)

    for i, t in enumerate(ts):
        prev_soc = init_soc if i == 0 else _val(storage.soc[ts[i - 1]])
        rhs = (
            prev_soc
            + (
                eff_c * _val(storage.charge[t])
                - (1 / eff_d) * _val(storage.discharge[t])
                - loss * prev_soc * cap
            )
            / cap
        )
        assert abs(_val(storage.soc[t]) - rhs) <= 1e-5


def test_pv_reduces_total_power_input(base_building_forecaster, pv_components):
    forecaster = base_building_forecaster
    forecaster.pv_profile = _series([2] * 8, forecaster.index.as_datetimeindex())
    building = _make_building(forecaster, pv_components)
    _, instance, _ = _solve_building_opt(building)

    for t in instance.time_steps:
        expected = _val(instance.inflex_demand[t]) - _val(
            instance.dsm_blocks["pv_plant"].power[t]
        )
        assert abs(_val(instance.total_power_input[t]) - expected) <= 1e-5


def test_variable_cost_equals_total_power_input_times_price(
    base_building_forecaster, pv_components
):
    forecaster = base_building_forecaster
    forecaster.pv_profile = _series([1] * 8, forecaster.index.as_datetimeindex())
    building = _make_building(forecaster, pv_components)
    _, instance, _ = _solve_building_opt(building)

    for t in instance.time_steps:
        expected = _val(instance.total_power_input[t]) * _val(
            instance.electricity_price[t]
        )
        assert abs(_val(instance.variable_cost[t]) - expected) <= 1e-5


def test_non_prosumer_cannot_export(base_building_forecaster, pv_components):
    forecaster = base_building_forecaster
    forecaster.load_profile = _series([5] * 8, forecaster.index.as_datetimeindex())
    forecaster.pv_profile = _series([2] * 8, forecaster.index.as_datetimeindex())

    building = _make_building(forecaster, pv_components, prosumer="No")
    _, instance, _ = _solve_building_opt(building)

    for t in instance.time_steps:
        assert _val(instance.total_power_input[t]) >= -1e-5


# ---------------------------------------------------------------------
# EV + charging-station fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def solved_building_with_cs(
    building_forecaster_with_ev_cs, building_components_with_cs
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_with_cs
    )
    return _solve_building_opt(building)


@pytest.fixture
def solved_building_without_cs(
    building_forecaster_with_ev_cs, building_components_without_cs
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_without_cs
    )
    return _solve_building_opt(building)


# ---------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------
def test_building_detects_multiple_evs_and_charging_stations(
    building_forecaster_with_ev_cs, building_components_with_cs
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_with_cs
    )
    assert building.has_ev is True
    assert building.has_charging_station is True
    assert len(building.evs) == 2
    assert len(building.charging_stations) == 2


def test_building_without_charging_station_detects_direct_ev_mode(
    building_forecaster_with_ev_cs, building_components_without_cs
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_without_cs
    )
    assert building.has_ev is True
    assert building.has_charging_station is False
    assert len(building.evs) == 2


# ---------------------------------------------------------------------
# EV functionality
# ---------------------------------------------------------------------
def test_ev_usage_matches_unavailability_and_range(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    ts = list(instance.time_steps)

    for ev in [k for k in instance.dsm_blocks if k.startswith("electric_vehicle")]:
        block = instance.dsm_blocks[ev]
        availability = getattr(instance, f"{ev}_availability")
        external_range = getattr(instance, f"{ev}_range")
        mileage = _val(block.mileage)

        for t in ts:
            expected = (1 - _val(availability[t])) * _val(external_range[t]) * mileage
            assert abs(_val(block.usage[t]) - expected) <= 1e-5


def test_ev_soc_balance_with_usage(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    ts = list(instance.time_steps)

    for ev in [k for k in instance.dsm_blocks if k.startswith("electric_vehicle")]:
        block = instance.dsm_blocks[ev]
        eff_c = _val(block.efficiency_charge)
        eff_d = _val(block.efficiency_discharge)
        cap = _val(block.capacity)
        init_soc = _val(block.initial_soc)
        loss = _val(block.storage_loss_rate)

        for i, t in enumerate(ts):
            prev_soc = init_soc if i == 0 else _val(block.soc[ts[i - 1]])
            rhs = (
                prev_soc
                + (
                    eff_c * _val(block.charge[t])
                    - (1 / eff_d) * _val(block.discharge[t])
                    - loss * prev_soc * cap
                    - _val(block.usage[t])
                )
                / cap
            )
            assert abs(_val(block.soc[t]) - rhs) <= 1e-5


def test_unidirectional_ev_never_discharges(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    ev = instance.dsm_blocks["electric_vehicle_2"]
    for t in instance.time_steps:
        assert abs(_val(ev.discharge[t])) <= 1e-5


def test_ev_cannot_charge_when_unavailable_if_connected_via_station(
    solved_building_with_cs,
):
    _, instance, _ = solved_building_with_cs
    for ev in instance.evs:
        availability = getattr(instance, f"{ev}_availability")
        for t in instance.time_steps:
            if _val(availability[t]) < 0.5:
                assert abs(_val(instance.dsm_blocks[ev].charge[t])) <= 1e-5


# ---------------------------------------------------------------------
# Charging-station functionality
# ---------------------------------------------------------------------
def test_charging_station_availability_limits_operation(solved_building_with_cs):
    building, instance, _ = solved_building_with_cs

    for cs in building.charging_stations:
        block = instance.dsm_blocks[cs]
        profile = building.forecaster[f"A360_{cs}_availability_profile"]

        for t in instance.time_steps:
            availability = int(profile.iloc[t])
            if availability == 0:
                assert abs(_val(block.charge[t])) <= 1e-5
                assert abs(_val(block.discharge[t])) <= 1e-5


def test_unidirectional_charging_station_never_discharges(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    cs = instance.dsm_blocks["charging_station_2"]
    for t in instance.time_steps:
        assert abs(_val(cs.discharge[t])) <= 1e-5


def test_charging_station_ramping_constraints(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    ts = list(instance.time_steps)

    for cs_name in ["charging_station_1", "charging_station_2"]:
        cs = instance.dsm_blocks[cs_name]
        ramp_up = _val(cs.ramp_up)
        ramp_down = _val(cs.ramp_down)

        for i in range(1, len(ts)):
            t0 = ts[i - 1]
            t1 = ts[i]

            assert _val(cs.charge[t1]) - _val(cs.charge[t0]) <= ramp_up + 1e-5
            assert _val(cs.charge[t0]) - _val(cs.charge[t1]) <= ramp_down + 1e-5
            assert _val(cs.discharge[t1]) - _val(cs.discharge[t0]) <= ramp_up + 1e-5
            assert _val(cs.discharge[t0]) - _val(cs.discharge[t1]) <= ramp_down + 1e-5


# ---------------------------------------------------------------------
# EV <-> charging-station assignments
# ---------------------------------------------------------------------
def test_one_ev_per_charging_station(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    for cs in instance.charging_stations:
        for t in instance.time_steps:
            total = sum(_val(instance.is_assigned[ev, cs, t]) for ev in instance.evs)
            assert total <= 1 + 1e-5


def test_one_charging_station_per_ev(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    for ev in instance.evs:
        for t in instance.time_steps:
            total = sum(
                _val(instance.is_assigned[ev, cs, t])
                for cs in instance.charging_stations
            )
            assert total <= 1 + 1e-5


def test_assignment_only_if_ev_available(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    for ev in instance.evs:
        availability = getattr(instance, f"{ev}_availability")
        for cs in instance.charging_stations:
            for t in instance.time_steps:
                if _val(availability[t]) < 0.5:
                    assert abs(_val(instance.is_assigned[ev, cs, t])) <= 1e-5


def test_ev_charge_equals_sum_of_charge_assignments(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    for ev in instance.evs:
        for t in instance.time_steps:
            lhs = _val(instance.dsm_blocks[ev].charge[t])
            rhs = sum(
                _val(instance.charge_assignment[ev, cs, t])
                for cs in instance.charging_stations
            )
            assert abs(lhs - rhs) <= 1e-5


def test_charging_station_charge_equals_sum_of_ev_assignments(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    for cs in instance.charging_stations:
        for t in instance.time_steps:
            lhs = _val(instance.dsm_blocks[cs].charge[t])
            rhs = sum(
                _val(instance.charge_assignment[ev, cs, t]) for ev in instance.evs
            )
            assert abs(lhs - rhs) <= 1e-5


# ---------------------------------------------------------------------
# Building power balance
# ---------------------------------------------------------------------
def test_total_power_input_with_charging_stations(solved_building_with_cs):
    building, instance, _ = solved_building_with_cs

    for t in instance.time_steps:
        expected = _val(instance.inflex_demand[t])

        for hp in getattr(building, "heat_pumps", []):
            expected += _val(instance.dsm_blocks[hp].power_in[t])

        for boiler in getattr(building, "boilers", []):
            if hasattr(instance.dsm_blocks[boiler], "power_in"):
                expected += _val(instance.dsm_blocks[boiler].power_in[t])

        for cs in building.charging_stations:
            expected += _val(instance.dsm_blocks[cs].charge[t])
            expected -= _val(instance.dsm_blocks[cs].discharge[t])

        for bat in getattr(building, "battery_storages", []):
            expected += _val(instance.dsm_blocks[bat].charge[t])
            expected -= _val(instance.dsm_blocks[bat].discharge[t])

        for pv in getattr(building, "pv_plants", []):
            expected -= _val(instance.dsm_blocks[pv].power[t])

        assert abs(_val(instance.total_power_input[t]) - expected) <= 1e-5


def test_total_power_input_without_charging_stations(solved_building_without_cs):
    building, instance, _ = solved_building_without_cs

    for t in instance.time_steps:
        expected = _val(instance.inflex_demand[t])

        for hp in getattr(building, "heat_pumps", []):
            expected += _val(instance.dsm_blocks[hp].power_in[t])

        for boiler in getattr(building, "boilers", []):
            if hasattr(instance.dsm_blocks[boiler], "power_in"):
                expected += _val(instance.dsm_blocks[boiler].power_in[t])

        for ev in building.evs:
            expected += _val(instance.dsm_blocks[ev].charge[t])
            expected -= _val(instance.dsm_blocks[ev].discharge[t])

        for bat in getattr(building, "battery_storages", []):
            expected += _val(instance.dsm_blocks[bat].charge[t])
            expected -= _val(instance.dsm_blocks[bat].discharge[t])

        for pv in getattr(building, "pv_plants", []):
            expected -= _val(instance.dsm_blocks[pv].power[t])

        assert abs(_val(instance.total_power_input[t]) - expected) <= 1e-5


# ---------------------------------------------------------------------
# Direct-grid fallback
# ---------------------------------------------------------------------
def test_without_charging_stations_no_assignment_variables_exist(
    solved_building_without_cs,
):
    _, instance, _ = solved_building_without_cs
    assert not hasattr(instance, "is_assigned")
    assert not hasattr(instance, "charge_assignment")


def test_without_charging_stations_evs_directly_affect_power_balance(
    solved_building_without_cs,
):
    building, instance, _ = solved_building_without_cs

    for t in instance.time_steps:
        ev_net = sum(
            _val(instance.dsm_blocks[ev].charge[t])
            - _val(instance.dsm_blocks[ev].discharge[t])
            for ev in building.evs
        )
        expected = _val(instance.inflex_demand[t]) + ev_net
        assert abs(_val(instance.total_power_input[t]) - expected) <= 1e-5


def test_building_forecaster_getitem_returns_component_profile(
    building_forecaster_with_ev_cs,
):
    prof = building_forecaster_with_ev_cs[
        "A360_electric_vehicle_1_availability_profile"
    ]
    rng = building_forecaster_with_ev_cs["A360_electric_vehicle_1_range"]

    assert len(prof) == 8
    assert len(rng) == 8
    assert prof.iloc[2] == 0
    assert rng.iloc[2] == 5


def test_building_forecaster_getitem_missing_key_raises_keyerror(
    building_forecaster_with_ev_cs,
):
    with pytest.raises(KeyError):
        _ = building_forecaster_with_ev_cs["A360_non_existing_profile"]


def test_building_classifies_components_by_prefix(
    building_forecaster_with_ev_cs,
    building_components_with_cs,
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_with_cs
    )

    assert building.evs == ["electric_vehicle_1", "electric_vehicle_2"]
    assert building.charging_stations == ["charging_station_1", "charging_station_2"]
    assert building.has_ev is True
    assert building.has_charging_station is True


def test_building_no_charging_station_mode_has_no_assignment_vars(
    building_forecaster_with_ev_cs,
    building_components_without_cs,
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_without_cs
    )
    _, instance, _ = _solve_building_opt(building)

    assert building.has_charging_station is False
    assert not hasattr(instance, "is_assigned")
    assert not hasattr(instance, "charge_assignment")


def test_building_with_charging_station_mode_has_assignment_vars(
    building_forecaster_with_ev_cs,
    building_components_with_cs,
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_with_cs
    )
    _, instance, _ = _solve_building_opt(building)

    assert building.has_charging_station is True
    assert hasattr(instance, "is_assigned")
    assert hasattr(instance, "charge_assignment")


def test_electricity_price_signal_replaces_building_price_signal(
    building_forecaster_with_ev_cs,
    building_components_without_cs,
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_without_cs
    )

    # setup model with flex logic included
    building.setup_model(presolve=True)

    for t in building.model.time_steps:
        assert (
            abs(
                _val(building.model.electricity_price[t])
                - float(building.forecaster.electricity_price_flex.iloc[t])
            )
            <= 1e-5
        )


def test_variable_cost_uses_flex_price_signal_in_building_mode(
    building_forecaster_with_ev_cs,
    building_components_without_cs,
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_without_cs
    )
    _, instance, _ = _solve_building_opt(building)

    for t in instance.time_steps:
        expected = _val(instance.total_power_input[t]) * _val(
            instance.electricity_price[t]
        )
        assert abs(_val(instance.variable_cost[t]) - expected) <= 1e-5


def test_assignment_variables_have_expected_dimensions(
    solved_building_with_cs,
):
    _, instance, _ = solved_building_with_cs

    for ev in instance.evs:
        for cs in instance.charging_stations:
            for t in instance.time_steps:
                _ = instance.is_assigned[ev, cs, t]
                _ = instance.charge_assignment[ev, cs, t]


if __name__ == "__main__":
    pytest.main(["-s", __file__])
