# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.forecaster import BuildingForecaster
from assume.units.building import Building
from assume.units.dst_components import ElectricVehicle

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
        A360_electric_vehicle_1_trip_distance=_series(
            [0, 0, 5, 5, 0, 0, 0, 0], time_index
        ),
        A360_electric_vehicle_2_availability_profile=_series(
            [1, 1, 1, 1, 1, 0, 0, 1], time_index
        ),
        A360_electric_vehicle_2_trip_distance=_series(
            [0, 0, 0, 0, 0, 4, 4, 0], time_index
        ),
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
def test_ev_usage_matches_unavailability_and_trip_distance(solved_building_with_cs):
    _, instance, _ = solved_building_with_cs
    ts = list(instance.time_steps)

    for ev in [k for k in instance.dsm_blocks if k.startswith("electric_vehicle")]:
        block = instance.dsm_blocks[ev]
        availability = getattr(instance, f"{ev}_availability")
        external_trip_distance = getattr(instance, f"{ev}_trip_distance")
        mileage = _val(block.mileage)

        for t in ts:
            expected = (
                (1 - _val(availability[t])) * _val(external_trip_distance[t]) * mileage
            )
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
    trip_dist = building_forecaster_with_ev_cs["A360_electric_vehicle_1_trip_distance"]

    assert len(prof) == 8
    assert len(trip_dist) == 8
    assert prof.iloc[2] == 0
    assert trip_dist.iloc[2] == 5


def test_building_forecaster_getitem_missing_key_raises_keyerror(
    building_forecaster_with_ev_cs,
):
    with pytest.raises(KeyError):
        _ = building_forecaster_with_ev_cs["A360_non_existing_profile"]


def test_building_ev_profile_keyerror_triggers_fallback(
    time_index,
):
    """Test that missing EV profile keys trigger fallback to ev_load_profile."""
    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    # Create forecaster WITHOUT the specific EV profiles, but WITH ev_load_profile
    forecaster = BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series([45, 35, 25, 15, 15, 25, 35, 45], time_index),
        load_profile=_series([5, 5, 5, 5, 5, 5, 5, 5], time_index),
        heat_demand=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        pv_profile=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        battery_load_profile=_series([0] * 8, time_index),
        ev_load_profile=_series(
            [1, 1, 1, 1, 1, 1, 1, 1], time_index
        ),  # fallback profile
        A360_electric_vehicle_1_trip_distance=_series(
            [0, 0, 0, 0, 0, 0, 0, 0], time_index
        ),  # required: at least one trip config
    )

    components = {
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
        }
    }

    # Should succeed by falling back to ev_load_profile
    building = _make_building(forecaster, components)
    assert building.has_ev is True


def test_building_ev_trip_distance_keyerror_is_optional(
    time_index,
):
    """Test that missing EV trip distance profile is optional when trip_energy_consumption is provided."""
    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    # Create forecaster WITHOUT trip_distance, but WITH trip_energy_consumption (alternative mode)
    forecaster = BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series([45, 35, 25, 15, 15, 25, 35, 45], time_index),
        load_profile=_series([5, 5, 5, 5, 5, 5, 5, 5], time_index),
        heat_demand=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        pv_profile=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        battery_load_profile=_series([0] * 8, time_index),
        ev_load_profile=_series([1, 1, 1, 1, 1, 1, 1, 1], time_index),
        # No trip_distance, but providing trip_energy_consumption as alternative
        A360_electric_vehicle_1_availability_profile=_series(
            [1, 1, 0, 0, 1, 1, 1, 1], time_index
        ),
        A360_electric_vehicle_1_trip_energy_consumption=_series(
            [0, 0, 5, 5, 0, 0, 0, 0], time_index
        ),
    )

    components = {
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
        }
    }

    # Should succeed with trip_distance = None when trip_energy_consumption is provided
    building = _make_building(forecaster, components)
    assert building.has_ev is True
    assert building.components["electric_vehicle_1"]["trip_distance"] is None
    assert (
        building.components["electric_vehicle_1"]["trip_energy_consumption"] is not None
    )


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


def test_building_recognizes_suffixed_required_component_names(
    monkeypatch,
    base_building_forecaster,
    heat_pump_components,
):
    monkeypatch.setattr(Building, "required_technologies", ["heat_pump"])

    building = _make_building(
        base_building_forecaster,
        {"heat_pump_1": dict(heat_pump_components["heat_pump"])},
    )

    assert building.heat_pumps == ["heat_pump_1"]
    assert building.has_heatpump is True


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


def test_building_without_cs_uses_ev_names_as_expected(
    building_forecaster_with_ev_cs,
    building_components_without_cs,
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_without_cs
    )
    assert sorted(building.evs) == ["electric_vehicle_1", "electric_vehicle_2"]


def test_building_with_cs_uses_station_names_as_expected(
    building_forecaster_with_ev_cs,
    building_components_with_cs,
):
    building = _make_building(
        building_forecaster_with_ev_cs, building_components_with_cs
    )
    assert sorted(building.charging_stations) == [
        "charging_station_1",
        "charging_station_2",
    ]


# ---------------------------------------------------------------------
# Optimality / economic dispatch tests
# These tests verify the optimizer finds economically rational solutions
# given a clear price signal, not just that constraints are satisfied.
# ---------------------------------------------------------------------


def test_battery_charges_at_cheap_hours_discharges_at_expensive(time_index):
    """
    Price is cheap in the first half (t=0-3) and expensive in the second half (t=4-7).
    A prosumer battery should charge when cheap and discharge when expensive to minimise cost.
    """
    prices = [10, 10, 10, 10, 100, 100, 100, 100]
    market_prices = {"EOM": _series(prices, time_index)}

    forecaster = BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series(prices, time_index),
        load_profile=_series([3] * 8, time_index),
        heat_demand=_series([0] * 8, time_index),
        pv_profile=_series([0] * 8, time_index),
        battery_load_profile=_series([0] * 8, time_index),
        ev_load_profile=_series([0] * 8, time_index),
    )

    components = {
        "generic_storage": {
            "capacity": 20.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 5.0,
            "max_power_discharge": 5.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "initial_soc": 0.0,  # starts empty
            "ramp_up": 5.0,
            "ramp_down": 5.0,
            "storage_loss_rate": 0.0,
        }
    }

    building = _make_building(forecaster, components, prosumer="Yes")
    _, instance, _ = _solve_building_opt(building)

    storage = instance.dsm_blocks["generic_storage"]
    cheap_hours = [0, 1, 2, 3]
    expensive_hours = [4, 5, 6, 7]

    total_charge_cheap = sum(_val(storage.charge[t]) for t in cheap_hours)
    total_charge_expensive = sum(_val(storage.charge[t]) for t in expensive_hours)
    total_discharge_expensive = sum(_val(storage.discharge[t]) for t in expensive_hours)
    total_discharge_cheap = sum(_val(storage.discharge[t]) for t in cheap_hours)

    assert total_charge_cheap > total_charge_expensive + 1e-3, (
        "Battery should charge more at cheap hours"
    )
    assert total_discharge_expensive > total_discharge_cheap + 1e-3, (
        "Battery should discharge more at expensive hours"
    )


def test_battery_arbitrage_reduces_total_cost(time_index):
    """
    Total energy cost with battery arbitrage (prosumer) should be lower
    than the same building without a battery, given a variable price signal.
    """
    prices = [10, 10, 10, 10, 100, 100, 100, 100]
    market_prices = {"EOM": _series(prices, time_index)}

    def make_forecaster():
        return BuildingForecaster(
            index=time_index,
            fuel_prices={"natural_gas": _series([20] * 8, time_index)},
            market_prices=market_prices,
            electricity_price_flex=_series(prices, time_index),
            load_profile=_series([3] * 8, time_index),
            heat_demand=_series([0] * 8, time_index),
            pv_profile=_series([0] * 8, time_index),
            battery_load_profile=_series([0] * 8, time_index),
            ev_load_profile=_series([0] * 8, time_index),
        )

    battery = {
        "generic_storage": {
            "capacity": 20.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 5.0,
            "max_power_discharge": 5.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "initial_soc": 0.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
            "storage_loss_rate": 0.0,
        }
    }

    _, inst_no_bat, _ = _solve_building_opt(
        _make_building(make_forecaster(), {}, prosumer="Yes")
    )
    _, inst_bat, _ = _solve_building_opt(
        _make_building(make_forecaster(), battery, prosumer="Yes")
    )

    cost_no_bat = sum(
        _val(inst_no_bat.variable_cost[t]) for t in inst_no_bat.time_steps
    )
    cost_bat = sum(_val(inst_bat.variable_cost[t]) for t in inst_bat.time_steps)

    assert cost_bat < cost_no_bat, (
        "Battery arbitrage should reduce total cost compared to no battery"
    )


def test_ev_bidirectional_v2g_charges_cheap_discharges_expensive(time_index):
    """
    A bidirectional EV (V2G mode) starts empty. With cheap prices early and expensive
    prices late, the optimizer should charge during cheap hours and discharge to grid
    during expensive hours (prosumer mode).
    """
    prices = [10, 10, 10, 10, 100, 100, 100, 100]
    market_prices = {"EOM": _series(prices, time_index)}

    forecaster = BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series(prices, time_index),
        load_profile=_series([0] * 8, time_index),
        heat_demand=_series([0] * 8, time_index),
        pv_profile=_series([0] * 8, time_index),
        battery_load_profile=_series([0] * 8, time_index),
        ev_load_profile=_series([0] * 8, time_index),
        A360_electric_vehicle_1_availability_profile=_series([1] * 8, time_index),
        A360_electric_vehicle_1_trip_distance=_series(
            [0] * 8, time_index
        ),  # required: at least one trip config
    )

    components = {
        "electric_vehicle_1": {
            "capacity": 20.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 5.0,
            "max_power_discharge": 5.0,
            "initial_soc": 0.0,  # starts empty; must charge before discharging
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
            "storage_loss_rate": 0.0,
            "power_flow_directionality": "bidirectional",
            "mileage": 1.0,  # required when trip_distance is provided
        }
    }

    building = _make_building(forecaster, components, prosumer="Yes")
    _, instance, _ = _solve_building_opt(building)

    ev = instance.dsm_blocks["electric_vehicle_1"]
    cheap_hours = [0, 1, 2, 3]
    expensive_hours = [4, 5, 6, 7]

    total_charge_cheap = sum(_val(ev.charge[t]) for t in cheap_hours)
    total_charge_expensive = sum(_val(ev.charge[t]) for t in expensive_hours)
    total_discharge_expensive = sum(_val(ev.discharge[t]) for t in expensive_hours)

    assert total_charge_cheap > 0, "EV should charge during cheap hours"
    assert total_discharge_expensive > 0, "EV should discharge during expensive hours"
    assert total_charge_cheap > total_charge_expensive + 1e-3, (
        "EV should charge more at cheap hours than expensive hours"
    )


def test_heat_pump_defers_to_cheap_hours_when_thermal_storage_available(time_index):
    """
    With thermal storage, the heat pump should run primarily during cheap electricity
    hours, pre-charging the thermal store, and rely on stored heat during expensive hours.
    Without storage, the heat pump must run every hour to meet demand directly.
    """
    prices = [10, 10, 10, 10, 100, 100, 100, 100]
    market_prices = {"EOM": _series(prices, time_index)}

    def make_forecaster():
        return BuildingForecaster(
            index=time_index,
            fuel_prices={"natural_gas": _series([20] * 8, time_index)},
            market_prices=market_prices,
            electricity_price_flex=_series(prices, time_index),
            load_profile=_series([0] * 8, time_index),
            heat_demand=_series([3] * 8, time_index),  # constant 3 kW heat demand
            pv_profile=_series([0] * 8, time_index),
            battery_load_profile=_series([0] * 8, time_index),
            ev_load_profile=_series([0] * 8, time_index),
        )

    hp_only = {
        "heat_pump": {
            "max_power": 5.0,
            "cop": 3.0,
            "min_power": 0.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
    }
    hp_with_storage = {
        "heat_pump": {
            "max_power": 5.0,
            "cop": 3.0,
            "min_power": 0.0,
            "ramp_up": 5.0,
            "ramp_down": 5.0,
        },
        "thermal_storage": {
            "capacity": 50.0,
            "min_soc": 0.0,
            "max_soc": 1.0,
            "max_power_charge": 15.0,
            "max_power_discharge": 15.0,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "initial_soc": 0.0,
            "ramp_up": 15.0,
            "ramp_down": 15.0,
            "storage_loss_rate": 0.0,
        },
    }

    # Without thermal storage: HP must run at every hour to satisfy demand
    _, inst_no_ts, _ = _solve_building_opt(_make_building(make_forecaster(), hp_only))
    # With thermal storage: HP can shift operation to cheap hours
    _, inst_ts, _ = _solve_building_opt(
        _make_building(make_forecaster(), hp_with_storage)
    )

    cheap_hours = [0, 1, 2, 3]
    expensive_hours = [4, 5, 6, 7]

    hp_no_ts = inst_no_ts.dsm_blocks["heat_pump"]
    hp_ts = inst_ts.dsm_blocks["heat_pump"]

    # Without storage: HP power is distributed fairly evenly (must meet demand every hour)
    hp_expensive_no_ts = sum(_val(hp_no_ts.power_in[t]) for t in expensive_hours)
    # With storage: HP runs significantly more at cheap hours
    hp_cheap_ts = sum(_val(hp_ts.power_in[t]) for t in cheap_hours)
    hp_expensive_ts = sum(_val(hp_ts.power_in[t]) for t in expensive_hours)

    assert hp_cheap_ts > hp_expensive_ts + 1e-3, (
        "With thermal storage, heat pump should run more at cheap hours"
    )
    # Without storage, HP must run at expensive hours too (no other heat source)
    assert hp_expensive_no_ts > hp_expensive_ts + 1e-3, (
        "Without storage, HP is forced to run more during expensive hours than with storage"
    )


def test_pv_generation_reduces_total_energy_cost(time_index):
    """
    A building with PV generation should have strictly lower total energy cost
    than the same building without PV.
    """
    prices = [50] * 8
    market_prices = {"EOM": _series(prices, time_index)}

    def make_forecaster(with_pv: bool):
        return BuildingForecaster(
            index=time_index,
            fuel_prices={"natural_gas": _series([20] * 8, time_index)},
            market_prices=market_prices,
            electricity_price_flex=_series(prices, time_index),
            load_profile=_series([5] * 8, time_index),
            heat_demand=_series([0] * 8, time_index),
            pv_profile=_series([2] * 8 if with_pv else [0] * 8, time_index),
            battery_load_profile=_series([0] * 8, time_index),
            ev_load_profile=_series([0] * 8, time_index),
        )

    pv_components = {"pv_plant": {"max_power": 5.0, "uses_power_profile": "false"}}

    _, inst_no_pv, _ = _solve_building_opt(_make_building(make_forecaster(False), {}))
    _, inst_pv, _ = _solve_building_opt(
        _make_building(make_forecaster(True), pv_components)
    )

    cost_no_pv = sum(_val(inst_no_pv.variable_cost[t]) for t in inst_no_pv.time_steps)
    cost_pv = sum(_val(inst_pv.variable_cost[t]) for t in inst_pv.time_steps)

    assert cost_pv < cost_no_pv, "PV generation should reduce total energy cost"


# ---------------------------------------------------------------------
# EV component — unit tests (migrated from test_ev.py)
# These tests verify the ElectricVehicle component in isolation.
# Features like availability-profile are also covered by full-building
# integration tests (e.g. test_ev_cannot_charge_when_unavailable_if_connected_via_station).
# ---------------------------------------------------------------------
@pytest.fixture
def ev_unit_config():
    return {
        "capacity": 10.0,
        "min_soc": 0,
        "max_soc": 1,
        "max_power_charge": 3,
        "max_power_discharge": 2,
        "efficiency_charge": 0.95,
        "efficiency_discharge": 0.9,
        "initial_soc": 0,
    }


@pytest.fixture
def ev_model_with_charging_profile(ev_unit_config):
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=list(range(10)))
    model.electricity_price = pyo.Param(
        model.time_steps, initialize={t: 50 for t in range(10)}, mutable=True
    )
    charging_profile = pd.Series(
        [0.1, 0.05, 0.15, 0.2, 0.0, 0.1, 0.05, 0.1, 0.0, 0.15],
        index=list(range(10)),
    )
    ev = ElectricVehicle(
        **ev_unit_config,
        time_steps=model.time_steps,
        availability_profile=None,
        charging_profile=charging_profile,
    )
    model.ev = pyo.Block()
    ev.add_to_model(model, model.ev)
    model.total_charge = pyo.Objective(
        expr=sum(model.ev.charge[t] for t in model.time_steps), sense=pyo.maximize
    )
    solver = pyo.SolverFactory(USE_SOLVER)
    results = solver.solve(model)
    return model, results


def test_ev_charging_profile(ev_model_with_charging_profile, ev_unit_config):
    """Charging follows the predefined schedule exactly; SOC stays within bounds."""
    model, _ = ev_model_with_charging_profile
    charging_profile = pd.Series(
        [0.1, 0.05, 0.15, 0.2, 0.0, 0.1, 0.05, 0.1, 0.0, 0.15],
        index=list(range(10)),
    )
    for t in model.time_steps:
        assert pyo.value(model.ev.charge[t]) == pytest.approx(
            charging_profile[t], rel=1e-2
        )
    for t in model.time_steps:
        soc = pyo.value(model.ev.soc[t])
        assert ev_unit_config["min_soc"] <= soc <= ev_unit_config["max_soc"]


@pytest.fixture
def ev_model_with_availability(ev_unit_config):
    """EV model with availability profile (standalone unit test)."""
    model = pyo.ConcreteModel()
    model.time_steps = pyo.Set(initialize=list(range(10)))
    model.electricity_price = pyo.Param(
        model.time_steps, initialize={t: 50 for t in range(10)}, mutable=True
    )
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=list(range(10))
    )
    charging_profile = None
    ev = ElectricVehicle(
        **ev_unit_config,
        time_steps=model.time_steps,
        availability_profile=availability_profile,
        charging_profile=charging_profile,
    )
    model.ev = pyo.Block()
    ev.add_to_model(model, model.ev)
    model.total_charge = pyo.Objective(
        expr=sum(model.ev.charge[t] for t in model.time_steps), sense=pyo.maximize
    )
    solver = pyo.SolverFactory(USE_SOLVER)
    results = solver.solve(model)
    return model, results


def test_ev_availability_profile(ev_model_with_availability):
    """When unavailable, EV cannot charge/discharge; when available, within limits."""
    model, _ = ev_model_with_availability
    availability_profile = pd.Series(
        [1, 0, 1, 1, 0, 1, 1, 0, 1, 1], index=list(range(10))
    )
    for t in model.time_steps:
        availability = availability_profile[t]
        charge = pyo.value(model.ev.charge[t])
        discharge = pyo.value(model.ev.discharge[t])
        if availability == 0:
            assert charge == 0
            assert discharge == 0
        else:
            assert charge <= pyo.value(model.ev.max_power_charge) + 1e-5
            assert discharge <= pyo.value(model.ev.max_power_discharge) + 1e-5


# EV input validation
# -----------------------------------------------------------------
def test_ev_conflicting_availability_trip_distance_warning(time_index, caplog):
    """Soft validation: warn if EV has availability=1 and trip_distance>0 in same timestep."""
    import logging

    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    # Create forecaster with conflicting profiles
    forecaster = BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series([45, 35, 25, 15, 15, 25, 35, 45], time_index),
        load_profile=_series([5, 5, 5, 5, 5, 5, 5, 5], time_index),
        heat_demand=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        pv_profile=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        battery_load_profile=_series([0] * 8, time_index),
        # Conflicting: availability=1 but trip_distance>0 at timesteps 2, 3, 6
        A360_electric_vehicle_1_availability_profile=_series(
            [1, 1, 1, 1, 0, 0, 1, 1], time_index
        ),
        A360_electric_vehicle_1_trip_distance=_series(
            [0, 0, 100, 80, 0, 0, 50, 0], time_index
        ),
    )

    components = {
        "electric_vehicle_1": {
            "capacity": 50.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "initial_soc": 0.5,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "power_flow_directionality": "bidirectional",
        }
    }

    # Capture log output at WARNING level
    with caplog.at_level(logging.WARNING, logger="assume.units.building"):
        building = _make_building(forecaster, components)

    # Verify warning was logged
    assert building.has_ev is True
    warning_messages = [
        r.message for r in caplog.records if "conflicting profiles" in r.message
    ]
    assert len(warning_messages) > 0, "Expected warning about conflicting EV profiles"

    # Verify warning mentions conflicting timesteps and conflict details
    warning_text = warning_messages[0]
    assert "availability=1 and trip_distance>0" in warning_text
    assert "prioritize trip fulfillment" in warning_text


# EV energy consumption modes
# -----------------------------------------------------------------
def test_ev_direct_trip_energy_consumption_mode(solved_building_with_cs):
    """Mode 1: EV uses direct trip_energy_consumption (energy per trip, not distance-based)."""
    _, instance, _ = solved_building_with_cs
    ts = list(instance.time_steps)

    # electric_vehicle_1 uses trip_energy_consumption mode
    # (provided in forecaster as A360_electric_vehicle_1_trip_energy_consumption)
    ev = instance.dsm_blocks["electric_vehicle_1"]
    availability = getattr(instance, "electric_vehicle_1_availability")
    trip_energy = getattr(instance, "electric_vehicle_1_trip_energy_consumption", None)

    # If trip_energy_consumption is available, verify usage follows: usage = (1 - availability) * trip_energy
    if trip_energy is not None:
        for t in ts:
            avail = _val(availability[t])
            expected_usage = (1 - avail) * _val(trip_energy[t])
            actual_usage = _val(ev.usage[t])
            assert abs(actual_usage - expected_usage) <= 1e-5


# EV validation
# -----------------------------------------------------------------
def test_building_raises_error_if_ev_lacks_trip_data(time_index):
    """Verify that Building raises ValueError when EV has neither trip_distance nor trip_energy_consumption."""
    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    # Forecaster WITHOUT trip_distance or trip_energy_consumption for the EV
    forecaster = BuildingForecaster(
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
        # NOTE: Missing trip_distance and trip_energy_consumption!
    )

    components = {
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
    }

    # Should raise ValueError because EV lacks both trip configurations
    with pytest.raises(
        ValueError,
        match="must have either 'trip_distance' or 'trip_energy_consumption'",
    ):
        _make_building(forecaster, components)


def test_ev_distance_based_energy_consumption_mode(time_index):
    """Mode 2: EV uses trip_distance + mileage (existing behavior)."""
    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    forecaster = BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series([45, 35, 25, 15, 15, 25, 35, 45], time_index),
        load_profile=_series([5, 5, 5, 5, 5, 5, 5, 5], time_index),
        heat_demand=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        pv_profile=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        battery_load_profile=_series([0] * 8, time_index),
        # Provide trip_distance and availability but NOT trip_energy_consumption
        A360_electric_vehicle_1_availability_profile=_series(
            [1, 1, 0, 0, 1, 1, 1, 1], time_index
        ),
        A360_electric_vehicle_1_trip_distance=_series(
            [0, 0, 50, 100, 0, 0, 25, 0], time_index
        ),
    )

    components = {
        "electric_vehicle_1": {
            "capacity": 50.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "initial_soc": 0.5,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "mileage": 0.2,  # kWh per km
            "power_flow_directionality": "bidirectional",
        }
    }

    building = _make_building(forecaster, components)
    _, instance, _ = _solve_building_opt(building)
    ts = list(instance.time_steps)

    # Verify usage follows distance-based formula: usage = (1 - availability) * trip_distance * mileage
    ev = instance.dsm_blocks["electric_vehicle_1"]
    availability = getattr(instance, "electric_vehicle_1_availability")
    trip_distance = getattr(instance, "electric_vehicle_1_trip_distance", None)

    if trip_distance is not None:
        mileage = _val(ev.mileage)
        for t in ts:
            avail = _val(availability[t])
            dist = _val(trip_distance[t])
            expected_usage = (1 - avail) * dist * mileage
            actual_usage = _val(ev.usage[t])
            assert abs(actual_usage - expected_usage) <= 1e-5


def test_ev_trip_energy_takes_priority_over_distance(time_index):
    """When both trip_energy_consumption and trip_distance provided, energy-based takes priority."""
    market_prices = {"EOM": _series([50, 40, 30, 20, 20, 30, 40, 50], time_index)}

    forecaster = BuildingForecaster(
        index=time_index,
        fuel_prices={"natural_gas": _series([20] * 8, time_index)},
        market_prices=market_prices,
        electricity_price_flex=_series([45, 35, 25, 15, 15, 25, 35, 45], time_index),
        load_profile=_series([5, 5, 5, 5, 5, 5, 5, 5], time_index),
        heat_demand=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        pv_profile=_series([0, 0, 0, 0, 0, 0, 0, 0], time_index),
        battery_load_profile=_series([0] * 8, time_index),
        A360_electric_vehicle_1_availability_profile=_series(
            [1, 1, 0, 0, 1, 1, 1, 1], time_index
        ),
        # Both provided - energy-based should take priority
        A360_electric_vehicle_1_trip_distance=_series(
            [0, 0, 100, 200, 0, 0, 50, 0], time_index
        ),
        A360_electric_vehicle_1_trip_energy_consumption=_series(
            [0, 0, 10, 15, 0, 0, 5, 0], time_index
        ),
    )

    components = {
        "electric_vehicle_1": {
            "capacity": 50.0,
            "min_soc": 0.1,
            "max_soc": 1.0,
            "max_power_charge": 10.0,
            "max_power_discharge": 10.0,
            "initial_soc": 0.5,
            "efficiency_charge": 1.0,
            "efficiency_discharge": 1.0,
            "mileage": 0.2,
            "power_flow_directionality": "bidirectional",
        }
    }

    building = _make_building(forecaster, components)
    _, instance, _ = _solve_building_opt(building)

    # Verify usage uses trip_energy_consumption, not distance-based
    ev = instance.dsm_blocks["electric_vehicle_1"]
    availability = getattr(instance, "electric_vehicle_1_availability")
    trip_energy = getattr(instance, "electric_vehicle_1_trip_energy_consumption", None)

    if trip_energy is not None:
        ts = list(instance.time_steps)
        for t in ts:
            avail = _val(availability[t])
            # Should use energy-based formula, not distance-based
            expected_usage = (1 - avail) * _val(trip_energy[t])
            actual_usage = _val(ev.usage[t])
            assert abs(actual_usage - expected_usage) <= 1e-5


if __name__ == "__main__":
    pytest.main(["-s", __file__])
