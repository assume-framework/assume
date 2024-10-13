# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from datetime import datetime, timedelta
from distutils.command.build import build

import pandas as pd
import pytest
import pyomo.environ as pyo

from assume.common.forecasts import CsvForecaster
from assume.strategies.naive_strategies import NaiveDABuildingStrategy
from assume.units import Building


@pytest.fixture
def index() -> pd.DatetimeIndex:
    return pd.date_range(
        "2019-01-01", periods=4, freq="h"
    )


@pytest.fixture
def forecast(index) -> CsvForecaster:
    forecaster = CsvForecaster(index=index)
    forecaster.forecasts = pd.DataFrame()
    forecaster.forecasts["test_building_load_profile"] = pd.Series(1, index=index)
    forecaster.forecasts["availability_Solar"] = pd.Series(0.25, index=index)
    forecaster.forecasts["price_EOM"] = pd.Series(2, index=index)

    return forecaster


@pytest.fixture
def building_components_1() -> dict:
    return {
        'battery_storage':
            {'charge_loss_rate': 0.0, 'charging_profile': 'No', 'discharge_loss_rate': 0.0,
             'initial_soc': 0.0, 'max_capacity': 8.0, 'max_charging_rate': 1.0, 'max_discharging_rate': 1.0,
             'min_capacity': 0.0, 'node': 'north', 'sells_energy_to_market': 'Yes',
             'unit_type': 'building'},
        'ev':
            {'availability_periods': '[("1/1/2019  1:00:00 AM", "1/1/2019  3:00:00 AM")]', 'charging_profile': 'No',
             'initial_soc': 0.0, 'max_capacity': 3.0, 'min_capacity': 0.0, 'node': 'north', 'ramp_down': 3.0,
             'ramp_up': 3.0, 'unit_type': 'building', 'max_charging_rate': 3.0},
        'pv_plant':
            {'bidding_EOM': 'naive_da_building', 'max_power': 16.0, 'min_power': 0.0, 'node': 'north',
             'objective': 'minimize_expenses', 'power_profile': 'No', 'unit_operator': 'test_operator',
             'unit_type': 'building'}
    }


@pytest.fixture
def building_components_2() -> dict:
    return {
        'battery_storage':
            {'charge_loss_rate': 0.0, 'charging_profile': 'No', 'discharge_loss_rate': 0.0,
             'initial_soc': 0.0, 'max_capacity': 8.0, 'max_charging_rate': 1.0, 'max_discharging_rate': 1.0,
             'min_capacity': 1.0, 'node': 'north', 'sells_energy_to_market': 'No',
             'unit_type': 'building'}
    }


@pytest.fixture
def building_1(building_components_1, forecast, index) -> Building:
    return Building(
        id="test_building",
        unit_operator="test_operator",
        bidding_strategies={"EOM": NaiveDABuildingStrategy()},
        index=index,
        objective="minimize_expenses",
        forecaster=forecast,
        components=building_components_1,
    )


@pytest.fixture
def building_2(building_components_2, forecast, index) -> Building:
    return Building(
        id="test_building",
        unit_operator="test_operator",
        bidding_strategies={"EOM": NaiveDABuildingStrategy()},
        index=index,
        objective="minimize_expenses",
        forecaster=forecast,
        components=building_components_2,
    )


def assert_is_pyo_var(attribute):
    assert isinstance(attribute, pyo.Var)


def assert_attribute_error(model, attribute):
    with pytest.raises(AttributeError):
        getattr(model, attribute)


def test_initialization(building_1, building_2):
    assert building_1.id == "test_building"
    assert building_1.unit_operator == "test_operator"
    assert building_1.objective == "minimize_expenses"
    assert building_1.forecaster is not None
    assert isinstance(building_1.bidding_strategies["EOM"], NaiveDABuildingStrategy)
    assert building_1.has_pv is True
    assert building_2.has_pv is False
    assert building_1.has_ev is True
    assert building_2.has_ev is False
    assert building_1.has_battery_storage is True
    assert building_1.has_heatpump is False
    assert building_1.has_boiler is False
    assert building_1.has_thermal_storage is False
    assert all(building_1.additional_electricity_load == 1.0)
    assert all(building_1.electricity_price == 2)
    assert building_1.sells_battery_energy_to_market is True
    assert building_2.sells_battery_energy_to_market is False

    # Check if components got initialized correctly
    assert len(building_1.model.dsm_blocks) == 3
    assert len(building_2.model.dsm_blocks) == 1
    assert "pv_plant" in building_1.model.dsm_blocks
    assert "ev" in building_1.model.dsm_blocks
    assert "battery_storage" in building_1.model.dsm_blocks

    # Check if additional variables got initialized
    assert_is_pyo_var(building_1.model.total_power_input)
    assert_is_pyo_var(building_1.model.variable_cost)
    assert_is_pyo_var(building_1.model.variable_revenue)
    assert_is_pyo_var(building_1.model.total_power_output)
    assert_is_pyo_var(building_1.model.total_power_self_usage)
    assert_is_pyo_var(building_1.model.additional_load_from_grid)
    assert_is_pyo_var(building_1.model.charge_ev_from_grid)
    assert_is_pyo_var(building_1.model.consumer_indicator)
    assert_attribute_error(building_2.model, "charge_ev_from_grid")
    assert_is_pyo_var(building_1.model.energy_self_consumption_pv)
    assert_attribute_error(building_2.model, "energy_self_consumption_pv")
    assert_is_pyo_var(building_1.model.energy_sell_pv)
    assert_attribute_error(building_2.model, "energy_sell_pv")
    assert_is_pyo_var(building_1.model.charge_battery_from_pv)
    assert_attribute_error(building_2.model, "charge_battery_from_pv")
    assert_is_pyo_var(building_1.model.charge_ev_from_pv)
    assert_attribute_error(building_2.model, "charge_ev_from_pv")
    assert_is_pyo_var(building_1.model.discharge_battery_self_consumption)
    assert_is_pyo_var(building_1.model.discharge_battery_sell)
    assert_attribute_error(building_2.model, "discharge_battery_sell")
    assert_is_pyo_var(building_1.model.charge_battery_from_grid)
    assert_is_pyo_var(building_1.model.charge_ev_from_battery)
    assert_attribute_error(building_2.model, "charge_ev_from_battery")


def test_optimal_operation(building_1, index):
    electricity_load = building_1.additional_electricity_load
    energy_prices = building_1.forecaster['price_EOM'].values
    instance = building_1.model.create_instance()
    building_1.calculate_optimal_operation_if_needed()
    assert building_1.opt_power_requirement is not None
    assert isinstance(building_1.opt_power_requirement, pd.Series)
    building_1.solver.solve(instance, tee=False)

    # Check additional load constraints are valid
    load_pv = pd.Series([pyo.value(instance.additional_load_from_pv[i]) for i in range(len(index))])
    load_battery = pd.Series([pyo.value(instance.additional_load_from_battery[i]) for i in range(len(index))])
    load_grid = pd.Series([pyo.value(instance.additional_load_from_grid[i]) for i in range(len(index))])
    assert all(load_pv + load_battery + load_grid == electricity_load.values)

    # Check if PV constraints valid
    pv_energy_out = pd.Series([pyo.value(instance.dsm_blocks['pv_plant'].energy_out[i]) for i in range(len(index))])
    pv_self = pd.Series([pyo.value(instance.energy_self_consumption_pv[i]) for i in range(len(index))])
    pv_sell = pd.Series([pyo.value(instance.energy_sell_pv[i]) for i in range(len(index))])
    assert all(pv_energy_out == 4.0)
    assert all(pv_self + pv_sell == pv_energy_out)

    # Check if battery constraints valid
    battery_soc = pd.Series([pyo.value(instance.dsm_blocks['battery_storage'].soc[i]) for i in range(len(index))])
    battery_charge = pd.Series([pyo.value(instance.dsm_blocks['battery_storage'].charge[i]) for i in range(len(index))])
    battery_discharge = pd.Series(
        [pyo.value(instance.dsm_blocks['battery_storage'].discharge[i]) for i in range(len(index))])
    charge_bat_from_pv = pd.Series([pyo.value(instance.charge_battery_from_pv[i]) for i in range(len(index))])
    charge_bat_from_grid = pd.Series([pyo.value(instance.charge_battery_from_grid[i]) for i in range(len(index))])
    discharge_sell = pd.Series([pyo.value(instance.discharge_battery_sell[i]) for i in range(len(index))])
    discharge_self = pd.Series([pyo.value(instance.discharge_battery_self_consumption[i]) for i in range(len(index))])
    assert battery_soc[0] == battery_charge[0]
    assert battery_soc[1] == battery_soc[0] - battery_discharge[1]
    assert all(charge_bat_from_grid + charge_bat_from_pv == battery_charge)
    assert all(discharge_sell + discharge_self == battery_discharge)

    # Check if EV coinstraints valid
    soc_ev = pd.Series([pyo.value(instance.dsm_blocks['ev'].ev_battery_soc[i]) for i in range(len(index))])
    charge_ev = pd.Series([pyo.value(instance.dsm_blocks['ev'].charge_ev[i]) for i in range(len(index))])
    charge_ev_from_pv = pd.Series([pyo.value(instance.charge_ev_from_pv[i]) for i in range(len(index))])
    charge_ev_from_bat = pd.Series([pyo.value(instance.charge_ev_from_battery[i]) for i in range(len(index))])
    charge_ev_from_grid = pd.Series([pyo.value(instance.charge_ev_from_grid[i]) for i in range(len(index))])
    assert all(soc_ev == charge_ev)
    assert all(charge_ev_from_grid + charge_ev_from_bat + charge_ev_from_pv == charge_ev)

    # Check if energy got distributed and charged right
    assert all(charge_bat_from_pv + charge_ev_from_pv + load_pv == pv_self)
    assert all(charge_ev_from_bat + load_battery == discharge_self)
    building_energy = ((load_grid + charge_ev_from_grid + charge_bat_from_grid)-(pv_sell + discharge_sell))
    power_input = pd.Series([instance.total_power_input[t].value for t in instance.time_steps])
    power_output = pd.Series([instance.total_power_output[t].value for t in instance.time_steps])
    assert all(power_input - power_output == building_energy)
    costs = pd.Series([instance.variable_cost[t].value for t in instance.time_steps])
    revenue = pd.Series([instance.variable_revenue[t].value for t in instance.time_steps])
    assert sum(building_energy * energy_prices) == sum(costs - revenue)


if __name__ == "__main__":
    # run pytest and enable prints
    pytest.main(["-s", __file__])
