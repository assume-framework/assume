# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecasts import NaiveForecast
from assume.strategies.naive_strategies import NaiveDADSMStrategy
from assume.units.steam_generation_plant import SteamPlant


@pytest.fixture
def steam_plant_components_with_hp():
    return {
        "heat_pump": {
            "max_power": 50,
            "cop": 2,
            "min_power": 0,
            "ramp_up": 50,
            "ramp_down": 50,
        },
    }


@pytest.fixture
def steam_plant_with_hp(steam_plant_components_with_hp) -> SteamPlant:
    # Define the time index and forecast data
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        electricity_price=[
            60,
            60,
            60,
            60,
            60,
            50,
            50,
            50,
            50,
            50,
            40,
            40,
            40,
            40,
            40,
            60,
            60,
            60,
            60,
            60,
            20,
            40,
            60,
            60,
        ],
        A360_thermal_demand=[8] * 24,  # Constant thermal demand
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": NaiveDADSMStrategy(),
    }

    # Initialize the HydrogenPlant with the specified components, forecast, and strategy
    return SteamPlant(
        id="A360",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=50,
        bidding_strategies=bidding_strategy,
        components=steam_plant_components_with_hp,
        forecaster=forecast,
    )


def test_optimal_operation_without_flex_initialization_hp(steam_plant_with_hp):
    # Run the initial without-flexibility operation to populate opt_power_requirement
    steam_plant_with_hp.determine_optimal_operation_without_flex()

    # Check that opt_power_requirement is populated and of the correct type
    assert (
        steam_plant_with_hp.opt_power_requirement is not None
    ), "opt_power_requirement should be populated"
    assert isinstance(steam_plant_with_hp.opt_power_requirement, FastSeries)

    # Create an instance of the model and switch to optimization mode
    instance = steam_plant_with_hp.model.create_instance()
    instance = steam_plant_with_hp.switch_to_opt(instance)
    steam_plant_with_hp.solver.solve(instance, tee=False)

    # Check that the total power input is greater than zero, indicating operation
    total_power_input = sum(
        instance.total_power_input[t].value
        for t in instance.time_steps
        if instance.total_power_input[t].value is not None
    )
    assert total_power_input > 0, "Total power input should be greater than zero"

    for t in instance.time_steps:
        heat_pump_output = instance.dsm_blocks["heat_pump"].heat_out[t].value
        thermal_demand = instance.thermal_demand[t]
        assert heat_pump_output == pytest.approx(
            thermal_demand, rel=1e-3
        ), f"Heat pump output does not meet thermal demand at time {t}"


# Pytest with Heatpump & Boiler


@pytest.fixture
def steam_plant_components_with_hp_b():
    return {
        "heat_pump": {
            "max_power": 30,
            "cop": 2,
            "min_power": 0,
            "ramp_up": 30,
            "ramp_down": 30,
        },
        "boiler": {
            "max_power": 50,
            "efficiency": 0.9,
            "fuel_type": "hydrogen_gas",
            "min_power": 0,
            "ramp_up": 50,
            "ramp_down": 50,
        },
    }


@pytest.fixture
def steam_plant_with_hp_b(steam_plant_components_with_hp_b) -> SteamPlant:
    # Define the time index and forecast data
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        electricity_price=[
            60,
            60,
            60,
            60,
            60,
            50,
            50,
            50,
            50,
            50,
            40,
            40,
            40,
            40,
            40,
            60,
            60,
            60,
            60,
            60,
            20,
            40,
            60,
            60,
        ],
        hydrogen_gas_price=[55] * 24,
        A360_thermal_demand=[8] * 24,  # Constant thermal demand
        # test_steam_plant_congestion_signal=[0] * 24,  # No congestion
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": NaiveDADSMStrategy(),
    }

    # Initialize the HydrogenPlant with the specified components, forecast, and strategy
    return SteamPlant(
        id="A360",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=50,
        bidding_strategies=bidding_strategy,
        components=steam_plant_components_with_hp_b,
        forecaster=forecast,
    )


def test_optimal_operation_without_flex_initialization_hp_b(steam_plant_with_hp_b):
    # Run the initial without-flexibility operation to populate opt_power_requirement
    steam_plant_with_hp_b.determine_optimal_operation_without_flex()

    # Check that opt_power_requirement is populated and of the correct type
    assert (
        steam_plant_with_hp_b.opt_power_requirement is not None
    ), "opt_power_requirement should be populated"
    assert isinstance(steam_plant_with_hp_b.opt_power_requirement, FastSeries)

    # Create an instance of the model and switch to optimization mode
    instance = steam_plant_with_hp_b.model.create_instance()
    instance = steam_plant_with_hp_b.switch_to_opt(instance)
    steam_plant_with_hp_b.solver.solve(instance, tee=False)

    # Check that the total power input is greater than zero, indicating operation
    total_power_input = sum(
        instance.total_power_input[t].value
        for t in instance.time_steps
        if instance.total_power_input[t].value is not None
    )
    assert total_power_input > 0, "Total power input should be greater than zero"

    for t in instance.time_steps:
        heat_pump_output = instance.dsm_blocks["heat_pump"].heat_out[t].value
        boiler_output = instance.dsm_blocks["boiler"].heat_out[t].value
        thermal_demand = instance.thermal_demand[t]
        assert heat_pump_output + boiler_output == pytest.approx(
            thermal_demand, rel=1e-3
        ), f"Heat pump and Boiler output does not meet thermal demand at time {t}"


# Pytest with Heatpump, Boiler & Thermal storage


@pytest.fixture
def steam_plant_components_with_hp_b_ts():
    return {
        "heat_pump": {
            "max_power": 30,
            "cop": 2,
            "min_power": 0,
            "ramp_up": 30,
            "ramp_down": 30,
        },
        "boiler": {
            "max_power": 50,
            "efficiency": 0.9,
            "fuel_type": "hydrogen_gas",
            "min_power": 0,
            "ramp_up": 50,
            "ramp_down": 50,
        },
        "thermal_storage": {
            "max_capacity": 100,
            "min_capacity": 0,
            "max_power_charge": 50,
            "max_power_discharge": 50,
            "efficiency_charge": 1,
            "efficiency_discharge": 1,
            "initial_soc": 0.5,
            "ramp_up": 50,
            "ramp_down": 50,
            "storage_loss_rate": 0.0,
            "storage_type": "short-term",
        },
    }


@pytest.fixture
def steam_plant_with_hp_b_ts(steam_plant_components_with_hp_b_ts) -> SteamPlant:
    # Define the time index and forecast data
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        electricity_price=[
            60,
            60,
            60,
            60,
            60,
            50,
            50,
            50,
            50,
            50,
            40,
            40,
            40,
            40,
            40,
            60,
            60,
            60,
            60,
            60,
            20,
            40,
            60,
            60,
        ],
        natural_gas_price=[55] * 24,
        A360_thermal_demand=[8] * 24,  # Constant thermal demand
        # test_steam_plant_congestion_signal=[0] * 24,  # No congestion
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": NaiveDADSMStrategy(),
    }

    # Initialize the HydrogenPlant with the specified components, forecast, and strategy
    return SteamPlant(
        id="A360",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=50,
        bidding_strategies=bidding_strategy,
        components=steam_plant_components_with_hp_b_ts,
        forecaster=forecast,
    )


def test_optimal_operation_without_flex_initialization_hp_b_ts(
    steam_plant_with_hp_b_ts,
):
    # Run the initial without-flexibility operation to populate opt_power_requirement
    steam_plant_with_hp_b_ts.determine_optimal_operation_without_flex()

    # Check that opt_power_requirement is populated and of the correct type
    assert (
        steam_plant_with_hp_b_ts.opt_power_requirement is not None
    ), "opt_power_requirement should be populated"
    assert isinstance(steam_plant_with_hp_b_ts.opt_power_requirement, FastSeries)

    # Create an instance of the model and switch to optimization mode
    instance = steam_plant_with_hp_b_ts.model.create_instance()
    instance = steam_plant_with_hp_b_ts.switch_to_opt(instance)
    steam_plant_with_hp_b_ts.solver.solve(instance, tee=False)

    # Check that the total power input is greater than zero, indicating operation
    total_power_input = sum(
        instance.total_power_input[t].value
        for t in instance.time_steps
        if instance.total_power_input[t].value is not None
    )
    assert total_power_input > 0, "Total power input should be greater than zero"

    for t in instance.time_steps:
        heat_pump_output = instance.dsm_blocks["heat_pump"].heat_out[t].value
        boiler_output = instance.dsm_blocks["boiler"].heat_out[t].value
        storage_discharge = instance.dsm_blocks["thermal_storage"].discharge[t].value
        storage_charge = instance.dsm_blocks["thermal_storage"].charge[t].value
        thermal_demand = instance.thermal_demand[t]
        # THE CORRECT THERMAL BALANCE:
        assert (
            heat_pump_output + boiler_output + storage_discharge - storage_charge
            == pytest.approx(thermal_demand, rel=1e-3)
        ), (
            f"Heat pump, Boiler & Storage output minus charging does not meet "
            f"thermal demand at time {t} "
            f"(HP={heat_pump_output:.2f}, B={boiler_output:.2f}, SD={storage_discharge:.2f}, SC={storage_charge:.2f}, D={thermal_demand:.2f})"
        )


@pytest.fixture
def steam_plant_components_with_hp_b_longterm_ts():
    return {
        "heat_pump": {
            "max_power": 30,
            "cop": 2,
            "min_power": 0,
            "ramp_up": 30,
            "ramp_down": 30,
        },
        "boiler": {
            "max_power": 50,
            "efficiency": 0.9,
            "fuel_type": "hydrogen_gas",
            "min_power": 0,
            "ramp_up": 50,
            "ramp_down": 50,
        },
        "thermal_storage": {
            "max_capacity": 200,
            "min_capacity": 0,
            "max_power_charge": 40,
            "max_power_discharge": 50,
            "efficiency_charge": 1,
            "efficiency_discharge": 1,
            "initial_soc": 1.0,  # Fully charged
            "ramp_up": 50,
            "ramp_down": 50,
            "storage_loss_rate": 0.0,
            "storage_type": "long-term",
            "storage_schedule_profile": pd.Series(
                # Only allow discharge from t=10 to t=15 (inclusive), rest is charging only
                [0] * 8 + [1] * 8 + [0] * 4 + [1] * 4,
            ),
        },
    }


@pytest.fixture
def steam_plant_with_hp_b_longterm_ts(
    steam_plant_components_with_hp_b_longterm_ts,
) -> SteamPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        electricity_price=[
            60,
            60,
            60,
            60,
            60,
            50,
            50,
            50,
            50,
            50,
            40,
            40,
            40,
            40,
            40,
            60,
            60,
            60,
            60,
            60,
            20,
            40,
            60,
            60,
        ],
        hydrogen_gas_price=[
            55,
            55,
            40,
            100,
            20,
            55,
            40,
            55,
            60,
            55,
            55,
            100,
            100,
            80,
            55,
            55,
            55,
            55,
            55,
            55,
            55,
            55,
            55,
            60,
        ],
        A360_thermal_demand=[8] * 24,
        # add schedule if your forecaster expects it, but for thermal demand it's enough
    )
    bidding_strategy = {"EOM": NaiveDADSMStrategy()}
    return SteamPlant(
        id="A360",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        cost_tolerance=50,
        bidding_strategies=bidding_strategy,
        components=steam_plant_components_with_hp_b_longterm_ts,
        forecaster=forecast,
    )


def test_optimal_operation_with_longterm_storage(steam_plant_with_hp_b_longterm_ts):
    steam_plant_with_hp_b_longterm_ts.determine_optimal_operation_without_flex()
    assert steam_plant_with_hp_b_longterm_ts.opt_power_requirement is not None
    assert isinstance(
        steam_plant_with_hp_b_longterm_ts.opt_power_requirement, FastSeries
    )

    instance = steam_plant_with_hp_b_longterm_ts.model.create_instance()
    instance = steam_plant_with_hp_b_longterm_ts.switch_to_opt(instance)
    steam_plant_with_hp_b_longterm_ts.solver.solve(instance, tee=False)

    total_power_input = sum(
        instance.total_power_input[t].value
        for t in instance.time_steps
        if instance.total_power_input[t].value is not None
    )
    assert total_power_input > 0, "Total power input should be greater than zero"

    # Print for diagnostics
    print("t | HeatPump | Boiler | StorageOut | StorageIn | LHS | RHS (Demand) | DIFF")
    print("--------------------------------------------------------------")
    for t in instance.time_steps:
        hp = instance.dsm_blocks["heat_pump"].heat_out[t].value
        b = instance.dsm_blocks["boiler"].heat_out[t].value
        sd = instance.dsm_blocks["thermal_storage"].discharge[t].value
        sc = instance.dsm_blocks["thermal_storage"].charge[t].value
        td = instance.thermal_demand[t]
        lhs = hp + b + sd - sc
        assert lhs == pytest.approx(td, rel=1e-3), (
            f"Heat pump, Boiler & Long-Term Storage output minus charging does not meet "
            f"thermal demand at time {t}"
        )


def test_all_assets_coordinated(steam_plant_with_hp_b_ts):
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    demand_profile = [20] * 10 + [60] + [20] * 13

    # Re-create the forecast object with new demand
    new_forecast = NaiveForecast(
        index,
        electricity_price=[60] * 24,  # adapt if needed
        hydrogen_gas_price=[55] * 24,  # adapt if needed
        A360_thermal_demand=demand_profile,
    )
    # Replace the forecaster of the plant with the new one
    steam_plant_with_hp_b_ts.forecaster = new_forecast

    # Now run the rest as usual
    steam_plant_with_hp_b_ts.determine_optimal_operation_without_flex()
    instance = steam_plant_with_hp_b_ts.model.create_instance()
    instance = steam_plant_with_hp_b_ts.switch_to_opt(instance)
    steam_plant_with_hp_b_ts.solver.solve(instance, tee=False)

    t_spike = 10
    heat_pump_output = instance.dsm_blocks["heat_pump"].heat_out[t_spike].value
    boiler_output = instance.dsm_blocks["boiler"].heat_out[t_spike].value
    storage_output = instance.dsm_blocks["thermal_storage"].discharge[t_spike].value
    thermal_demand = instance.thermal_demand[t_spike]
    storage_charge = instance.dsm_blocks["thermal_storage"].charge[t_spike].value
    total_supply = heat_pump_output + boiler_output + storage_output - storage_charge

    print(
        f"Spike: HP={heat_pump_output}, Boiler={boiler_output}, StorageDischarge={storage_output}, StorageCharge={storage_charge}, Demand={thermal_demand}"
    )

    assert total_supply == pytest.approx(thermal_demand, rel=1e-3)


if __name__ == "__main__":
    pytest.main(["-s", __file__])
