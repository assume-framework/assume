# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from assume.common.fast_pandas import FastSeries
from assume.common.forecaster import SteamgenerationForecaster
from assume.strategies.naive_strategies import (
    DsmCapacityHeuristicBalancingNegStrategy,
    DsmCapacityHeuristicBalancingPosStrategy,
    DsmEnergyOptimizationStrategy,
)
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
    forecast = SteamgenerationForecaster(
        index=index,
        demand=0,
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
        electricity_price_flex=0,
        fuel_prices={},
        thermal_demand=[100] * 24,
        congestion_signal=0,
        renewable_utilisation_signal=0,
        thermal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": 0},
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": DsmEnergyOptimizationStrategy(),
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
    assert steam_plant_with_hp.opt_power_requirement is not None, (
        "opt_power_requirement should be populated"
    )
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
        assert heat_pump_output == pytest.approx(thermal_demand, rel=1e-3), (
            f"Heat pump output does not meet thermal demand at time {t}"
        )


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
    forecast = SteamgenerationForecaster(
        index=index,
        demand=0,
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
        electricity_price_flex=0,
        fuel_prices={},
        thermal_demand=[100] * 24,
        congestion_signal=0,
        renewable_utilisation_signal=0,
        thermal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": 0},
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": DsmEnergyOptimizationStrategy(),
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
    assert steam_plant_with_hp_b.opt_power_requirement is not None, (
        "opt_power_requirement should be populated"
    )
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
            "capacity": 100,
            "min_soc": 0,
            "max_soc": 1,
            "max_power_charge": 50,
            "max_power_discharge": 50,
            "efficiency_charge": 1,
            "efficiency_discharge": 1,
            "initial_soc": 0,
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
    forecast = SteamgenerationForecaster(
        index,
        demand=0,
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
        electricity_price_flex=0,
        fuel_prices={},
        thermal_demand=[100] * 24,
        congestion_signal=0,
        renewable_utilisation_signal=0,
        thermal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": 0},
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": DsmEnergyOptimizationStrategy(),
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
    assert steam_plant_with_hp_b_ts.opt_power_requirement is not None, (
        "opt_power_requirement should be populated"
    )
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
            "capacity": 200,
            "min_soc": 0,
            "max_soc": 1,
            "max_power_charge": 40,
            "max_power_discharge": 50,
            "efficiency_charge": 1,
            "efficiency_discharge": 1,
            "initial_soc": 0,
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
    forecast = SteamgenerationForecaster(
        index,
        demand=0,
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
        electricity_price_flex=0,
        fuel_prices={
            "hydrogen": [
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
            ]
        },
        thermal_demand=[100] * 24,
        congestion_signal=0,
        renewable_utilisation_signal=0,
        thermal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": 0},
    )
    bidding_strategy = {"EOM": DsmEnergyOptimizationStrategy()}
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
    new_forecast = SteamgenerationForecaster(
        index,
        electricity_price=[60] * 24,  # adapt if needed
        fuel_prices={"hydrogen_gas": [55] * 24},  # adapt if needed
        thermal_demand=demand_profile,
        demand=0,
        electricity_price_flex=0,
        congestion_signal=0,
        renewable_utilisation_signal=0,
        thermal_storage_schedule=0,
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

    assert total_supply == pytest.approx(thermal_demand, rel=1e-3)


@pytest.fixture
def steam_plant_with_crm_flex(steam_plant_components_with_hp_b):
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = SteamgenerationForecaster(
        index,
        demand=0,
        electricity_price=[60] * 24,
        electricity_price_flex=0,
        fuel_prices={},
        thermal_demand=[100] * 24,
        congestion_signal=0,
        renewable_utilisation_signal=0,
        thermal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": 0},
    )
    bidding_strategy = {
        "CRM_pos": DsmCapacityHeuristicBalancingPosStrategy(),
        "CRM_neg": DsmCapacityHeuristicBalancingNegStrategy(),
    }
    return SteamPlant(
        id="A360",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="symmetric_flexible_block",
        cost_tolerance=5,  # Use low tolerance for strict test
        bidding_strategies=bidding_strategy,
        components=steam_plant_components_with_hp_b,
        forecaster=forecast,
    )


def test_crm_block_flexibility_and_bidding(steam_plant_with_crm_flex):
    # 1. Run optimal operation with flexibility
    steam_plant_with_crm_flex.determine_optimal_operation_with_flex()
    assert steam_plant_with_crm_flex.flex_power_requirement is not None

    # 2. Access the plant's CRM block variables after solving
    instance = steam_plant_with_crm_flex.model.create_instance()
    instance = steam_plant_with_crm_flex.switch_to_flex(instance)
    steam_plant_with_crm_flex.solver.solve(instance, tee=False)

    # 3. Test the CRM bidding strategies
    market_config = {}  # add if you need for your bidding function
    # Define possible blocks (simulate market product_tuples as (start, end, only_hours))
    index = steam_plant_with_crm_flex.index
    product_tuples = []
    block_length = 4
    for i in range(len(index) - block_length):
        product_tuples.append((index[i], index[i + block_length], None))

    # --- POSITIVE CRM ---
    pos_strategy = DsmCapacityHeuristicBalancingPosStrategy()
    pos_bids = pos_strategy.calculate_bids(
        steam_plant_with_crm_flex, market_config, product_tuples
    )
    assert isinstance(pos_bids, list)
    # --- NEGATIVE CRM ---
    neg_strategy = DsmCapacityHeuristicBalancingNegStrategy()
    neg_bids = neg_strategy.calculate_bids(
        steam_plant_with_crm_flex, market_config, product_tuples
    )
    assert isinstance(neg_bids, list)
    # Optional: check at least one bid has positive volume if possible
    assert any(bid["volume"] > 0 for bid in pos_bids + neg_bids)


@pytest.fixture
def steam_plant_with_price_signal_flex(steam_plant_components_with_hp_b):
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    # New price signal: simulate low prices at night, high in afternoon
    price_flex = [30] * 6 + [45] * 6 + [80] * 6 + [50] * 6
    forecast = SteamgenerationForecaster(
        index,
        demand=0,
        electricity_price=[60] * 24,
        electricity_price_flex=0,
        fuel_prices={},
        thermal_demand=[100] * 24,
        congestion_signal=0,
        renewable_utilisation_signal=0,
        thermal_storage_schedule=0,
        availability=0,
        market_prices={"EOM": 0},
    )
    bidding_strategy = {}
    plant = SteamPlant(
        id="A360",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="electricity_price_signal",
        cost_tolerance=5,
        bidding_strategies=bidding_strategy,
        components=steam_plant_components_with_hp_b,
        forecaster=forecast,
    )
    plant.electricity_price_flex = price_flex
    return plant


def test_electricity_price_signal_flexibility(steam_plant_with_price_signal_flex):
    # Solve with price signal flexibility
    steam_plant_with_price_signal_flex.determine_optimal_operation_with_flex()
    flex_profile = list(steam_plant_with_price_signal_flex.flex_power_requirement)

    # Check the profile is the correct length
    assert len(flex_profile) == 24

    # Optional: check for non-trivial operation
    # e.g., did the plant shift operation at low/high prices? (Heuristic)
    low_price_avg = sum(flex_profile[:6]) / 6
    high_price_avg = sum(flex_profile[12:18]) / 6
    print(f"Low price avg: {low_price_avg:.2f}, High price avg: {high_price_avg:.2f}")
    # Should use more power when price is low than when it's high
    assert low_price_avg >= high_price_avg - 1e-3


if __name__ == "__main__":
    pytest.main(["-s", __file__])
