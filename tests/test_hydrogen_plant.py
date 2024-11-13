# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies.naive_strategies import NaiveDADSMStrategy
from assume.units.hydrogen_plant import HydrogenPlant


@pytest.fixture
def hydrogen_components():
    # Define full component configuration for hydrogen plant components including electrolyser and seasonal storage
    return {
        "electrolyser": {
            "max_power": 100,  # Maximum power input in MW
            "min_power": 0,  # Minimum power input in MW
            "ramp_up": 50,  # Ramp-up rate in MW per time step
            "ramp_down": 50,  # Ramp-down rate in MW per time step
            "efficiency": 0.7,  # Efficiency of the electrolyser
            "min_operating_time": 1,  # Minimum number of operating steps
            "min_down_time": 1,  # Minimum downtime steps
        },
        "h2_seasonal_storage": {
            "max_capacity": 500,  # Maximum storage capacity in MWh
            "min_capacity": 50,  # Minimum storage capacity in MWh
            "max_power_charge": 30,  # Maximum charging power in MW
            "max_power_discharge": 30,  # Maximum discharging power in MW
            "efficiency_charge": 0.9,  # Efficiency for charging
            "efficiency_discharge": 0.9,  # Efficiency for discharging
            "initial_soc": 1.0,  # Initial state of charge (SOC) as fraction of max capacity
            "final_soc_target": 0.5,  # Target SOC at end of horizon, as a fraction of max capacity
            "ramp_up": 10,  # Maximum increase in charge/discharge per step
            "ramp_down": 10,
            "horizon": 23,  # Maximum decrease in charge/discharge per step
            "storage_loss_rate": 0.01,  # 1% storage loss per time step
            "off_season_start": "0",  # Off-season start times, as comma-separated values
            "off_season_end": "12",  # Off-season end times, as comma-separated values
            "on_season_start": "13",  # On-season start times, as comma-separated values
            "on_season_end": "23",  # On-season end times, as comma-separated values
        },
    }


@pytest.fixture
def hydrogen_plant(hydrogen_components) -> HydrogenPlant:
    # Define the time index and forecast data
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        price_EOM=[60] * 24,  # Forecast electricity prices
    )

    # Define a bidding strategy for the hydrogen plant
    bidding_strategy = {
        "EOM": NaiveDADSMStrategy(),
    }

    # Initialize the HydrogenPlant with the specified components, forecast, and strategy
    return HydrogenPlant(
        id="test_hydrogen_plant",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="max_load_shift",
        bidding_strategies=bidding_strategy,
        index=index,
        components=hydrogen_components,
        forecaster=forecast,
        demand=800,  # Total hydrogen demand over the horizon
    )


def test_optimal_operation_without_flex_initialization(hydrogen_plant):
    hydrogen_plant.determine_optimal_operation_without_flex()
    assert (
        hydrogen_plant.opt_power_requirement is not None
    ), "opt_power_requirement should be populated"
    assert isinstance(hydrogen_plant.opt_power_requirement, pd.Series)

    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    total_power_input = sum(
        instance.total_power_input[t].value
        for t in instance.time_steps
        if instance.total_power_input[t].value is not None
    )
    assert total_power_input > 0, "Total power input should be greater than zero"

    # Check hydrogen balance after solving
    for t in instance.time_steps:
        electrolyser_out = instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
        hydrogen_demand = (
            instance.hydrogen_demand[t].value
            if hasattr(instance.hydrogen_demand[t], "value")
            else instance.hydrogen_demand[t]
        )
        storage_charge = instance.dsm_blocks["h2_seasonal_storage"].charge[t].value
        storage_discharge = (
            instance.dsm_blocks["h2_seasonal_storage"].discharge[t].value
        )

        # Check that the balance holds within a small tolerance
        balance_difference = abs(
            electrolyser_out - (hydrogen_demand + storage_charge - storage_discharge)
        )
        assert (
            balance_difference < 1e-3
        ), f"Hydrogen balance mismatch at time {t}, difference: {balance_difference}"


def test_ramping_constraints_without_flex(hydrogen_plant):
    # Test that ramping constraints are respected in operation without flexibility
    hydrogen_plant.determine_optimal_operation_without_flex()
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    # Access ramp_up and ramp_down as attributes
    ramp_up = hydrogen_plant.components["h2_seasonal_storage"].ramp_up
    ramp_down = hydrogen_plant.components["h2_seasonal_storage"].ramp_down

    for t in list(instance.time_steps)[1:]:
        charge_prev = instance.dsm_blocks["h2_seasonal_storage"].charge[t - 1].value
        charge_curr = instance.dsm_blocks["h2_seasonal_storage"].charge[t].value
        discharge_prev = (
            instance.dsm_blocks["h2_seasonal_storage"].discharge[t - 1].value
        )
        discharge_curr = instance.dsm_blocks["h2_seasonal_storage"].discharge[t].value

        # Check charge ramping
        if charge_prev is not None and charge_curr is not None:
            assert (
                abs(charge_curr - charge_prev) <= ramp_up
            ), f"Charge ramp-up at time {t} exceeds limit"
        if discharge_prev is not None and discharge_curr is not None:
            assert (
                abs(discharge_curr - discharge_prev) <= ramp_down
            ), f"Discharge ramp-down at time {t} exceeds limit"


def test_final_soc_target_without_flex(hydrogen_plant):
    hydrogen_plant.determine_optimal_operation_without_flex()
    instance = hydrogen_plant.model.create_instance()
    instance = hydrogen_plant.switch_to_opt(instance)
    hydrogen_plant.solver.solve(instance, tee=False)

    # Access final SOC using integer index
    final_step_index = instance.time_steps[-1]
    final_soc = instance.dsm_blocks["h2_seasonal_storage"].soc[final_step_index].value
    final_soc_target = (
        hydrogen_plant.components["h2_seasonal_storage"].final_soc_target
        * hydrogen_plant.components["h2_seasonal_storage"].max_capacity
    )

    assert final_soc is not None, "Final SOC should not be None"
    assert final_soc >= final_soc_target, "Final SOC does not meet target"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
