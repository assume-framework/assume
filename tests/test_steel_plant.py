# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pyomo.environ as pyo
import pytest

from assume.common.forecasts import NaiveForecast
from assume.strategies.naive_strategies import (
    NaiveDASteelplantStrategy,
    NaiveRedispatchSteelplantStrategy,
)
from assume.units.steel_plant import SteelPlant


@pytest.fixture
def dsm_components():
    return {
        "electrolyser": {
            "max_power": 100,
            "min_power": 0,
            "ramp_up": 100,
            "ramp_down": 100,
            "min_operating_time": 0,
            "min_down_time": 0,
            "efficiency": 1,
        },
        "dri_plant": {
            "specific_hydrogen_consumption": 1,
            "specific_natural_gas_consumption": 1,
            "specific_electricity_consumption": 1,
            "specific_iron_ore_consumption": 1,
            "max_power": 100,
            "min_power": 0,
            "fuel_type": "hydrogen",
            "ramp_up": 100,
            "ramp_down": 100,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
        "eaf": {
            "max_power": 100,
            "min_power": 0,
            "specific_electricity_consumption": 1,
            "specific_dri_demand": 1,
            "specific_lime_demand": 1,
            "lime_co2_factor": 0.1,
            "ramp_up": 100,
            "ramp_down": 100,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
    }


@pytest.fixture
def steel_plant(dsm_components) -> SteelPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        price_EOM=[50] * 24,
        fuel_price_natural_gas=[30] * 24,
        co2_price=[20] * 24,
        east_congestion_severity=[0.5] * 8 + [0.9] * 8 + [0.2] * 8,
        south_renewable_utilisation=[0.1 * i for i in range(24)],
    )

    bidding_strategies = {
        "EOM": NaiveDASteelplantStrategy(),
        "RD": NaiveRedispatchSteelplantStrategy(),
    }
    return SteelPlant(
        id="test_steel_plant",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies=bidding_strategies,
        index=index,
        node="south",
        components=dsm_components,
        forecaster=forecast,
        demand=1000,
        technology="steel_plant",
        cost_tolerance=10,
        congestion_threshold=0.8,
        peak_load_cap=95,
    )


def test_initialize_components(steel_plant):
    assert "electrolyser" in steel_plant.model.dsm_blocks.keys()
    assert "dri_plant" in steel_plant.model.dsm_blocks.keys()
    assert "eaf" in steel_plant.model.dsm_blocks.keys()


def test_determine_optimal_operation_without_flex(steel_plant):
    steel_plant.determine_optimal_operation_without_flex()
    assert steel_plant.opt_power_requirement is not None
    assert isinstance(steel_plant.opt_power_requirement, pd.Series)

    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_opt(instance)
    steel_plant.solver.solve(instance, tee=False)

    total_power_input = sum(
        instance.total_power_input[t].value for t in instance.time_steps
    )

    assert total_power_input == 3000

    for t in instance.time_steps:
        hydrogen_out = instance.dsm_blocks["electrolyser"].hydrogen_out[t].value
        hydrogen_in = instance.dsm_blocks["dri_plant"].hydrogen_in[t].value
        assert (
            hydrogen_out >= hydrogen_in
        ), f"Hydrogen out at time {t} is less than hydrogen in"

    for t in instance.time_steps:
        dri_output = instance.dsm_blocks["dri_plant"].dri_output[t].value
        dri_input = instance.dsm_blocks["eaf"].dri_input[t].value
        assert (
            dri_output == dri_input
        ), f"DRI output at time {t} does not match DRI input"

    # for t in instance.time_steps:
    #     dri_output = instance.dsm_blocks["dri_plant"].dri_output[t].value
    #     dri_input = instance.dsm_blocks["eaf"].dri_input[t].value
    #     assert (
    #         dri_output == dri_input
    #     ), f"Material flow from DRI plant to EAF at time {t} is inconsistent"

    total_steel_output = sum(
        instance.dsm_blocks["eaf"].steel_output[t].value for t in instance.time_steps
    )
    assert (
        total_steel_output == instance.steel_demand
    ), f"Total steel output {total_steel_output} does not match steel demand {instance.steel_demand}"


def test_ramping_constraints(steel_plant):
    steel_plant.determine_optimal_operation_without_flex()
    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_opt(instance)
    steel_plant.solver.solve(instance, tee=False)

    # Loop through time steps to check that the electrolyser ramp constraints hold
    for t in list(instance.time_steps)[1:]:
        # Current and previous power values for electrolyser
        power_prev = instance.dsm_blocks["electrolyser"].power_in[t - 1].value
        power_curr = instance.dsm_blocks["electrolyser"].power_in[t].value

        # Access ramp constraints using dot notation
        ramp_up = steel_plant.components["electrolyser"].ramp_up
        ramp_down = steel_plant.components["electrolyser"].ramp_down

        # Verify ramping constraints
        assert (
            power_curr - power_prev <= ramp_up
        ), f"Electrolyser ramp-up constraint violated at time {t}"
        assert (
            power_prev - power_curr <= ramp_down
        ), f"Electrolyser ramp-down constraint violated at time {t}"


def test_handle_missing_components():
    # Check for handling missing required components, as the SteelPlant requires dri_plant for initialization
    with pytest.raises(
        ValueError, match="Component dri_plant is required for the steel plant unit."
    ):
        _ = SteelPlant(
            id="test_steel_plant",
            unit_operator="test_operator",
            objective="min_variable_cost",
            flexibility_measure="cost_based_load_shift",
            bidding_strategies={"EOM": NaiveDASteelplantStrategy()},
            index=pd.date_range("2023-01-01", periods=24, freq="h"),
            node="south",
            components={},  # No components provided
            forecaster=None,
            demand=1000,
            technology="unknown_tech",
            cost_tolerance=10,
            congestion_threshold=0.8,
            peak_load_cap=15,
        )


def test_determine_optimal_operation_with_flex(steel_plant):
    # Ensure that the optimal operation without flexibility is determined first
    steel_plant.determine_optimal_operation_without_flex()
    assert steel_plant.opt_power_requirement is not None
    assert isinstance(steel_plant.opt_power_requirement, pd.Series)

    steel_plant.determine_optimal_operation_with_flex()
    assert steel_plant.flex_power_requirement is not None
    assert isinstance(steel_plant.flex_power_requirement, pd.Series)

    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_flex(instance)
    steel_plant.solver.solve(instance, tee=False)

    total_power_input = sum(
        instance.total_power_input[t].value for t in instance.time_steps
    )

    assert total_power_input == 3000


# New fixture without electrolyser to test line 89
@pytest.fixture
def steel_plant_without_electrolyser(dsm_components) -> SteelPlant:
    index = pd.date_range("2023-01-01", periods=24, freq="h")
    forecast = NaiveForecast(
        index,
        price_EOM=[50] * 24,
        fuel_price_natural_gas=[30] * 24,
        co2_price=[20] * 24,
        east_congestion_severity=[0.5] * 8 + [0.9] * 8 + [0.2] * 8,
        south_renewable_utilisation=[
            0,
            1,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
    )

    # Remove 'electrolyser' from the components to trigger line 89
    dsm_components_without_electrolyser = dsm_components.copy()
    dsm_components_without_electrolyser.pop("electrolyser", None)

    bidding_strategies = {
        "EOM": NaiveDASteelplantStrategy(),
        "RD": NaiveRedispatchSteelplantStrategy(),
    }
    return SteelPlant(
        id="test_steel_plant_without_electrolyser",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="cost_based_load_shift",
        bidding_strategies=bidding_strategies,
        index=index,
        components=dsm_components_without_electrolyser,
        forecaster=forecast,
        demand=1000,
        technology="steel_plant",
    )


# Test cases


def test_handle_missing_electrolyser(steel_plant_without_electrolyser):
    # This should raise an error because 'electrolyser' is required for steel plant
    steel_plant_without_electrolyser.determine_optimal_operation_without_flex()
    instance = steel_plant_without_electrolyser.model.create_instance()
    instance = steel_plant_without_electrolyser.switch_to_opt(instance)
    steel_plant_without_electrolyser.solver.solve(instance, tee=False)

    # Loop through time steps to check that the electrolyser ramp constraints hold
    for t in list(instance.time_steps)[1:]:
        # Current and previous power values for electrolyser
        power_prev = instance.dsm_blocks["dri_plant"].power_in[t - 1].value
        power_curr = instance.dsm_blocks["dri_plant"].power_in[t].value

        # Access ramp constraints using dot notation
        ramp_up = steel_plant_without_electrolyser.components["dri_plant"].ramp_up
        ramp_down = steel_plant_without_electrolyser.components["dri_plant"].ramp_down

        # Verify ramping constraints
        assert (
            power_curr - power_prev <= ramp_up
        ), f"Dri plant ramp-up constraint violated at time {t}"
        assert (
            power_prev - power_curr <= ramp_down
        ), f"Dri plant ramp-down constraint violated at time {t}"


def test_congestion_management_flexibility(steel_plant):
    steel_plant.flexibility_measure = "congestion_management_flexibility"
    steel_plant.determine_optimal_operation_with_flex()

    # Calculate the congestion indicator
    congestion_indicator = {
        t: int(steel_plant.congestion_signal[t] > steel_plant.congestion_threshold)
        for t in range(len(steel_plant.index))
    }

    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_flex(instance)

    # Set the congestion_indicator in the instance
    instance.congestion_indicator = pyo.Param(
        instance.time_steps,
        initialize=congestion_indicator,
        within=pyo.Binary,
    )

    # Solve the instance
    steel_plant.solver.solve(instance, tee=False)

    # Calculate adjusted total power input
    adjusted_total_power_input = [
        instance.total_power_input[t].value
        + instance.load_shift_pos[t].value
        - instance.load_shift_neg[t].value
        for t in instance.time_steps
    ]

    # Assert load shifting respects congestion indicator
    for t in instance.time_steps:
        if instance.congestion_indicator[t] == 1:  # Congestion period
            assert (
                adjusted_total_power_input[t] <= instance.total_power_input[t].value
            ), f"Load shift not aligned with congestion signal at time {t}"


def test_peak_load_shifting(steel_plant):
    steel_plant.flexibility_measure = "peak_load_shifting"
    steel_plant.determine_optimal_operation_with_flex()

    # Calculate the peak load cap value
    max_load = steel_plant.opt_power_requirement.max()
    peak_load_cap_value = max_load * (steel_plant.peak_load_cap / 100)
    peak_indicator = {
        t: int(steel_plant.opt_power_requirement[t] > peak_load_cap_value)
        for t in range(len(steel_plant.opt_power_requirement))
    }

    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_flex(instance)

    # Set the peak_load_cap_value in the instance
    instance.peak_load_cap_value = pyo.Param(
        initialize=peak_load_cap_value,
        within=pyo.NonNegativeReals,
    )
    instance.peak_indicator = pyo.Param(
        instance.time_steps,
        initialize=peak_indicator,
        within=pyo.Binary,
    )

    # Solve the instance
    steel_plant.solver.solve(instance, tee=False)

    # Calculate adjusted total power input
    adjusted_total_power_input = [
        instance.total_power_input[t].value
        + instance.load_shift_pos[t].value
        - instance.load_shift_neg[t].value
        for t in instance.time_steps
    ]

    # Assert load shifting respects peak load cap
    for t in instance.time_steps:
        if instance.peak_indicator[t] == 1:
            assert (
                adjusted_total_power_input[t] <= instance.peak_load_cap_value
            ), f"Peak load exceeded at time {t}"


def test_renewable_utilisation(steel_plant):
    """
    Tests the renewable utilisation flexibility measure by ensuring that the load increase aligns
    with the renewable signal intensity and does not exceed allowable thresholds.
    """
    # Set the flexibility measure to renewable utilisation
    steel_plant.flexibility_measure = "renewable_utilisation"
    steel_plant.determine_optimal_operation_with_flex()

    # Safeguard normalization of renewable_utilisation_signal
    min_signal = steel_plant.renewable_utilisation_signal.min()
    max_signal = steel_plant.renewable_utilisation_signal.max()

    if max_signal - min_signal > 0:
        renewable_signal_normalised = (
            steel_plant.renewable_utilisation_signal - min_signal
        ) / (max_signal - min_signal)
    else:
        renewable_signal_normalised = pd.Series(
            1, index=steel_plant.renewable_utilisation_signal.index
        )

    # Map normalized renewable signals to a dictionary for Pyomo parameters
    renewable_signal_dict = {
        t: renewable_signal_normalised[t]
        for t in range(len(renewable_signal_normalised))
    }

    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_flex(instance)

    # Set the normalized renewable signal in the instance
    instance.renewable_signal = pyo.Param(
        instance.time_steps,
        initialize=renewable_signal_dict,
        within=pyo.NonNegativeReals,
    )

    # Solve the instance
    steel_plant.solver.solve(instance, tee=False)

    # Calculate adjusted total power input
    adjusted_total_power_input = [
        instance.total_power_input[t].value
        + instance.load_shift_pos[t].value
        - instance.load_shift_neg[t].value
        for t in instance.time_steps
    ]

    # Assert load shifting respects renewable signal intensity
    for t in instance.time_steps:
        assert (
            adjusted_total_power_input[0] <= adjusted_total_power_input[3]
        ), f"Load shift exceeds renewable intensity signal at time {t}"


def test_total_cost_upper_limit_with_flexibility(steel_plant):
    """
    Test if the total cost with flexibility adheres to the cost tolerance limits.
    """
    steel_plant.determine_optimal_operation_with_flex()

    instance = steel_plant.model.create_instance()
    instance = steel_plant.switch_to_flex(instance)
    steel_plant.solver.solve(instance, tee=False)

    total_cost = sum(instance.variable_cost[t].value for t in instance.time_steps)
    assert total_cost <= steel_plant.total_cost * (
        1 + steel_plant.cost_tolerance / 100
    ), "Total cost exceeds the cost tolerance limit"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
