# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
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
            "rated_power": 100,
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
            "rated_power": 100,
            "min_power": 0,
            "fuel_type": "hydrogen",
            "ramp_up": 100,
            "ramp_down": 100,
            "min_operating_time": 0,
            "min_down_time": 0,
        },
        "eaf": {
            "rated_power": 100,
            "min_power": 0,
            "specific_electricity_consumption": 1,
            "specific_dri_demand": 1,
            "specific_lime_demand": 1,
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
    )

    bidding_strategies = {
        "EOM": NaiveDASteelplantStrategy(),
        "RD": NaiveRedispatchSteelplantStrategy(),
    }
    return SteelPlant(
        id="test_steel_plant",
        unit_operator="test_operator",
        objective="min_variable_cost",
        flexibility_measure="max_load_shift",
        bidding_strategies=bidding_strategies,
        index=index,
        components=dsm_components,
        forecaster=forecast,
        demand=1000,
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

    for t in instance.time_steps:
        dri_output = instance.dsm_blocks["dri_plant"].dri_output[t].value
        dri_input = instance.dsm_blocks["eaf"].dri_input[t].value
        assert (
            dri_output == dri_input
        ), f"Material flow from DRI plant to EAF at time {t} is inconsistent"

    total_steel_output = sum(
        instance.dsm_blocks["eaf"].steel_output[t].value for t in instance.time_steps
    )
    assert (
        total_steel_output == instance.steel_demand
    ), f"Total steel output {total_steel_output} does not match steel demand {instance.steel_demand}"


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


if __name__ == "__main__":
    pytest.main(["-s", __file__])
