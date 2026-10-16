# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from pathlib import Path

import pandas as pd
import pytest

from assume.scenario.loader_csv import (
    build_flexible_loads,
    get_unit_forecast_algorithms,
    load_config_and_create_forecaster,
    setup_world,
)
from assume.world import World


def test_csv_loader_validation():
    with pytest.raises(
        ValueError,
        match="min_power and max_power must both be either negative or positive",
    ):
        load_config_and_create_forecaster(
            inputs_path="tests/fixtures", scenario="invalid_units", study_case="base"
        )
    with pytest.raises(
        ValueError, match="No power plant or no demand units were provided!"
    ):
        load_config_and_create_forecaster(
            inputs_path="tests/fixtures", scenario="missing_units", study_case="base"
        )


def test_csv_loader_forecaster_algorithms():
    """
    Testing forecast_algorithm loading. This includes:
        setting via config
        setting via csv (overwrites config if not None)
    for forecast, preprocess and update algorithms.

    Special cases:
        setting specific forecasts for config and csv that are not present in the other one
    """
    scenario_data = load_config_and_create_forecaster(
        inputs_path="tests/fixtures", scenario="different_forecasts", study_case="base"
    )

    forecasters = scenario_data["unit_forecasts"]

    expected_forcast_algorithms = {
        "Unit 1": {
            "price": "price_naive_forecast",
            "residual_load": "not_yet_implemented_learnable_forecast",
            "congestion_signal": "congestion_signal_special_forecast",
            "new_specific_forecast": "new_naive_forecast",
            "other": None,
        },
        "Unit 2": {
            "price": "price_naive_forecast",
            "residual_load": "default_test",
            "congestion_signal": "congestion_signal_special_forecast",
            "new_specific_forecast": "new_naive_forecast",
            "other": "default_test",
        },
        "Unit 3": {
            "price": "price_naive_forecast",
            "residual_load": "residual_load_naive_forecast",
            "congestion_signal": "congestion_signal_special_forecast",
            "new_specific_forecast": "new_naive_forecast",
            "other": "nothing_special",
        },
        "Unit 4": {
            "price": "default_test",
            "residual_load": "residual_load_naive_forecast",
            "congestion_signal": "congestion_signal_special_forecast",
            "new_specific_forecast": "new_naive_forecast",
            "other": "nothing_special",
        },
        "demand_EOM": {
            "price": "price_naive_forecast",
            "residual_load": "not_yet_implemented_learnable_forecast",
            "congestion_signal": "congestion_signal_special_forecast",
            "new_specific_forecast": "new_naive_forecast",
        },
    }

    # Also include preprocess and update algorithms
    for key in expected_forcast_algorithms:
        expected_forcast_algorithms[key]["preprocess_price"] = "price_default"
        expected_forcast_algorithms[key]["preprocess_residual_load"] = (
            "residual_load_default"
        )
        expected_forcast_algorithms[key]["preprocess_congestion_signal"] = (
            "congestion_signal_default"
        )
        expected_forcast_algorithms[key]["preprocess_renewable_utilisation"] = (
            "renewable_utilisation_special"
        )

        expected_forcast_algorithms[key]["update_price"] = "price_default"
        expected_forcast_algorithms[key]["update_residual_load"] = "residual_load_fancy"
        expected_forcast_algorithms[key]["update_congestion_signal"] = (
            "congestion_signal_default"
        )
        expected_forcast_algorithms[key]["update_renewable_utilisation"] = (
            "renewable_utilisation_special"
        )

    expected_forcast_algorithms["Unit 1"]["preprocess_residual_load"] = (
        "residual_load_prepare_multiple"
    )
    expected_forcast_algorithms["Unit 2"]["update_congestion_signal"] = (
        "congestion_signal_neural_net"
    )
    expected_forcast_algorithms["Unit 3"]["update_congestion_signal"] = (
        "congestion_signal_special"
    )

    for unit, forecaster in forecasters.items():
        for (
            forecast_type,
            forecast_algorithm_id,
        ) in forecaster.forecast_algorithms.items():
            assert (
                forecast_algorithm_id
                == expected_forcast_algorithms[unit][forecast_type]
            ), (
                f"{unit}, forecast type: {forecast_type}, {forecast_algorithm_id} != {expected_forcast_algorithms[unit][forecast_type]}"
            )


def test_get_unit_forecast_algorithms():
    """
    Test the function: get_unit_forecast_algorithms
        This function gets correct forecast_names from config and csvs.
        csv will overwrite config values (except if csv has None types)
        Further, this checks, that only forecast_names with prefix "forecast_" are used from csv
    """
    powerplant_dict = {
        "forecast_test": "value1",
        "forecast_test2": "value2",
        "test3": "value3",
        "forecast_test5": None,  # Make sure that config values can overwrite None!
    }
    powerplant_dict_copy = powerplant_dict.copy()

    config_dict = {"test": "other", "test4": "other2", "test5": "other3"}
    config_dict_copy = config_dict.copy()

    expected_output = {
        "test": "value1",  # from powerplant_dict
        "test2": "value2",  # from powerplant_dict
        "test4": "other2",  # from config_dict
        "test5": "other3",  # from config_dict (powerplant_dict has None)
    }

    output = get_unit_forecast_algorithms(config_dict, powerplant_dict)

    # Make sure powerplant_dict is not overwritten
    for key in powerplant_dict:
        assert powerplant_dict[key] == powerplant_dict_copy[key]

    # Make sure config_dict is not overwritten
    for key in config_dict:
        assert config_dict[key] == config_dict_copy[key]

    # Make sure only strings with "forecast_" are accepted
    for key in expected_output:
        assert expected_output[key] == output[key]


def test_forecast_interface__save_forecasts():
    """
    Tests that forecasts are saved correctly
    """
    path = Path("./tests/fixtures/forecast_save")
    expected_price = pd.read_csv(path / "results/price.csv")
    expected_load = pd.read_csv(path / "results/load_forecast.csv")
    world = World()
    world.scenario_data = load_config_and_create_forecaster(
        inputs_path="tests/fixtures", scenario="forecast_save", study_case="base"
    )
    setup_world(world=world)

    saved_forecasts = pd.read_csv(path / "outputs/saved_forecasts.csv")
    assert (
        saved_forecasts["price_naive_forecast_EOM"] == expected_price["price"]
    ).all()
    assert (
        saved_forecasts["residual_load_naive_forecast_EOM"]
        == expected_load["load_forecast"]
    ).all()


def test_build_flexible_loads_derives_electrical_p_nom_from_components():
    """
    build_flexible_loads() derives the grid-facing nominal electrical
    capacity from directly electricity-consuming components.
    """
    dsm_units = {
        "sector_coupled_system": pd.DataFrame(
            {
                "node": ["N", "E", "S"],
                "components": [
                    {
                        "heat_pump": {"max_power": 2.0},
                        "boiler_gas": {"max_power": 5.0, "fuel_type": "natural_gas"},
                        "thermal_storage": {"capacity": 8.0},
                    },
                    {
                        "heat_pump": {"max_power": 3.0},
                        "boiler_gas": {"max_power": 7.0, "fuel_type": "natural_gas"},
                    },
                    {"electrolyser": {"max_power": 8.0}},
                ],
            },
            index=["hs_north", "hs_east", "h2_south"],
        )
    }

    flexible_loads = build_flexible_loads(dsm_units)

    assert set(flexible_loads.index) == {"hs_north", "hs_east", "h2_south"}
    assert flexible_loads.loc["hs_north", "node"] == "N"
    assert flexible_loads.loc["hs_north", "p_nom"] == 2.0
    assert flexible_loads.loc["hs_east", "node"] == "E"
    assert flexible_loads.loc["hs_east", "p_nom"] == 3.0
    assert flexible_loads.loc["h2_south", "node"] == "S"
    assert flexible_loads.loc["h2_south", "p_nom"] == 8.0
    assert (flexible_loads["unit_type"] == "sector_coupled_system").all()


def test_build_flexible_loads_excludes_non_electrical_dsm_units():
    """
    A DSM parent without a directly electricity-consuming component
    must not create a redispatch flexible-load asset.
    """
    dsm_units = {
        "sector_coupled_system": pd.DataFrame(
            {
                "node": ["N", "S"],
                "components": [
                    {
                        "heat_pump": {"max_power": 8.0},
                        "thermal_storage": {"capacity": 20.0},
                    },
                    {
                        "boiler_gas": {"max_power": 8.0, "fuel_type": "natural_gas"},
                        "thermal_storage": {"capacity": 20.0},
                    },
                ],
            },
            index=["heat_with_hp", "thermal_only"],
        )
    }

    flexible_loads = build_flexible_loads(dsm_units)

    assert "heat_with_hp" in flexible_loads.index
    assert flexible_loads.loc["heat_with_hp", "p_nom"] == 8.0
    assert "thermal_only" not in flexible_loads.index


def test_build_flexible_loads_handles_empty_dsm_input():
    flexible_loads = build_flexible_loads({})
    assert flexible_loads.empty
    assert {"node", "p_nom"}.issubset(flexible_loads.columns)


if __name__ == "__main__":
    pytest.main(["-s", __file__])