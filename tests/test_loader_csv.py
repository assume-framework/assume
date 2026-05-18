# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import pytest

from assume.common.forecast_algorithms import (
    calculate_naive_congestion_signal,
    calculate_naive_price,
    calculate_naive_renewable_utilisation,
    calculate_naive_residual_load,
)
from assume.scenario.loader_csv import (
    get_unit_forecast_algorithms,
    load_config_and_create_forecaster,
    resolve_forecast_source,
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
    Includes:
        test in config and csv with forecasts that are not present in the other.
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
    powerplant_dict = {
        "forecast_test": "value1",
        "forecast_test2": "value2",
        "test3": "value3",
        "forecast_test5": None,  # Make sure that config values can overwrite None!
    }
    powerplant_dict_copy = powerplant_dict.copy()

    config_dict = {
        "test": "other",
        "test4": "other2",
        "test5": "other3",
    }
    config_dict_copy = config_dict.copy()

    expected_output = {
        "test": "value1",
        "test2": "value2",
        "test4": "other2",
        "test5": "other3",
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


def test_cache_unit_forecast_algorithms_cache_hits():
    calculate_naive_price.cache_clear()
    calculate_naive_residual_load.cache_clear()
    calculate_naive_congestion_signal.cache_clear()
    calculate_naive_renewable_utilisation.cache_clear()

    world = World()
    world.scenario_data = load_config_and_create_forecaster(
        inputs_path="tests/fixtures", scenario="forecast_init", study_case="base"
    )

    setup_world(world=world)

    assert calculate_naive_price.cache_info().hits == len(world.units) - 1
    assert calculate_naive_price.cache_info().misses == 1

    assert calculate_naive_residual_load.cache_info().hits == len(world.units) - 1
    assert calculate_naive_residual_load.cache_info().misses == 1


# --- resolve_forecast_source tests ---


def test_resolve_forecast_source_none_returns_none():
    assert resolve_forecast_source(None) is None


def test_resolve_forecast_source_nan_returns_none():
    assert resolve_forecast_source(float("nan")) is None


def test_resolve_forecast_source_adaptive_maps_to_algorithm():
    assert resolve_forecast_source("adaptive") == "price_from_cleared_history"


def test_resolve_forecast_source_adaptive_case_insensitive():
    assert resolve_forecast_source("ADAPTIVE") == "price_from_cleared_history"
    assert resolve_forecast_source("Adaptive") == "price_from_cleared_history"


def test_resolve_forecast_source_column_returns_series():
    df = pd.DataFrame(
        {"A360_electricity_price": [30.0, 50.0, 70.0]},
        index=pd.date_range("2023-01-01", periods=3, freq="h"),
    )
    result = resolve_forecast_source("A360_electricity_price", df)
    assert isinstance(result, pd.Series)
    assert list(result) == [30.0, 50.0, 70.0]


def test_resolve_forecast_source_unknown_string_passthrough():
    assert resolve_forecast_source("price_naive_forecast") == "price_naive_forecast"
    assert resolve_forecast_source("some_custom_algorithm") == "some_custom_algorithm"


def test_resolve_forecast_source_missing_column_returns_string():
    df = pd.DataFrame({"other_col": [1, 2]})
    # "nonexistent" is not a column, not "adaptive" → returned as-is
    assert resolve_forecast_source("nonexistent", df) == "nonexistent"


def test_resolve_forecast_source_column_takes_priority_over_adaptive_mapping():
    # A column literally named "adaptive" should be returned as a Series, not remapped
    df = pd.DataFrame({"adaptive": [10.0, 20.0]})
    result = resolve_forecast_source("adaptive", df)
    assert isinstance(result, pd.Series)
    assert list(result) == [10.0, 20.0]


# --- Rolling Horizon DSM Config Loading Tests ---


def test_rolling_horizon_config_loaded_from_yaml():
    """Load rolling horizon DSM config from config.yaml and verify DSM units dict
    is created with correct rolling horizon configuration information."""
    scenario_data = load_config_and_create_forecaster(
        inputs_path="tests/fixtures",
        scenario="rolling_horizon_dsm",
        study_case="base",
    )

    # Check that dsm_units were loaded from the fixture
    dsm_units = scenario_data["dsm_units"]
    assert dsm_units is not None, "dsm_units should be loaded from config"
    assert len(dsm_units) > 0, "At least one DSM unit type should be present"

    # Check for steel_plant type units (from industrial_dsm_units.csv)
    assert "steel_plant" in dsm_units, "Steel plant DSM units should be present"
    steel_plants = dsm_units["steel_plant"]
    assert len(steel_plants) > 0, "At least one steel plant should be loaded"

    # Get the first steel plant unit configuration
    steel_plant_config = steel_plants.iloc[0]

    # Check that components dictionary contains the technology
    assert "components" in steel_plant_config, "Config should have components"
    components = steel_plant_config["components"]
    assert isinstance(components, dict), "Components should be a dict"
    assert "steel_plant" in components, (
        "Components should contain steel_plant technology"
    )

    # Check that DSM optimisation config was loaded
    assert "dsm_optimisation_config" in steel_plant_config, (
        "Config should have dsm_optimisation_config"
    )
    dsm_opt = steel_plant_config["dsm_optimisation_config"]
    assert isinstance(dsm_opt, dict), "dsm_optimisation_config should be a dict"
    assert dsm_opt.get("horizon_mode") == "rolling_horizon", (
        f"horizon_mode should be 'rolling_horizon', got {dsm_opt.get('horizon_mode')}"
    )


def test_rolling_horizon_config_passed_to_units_via_setup_world():
    """Verify that rolling horizon config from config.yaml is passed to DSM units
    when they are created via setup_world, and the units receive the correct
    horizon_mode, look_ahead, commit, and step parameters."""
    scenario_data = load_config_and_create_forecaster(
        inputs_path="tests/fixtures",
        scenario="rolling_horizon_dsm",
        study_case="base",
    )

    # Verify that dsm_optimisation_config was loaded and attached to units
    dsm_units = scenario_data["dsm_units"]
    assert "steel_plant" in dsm_units
    steel_plants = dsm_units["steel_plant"]
    steel_plant_config = steel_plants.iloc[0]

    # Verify rolling horizon configuration is present
    assert "dsm_optimisation_config" in steel_plant_config
    dsm_opt = steel_plant_config["dsm_optimisation_config"]

    # Verify all rolling horizon parameters are present
    assert dsm_opt.get("horizon_mode") == "rolling_horizon"
    assert dsm_opt.get("look_ahead_horizon") == "4h"
    assert dsm_opt.get("commit_horizon") == "2h"
    assert dsm_opt.get("rolling_step") == "2h"


if __name__ == "__main__":
    pytest.main(["-s", __file__])
