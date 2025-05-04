# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import logging
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import dateutil.rrule as rr
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from assume.common.base import LearningConfig
from assume.common.exceptions import AssumeException
from assume.common.forecasts import CsvForecaster, Forecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import (
    adjust_unit_operator_for_learning,
    convert_to_rrule_freq,
    normalize_availability,
)
from assume.strategies import BaseStrategy
from assume.world import World

logger = logging.getLogger(__name__)


def load_file(
    path: str,
    config: dict,
    file_name: str,
    index: pd.DatetimeIndex | None = None,
    check_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Loads a csv file from the given path and returns a dataframe.

    The config file is used to check if the file name is specified in the config file,
    otherwise defaults to the file name.

    If the index is specified, the dataframe is resampled to the index, if possible. If not, None is returned.

    Args:
        path (str): The path to the csv file.
        config (dict): The config file containing file mappings.
        file_name (str): The name of the csv file.
        index (pd.DatetimeIndex, optional): The index of the dataframe. Defaults to None.
        check_duplicates (bool, optional): Whether to check for duplicate unit names. Defaults to True.

    Returns:
        pandas.DataFrame: The dataframe containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file is not found, returns None.
    """
    df = None

    if file_name in config:
        if config[file_name] is None:
            return None
        file_path = f"{path}/{config[file_name]}"
    else:
        file_path = f"{path}/{file_name}.csv"

    try:
        df = pd.read_csv(
            file_path,
            index_col=0,
            encoding="utf-8",
            na_values=["n.a.", "None", "-", "none", "nan"],
            parse_dates=index is not None,
        )

        for col in df:
            # check if the column is of dtype int
            if df[col].dtype == "int":
                # convert the column to float
                df[col] = df[col].astype(float)

        if index is not None:
            if len(df.index) == 1:
                return df

            if len(df.index) != len(index) and not isinstance(
                df.index, pd.DatetimeIndex
            ):
                logger.warning(
                    f"{file_name}: simulation time line does not match length of dataframe and index is not a datetimeindex. Returning None."
                )
                return None

            df.index.freq = df.index.inferred_freq

            if len(df.index) < len(index) and df.index.freq == index.freq:
                logger.warning(
                    f"{file_name}: simulation time line is longer than length of the dataframe. Returning None."
                )
                return None

            if df.index.freq < index.freq:
                logger.warning(
                    f"Resolution of {file_name} ({df.index.freq}) is higher than the simulation ({index.freq}). "
                    "Resampling using mean(). Make sure this is what you want."
                )
                df = df.resample(index.freq).mean()
                logger.info(f"Downsampling {file_name} successful.")

            elif df.index.freq > index.freq or len(df.index) < len(index):
                logger.warning("Upsampling not implemented yet. Returning None.")
                return None

            df = df.loc[index]

        elif check_duplicates:
            # Check if duplicate unit names exist and raise an error
            duplicates = df.index[df.index.duplicated()].unique()

            if len(duplicates) > 0:
                duplicate_names = ", ".join(duplicates)
                raise ValueError(
                    f"Duplicate unit names found in {file_name}: {duplicate_names}. Please rename them to avoid conflicts."
                )

        return df

    except FileNotFoundError:
        logger.info(f"{file_name} not found. Returning None")
        return None


def load_dsm_units(
    path: str,
    config: dict,
    file_name: str,
) -> dict:
    """
    Loads and processes a CSV file containing DSM unit data, where each unit may consist of multiple components
    (technologies) under the same plant name. The function groups data by plant name, processes each group to
    handle different technologies, and organizes the data into a structured DataFrame. It then splits the DataFrame
    based on unique unit_types.

    Parameters:
        path (str): The directory path where the CSV file is located.
        config (dict): Configuration dictionary, potentially used for specifying additional options or behaviors
                       (not used in the current implementation but provides flexibility for future enhancements).
        file_name (str): The name of the CSV file to be loaded.

    Returns:
        dict: A dictionary where each key is a unique unit_type and the value is a DataFrame containing
              the corresponding DSM units of that type.

    Notes:
        - The CSV file is expected to have columns such as 'name', 'technology', 'unit_type', and other operational parameters.
        - The function assumes that the first non-null value in common and bidding columns is representative if multiple
          entries exist for the same plant.
        - It is crucial that the input CSV file follows the expected structure for the function to process it correctly.
    """

    # Load the DSM units file
    # Note: check_duplicates is set to False to avoid raising an error for duplicate unit names
    dsm_units = load_file(
        path=path,
        config=config,
        file_name=file_name,
        check_duplicates=False,
    )

    if dsm_units is None:
        return None

    # Define columns that are common across different technologies within the same plant
    common_columns = [
        "unit_operator",
        "objective",
        "demand",
        "cost_tolerance",
        "unit_type",
        "node",
        "flexibility_measure",
        "is_prosumer",
        "congestion_threshold",
        "peak_load_cap",
    ]
    # Filter the common columns to only include those that exist in the DataFrame
    common_columns = [col for col in common_columns if col in dsm_units.columns]

    # Get bidding columns dynamically
    bidding_columns = [col for col in dsm_units.columns if col.startswith("bidding_")]

    # Initialize the dictionary to hold the final structured data
    dsm_units_dict = {}

    # Process each group of components by plant name or building name
    for name, group in dsm_units.groupby(dsm_units.index):
        dsm_unit = {}

        # Aggregate or select appropriate data for available common and bidding columns
        for col in common_columns + bidding_columns:
            non_null_values = group[col].dropna()
            if not non_null_values.empty:
                dsm_unit[col] = non_null_values.iloc[0]

        # Process each technology within the plant
        components = {}
        for tech, tech_data in group.groupby("technology"):
            # Clean the technology-specific data: drop all-NaN columns and drop 'technology', common, and bidding columns
            cleaned_data = tech_data.dropna(axis=1, how="all").drop(
                columns=["technology"] + common_columns + bidding_columns,
                errors="ignore",
            )
            # Ensure that there is at least one record before adding to components
            if not cleaned_data.empty:
                components[tech] = cleaned_data.to_dict(orient="records")[0]

        dsm_unit["components"] = components
        dsm_units_dict[name] = dsm_unit

    # Convert the structured dictionary into a DataFrame
    dsm_units_df = pd.DataFrame.from_dict(dsm_units_dict, orient="index")

    # Split the DataFrame based on unit_type
    unit_type_dict = {}
    if "unit_type" in dsm_units_df.columns:
        for unit_type in dsm_units_df["unit_type"].unique():
            unit_type_dict[unit_type] = dsm_units_df[
                dsm_units_df["unit_type"] == unit_type
            ]

    return unit_type_dict


def replace_paths(config: dict, inputs_path: str):
    """
    This function replaces all config items which end with "_path"
    to one starting with the given inputs_path.
    So that paths in the config are relative to the inputs_path where the config is read from.

    Args:
        config (dict): the config dict read from yaml
        inputs_path (str): the base path from the config

    Returns:
        dict: the adjusted config dict
    """

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict | list):
                config[key] = replace_paths(value, inputs_path)
            elif isinstance(key, str) and key.endswith("_path") and value is not None:
                if not value.startswith(inputs_path):
                    config[key] = inputs_path + "/" + value
    elif isinstance(config, list):
        for i, item in enumerate(config):
            config[i] = replace_paths(item, inputs_path)
    return config


def make_market_config(
    id: str,
    market_params: dict,
    world_start: datetime,
    world_end: datetime,
) -> MarketConfig:
    """
    Create a market config from a given dictionary.

    Args:
    id (str): The id of the market.
    market_params (dict): The market parameters.
    world_start (datetime.datetime): The start time of the world.
    world_end (datetime.datetime): The end time of the world.

    Returns:
    MarketConfig: The market config.
    """
    freq, interval = convert_to_rrule_freq(market_params["opening_frequency"])
    start = market_params.get("start_date")
    end = market_params.get("end_date")
    if start:
        start = pd.Timestamp(start)
    if end:
        end = pd.Timestamp(end)
    start = start or world_start
    end = end or world_end

    market_products = [
        MarketProduct(
            duration=pd.Timedelta(product["duration"]),
            count=product["count"],
            first_delivery=pd.Timedelta(product["first_delivery"]),
        )
        for product in market_params["products"]
    ]
    market_config = MarketConfig(
        market_id=id,
        market_products=market_products,
        product_type=market_params.get("product_type", "energy"),
        opening_hours=rr.rrule(
            freq=freq,
            interval=interval,
            dtstart=start,
            until=end,
            cache=True,
        ),
        opening_duration=pd.Timedelta(market_params["opening_duration"]),
        market_mechanism=market_params["market_mechanism"],
        maximum_bid_volume=market_params.get("maximum_bid_volume", 1e6),
        maximum_bid_price=market_params.get("maximum_bid_price", 3000),
        minimum_bid_price=market_params.get("minimum_bid_price", -3000),
        maximum_gradient=market_params.get("max_gradient"),
        volume_unit=market_params.get("volume_unit", "MW"),
        volume_tick=market_params.get("volume_tick"),
        price_unit=market_params.get("price_unit", "€/MWh"),
        price_tick=market_params.get("price_tick"),
        additional_fields=market_params.get("additional_fields", []),
        supports_get_unmatched=market_params.get("supports_get_unmatched", False),
        param_dict=market_params.get("param_dict", {}),
    )

    return market_config


def read_grid(network_path: str | Path) -> dict[str, pd.DataFrame]:
    network_path = Path(network_path)
    buses = pd.read_csv(network_path / "buses.csv", index_col=0)
    lines = pd.read_csv(network_path / "lines.csv", index_col=0)
    generators = pd.read_csv(network_path / "powerplant_units.csv", index_col=0)
    loads = pd.read_csv(network_path / "demand_units.csv", index_col=0)

    return {
        "buses": buses,
        "lines": lines,
        "generators": generators,
        "loads": loads,
    }


def add_units(
    units_df: pd.DataFrame,
    unit_type: str,
    world: World,
    forecaster: Forecaster,
) -> None:
    """
    Add units to the world from a given dataframe.
    The callback is used to adjust unit_params depending on the unit_type, before adding the unit to the world.

    Args:
        units_df (pandas.DataFrame): The dataframe containing the units.
        unit_type (str): The type of the unit.
        world (World): The world to which the units will be added.
        forecaster (Forecaster): The forecaster used for adding the units.
    """
    if units_df is None:
        return

    logger.info(f"Adding {unit_type} units")

    units_df = units_df.fillna(0)
    for unit_name, unit_params in units_df.iterrows():
        bidding_strategies = {
            key.split("bidding_")[1]: unit_params[key]
            for key in unit_params.keys()
            if key.startswith("bidding_")
        }
        unit_params["bidding_strategies"] = bidding_strategies
        operator_id = unit_params["unit_operator"]
        del unit_params["unit_operator"]
        world.add_unit(
            id=unit_name,
            unit_type=unit_type,
            unit_operator_id=operator_id,
            unit_params=unit_params,
            forecaster=forecaster,
        )


def read_units(
    units_df: pd.DataFrame,
    unit_type: str,
    forecaster: Forecaster,
    world_bidding_strategies: dict[str, BaseStrategy],
    learning_mode: bool = False,
) -> dict[str, list[dict]]:
    """
    Read units from a dataframe and only add them to a dictionary.
    The dictionary contains the operator ids as keys and the list of units belonging to the operator as values.

    Args:
        units_df (pandas.DataFrame): The dataframe containing the units.
        unit_type (str): The type of the unit.
        forecaster (Forecaster): The forecaster used for adding the units.
        world_bidding_strategies (dict[str, BaseStrategy]): The strategies available in the world
        learning_mode (bool, optional): Whether the world is in learning mode. Defaults to False.
    """
    if units_df is None:
        return {}

    logger.info(f"Adding {unit_type} units")
    units_dict = defaultdict(list)

    units_df = units_df.fillna(0)
    for unit_name, unit_params in units_df.iterrows():
        bidding_strategies = {
            key.split("bidding_")[1]: unit_params[key]
            for key in unit_params.keys()
            if key.startswith("bidding_") and unit_params[key]
        }
        unit_params["bidding_strategies"] = bidding_strategies

        # adjust the unit operator to Operator-RL if learning mode is enabled
        if learning_mode:
            operator_id = adjust_unit_operator_for_learning(
                bidding_strategies,
                world_bidding_strategies,
                unit_params["unit_operator"],
            )
        else:
            operator_id = unit_params["unit_operator"]

        del unit_params["unit_operator"]
        units_dict[operator_id].append(
            dict(
                id=unit_name,
                unit_type=unit_type,
                unit_operator_id=operator_id,
                unit_params=unit_params.to_dict(),
                forecaster=forecaster,
            )
        )
    return units_dict


def load_config_and_create_forecaster(
    inputs_path: str,
    scenario: str,
    study_case: str,
) -> dict[str, object]:
    """
    Load the configuration and files for a given scenario and study case. This function
    allows us to load the files and config only once when running multiple iterations of the same scenario.

    Args:
        inputs_path (str): The path to the folder containing input files necessary for the scenario.
        scenario (str): The name of the scenario to be loaded.
        study_case (str): The specific study case within the scenario to be loaded.

    Returns:
        dict[str, object]:: A dictionary containing the configuration and loaded files for the scenario and study case.
    """

    path = f"{inputs_path}/{scenario}"
    logger.info(f"Input files path: {path}")
    logger.info(f"Study case: {study_case}")

    with open(f"{path}/config.yaml") as f:
        config = yaml.safe_load(f)
    if not study_case:
        study_case = list(config.keys())[0]
    config = config[study_case]

    simulation_id = config.get("simulation_id", f"{scenario}_{study_case}")

    logger.info(f"Simulation ID: {simulation_id}")

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    index = pd.date_range(
        start=start,
        end=end,
        freq=config["time_step"],
    )

    powerplant_units = load_file(path=path, config=config, file_name="powerplant_units")
    storage_units = load_file(path=path, config=config, file_name="storage_units")
    demand_units = load_file(path=path, config=config, file_name="demand_units")
    exchange_units = load_file(path=path, config=config, file_name="exchange_units")

    # Initialize an empty dictionary to combine the DSM units
    dsm_units = {}
    for unit_type in ["industrial_dsm_units", "residential_dsm_units"]:
        units = load_dsm_units(
            path=path,
            config=config,
            file_name=unit_type,
        )
        if units is not None:
            dsm_units.update(units)

    if powerplant_units is None or demand_units is None:
        raise ValueError("No power plant or no demand units were provided!")

    forecasts_df = load_file(
        path=path, config=config, file_name="forecasts_df", index=index
    )
    demand_df = load_file(path=path, config=config, file_name="demand_df", index=index)
    if demand_df is None:
        logger.warning(
            "!! No demand_df timeseries provided !! Filling demand_df with zeros. Make sure this is what you actually want."
        )
        demand_df = pd.DataFrame(index=index, columns=demand_units.index, data=0.0)

    exchanges_df = load_file(
        path=path, config=config, file_name="exchanges_df", index=index
    )
    availability = load_file(
        path=path, config=config, file_name="availability_df", index=index
    )
    # check if availability contains any values larger than 1 and raise a warning
    if availability is not None and availability.max().max() > 1:
        # warn the user that the availability contains values larger than 1
        # and normalize the availability
        logger.warning(
            "Availability contains values larger than 1. This is not allowed. "
            "The availability will be normalized automatically. "
            "The quality of the automatic normalization is not guaranteed."
        )
        availability = normalize_availability(powerplant_units, availability)

    fuel_prices_df = load_file(
        path=path, config=config, file_name="fuel_prices_df", index=index
    )

    buses = load_file(path=path, config=config, file_name="buses")
    lines = load_file(path=path, config=config, file_name="lines")

    learning_config: LearningConfig = config.get("learning_config", {})

    # Check if simulation length is divisible by train_freq in learning config and adjust if not
    if config.get("learning_mode"):
        train_freq_str = learning_config.get("train_freq", "24h")
        train_freq = pd.Timedelta(train_freq_str)
        total_length = end - start

        # Compute remainder and determine the required intervals
        quotient, remainder = divmod(total_length, train_freq)

        if remainder != pd.Timedelta(0):
            # Adjust train_freq so that it evenly divides total_length
            n_intervals = quotient + 1
            new_train_freq = (total_length / n_intervals).total_seconds() / 3600
            new_train_freq_str = f"{int(new_train_freq)}h"  # Directly accessing hours

            # Update the configuration
            learning_config["train_freq"] = new_train_freq_str

            logger.warning(
                f"Simulation length ({total_length}) is not divisible by train_freq ({train_freq_str}). This will lead to a loss of training experience."
                f"Adjusting train_freq to {new_train_freq_str}. Consider modifying simulation length or train_freq in the config to avoid this adjustment."
            )

    forecaster = CsvForecaster(
        index=index,
        powerplants_units=powerplant_units,
        demand_units=demand_units,
        exchange_units=exchange_units,
        market_configs=config["markets_config"],
        buses=buses,
        lines=lines,
    )

    forecaster.set_forecast(forecasts_df)
    forecaster.set_forecast(demand_df)
    forecaster.set_forecast(exchanges_df)
    forecaster.set_forecast(availability, prefix="availability_")
    forecaster.set_forecast(fuel_prices_df, prefix="fuel_price_")
    forecaster.calc_forecast_if_needed()

    forecaster.convert_forecasts_to_fast_series()

    return {
        "config": config,
        "simulation_id": simulation_id,
        "path": path,
        "start": start,
        "end": end,
        "powerplant_units": powerplant_units,
        "storage_units": storage_units,
        "demand_units": demand_units,
        "exchange_units": exchange_units,
        "dsm_units": dsm_units,
        "forecaster": forecaster,
    }


def setup_world(
    world: World,
    evaluation_mode: bool = False,
    terminate_learning: bool = False,
    episode: int = 1,
    eval_episode: int = 1,
) -> None:
    """
    Load a scenario from a given path.

    This function loads a scenario within a specified study case from a given path, setting up the world environment for simulation and learning.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        scenario_data (dict): A dictionary containing the configuration and loaded files for the scenario and study case.
        evaluation_mode (bool, optional): A flag indicating whether evaluation should be performed. Defaults to False.
        terminate_learning (bool, optional): An automatically set flag indicating that we terminated the learning process now, either because we reach the end of the episode iteration or because we triggered an early stopping.
        episode (int, optional): The episode number for learning. Defaults to 1.
        eval_episode (int, optional): The episode number for evaluation. Defaults to 1.

    Raises:
        ValueError: If the specified scenario or study case is not found in the provided inputs.

    """
    # make a deep copy of the scenario data to avoid changing the original data
    scenario_data = copy.deepcopy(world.scenario_data)

    simulation_id = scenario_data["simulation_id"]
    config = scenario_data["config"]
    start = scenario_data["start"]
    end = scenario_data["end"]
    powerplant_units = scenario_data["powerplant_units"]
    storage_units = scenario_data["storage_units"]
    demand_units = scenario_data["demand_units"]
    exchange_units = scenario_data["exchange_units"]
    dsm_units = scenario_data["dsm_units"]
    forecaster = scenario_data["forecaster"]

    # save every thousand steps by default to free up memory
    save_frequency_hours = config.get("save_frequency_hours", 48)
    # if save_frequency_hours is set to 0, disable saving
    save_frequency_hours = None if save_frequency_hours == 0 else save_frequency_hours
    # check that save_frequency_hours is either None or an integer and raise an error if not with a hint for the user
    if save_frequency_hours is not None and (
        not isinstance(save_frequency_hours, int) or save_frequency_hours <= 0
    ):
        raise ValueError(
            f"save_frequency_hours argument in the config file must be either null or a positive integer. "
            f"Current value: {save_frequency_hours}."
        )

    # Disable save frequency if CSV export is enabled
    if world.export_csv_path and save_frequency_hours is not None:
        save_frequency_hours = None
        logger.info(
            "save_frequency_hours is disabled due to CSV export being enabled. "
            "Data will be stored in the CSV files at the end of the simulation."
        )

        # If PostgreSQL database is in use, warn the user about end-of-simulation saving
        if world.db_uri is not None and "postgresql" in world.db_uri:
            logger.warning(
                "Data will be stored in the PostgreSQL database only at the end of the simulation due to CSV export being enabled. "
                "Disable CSV export to save data at regular intervals (export_csv_path = '')."
            )

    learning_config: LearningConfig = config.get("learning_config", {})
    bidding_strategy_params = config.get("bidding_strategy_params", {})

    learning_config["learning_mode"] = config.get("learning_mode", False)
    learning_config["evaluation_mode"] = evaluation_mode

    if terminate_learning:
        learning_config["learning_mode"] = False
        learning_config["evaluation_mode"] = False

    if not learning_config.get("trained_policies_save_path"):
        learning_config["trained_policies_save_path"] = (
            f"learned_strategies/{simulation_id}"
        )

    if not learning_config.get("trained_policies_load_path"):
        learning_config["trained_policies_load_path"] = (
            f"learned_strategies/{simulation_id}/avg_reward_eval_policies"
        )

    config = replace_paths(config, scenario_data["path"])

    world.reset()

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=simulation_id,
        learning_config=learning_config,
        episode=episode,
        eval_episode=eval_episode,
        bidding_params=bidding_strategy_params,
        forecaster=forecaster,
    )

    # get the market config from the config file and add the markets
    logger.info("Adding markets")
    for market_id, market_params in config["markets_config"].items():
        market_config = make_market_config(
            id=market_id,
            market_params=market_params,
            world_start=start,
            world_end=end,
        )
        if "network_path" in market_config.param_dict.keys():
            grid_data = read_grid(market_config.param_dict["network_path"])
            market_config.param_dict["grid_data"] = grid_data

        operator_id = str(market_params["operator"])
        if operator_id not in world.market_operators:
            world.add_market_operator(id=operator_id)

        world.add_market(
            market_operator_id=operator_id,
            market_config=market_config,
        )

    # create list of units from dataframes before adding actual operators
    logger.info("Read units from file")

    units = defaultdict(list)
    powerplant_units = read_units(
        units_df=powerplant_units,
        unit_type="power_plant",
        forecaster=forecaster,
        world_bidding_strategies=world.bidding_strategies,
        learning_mode=learning_config["learning_mode"],
    )

    storage_units = read_units(
        units_df=storage_units,
        unit_type="storage",
        forecaster=forecaster,
        world_bidding_strategies=world.bidding_strategies,
        learning_mode=learning_config["learning_mode"],
    )

    demand_units = read_units(
        units_df=demand_units,
        unit_type="demand",
        forecaster=forecaster,
        world_bidding_strategies=world.bidding_strategies,
        learning_mode=learning_config["learning_mode"],
    )

    exchange_units = read_units(
        units_df=exchange_units,
        unit_type="exchange",
        forecaster=forecaster,
        world_bidding_strategies=world.bidding_strategies,
    )

    if dsm_units is not None:
        for unit_type, units_df in dsm_units.items():
            dsm_units = read_units(
                units_df=units_df,
                unit_type=unit_type,
                forecaster=forecaster,
                world_bidding_strategies=world.bidding_strategies,
                learning_mode=learning_config["learning_mode"],
            )
        for op, op_units in dsm_units.items():
            units[op].extend(op_units)

    for op, op_units in powerplant_units.items():
        units[op].extend(op_units)
    for op, op_units in storage_units.items():
        units[op].extend(op_units)
    for op, op_units in demand_units.items():
        units[op].extend(op_units)
    for op, op_units in exchange_units.items():
        units[op].extend(op_units)

    # if distributed_role is true - there is a manager available
    # and we can add each units_operator as a separate process
    if world.distributed_role is True:
        logger.info("Adding unit operators and units - with subprocesses")
        for op, op_units in units.items():
            world.add_units_with_operator_subprocess(op, op_units)
    else:
        logger.info("Adding unit operators and units")
        for company_name in set(units.keys()):
            if company_name == "Operator-RL" and world.learning_mode:
                world.add_rl_unit_operator(id="Operator-RL")
            else:
                world.add_unit_operator(id=str(company_name))

        # add the units to corresponding unit operators
        for op, op_units in units.items():
            for unit in op_units:
                world.add_unit(**unit)

    if world.learning_mode or world.evaluation_mode:
        world.add_learning_strategies_to_learning_role()

    if (
        world.learning_mode
        and world.learning_role is not None
        and len(world.learning_role.rl_strats) == 0
    ):
        raise ValueError("No RL units/strategies were provided!")


def load_scenario_folder(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
):
    """
    Load a scenario from a given path.

    This function loads a scenario within a specified study case from a given path, setting up the world environment for simulation and learning.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        inputs_path (str): The path to the folder containing input files necessary for the scenario.
        scenario (str): The name of the scenario to be loaded.
        study_case (str): The specific study case within the scenario to be loaded.

    Raises:
        ValueError: If the specified scenario or study case is not found in the provided inputs.

    Notes:
        - The function sets up the world environment based on the provided inputs and configuration files.
        - The function utilizes the specified inputs to configure the simulation environment, including market parameters, unit operators, and forecasting data.
        - After calling this function, the world environment is prepared for further simulation and analysis.

    """

    world.scenario_data = load_config_and_create_forecaster(
        inputs_path, scenario, study_case
    )

    setup_world(world=world)


def load_custom_units(
    world: World,
    inputs_path: str,
    scenario: str,
    file_name: str,
    unit_type: str,
) -> None:
    """
    Load custom units from a given path.

    This function loads custom units of a specified type from a given path within a scenario, adding them to the world environment for simulation.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        inputs_path (str): The path to the folder containing input files necessary for the custom units.
        scenario (str): The name of the scenario from which the custom units are to be loaded.
        file_name (str): The name of the file containing the custom units.
        unit_type (str): The type of the custom units to be loaded.

    Example:
        >>> load_custom_units(
            world=world,
            inputs_path="/path/to/inputs",
            scenario="scenario_name",
            file_name="custom_units.csv",
            unit_type="custom_type"
        )

    Notes:
        - The function loads custom units from the specified file within the given scenario and adds them to the world environment for simulation.
        - If the specified custom units file is not found, a warning is logged.
        - Each unique unit operator in the custom units is added to the world's unit operators.
        - The custom units are added to the world environment based on their type for use in simulations.
    """
    path = f"{inputs_path}/{scenario}"

    custom_units = load_file(
        path=path,
        config={},
        file_name=file_name,
    )

    if custom_units is None:
        logger.warning(f"No {file_name} units were provided!")

    operators = custom_units.unit_operator.unique()
    for operator in operators:
        if operator not in world.unit_operators:
            world.add_unit_operator(id=str(operator))

    add_units(
        units_df=custom_units,
        unit_type=unit_type,
        world=world,
        forecaster=world.forecaster,
    )


def run_learning(
    world: World,
    verbose: bool = False,
) -> None:
    """
    Train Deep Reinforcement Learning (DRL) agents to act in a simulated market environment.

    This function runs multiple episodes of simulation to train DRL agents, performs evaluation, and saves the best runs. It maintains the buffer and learned agents in memory to avoid resetting them with each new run.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        inputs_path (str): The path to the folder containing input files necessary for the simulation.
        scenario (str): The name of the scenario for the simulation.
        study_case (str): The specific study case for the simulation.

    Note:
        - The function uses a ReplayBuffer to store experiences for training the DRL agents.
        - It iterates through training episodes, updating the agents and evaluating their performance at regular intervals.
        - Initial exploration is active at the beginning and is disabled after a certain number of episodes to improve the performance of DRL algorithms.
        - Upon completion of training, the function performs an evaluation run using the best policy learned during training.
        - The best policies are chosen based on the average reward obtained during the evaluation runs, and they are saved for future use.
    """
    from assume.reinforcement_learning.buffer import ReplayBuffer

    if not verbose:
        logger.setLevel(logging.WARNING)

    # remove csv path so that nothing is written while learning
    temp_csv_path = world.export_csv_path
    world.export_csv_path = ""

    # initialize policies already here to set the obs_dim and act_dim in the learning role
    world.learning_role.rl_algorithm.initialize_policy()

    # check if we already stored policies for this simulation
    save_path = world.learning_config["trained_policies_save_path"]

    if Path(save_path).is_dir():
        if world.learning_config.get("continue_learning", False):
            logger.warning(
                f"Save path '{save_path}' exists.\n"
                "You are in continue learning mode. New strategies may overwrite previous ones.\n"
                "It is recommended to use a different save path to avoid unintended overwrites.\n"
                "You can set 'trained_policies_save_path' in the config."
            )
            proceed = input(
                "Do you still want to proceed with the existing save path? (y/N) "
            )
            if not proceed.lower().startswith("y"):
                raise AssumeException(
                    "Simulation aborted by user to avoid overwriting previous learned strategies. "
                    "Consider setting a new 'simulation_id' or 'trained_policies_save_path' in the config."
                )
        else:
            logger.warning(
                f"Save path '{save_path}' exists. Previous training data will be deleted to start fresh."
            )
            accept = input("Do you want to overwrite and start fresh? (y/N) ")
            if accept.lower().startswith("y"):
                shutil.rmtree(save_path, ignore_errors=True)
                logger.info(
                    f"Previous strategies at '{save_path}' deleted. Starting fresh training."
                )
            else:
                raise AssumeException(
                    "Simulation aborted by user not to overwrite existing learned strategies. "
                    "You can set a different 'simulation_id' or 'trained_policies_save_path' in the config."
                )

    # also remove tensorboard logs
    tensorboard_path = f"tensorboard/{world.scenario_data['simulation_id']}"
    if os.path.exists(tensorboard_path):
        shutil.rmtree(tensorboard_path, ignore_errors=True)

    # -----------------------------------------
    # Information that needs to be stored across episodes, aka one simulation run
    inter_episodic_data = {
        "buffer": ReplayBuffer(
            buffer_size=int(world.learning_config.get("replay_buffer_size", 5e5)),
            obs_dim=world.learning_role.rl_algorithm.obs_dim,
            act_dim=world.learning_role.rl_algorithm.act_dim,
            n_rl_units=len(world.learning_role.rl_strats),
            device=world.learning_role.device,
            float_type=world.learning_role.float_type,
        ),
        "actors_and_critics": None,
        "max_eval": defaultdict(lambda: -1e9),
        "all_eval": defaultdict(list),
        "avg_all_eval": [],
        "episodes_done": 0,
        "eval_episodes_done": 0,
    }

    world.learning_role.load_inter_episodic_data(inter_episodic_data)

    # -----------------------------------------

    validation_interval = min(
        world.learning_role.training_episodes,
        world.learning_config.get("validation_episodes_interval", 5),
    )

    # Ensure training episodes exceed the sum of initial experience and one evaluation interval
    min_required_episodes = (
        world.learning_role.episodes_collecting_initial_experience + validation_interval
    )

    if world.learning_role.training_episodes < min_required_episodes:
        raise ValueError(
            f"Training episodes ({world.learning_role.training_episodes}) must be greater than the sum of initial experience episodes ({world.learning_role.episodes_collecting_initial_experience}) and evaluation interval ({validation_interval})."
        )

    eval_episode = 1

    for episode in tqdm(
        range(1, world.learning_role.training_episodes + 1),
        desc="Training Episodes",
    ):
        # -----------------------------------------
        # Give the newly initialized learning role the needed information across episodes
        if episode != 1:
            setup_world(
                world=world,
                episode=episode,
            )
            world.learning_role.load_inter_episodic_data(inter_episodic_data)

        world.run()

        world.learning_role.tensor_board_logger.update_tensorboard()

        # -----------------------------------------
        # Store updated information across episodes
        inter_episodic_data = world.learning_role.get_inter_episodic_data()
        inter_episodic_data["episodes_done"] = episode

        # evaluation run:
        if (
            episode % validation_interval == 0
            and episode
            >= world.learning_role.episodes_collecting_initial_experience
            + validation_interval
        ):
            world.reset()

            # load evaluation run
            setup_world(
                world=world,
                evaluation_mode=True,
                episode=episode,
                eval_episode=eval_episode,
            )

            world.learning_role.load_inter_episodic_data(inter_episodic_data)

            world.run()

            world.learning_role.tensor_board_logger.update_tensorboard()

            total_rewards = world.output_role.get_sum_reward(episode=eval_episode)

            if len(total_rewards) == 0:
                raise AssumeException("No rewards were collected during evaluation run")

            avg_reward = np.mean(total_rewards)

            # check reward improvement in evaluation run
            # and store best run in eval folder
            terminate = world.learning_role.compare_and_save_policies(
                {"avg_reward": avg_reward}
            )

            inter_episodic_data["eval_episodes_done"] = eval_episode

            # if we have not improved in the last x evaluations, we stop loop
            if terminate:
                break

            eval_episode += 1

        world.reset()

        # save the policies after each episode in case the simulation is stopped or crashes
        if (
            episode
            >= world.learning_role.episodes_collecting_initial_experience
            + validation_interval
        ):
            world.learning_role.rl_algorithm.save_params(
                directory=f"{world.learning_role.trained_policies_save_path}/last_policies"
            )

    # container shutdown implicitly with new initialisation
    logger.info("################")
    logger.info("Training finished, Start evaluation run")
    world.export_csv_path = temp_csv_path

    world.reset()

    # Set 'trained_policies_load_path' to None in order to load the most recent policies,
    # especially if previous strategies were loaded from an external source.
    # This is useful when continuing from a previous learning session.
    world.scenario_data["config"]["learning_config"]["trained_policies_load_path"] = (
        f"{world.learning_role.trained_policies_save_path}/avg_reward_eval_policies"
    )

    # load scenario for evaluation
    setup_world(
        world=world,
        terminate_learning=True,
    )


if __name__ == "__main__":
    data = read_grid(Path("examples/inputs/example_01d"))
