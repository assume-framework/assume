# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import logging
from collections import defaultdict
from datetime import datetime, timedelta
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
                df = df.resample(index.freq).mean()
                logger.info(f"Downsampling {file_name} successful.")

            elif df.index.freq > index.freq or len(df.index) < len(index):
                logger.warning("Upsampling not implemented yet. Returning None.")
                return None

            df = df.loc[index]

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

    industrial_dsm_units = load_file(
        path=path,
        config=config,
        file_name=file_name,
    )

    if industrial_dsm_units is None:
        return None

    # Define columns that are common across different technologies within the same plant
    common_columns = [
        "unit_operator",
        "objective",
        "demand",
        "cost_tolerance",
        "unit_type",
    ]
    bidding_columns = [
        col for col in industrial_dsm_units.columns if col.startswith("bidding_")
    ]

    # Initialize the dictionary to hold the final structured data
    dsm_units_dict = {}

    # Process each group of components by plant name
    for name, group in industrial_dsm_units.groupby(industrial_dsm_units.index):
        dsm_unit = {}

        # Aggregate or select appropriate data for common and bidding columns
        # We take the first non-null entry
        for col in common_columns + bidding_columns:
            non_null_values = group[col].dropna()
            if not non_null_values.empty:
                dsm_unit[col] = non_null_values.iloc[0]

        # Process each technology within the plant
        components = {}
        for tech, tech_data in group.groupby("technology"):
            # Clean the technology-specific data: drop all-NaN columns and 'technology' column
            cleaned_data = tech_data.dropna(axis=1, how="all").drop(
                columns=["technology"]
            )
            components[tech] = cleaned_data.to_dict(orient="records")[0]

        dsm_unit["components"] = components
        dsm_units_dict[name] = dsm_unit

    # Convert the structured dictionary into a DataFrame
    industrial_dsm_units = pd.DataFrame.from_dict(dsm_units_dict, orient="index")

    # Split the DataFrame based on unit_type
    unit_type_dict = {}
    for unit_type in industrial_dsm_units["unit_type"].unique():
        unit_type_dict[unit_type] = industrial_dsm_units[
            industrial_dsm_units["unit_type"] == unit_type
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
) -> dict[str, list[dict]]:
    """
    Read units from a dataframe and only add them to a dictionary.
    The dictionary contains the operator ids as keys and the list of units belonging to the operator as values.

    Args:
        units_df (pandas.DataFrame): The dataframe containing the units.
        unit_type (str): The type of the unit.
        forecaster (Forecaster): The forecaster used for adding the units.
        world_bidding_strategies (dict[str, BaseStrategy]): The strategies available in the world
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
        operator_id = adjust_unit_operator_for_learning(
            bidding_strategies,
            world_bidding_strategies,
            unit_params["unit_operator"],
        )
        del unit_params["unit_operator"]
        units_dict[operator_id].append(
            dict(
                id=unit_name,
                unit_type=unit_type,
                unit_operator_id=operator_id,
                unit_params=unit_params,
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
    with open(f"{path}/config.yaml") as f:
        config = yaml.safe_load(f)
    if not study_case:
        study_case = list(config.keys())[0]
    config = config[study_case]

    sim_id = config.get("simulation_id", f"{scenario}_{study_case}")

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    index = pd.date_range(
        start=start,
        end=end + timedelta(days=1),
        freq=config["time_step"],
    )

    powerplant_units = load_file(path=path, config=config, file_name="powerplant_units")
    storage_units = load_file(path=path, config=config, file_name="storage_units")
    demand_units = load_file(path=path, config=config, file_name="demand_units")

    industrial_dsm_units = load_dsm_units(
        path=path,
        config=config,
        file_name="industrial_dsm_units",
    )

    if powerplant_units is None or demand_units is None:
        raise ValueError("No power plant or no demand units were provided!")

    forecasts_df = load_file(
        path=path, config=config, file_name="forecasts_df", index=index
    )
    demand_df = load_file(path=path, config=config, file_name="demand_df", index=index)
    if demand_df is None:
        raise ValueError("No demand time series was provided!")
    cross_border_flows_df = load_file(
        path=path, config=config, file_name="cross_border_flows", index=index
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

    electricity_prices_df = load_file(
        path=path, config=config, file_name="electricity_prices", index=index
    )
    price_forecast_df = load_file(
        path=path, config=config, file_name="price_forecasts", index=index
    )
    fuel_prices_df = load_file(
        path=path, config=config, file_name="fuel_prices_df", index=index
    )
    temperature_df = load_file(
        path=path, config=config, file_name="temperature", index=index
    )

    forecaster = CsvForecaster(
        index=index,
        powerplants_units=powerplant_units,
        demand_units=demand_units,
        market_configs=config["markets_config"],
    )

    forecaster.set_forecast(forecasts_df)
    forecaster.set_forecast(demand_df)
    forecaster.set_forecast(cross_border_flows_df)
    forecaster.set_forecast(availability, prefix="availability_")
    forecaster.set_forecast(electricity_prices_df)
    forecaster.set_forecast(price_forecast_df, "price_")
    forecaster.set_forecast(fuel_prices_df, prefix="fuel_price_")
    forecaster.set_forecast(temperature_df)
    forecaster.calc_forecast_if_needed()

    return {
        "config": config,
        "sim_id": sim_id,
        "path": path,
        "start": start,
        "end": end,
        "index": index,
        "powerplant_units": powerplant_units,
        "storage_units": storage_units,
        "demand_units": demand_units,
        "industrial_dsm_units": industrial_dsm_units,
        "forecaster": forecaster,
    }


async def async_setup_world(
    world: World,
    scenario_data: dict[str, object],
    study_case: str,
    perform_evaluation: bool = False,
    terminate_learning: bool = False,
    episode: int = 0,
    eval_episode: int = 0,
) -> None:
    """
    Load a scenario from a given path.

    This function loads a scenario within a specified study case from a given path, setting up the world environment for simulation and learning.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        scenario_data (dict): A dictionary containing the configuration and loaded files for the scenario and study case.
        study_case (str): The specific study case within the scenario to be loaded.
        perform_evaluation (bool, optional): A flag indicating whether evaluation should be performed. Defaults to False.
        terminate_learning (bool, optional): An automatically set flag indicating that we terminated the learning process now, either because we reach the end of the episode itteration or because we triggered an early stopping.
        episode (int, optional): The episode number for learning. Defaults to 0.
        eval_episode (int, optional): The episode number for evaluation. Defaults to 0.

    Raises:
        ValueError: If the specified scenario or study case is not found in the provided inputs.

    """
    # make a deep copy of the scenario data to avoid changing the original data
    scenario_data = copy.deepcopy(scenario_data)

    sim_id = scenario_data["sim_id"]
    config = scenario_data["config"]
    start = scenario_data["start"]
    end = scenario_data["end"]
    index = scenario_data["index"]
    powerplant_units = scenario_data["powerplant_units"]
    storage_units = scenario_data["storage_units"]
    demand_units = scenario_data["demand_units"]
    industrial_dsm_units = scenario_data["industrial_dsm_units"]
    forecaster = scenario_data["forecaster"]

    save_frequency_hours = config.get("save_frequency_hours", 48)

    learning_config: LearningConfig = config.get("learning_config", {})
    bidding_strategy_params = config.get("bidding_strategy_params", {})

    learning_config["learning_mode"] = config.get("learning_mode", False)
    learning_config["perform_evaluation"] = perform_evaluation

    if terminate_learning:
        learning_config["learning_mode"] = False
        learning_config["perform_evaluation"] = False

    if not learning_config.get("trained_policies_save_path"):
        if learning_config["learning_mode"]:
            path = f"learned_strategies/{study_case}"
        else:
            path = f"learned_strategies/{study_case}/last_policies"

        learning_config["trained_policies_save_path"] = path

    config = replace_paths(config, scenario_data["path"])

    if learning_config.get("learning_mode", False) and not learning_config.get(
        "perform_evaluation", False
    ):
        sim_id = f"{sim_id}_{episode}"

    elif learning_config.get("learning_mode", False) and learning_config.get(
        "perform_evaluation", False
    ):
        sim_id = f"{sim_id}_eval_{eval_episode}"

    world.reset()

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=sim_id,
        learning_config=learning_config,
        bidding_params=bidding_strategy_params,
        index=index,
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
    pwp_plants = read_units(
        units_df=powerplant_units,
        unit_type="power_plant",
        forecaster=forecaster,
        world_bidding_strategies=world.bidding_strategies,
    )

    str_plants = read_units(
        units_df=storage_units,
        unit_type="storage",
        forecaster=forecaster,
        world_bidding_strategies=world.bidding_strategies,
    )

    dem_plants = read_units(
        units_df=demand_units,
        unit_type="demand",
        forecaster=forecaster,
        world_bidding_strategies=world.bidding_strategies,
    )

    if industrial_dsm_units is not None:
        for unit_type, units_df in industrial_dsm_units.items():
            dsm_units = read_units(
                units_df=units_df,
                unit_type=unit_type,
                forecaster=forecaster,
                world_bidding_strategies=world.bidding_strategies,
            )
        for op, op_units in dsm_units.items():
            units[op].extend(op_units)

    for op, op_units in pwp_plants.items():
        units[op].extend(op_units)
    for op, op_units in str_plants.items():
        units[op].extend(op_units)
    for op, op_units in dem_plants.items():
        units[op].extend(op_units)

    # if distributed_role is true - there is a manager available
    # and we cann add each units_operator as a separate process
    if world.distributed_role is True:
        logger.info("Adding unit operators and units - with subprocesses")
        for op, op_units in units.items():
            await world.add_units_with_operator_subprocess(op, op_units)
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
                await world.async_add_unit(**unit)

    if (
        world.learning_mode
        and world.learning_role is not None
        and len(world.learning_role.rl_strats) == 0
    ):
        raise ValueError("No RL units/strategies were provided!")


def setup_world(
    world: World,
    scenario_data: dict[str, object],
    study_case: str,
    perform_evaluation: bool = False,
    terminate_learning: bool = False,
    episode: int = 0,
    eval_episode: int = 0,
) -> None:
    world.loop.run_until_complete(
        async_setup_world(
            world=world,
            scenario_data=scenario_data,
            study_case=study_case,
            perform_evaluation=perform_evaluation,
            terminate_learning=terminate_learning,
            episode=episode,
            eval_episode=eval_episode,
        )
    )


def load_scenario_folder(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
    perform_evaluation: bool = False,
    terminate_learning: bool = False,
    episode: int = 1,
    eval_episode: int = 1,
):
    """
    Load a scenario from a given path.

    This function loads a scenario within a specified study case from a given path, setting up the world environment for simulation and learning.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        inputs_path (str): The path to the folder containing input files necessary for the scenario.
        scenario (str): The name of the scenario to be loaded.
        study_case (str): The specific study case within the scenario to be loaded.
        perform_evaluation (bool, optional): A flag indicating whether evaluation should be performed. Defaults to False.
        terminate_learning (bool, optional): An automatically set flag indicating that we terminated the learning process now, either because we reach the end of the episode itteration or because we triggered an early stopping.
        episode (int, optional): The episode number for learning. Defaults to 0.
        eval_episode (int, optional): The episode number for evaluation. Defaults to 0.

    Raises:
        ValueError: If the specified scenario or study case is not found in the provided inputs.

    Example:
        >>> load_scenario_folder(
            world=world,
            inputs_path="/path/to/inputs",
            scenario="scenario_name",
            study_case="study_case_name",
            perform_evaluation=False,
            episode=1,
            eval_episode=1,
            trained_policies_save_path="",
        )

    Notes:
        - The function sets up the world environment based on the provided inputs and configuration files.
        - If `perform_evaluation` is set to True, the function performs evaluation using the specified evaluation episode number.
        - The function utilizes the specified inputs to configure the simulation environment, including market parameters, unit operators, and forecasting data.
        - After calling this function, the world environment is prepared for further simulation and analysis.

    """
    logger.info(f"Starting Scenario {scenario}/{study_case} from {inputs_path}")

    scenario_data = load_config_and_create_forecaster(inputs_path, scenario, study_case)

    setup_world(
        world=world,
        scenario_data=scenario_data,
        study_case=study_case,
        perform_evaluation=perform_evaluation,
        terminate_learning=terminate_learning,
        episode=episode,
        eval_episode=eval_episode,
    )


async def async_load_custom_units(
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
    world.loop.run_until_complete(
        async_load_custom_units(
            world=world,
            inputs_path=inputs_path,
            scenario=scenario,
            file_name=file_name,
            unit_type=unit_type,
        )
    )


def run_learning(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
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
    actors_and_critics = None
    world.learning_role.initialize_policy(actors_and_critics=actors_and_critics)
    world.output_role.del_similar_runs()

    # check if we already stored policies for this simualtion
    save_path = world.learning_config["trained_policies_save_path"]

    if Path(save_path).is_dir():
        # we are in learning mode and about to train new policies, which might overwrite existing ones
        accept = input(
            f"{save_path=} exists - should we overwrite current learnings? (y/N) "
        )
        if not accept.lower().startswith("y"):
            # stop here - do not start learning or save anything
            raise AssumeException("don't overwrite existing strategies")

    # -----------------------------------------
    # Load scenario data to reuse across episodes
    scenario_data = load_config_and_create_forecaster(inputs_path, scenario, study_case)

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
        "noise_scale": world.learning_config.get("noise_scale", 1.0),
    }

    # -----------------------------------------

    validation_interval = min(
        world.learning_role.training_episodes,
        world.learning_config.get("validation_episodes_interval", 5),
    )

    eval_episode = 1

    for episode in tqdm(
        range(1, world.learning_role.training_episodes + 1),
        desc="Training Episodes",
    ):
        # TODO normally, loading twice should not create issues, somehow a scheduling issue is raised currently
        if episode != 1:
            setup_world(
                world=world,
                scenario_data=scenario_data,
                study_case=study_case,
                episode=episode,
            )

        # -----------------------------------------
        # Give the newly initliazed learning role the needed information across episodes
        world.learning_role.load_inter_episodic_data(inter_episodic_data)

        world.run()

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
                scenario_data=scenario_data,
                study_case=study_case,
                perform_evaluation=True,
                eval_episode=eval_episode,
            )

            world.learning_role.load_inter_episodic_data(inter_episodic_data)

            world.run()

            total_rewards = world.output_role.get_sum_reward()
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

        # if at end of simulation save last policies
        if episode == (world.learning_role.training_episodes):
            world.learning_role.rl_algorithm.save_params(
                directory=f"{world.learning_role.trained_policies_save_path}/last_policies"
            )

        # container shutdown implicitly with new initialisation
    logger.info("################")
    logger.info("Training finished, Start evaluation run")
    world.export_csv_path = temp_csv_path

    world.reset()

    # load scenario for evaluation
    setup_world(
        world=world,
        scenario_data=scenario_data,
        study_case=study_case,
        terminate_learning=True,
    )

    world.learning_role.load_inter_episodic_data(inter_episodic_data)


if __name__ == "__main__":
    data = read_grid(Path("examples/inputs/example_01d"))
