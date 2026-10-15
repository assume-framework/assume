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

from assume.common.exceptions import AssumeException
from assume.common.fast_pandas import FastIndex
from assume.common.forecaster import (
    BuildingForecaster,
    CementForecaster,
    CustomUnitForecaster,
    DemandForecaster,
    DsmUnitForecaster,
    ExchangeForecaster,
    SectorCoupledForecaster,
    HydrogenForecaster,
    PowerplantForecaster,
    SteamgenerationForecaster,
    SteelplantForecaster,
    UnitForecaster,
    UnitsOperatorForecaster,
)
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import (
    adjust_unit_operator_for_learning,
    confirm_learning_save_path,
    convert_to_rrule_freq,
    load_index_file,
    normalize_availability,
    set_random_seed,
)
from assume.strategies import BaseStrategy
from assume.world import World

logger = logging.getLogger(__name__)


def get_unit_forecast_column(
    forecasts_df: pd.DataFrame | None,
    unit_id: str,
    column_name: str,
) -> pd.Series | None:
    """Return a forecast column, preferring ``{unit_id}_{column_name}`` over ``column_name``."""
    if forecasts_df is None:
        return None
    prefixed = f"{unit_id}_{column_name}"
    if prefixed in forecasts_df.columns:
        return forecasts_df[prefixed]
    if column_name in forecasts_df.columns:
        return forecasts_df[column_name]
    return None


def bidding_strategies_from_param_dict(param_dict: dict):
    return {
        ident.split("bidding_")[1]: strategy
        for ident, strategy in param_dict.items()
        if ident.startswith("bidding_")
    }


def forecast_algorithm_from_param_dict(param_dict: dict) -> dict[str, str]:
    return {
        ident.split("forecast_")[1]: forecast_algorithm
        for ident, forecast_algorithm in param_dict.items()
        if ident.startswith("forecast_")
    }


def get_unit_forecast_algorithms(
    forecast_algorithms: dict[str, str], plant: dict
) -> dict[str, str]:
    unit_forecast_algorithms = forecast_algorithm_from_param_dict(
        plant
    )  # get forecast specific parts

    # overwrite None in plant csv with values from config if it exists!
    for key, forecast_alg in unit_forecast_algorithms.items():
        if forecast_alg is None or pd.isna(forecast_alg):
            unit_forecast_algorithms[key] = forecast_algorithms.get(key)

    return forecast_algorithms | unit_forecast_algorithms  # merge dicts together


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
    if file_name in config:
        if config[file_name] is None:
            return None
        file_path = Path(path) / config[file_name]
    else:
        file_path = Path(path) / f"{file_name}.csv"

    try:
        if index is not None:
            df = load_index_file(file_path, index)
        else:
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

            if check_duplicates:
                # Check if duplicate unit names exist and raise an error
                duplicates = df.index[df.index.duplicated()].unique()

                if len(duplicates) > 0:
                    duplicate_names = ", ".join(map(str, duplicates))
                    raise ValueError(
                        f"Duplicate unit names found in {file_name}: {duplicate_names}. Please rename them to avoid conflicts."
                    )
        return df

    except FileNotFoundError:
        logger.info(f"{file_path} not found. Returning None")
        return None


def load_dsm_units(path: str, config: dict, file_name: str) -> dict:
    """
    Loads and processes a CSV file containing DSM unit data, where each unit may
    consist of multiple components (technologies) under the same plant name. The
    function groups data by plant name, processes each group to handle different
    technologies, and organizes the data into a structured DataFrame. It then splits
    the DataFrame based on unique unit_types.

    Args:
        path (str): The directory path where the CSV file is located.
        config (dict): Configuration dictionary, potentially used for specifying
            additional options or behaviors (not used in the current implementation
            but provides flexibility for future enhancements).
        file_name (str): The name of the CSV file to be loaded.

    Returns:
        dict: A dictionary where each key is a unique unit_type and the value is a
            DataFrame containing the corresponding DSM units of that type.

    Note:
        - The CSV file is expected to have columns such as 'name', 'technology',
          'unit_type', and other operational parameters.
        - The function assumes that the first non-null value in common and bidding
          columns is representative if multiple entries exist for the same plant.
        - Rolling-horizon optimisation settings (``horizon_mode``,
          ``look_ahead_horizon``, ``commit_horizon``, ``rolling_step``) are read as
          optional per-plant columns. They are assembled into the
          ``dsm_optimisation_config`` dict passed to the unit constructor.
        - It is crucial that the input CSV file follows the expected structure for
          the function to process it correctly.
    """
    # Load the DSM units file
    # Note: check_duplicates is set to False to avoid raising an error for duplicate unit names
    dsm_units = load_file(
        path=path, config=config, file_name=file_name, check_duplicates=False
    )
    if dsm_units is None:
        return None

    # Define columns that are common across different technologies within the same plant
    common_columns = [
        "unit_operator", "objective", "demand", "cost_tolerance", "unit_type", "node",
        "flexibility_measure", "is_prosumer", "congestion_threshold", "peak_load_cap",
        "load_profile_deviation", "system_type", "market_representation",
        "enforce_terminal_soc", "primary_carrier",
    ]  # fmt: skip
    # Filter the common columns to only include those that exist in the DataFrame
    common_columns = [col for col in common_columns if col in dsm_units.columns]

    # Get bidding columns dynamically
    bidding_columns = [col for col in dsm_units.columns if col.startswith("bidding_")]

    # Rolling-horizon optimisation columns (per-plant, optional). Filled on the first
    # technology row of each plant; assembled into a dsm_optimisation_config dict below.
    dsm_opt_columns = [
        col
        for col in ["horizon_mode", "look_ahead_horizon", "commit_horizon", "rolling_step"]
        if col in dsm_units.columns
    ]

    # Initialize the dictionary to hold the final structured data
    dsm_units_dict = {}

    # Parent/component grouping
    if "parent_unit" in dsm_units.columns:
        grouping_key = dsm_units["parent_unit"].copy()
        missing_parent = grouping_key.isna() | (grouping_key.astype(str).str.strip() == "")
        grouping_key = grouping_key.astype(object)
        grouping_key.loc[missing_parent] = dsm_units.index[missing_parent].astype(str)
    else:
        grouping_key = pd.Series(dsm_units.index, index=dsm_units.index, dtype=object)

    # Process each physical parent unit.
    for name, group in dsm_units.groupby(grouping_key, sort=False):
        dsm_unit = {}

        # Aggregate or select parent-level data.
        for col in common_columns + bidding_columns:
            non_null_values = group[col].dropna()
            if not non_null_values.empty:
                dsm_unit[col] = non_null_values.iloc[0]

        # Process component rows.
        components = {}
        metadata_columns = (
            ["technology", "parent_unit", "name"]
            + common_columns + bidding_columns + dsm_opt_columns
        )  # fmt: skip

        for row_index, row in group.iterrows():
            tech = row.get("technology")
            if tech is None or pd.isna(tech):
                raise ValueError(f"DSM component row '{row_index}' has no technology.")

            component_name = str(tech)

            # If the same technology occurs more than once under one parent,
            # retain both components by adding a deterministic suffix.
            if component_name in components:
                suffix = 2
                candidate = f"{component_name}_{suffix}"
                while candidate in components:
                    suffix += 1
                    candidate = f"{component_name}_{suffix}"
                component_name = candidate

            component_data = row.drop(labels=metadata_columns, errors="ignore").dropna()
            component_config = component_data.to_dict()

            # `name` is the preferred explicit market-facing component ID.
            # If it is absent, the CSV row index is a safe fallback.
            market_unit_id = row.get("name")
            if (
                market_unit_id is None
                or pd.isna(market_unit_id)
                or str(market_unit_id).strip() == ""
            ):
                market_unit_id = row_index

            component_config["market_unit_id"] = str(market_unit_id)
            components[component_name] = component_config

        dsm_unit["components"] = components

        # Assemble per-plant rolling-horizon config from CSV columns (if any values present)
        if dsm_opt_columns:
            opt_cfg = {}
            for col in dsm_opt_columns:
                non_null_values = group[col].dropna()
                if not non_null_values.empty:
                    opt_cfg[col] = non_null_values.iloc[0]
            if opt_cfg:
                dsm_unit["dsm_optimisation_config"] = opt_cfg

        dsm_units_dict[name] = dsm_unit

    # Convert the structured dictionary into a DataFrame
    dsm_units_df = pd.DataFrame.from_dict(dsm_units_dict, orient="index")

    # Split the DataFrame based on unit_type
    unit_type_dict = {}
    if "unit_type" in dsm_units_df.columns:
        for unit_type in dsm_units_df["unit_type"].unique():
            unit_type_dict[unit_type] = dsm_units_df[dsm_units_df["unit_type"] == unit_type]

    return unit_type_dict


def build_flexible_loads(dsm_units: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the static grid-facing representation of DSM units
    that may participate in redispatch as flexible consumers.

    Returns one row per physical DSM parent.
    """
    records = []

    for unit_type, units_df in dsm_units.items():
        if units_df is None or units_df.empty:
            continue

        for unit_id, unit in units_df.iterrows():
            node = unit.get("node")
            if node is None or pd.isna(node):
                continue

            components = unit.get("components", {})
            p_nom = 0.0

            for component_name, component in components.items():
                # Components with direct electrical consumption
                max_power = component.get("max_power", 0.0)
                try:
                    max_power = abs(float(max_power))
                except (TypeError, ValueError):
                    max_power = 0.0

                # We still need a robust way of identifying
                # which component types consume electricity.
                if component_name in {"heat_pump", "electrolyser", "electric_boiler"}:
                    p_nom += max_power

            if p_nom <= 0:
                continue

            records.append(
                {"unit_id": unit_id, "node": node, "p_nom": p_nom, "unit_type": unit_type}
            )

    if not records:
        return pd.DataFrame(columns=["node", "p_nom", "unit_type"])

    return pd.DataFrame(records).set_index("unit_id")


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


def read_grid(network_path: str | Path) -> dict[str, pd.DataFrame | None]:
    network_path = Path(network_path)
    buses = None
    lines = None
    generators = None
    loads = None
    storage_units = None

    if (network_path / "buses.csv").exists():
        buses = pd.read_csv(network_path / "buses.csv", index_col=0)
    if (network_path / "lines.csv").exists():
        lines = pd.read_csv(network_path / "lines.csv", index_col=0)
    if (network_path / "powerplant_units.csv").exists():
        generators = pd.read_csv(network_path / "powerplant_units.csv", index_col=0)
    if (network_path / "demand_units.csv").exists():
        loads = pd.read_csv(network_path / "demand_units.csv", index_col=0)
    if (network_path / "storage_units.csv").exists():
        storage_units = pd.read_csv(network_path / "storage_units.csv", index_col=0)

    return {
        "buses": buses,
        "lines": lines,
        "generators": generators,
        "loads": loads,
        "storage_units": storage_units,
    }


def add_units(
    units_df: pd.DataFrame,
    unit_type: str,
    world: World,
    forecaster: UnitForecaster,
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
        bidding_strategies = bidding_strategies_from_param_dict(unit_params)
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
    forecaster: dict[str, UnitForecaster],
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
                forecaster=forecaster[unit_name],
            )
        )
    return units_dict


def save_unique_forecasts(units, save_path: Path) -> None:
    """Collect unique forecasts computed by unit forecasters and write them to CSV.

    Since there is one forecaster per unit but forecasts are shared across units
    (via ``@lru_cache`` on the underlying algorithms), forecasts are deduplicated
    by column name. Column names mirror the ``forecasts_df.csv`` convention so the
    resulting file can be consumed as a drop-in input in a later run.
    """
    unique_forecasts = {
        "price": {},
        "residual_load": {},
        "congestion_signal": {},
        "renewable_utilisation": {},
    }
    default_values = {
        "price": "price_naive_forecast",
        "residual_load": "residual_load_naive_forecast",
        "congestion_signal": "congestion_signal_naive_forecast",
        "renewable_utilisation": "renewable_utilisation_naive_forecast",
    }
    for unit in units:
        algs = unit.forecaster.forecast_algorithms
        if isinstance(unit.forecaster, DsmUnitForecaster):
            for key in unique_forecasts:
                forecast_name = algs.get(key, default_values[key])
                unique_forecasts[key][forecast_name] = unit
        else:
            for key in ["price", "residual_load"]:
                forecast_name = algs.get(key, default_values[key])
                unique_forecasts[key][forecast_name] = unit

    forecast_dict = {}
    for f_type in unique_forecasts:  # price, residual_load, ...
        for f_name in unique_forecasts[f_type]:  #
            unit = unique_forecasts[f_type][f_name]
            attr_name = (
                "renewable_utilisation_signal"
                if f_type == "renewable_utilisation"
                else f_type
            )
            forecast = getattr(unit.forecaster, attr_name)
            if isinstance(forecast, dict):
                for f_key in forecast:
                    forecast_dict[f"{f_name}_{f_key}"] = forecast[f_key].as_pd_series(
                        name=f"{f_name}_{f_key}"
                    )
            else:
                forecast_dict[f"{f_name}"] = forecast.as_pd_series(name=f"{f_name}")

    if not forecast_dict:
        logger.info("No unique forecasts to save.")
        return

    df = pd.concat(forecast_dict.values(), axis=1, names=forecast_dict.keys())
    df.index.name = "datetime"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path)

    logger.info(f"Saved {len(df.columns)} unique forecasts to {save_path}")


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
    logger.info(f"Loaded config from: {path}/config.yaml")
    logger.info(f"heat_system config value: {config.get('heat_system', 'MISSING')!r}")
    learning_config = config.get("learning_config", {})
    learning_mode = learning_config.get("learning_mode", False) or learning_config.get(
        "continue_learning", False
    )

    # Set seed, or disable with `seed: null` in config
    set_random_seed(seed=config.get("seed", 42), learning_mode=learning_mode)

    simulation_id = config.get("simulation_id", f"{scenario}_{study_case}")

    logger.info(f"Simulation ID: {simulation_id}")

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    index = pd.date_range(
        start=start,
        end=end,
        freq=config["time_step"],
    )

    unit_operators = load_file(path=path, config=config, file_name="unit_operators")
    powerplant_units = load_file(path=path, config=config, file_name="powerplant_units")
    storage_units = load_file(path=path, config=config, file_name="storage_units")
    demand_units = load_file(path=path, config=config, file_name="demand_units")
    exchange_units = load_file(path=path, config=config, file_name="exchange_units")

    if powerplant_units is None or demand_units is None:
        raise ValueError("No power plant or no demand units were provided!")

    if ((demand_units["min_power"] < 0) & (demand_units["max_power"] > 0)).any() or (
        (demand_units["min_power"] > 0) & (demand_units["max_power"] < 0)
    ).any():
        raise ValueError(
            "min_power and max_power must both be either negative or positive"
        )
    demand_units["min_power"] = -abs(demand_units["min_power"])
    demand_units["max_power"] = -abs(demand_units["max_power"])

    if storage_units is not None:
        if "max_power_charge" in storage_units.columns:
            storage_units["max_power_charge"] = -abs(storage_units["max_power_charge"])
        if "min_power_charge" in storage_units.columns:
            storage_units["min_power_charge"] = -abs(storage_units["min_power_charge"])
        if "capacity" not in storage_units.columns:
            raise ValueError("No capacity column provided for storage units!")

    # Initialize an empty dictionary to combine the DSM units
    dsm_units = {}
    for unit_type in ["industrial_dsm_units", "residential_dsm_units"]:
        units = load_dsm_units(path=path, config=config, file_name=unit_type)
        if units is not None:
            dsm_units.update(units)
    """
    # Load heat system units only if enabled in config
    if config.get("heat_system", False):
        logger.info("Loading heat_system_units.csv")
        units = load_dsm_units(path=path,config=config,file_name="heat_system_units",)
        if units is None:
            raise ValueError("heat_system is enabled in config but heat_system_units.csv was not found.")
        dsm_units.update(units)
    """
    # Generic sector-coupled systems
    units = load_dsm_units(
        path=path,
        config=config,
        file_name="sector_coupled_units",
    )

    if units is None:
        logger.info(
            "No sector-coupled systems found. "
            "sector_coupled_units.csv is not present."
        )
    else:
        dsm_units.update(units)

        logger.info(
            "Loaded sector-coupled systems: %s",
            list(units.keys()),
        )

    # Load heat system units only if enabled in config
    logger.info(
        f"Checking HeatSystem: heat_system={config.get('heat_system', False)!r}"
    )

    if config.get("heat_system", False):
        logger.info("HeatSystem enabled - loading heat_system_units.csv")

        units = load_dsm_units(
            path=path,
            config=config,
            file_name="heat_system_units",
        )

        logger.info(
            f"Result from load_dsm_units for HeatSystem: "
            f"{list(units.keys()) if units is not None else None}"
        )

        if units is None:
            raise ValueError(
                "heat_system is enabled in config but "
                "heat_system_units.csv was not found."
            )

        dsm_units.update(units)

        logger.info(f"DSM unit types after HeatSystem loading: {list(dsm_units.keys())}")


    forecasts_df = load_file(
        path=path, config=config, file_name="forecasts_df", index=index
    )
    demand_df = load_file(path=path, config=config, file_name="demand_df", index=index)
    if demand_df is None:
        # no demand timeseries exist, all demand is elastic. Fill missing demand timeseries with zeros and raise a warning.
        logger.warning(
            "!! No demand_df timeseries provided !! Filling demand_df with zeros. Make sure this is what you actually want."
        )
        demand_df = pd.DataFrame(index=index, columns=demand_units.index, data=0.0)
    elif not demand_df.columns.equals(demand_units.index):
        # there exist demand timeseries, but not for all demand units. Some demand is elastic, some is not. Fill missing demand timeseries with zeros and raise a warning.
        logger.warning(
            "!! Incomplete demand_df timeseries provided !! Filling demand_df for some units with zeros. Make sure this is what you actually want."
        )
        missing_columns = demand_units.index.difference(demand_df.columns)
        for col in missing_columns:
            demand_df[col] = 0.0

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

    if availability is None:
        availability = pd.DataFrame(index=index)

    fuel_prices_df = load_file(
        path=path, config=config, file_name="fuel_prices_df", index=index
    )
    if fuel_prices_df is None:
        fuel_prices_df = pd.DataFrame(index=index)

    if len(fuel_prices_df) <= 1:  # single value provided, extend to full index
        fuel_prices_df.index = index[:1]
        fuel_prices_df = fuel_prices_df.reindex(index, method="ffill")

    forecast_algorithms = config.get("forecast_algorithms", {})

    # create shared unit index for caching!
    shared_unit_index = FastIndex(
        start=index[0], end=index[-1], freq=pd.infer_freq(index)
    )
    unit_forecasts: dict[str, UnitForecaster] = {}
    if powerplant_units is not None:
        for id, plant in powerplant_units.iterrows():
            unit_forecasts[id] = PowerplantForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                fuel_prices=fuel_prices_df,
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, plant
                ),
            )
    if demand_units is not None:
        for id, demand in demand_units.iterrows():
            unit_forecasts[id] = DemandForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                demand=-demand_df[id].abs(),
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, demand
                ),
            )
    if storage_units is not None:
        for id, storage in storage_units.iterrows():
            unit_forecasts[id] = UnitForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, storage
                ),
            )
    if exchange_units is not None:
        for id, exchange in exchange_units.iterrows():
            unit_forecasts[id] = ExchangeForecaster(
                index=shared_unit_index,
                availability=availability.get(id, pd.Series(1.0, index, name=id)),
                forecast_algorithms=get_unit_forecast_algorithms(
                    forecast_algorithms, exchange
                ),
                volume_export=exchanges_df[f"{id}_export"],
                volume_import=exchanges_df[f"{id}_import"],
            )
    if dsm_units is not None:
        for type, dsm in dsm_units.items():
            for id, unit in dsm.iterrows():
                unit_forecast_algorithms = get_unit_forecast_algorithms(
                    forecast_algorithms, unit
                )
                if type == "building":

                    def get_building_profile(column_name: str) -> pd.Series:
                        default_profile = pd.Series(0.0, index=index, name=column_name)
                        if forecasts_df is None:
                            return default_profile
                        return forecasts_df.get(column_name, default_profile)

                    # Base aggregate building profiles
                    building_load_profile = get_building_profile(f"{id}_load_profile")
                    building_heat_demand = get_building_profile(f"{id}_heat_demand")
                    building_pv_profile = get_building_profile(f"{id}_pv_profile")
                    building_battery_profile = get_building_profile(
                        f"{id}_battery_load_profile"
                    )
                    building_ev_profile = get_building_profile(f"{id}_ev_load_profile")
                    building_electricity_price_flex = get_building_profile(
                        f"{id}_electricity_price_flex"
                    )

                    # collect arbitrary component-level forecasts for this building
                    extra_building_profiles = {}

                    if forecasts_df is not None:
                        building_prefix = f"{id}_"
                        for col in forecasts_df.columns:
                            if col.startswith(building_prefix):
                                extra_building_profiles[col] = forecasts_df[col]

                    unit_forecasts[id] = BuildingForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        fuel_prices=fuel_prices_df,
                        load_profile=building_load_profile,
                        ev_load_profile=building_ev_profile,
                        heat_demand=building_heat_demand,
                        battery_load_profile=building_battery_profile,
                        pv_profile=building_pv_profile,
                        electricity_price_flex=building_electricity_price_flex,
                        **extra_building_profiles,
                    )
                if type == "steel_plant":
                    normalized_profile = get_unit_forecast_column(
                        forecasts_df, id, "normalized_load_profile"
                    )
                    steel_demand = get_unit_forecast_column(
                        forecasts_df, id, "steel_demand"
                    )

                    unit_forecasts[id] = SteelplantForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        market_prices=unit.get("market_prices"),
                        forecast_algorithms=unit_forecast_algorithms,
                        forecast_registries=None,
                        fuel_prices=fuel_prices_df,
                        normalized_load_profile=normalized_profile,
                        steel_demand=steel_demand,
                    )
                if type == "cement_plant":
                    storage_schedule = get_unit_forecast_column(
                        forecasts_df, id, "thermal_storage_schedule"
                    )
                    unit_forecasts[id] = CementForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        market_prices=unit.get("market_prices"),
                        forecast_algorithms=unit_forecast_algorithms,
                        forecast_registries=None,
                        fuel_prices=fuel_prices_df,
                        normalized_load_profile=get_unit_forecast_column(
                            forecasts_df, id, "normalized_load_profile"
                        ),
                        clinker_demand=get_unit_forecast_column(
                            forecasts_df, id, "clinker_demand"
                        ),
                        electricity_price_flex=get_unit_forecast_column(
                            forecasts_df, id, "electricity_price_flex"
                        ),
                        thermal_storage_schedule=(
                            storage_schedule if storage_schedule is not None else 0
                        ),
                        availability_profiles={
                            tech: get_unit_forecast_column(
                                forecasts_df, id, f"{tech}_availability"
                            )
                            for tech in (
                                "preheater",
                                "calciner",
                                "kiln",
                            )
                        },
                    )
                if type == "sector_coupled_system":
                    system_type = str(
                        unit.get("system_type", "heat")
                    ).strip().lower()

                    if system_type == "heat":
                        heat_demand = get_unit_forecast_column(
                            forecasts_df,
                            id,
                            "heat",
                        )
                        if heat_demand is None:
                            heat_demand = get_unit_forecast_column(
                                forecasts_df,
                                id,
                                "heat_demand",
                            )

                        if heat_demand is None:
                            raise ValueError(
                                f"No heat-demand forecast found for "
                                f"SectorCoupledSystem '{id}'. Expected "
                                f"'{id}_heat' or '{id}_heat_demand' "
                                "in forecasts_df.csv."
                            )

                        unit_forecasts[id] = SectorCoupledForecaster(
                            index=shared_unit_index,
                            availability=availability.get(
                                id,
                                pd.Series(1.0, index, name=id),
                            ),
                            forecast_algorithms=unit_forecast_algorithms,
                            fuel_prices=fuel_prices_df,
                            heat_demand=heat_demand,
                        )

                    elif system_type == "hydrogen":
                        hydrogen_demand = get_unit_forecast_column(
                            forecasts_df,
                            id,
                            "hydrogen_demand",
                        )
                        if hydrogen_demand is None:
                            hydrogen_demand = get_unit_forecast_column(
                                forecasts_df,
                                id,
                                "hydrogen",
                            )

                        # Backward-compatible fallback for scenarios that
                        # specify a constant per-step H2 demand in the unit CSV.
                        if hydrogen_demand is None:
                            raw_demand = unit.get("demand", 0.0)
                            try:
                                raw_demand = float(raw_demand)
                            except (TypeError, ValueError):
                                raw_demand = 0.0

                            hydrogen_demand = pd.Series(
                                raw_demand,
                                index=index,
                                name=f"{id}_hydrogen_demand",
                            )

                        seasonal_storage_schedule = (
                            get_unit_forecast_column(
                                forecasts_df,
                                id,
                                "seasonal_storage_schedule",
                            )
                        )
                        if seasonal_storage_schedule is None:
                            seasonal_storage_schedule = 0

                        unit_forecasts[id] = HydrogenForecaster(
                            index=shared_unit_index,
                            availability=availability.get(
                                id,
                                pd.Series(1.0, index, name=id),
                            ),
                            forecast_algorithms=unit_forecast_algorithms,
                            hydrogen_demand=hydrogen_demand,
                            seasonal_storage_schedule=(
                                seasonal_storage_schedule
                            ),
                        )

                    else:
                        raise ValueError(
                            f"Unsupported system_type '{system_type}' "
                            f"for SectorCoupledSystem '{id}'. "
                            "Currently supported presets are 'heat' "
                            "and 'hydrogen'."
                        )

                if type == "hydrogen_plant":
                    unit_forecasts[id] = HydrogenForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        hydrogen_demand=unit["demand"],
                        seasonal_storage_schedule=0,  # TODO
                    )
                if type == "steam_plant":
                    unit_forecasts[id] = SteamgenerationForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id, pd.Series(1.0, index, name=id)
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        demand=unit["demand"],
                        fuel_prices=fuel_prices_df,
                        electricity_price_flex=0,  # TODO
                        thermal_storage_schedule=0,  # TODO
                        thermal_demand=0,  # TODO
                    )
                if type == "heat_system":
                    heat_demand = get_unit_forecast_column(
                        forecasts_df,
                        id,
                        "heat",
                    )
                    if heat_demand is None:
                        raise ValueError(
                            f"No heat demand forecast found for HeatSystem '{id}'. "
                            f"Expected column '{id}_heat' in forecasts_df.csv."
                        )
                    unit_forecasts[id] = SectorCoupledForecaster(
                        index=shared_unit_index,
                        availability=availability.get(
                            id,
                            pd.Series(1.0, index, name=id),
                        ),
                        forecast_algorithms=unit_forecast_algorithms,
                        fuel_prices=fuel_prices_df,
                        heat_demand=heat_demand,
                    )
    # shared inputs used to build one UnitsOperatorForecaster per operator in
    # setup_world. An operator has no availability of its own (that is a
    # per-unit concept), so only the index and algorithms are shared here.
    units_operator_forecast_data = {
        "shared_unit_index": shared_unit_index,
        "forecast_algorithms": forecast_algorithms,
    }
    return {
        "config": config,
        "simulation_id": simulation_id,
        "path": path,
        "start": start,
        "end": end,
        "unit_operators": unit_operators,
        "powerplant_units": powerplant_units,
        "storage_units": storage_units,
        "demand_units": demand_units,
        "exchange_units": exchange_units,
        "dsm_units": dsm_units,
        "unit_forecasts": unit_forecasts,
        "index": index,
        "forecasts_df": forecasts_df,
        "units_operator_forecast_data": units_operator_forecast_data,
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
    unit_operators = scenario_data["unit_operators"]
    powerplant_units = scenario_data["powerplant_units"]
    storage_units = scenario_data["storage_units"]
    demand_units = scenario_data["demand_units"]
    exchange_units = scenario_data["exchange_units"]