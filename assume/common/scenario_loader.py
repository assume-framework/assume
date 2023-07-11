import logging
from datetime import datetime
from typing import Callable

import dateutil.rrule as rr
import numpy as np
import pandas as pd
import yaml
from mango import RoleAgent

from assume.common import ForecastProvider
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.world import World

logger = logging.getLogger(__name__)

freq_map = {
    "h": rr.HOURLY,
    "m": rr.MINUTELY,
    "d": rr.DAILY,
    "w": rr.WEEKLY,
}


def attempt_resample(
    df: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if len(df.index) > len(index):
        temp_index = pd.date_range(start=index[0], end=index[-1], periods=len(df))
        df.index = temp_index
        df = df.resample(index.freq).mean()
        logger.info("Resampling successful.")
        return df
    elif len(df.index) < len(index):
        logger.warning(
            "Index length mismatch. Upsampling not supported. Returning None"
        )
        return None
    else:
        return df


def load_file(
    path: str,
    config: dict,
    file_name: str,
    index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    df = None

    if file_name in config:
        file_path = f"{path}/{config[file_name]}"
    else:
        file_path = f"{path}/{file_name}.csv"

    try:
        df = pd.read_csv(
            file_path,
            index_col=0,
            encoding="utf-8",
            na_values=["n.a.", "None", "-", "none", "nan"],
        )

        for col in df:
            # check if the column is of dtype int
            if df[col].dtype == "int":
                # convert the column to float
                df[col] = df[col].astype(float)

        if index is not None:
            if len(df.index) == 1:
                return df

            if len(df.index) != len(index):
                logger.warning(
                    f"Length of index does not match length of {file_name} dataframe. Attempting to resample."
                )
                df = attempt_resample(df, index)
            else:
                df.index = index

        return df

    except FileNotFoundError:
        logger.warning(f"File {file_name} not found. Returning None")


def convert_to_rrule_freq(string):
    freq = freq_map[string[-1]]
    interval = int(string[:-1])
    return freq, interval


def make_market_config(
    id: str,
    market_params: dict,
    world_start: datetime,
    world_end: datetime,
):
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
        name=id,
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
    )

    return market_config


async def add_units(
    dataframe: pd.DataFrame,
    unit_type: str,
    callback: Callable[[str, dict], dict],
    world: World,
    bidding_strategies_df: pd.DataFrame,
):
    """
    This function adds units to the world from a given dataframe.
    The callback is used to adjust unit_params depending on the unit_type, before adding the unit to the world.
    """
    if dataframe is None:
        return
    logger.info(f"Adding {unit_type} units")
    for unit_name, unit_params in dataframe.iterrows():
        if (
            bidding_strategies_df is not None
            and unit_name in bidding_strategies_df.index
        ):
            unit_params["bidding_strategies"] = bidding_strategies_df.loc[
                unit_name
            ].to_dict()
        else:
            logger.warning(
                f"No bidding strategies specified for {unit_name}. Using default strategies."
            )
            unit_params["bidding_strategies"] = {
                market.product_type: "naive" for market in world.markets.values()
            }

        unit_params = callback(unit_name, unit_params)

        await world.add_unit(
            id=unit_name,
            unit_type=unit_type,
            unit_operator_id=unit_params["unit_operator"],
            unit_params=unit_params,
        )


async def load_scenario_folder_async(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
):
    """Load a scenario from a given path.

    Args:
        inputs_path (str): Path to the inputs folder.
        scenario (str): Name of the scenario.
        study_case (str): Name of the study case.

    Raises:
        ValueError: If the scenario or study case is not found.
    """

    # load the config file
    path = f"{inputs_path}/{scenario}"
    with open(f"{path}/config.yml", "r") as f:
        config = yaml.safe_load(f)
        if not study_case:
            study_case = list(config.keys())[0]
        config = config[study_case]
    logger.info(f"Starting Scenario {scenario}/{study_case} from {inputs_path}")

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    index = pd.date_range(
        start=start,
        end=end + pd.Timedelta(hours=4),
        freq=config["time_step"],
    )
    # get extra parameters for bidding strategies

    # load the data from the csv files
    # tries to load all files, returns a warning if file does not exist
    # also attempts to resample the inputs if their resolution is higher than user specified time step
    logger.info("Loading input data")
    powerplant_units = load_file(
        path=path,
        config=config,
        file_name="powerplant_units",
    )

    storage_units = load_file(
        path=path,
        config=config,
        file_name="storage_units",
    )

    demand_units = load_file(
        path=path,
        config=config,
        file_name="demand_units",
    )

    heatpump_units = load_file(path=path, config=config, file_name="heatpumps")

    fuel_prices_df = load_file(
        path=path,
        config=config,
        file_name="fuel_prices",
        index=index,
    )

    temperature_df = load_file(
        path=path, config=config, file_name="temperature", index=index
    )

    electricity_prices_df = load_file(
        path=path, config=config, file_name="electricity_prices", index=index
    )

    demand_df = load_file(
        path=path,
        config=config,
        file_name="demand_df",
        index=index,
    )

    vre_cf_df = load_file(
        path=path,
        config=config,
        file_name="vre_cf_df",
        index=index,
    )

    bidding_strategies_df = load_file(
        path=path, config=config, file_name="bidding_strategies"
    ).fillna(0)

    cross_border_flows_df = load_file(
        path=path,
        config=config,
        file_name="cross_border_flows",
        index=index,
    )
    cross_border_flows_df

    price_forecast_df = load_file(
        path=path,
        config=config,
        file_name="price_forecasts",
        index=index,
    )

    if powerplant_units is None or demand_units is None:
        raise ValueError("No power plant or no demand units were provided!")

    if demand_df is None:
        raise ValueError("No demand time series was provided!")

    # Initialize world

    save_frequency_hours = config.get("save_frequency_hours", None)
    sim_id = f"{scenario}_{study_case}"

    bidding_params = config.get("bidding_strategy_params", {})
    await world.setup(start, end, save_frequency_hours, sim_id, bidding_params, index)

    # add Agents to world

    # get the market config from the config file and add the markets
    logger.info("Adding markets")
    for market_id, market_params in config["markets_config"].items():
        market_config = make_market_config(
            id=market_id,
            market_params=market_params,
            world_start=start,
            world_end=end,
        )

        operator_id = str(market_params["operator"])
        if operator_id not in world.market_operators:
            world.add_market_operator(id=operator_id)

        world.add_market(
            market_operator_id=operator_id,
            market_config=market_config,
        )

    # add forecast providers for each market
    logger.info("Adding forecast providers")
    temp = pd.DataFrame()
    for market_id, market_config in world.markets.items():
        if price_forecast_df is not None and market_id in price_forecast_df.columns:
            market_price_forecast = price_forecast_df[market_id]
        else:
            market_price_forecast = None

        forecast_provider = ForecastProvider(
            market_id=market_id,
            price_forecast_df=market_price_forecast,
            fuel_prices_df=fuel_prices_df,
            vre_cf_df=vre_cf_df,
            powerplants=powerplant_units,
            demand_df=demand_df[f"demand_{market_id}"],
        )
        forecast_agent = RoleAgent(
            world.container, suggested_aid=f"forecast_agent_{market_id}"
        )
        forecast_agent.add_role(forecast_provider)
        world.forecast_providers[market_id] = forecast_provider

        temp[market_id] = forecast_provider.price_forecast_df

    temp.to_csv(f"{path}/price_forecasts.csv", index=True)

    # add the unit operators using unique unit operator names in the powerplants csv
    logger.info("Adding unit operators")
    all_operators = np.concatenate(
        [
            powerplant_units.unit_operator.unique(),
            demand_units.unit_operator.unique(),
            # heatpump_units.unit_operator.unique(),
            # storage_units.unit_operator.unique(),
        ]
    )

    if storage_units is not None:
        all_operators = np.concatenate(
            [all_operators, storage_units.unit_operator.unique()]
        )

    for company_name in set(all_operators):
        world.add_unit_operator(id=str(company_name))

    # add the units to corresponsing unit operators
    # if fuel prices are provided, add them to the unit params
    # if vre generation is provided, add them to the vre units
    # if we have RL strategy, add price forecast to unit_params
    def powerplant_callback(unit_name, unit_params):
        # check if we have RL bidding strategy
        if (
            unit_params["bidding_strategies"]["energy"] == "rl_strategy"
            and world.forecast_providers["EOM"].price_forecast_df is not None
        ):
            unit_params["price_forecast"] = world.price_forecast["mcp"]
            unit_params["res_demand_forecast"] = world.price_forecast["residual_demand"]

        if (
            fuel_prices_df is not None
            and unit_params["fuel_type"] in fuel_prices_df.columns
        ):
            unit_params["fuel_price"] = fuel_prices_df[unit_params["fuel_type"]]
            unit_params["co2_price"] = fuel_prices_df["co2"]

        if vre_cf_df is not None and unit_name in vre_cf_df.columns:
            unit_params["capacity_factor"] = vre_cf_df[unit_name]
        return unit_params

    await add_units(
        powerplant_units,
        "power_plant",
        powerplant_callback,
        world,
        bidding_strategies_df,
    )

    def heatpump_callback(unit_name, unit_params):
        if electricity_prices_df is not None:
            unit_params["electricity_price"] = electricity_prices_df[
                "electricity_price"
            ]

        if temperature_df is not None:
            unit_params["source_temp"] = temperature_df["source_temperature"]
            unit_params["sink_temp"] = temperature_df["sink_temperature"]
        return unit_params

    await add_units(
        heatpump_units, "heatpump", heatpump_callback, world, bidding_strategies_df
    )

    def storage_callback(unit_name, unit_params):
        unit_params["price_forecast"] = world.forecast_providers[
            "EOM"
        ].price_forecast_df
        return unit_params

    await add_units(
        storage_units, "storage", storage_callback, world, bidding_strategies_df
    )

    def demand_callback(unit_name, unit_params):
        if demand_df is not None and unit_name in demand_df.columns:
            unit_params["volume"] = demand_df[unit_name]
        return unit_params

    await add_units(
        demand_units, "demand", demand_callback, world, bidding_strategies_df
    )


def load_scenario_folder(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
):
    """
    Load a scenario from a given path.
    """
    return world.loop.run_until_complete(
        load_scenario_folder_async(
            world,
            inputs_path,
            scenario,
            study_case,
        )
    )
