import logging
from datetime import datetime

import dateutil.rrule as rr
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from assume.common.forecasts import CsvForecaster, Forecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.world import World

logger = logging.getLogger(__name__)

freq_map = {
    "h": rr.HOURLY,
    "m": rr.MINUTELY,
    "d": rr.DAILY,
    "w": rr.WEEKLY,
}


def load_file(
    path: str,
    config: dict,
    file_name: str,
    index: pd.DatetimeIndex = None,
) -> pd.DataFrame:
    """
    This function loads a csv file from a given path and returns a dataframe.

    :param path: the path to the csv file
    :type path: str
    :param config: the config file
    :type config: dict
    :param file_name: the name of the csv file
    :type file_name: str
    :param index: the index of the dataframe
    :type index: pd.DatetimeIndex
    :return: the dataframe
    :rtype: pd.DataFrame
    """
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


def convert_to_rrule_freq(string):
    """
    This function converts a string to a rrule frequency and interval.
    The string should be in the format of "1h" or "1d" or "1w".

    :param string: the string to be converted
    :type string: str
    :return: the rrule frequency and interval
    :rtype: tuple[int, int]
    """
    freq = freq_map[string[-1]]
    interval = int(string[:-1])
    return freq, interval


def make_market_config(
    id: str,
    market_params: dict,
    world_start: datetime,
    world_end: datetime,
):
    """
    This function creates a market config from a given dictionary.

    :param id: the id of the market
    :type id: str
    :param market_params: the market parameters
    :type market_params: dict
    :param world_start: the start time of the world
    :type world_start: datetime
    :param world_end: the end time of the world
    :type world_end: datetime
    :return: the market config
    :rtype: MarketConfig
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


def add_units(
    units_df: pd.DataFrame,
    unit_type: str,
    world: World,
    forecaster: Forecaster,
):
    """
    This function adds units to the world from a given dataframe.
    The callback is used to adjust unit_params depending on the unit_type, before adding the unit to the world.

    :param units_df: the dataframe containing the units
    :type units_df: pd.DataFrame
    :param unit_type: the type of the unit
    :type unit_type: str
    :param world: the world
    :type world: World
    :param forecaster: the forecaster
    :type forecaster: Forecaster
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


async def load_scenario_folder_async(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
    disable_learning: bool = False,
    episode: int = 0,
):
    """Load a scenario from a given path. Raises: ValueError: If the scenario or study case is not found.

    :param world: The world.
    :type world: World
    :param inputs_path: Path to the inputs folder.
    :type inputs_path: str
    :param scenario: Name of the scenario.
    :type scenario: str
    :param study_case: Name of the study case.
    :type study_case: str
    """

    # load the config file
    path = f"{inputs_path}/{scenario}"
    with open(f"{path}/config.yml", "r") as f:
        config = yaml.safe_load(f)
        if not study_case:
            study_case = list(config.keys())[0]
        config = config[study_case]
    logger.info(f"Starting Scenario {scenario}/{study_case} from {inputs_path}")

    world.reset()

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])

    index = pd.date_range(
        start=start,
        end=end + pd.Timedelta(hours=24),
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

    heatpump_units = load_file(
        path=path,
        config=config,
        file_name="heatpump_units",
    )

    if powerplant_units is None or demand_units is None:
        raise ValueError("No power plant or no demand units were provided!")

    # Initialize world

    save_frequency_hours = config.get("save_frequency_hours", 48)
    sim_id = f"{scenario}_{study_case}"

    learning_config = config.get("learning_config", {})
    bidding_strategy_params = config.get("bidding_strategy_params", {})

    if "load_learned_path" not in learning_config.keys():
        learning_config[
            "load_learned_path"
        ] = f"{inputs_path}/learned_strategies/{sim_id}/"

    if disable_learning:
        learning_config = {}

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=sim_id,
        episode=episode,
        learning_config=learning_config,
        bidding_params=bidding_strategy_params,
        index=index,
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

        operator_id = str(market_params["operator"])
        if operator_id not in world.market_operators:
            world.add_market_operator(id=operator_id)

        world.add_market(
            market_operator_id=operator_id,
            market_config=market_config,
        )

    # add forecast providers for each market
    logger.info("Adding forecast")
    forecaster = CsvForecaster(
        index=index,
        powerplants=powerplant_units,
    )

    forecasts_df = load_file(
        path=path,
        config=config,
        file_name="forecasts_df",
        index=index,
    )
    forecaster.set_forecast(forecasts_df)

    demand_df = load_file(
        path=path,
        config=config,
        file_name="demand_df",
        index=index,
    )
    if demand_df is None:
        raise ValueError("No demand time series was provided!")
    forecaster.set_forecast(demand_df)

    cross_border_flows_df = load_file(
        path=path,
        config=config,
        file_name="cross_border_flows",
        index=index,
    )
    forecaster.set_forecast(cross_border_flows_df)

    availability = load_file(
        path=path,
        config=config,
        file_name="availability_df",
        index=index,
    )
    forecaster.set_forecast(availability, prefix="availability_")
    electricity_prices_df = load_file(
        path=path, config=config, file_name="electricity_prices", index=index
    )
    forecaster.set_forecast(electricity_prices_df)

    price_forecast_df = load_file(
        path=path, config=config, file_name="price_forecasts", index=index
    )
    forecaster.set_forecast(price_forecast_df, "price_")
    forecaster.set_forecast(
        load_file(
            path=path,
            config=config,
            file_name="fuel_prices_df",
            index=index,
        ),
        prefix="fuel_price_",
    )
    forecaster.set_forecast(
        load_file(path=path, config=config, file_name="temperature", index=index)
    )
    forecaster.calc_forecast_if_needed()
    forecaster.save_forecasts(path)

    # add the unit operators using unique unit operator names in the powerplants csv
    logger.info("Adding unit operators")
    all_operators = np.concatenate(
        [
            powerplant_units.unit_operator.unique(),
            demand_units.unit_operator.unique(),
        ]
    )

    if storage_units is not None:
        all_operators = np.concatenate(
            [all_operators, storage_units.unit_operator.unique()]
        )

    if heatpump_units is not None:
        all_operators = np.concatenate(
            [all_operators, heatpump_units.unit_operator.unique()]
        )

    for company_name in set(all_operators):
        world.add_unit_operator(id=str(company_name))

    # add the units to corresponsing unit operators
    # if fuel prices are provided, add them to the unit params
    # if vre generation is provided, add them to the vre units
    # if we have RL strategy, add price forecast to unit_params
    def empty_callback(unit_name, unit_params):
        return unit_params

    add_units(
        powerplant_units,
        "power_plant",
        world,
        forecaster,
    )

    add_units(
        heatpump_units,
        "heatpump",
        world,
        forecaster,
    )

    add_units(
        storage_units,
        "storage",
        world,
        forecaster,
    )

    def demand_callback(unit_name, unit_params):
        if demand_df is not None and unit_name in demand_df.columns:
            unit_params["volume"] = demand_df[unit_name]

        return unit_params

    add_units(
        demand_units,
        "demand",
        world,
        forecaster,
    )


def load_scenario_folder(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
):
    """
    Load a scenario from a given path.

    :param world: The world.
    :type world: World
    :param inputs_path: Path to the inputs folder.
    :type inputs_path: str
    :param scenario: Name of the scenario.
    :type scenario: str
    :param study_case: Name of the study case.
    :type study_case: str
    """
    world.loop.run_until_complete(
        load_scenario_folder_async(
            world,
            inputs_path,
            scenario,
            study_case,
        )
    )
    # check if learning mode
    if world.learning_config.get("learning_mode"):
        # initiate buffer for rl agent
        from assume.reinforcement_learning.buffer import ReplayBuffer

        buffer = ReplayBuffer(
            buffer_size=int(5e5),
            obs_dim=world.learning_role.obs_dim,
            act_dim=world.learning_role.act_dim,
            n_rl_units=len(world.learning_role.rl_units),
            device=world.learning_role.device,
        )

        for episode in tqdm(
            range(world.learning_role.training_episodes),
            desc="Training Episodes",
        ):
            # give the newly created rl_agent the buffer that we stored from the beginning
            world.learning_role.set_buffer(buffer)

            world.run()
            world.reset()

            world.learning_role.episodes_done = episode + 1

            disable_learning = episode == world.learning_role.training_episodes - 1

            # in load_scenario_folder_async, we initiate new container and kill old if present
            # as long as we do not skip setup container should be handled correctly
            world.loop.run_until_complete(
                load_scenario_folder_async(
                    world,
                    inputs_path,
                    scenario,
                    study_case,
                    episode=world.learning_role.episodes_done,
                    disable_learning=disable_learning,
                )
            )

            # container shutdown implicitly with new initialisation

        logger.info("################")
        logger.info(f"Training finished, Start evaluation run")
