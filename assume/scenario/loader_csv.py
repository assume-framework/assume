# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import dateutil.rrule as rr
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from assume.common.base import LearningConfig
from assume.common.exceptions import AssumeException
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
    index: Optional[pd.DatetimeIndex] = None,
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
        pd.DataFrame: The dataframe containing the loaded data.

    Raises:
        FileNotFoundError: If the specified file is not found, returns None.
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


def convert_to_rrule_freq(string: str) -> Tuple[int, int]:
    """
    Convert a string to a rrule frequency and interval.

    Args:
        string (str): The string to be converted. Should be in the format of "1h" or "1d" or "1w".

    Returns:
        Tuple[int, int]: The rrule frequency and interval.
    """
    freq = freq_map[string[-1]]
    interval = int(string[:-1])
    return freq, interval


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
) -> None:
    """
    Add units to the world from a given dataframe.
    The callback is used to adjust unit_params depending on the unit_type, before adding the unit to the world.

    Args:
        units_df (pd.DataFrame): The dataframe containing the units.
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


async def load_scenario_folder_async(
    world: World,
    inputs_path: str,
    scenario: str,
    study_case: str,
    perform_learning: bool = True,
    perform_evaluation: bool = False,
    episode: int = 0,
    eval_episode: int = 0,
) -> None:
    """
    Load a scenario from a given path.

    This function loads a scenario within a specified study case from a given path, setting up the world environment for simulation and learning.

    Args:
        world (World): An instance of the World class representing the simulation environment.
        inputs_path (str): The path to the folder containing input files necessary for the scenario.
        scenario (str): The name of the scenario to be loaded.
        study_case (str): The specific study case within the scenario to be loaded.
        perform_learning (bool, optional): A flag indicating whether learning should be performed. Defaults to True.
        perform_evaluation (bool, optional): A flag indicating whether evaluation should be performed. Defaults to False.
        episode (int, optional): The episode number for learning. Defaults to 0.
        eval_episode (int, optional): The episode number for evaluation. Defaults to 0.

    Raises:
        ValueError: If the specified scenario or study case is not found in the provided inputs.

    """

    # load the config file
    path = f"{inputs_path}/{scenario}"
    with open(f"{path}/config.yaml", "r") as f:
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
        # end time needs to be a little ahead for forecasts
        end=end + timedelta(days=1),
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

    if powerplant_units is None or demand_units is None:
        raise ValueError("No power plant or no demand units were provided!")

    save_frequency_hours = config.get("save_frequency_hours", 48)
    sim_id = f"{scenario}_{study_case}"

    learning_config: LearningConfig = config.get("learning_config", {})
    bidding_strategy_params = config.get("bidding_strategy_params", {})

    learning_config["learning_mode"] = (
        config.get("learning_mode", False) and perform_learning
    )
    learning_config["evaluation_mode"] = perform_evaluation

    if learning_config.get("trained_policies_save_path"):
        learning_config[
            "trained_policies_save_path"
        ] = f"{inputs_path}/{learning_config['trained_policies_save_path']}"
    else:
        learning_config[
            "trained_policies_save_path"
        ] = f"{inputs_path}/learned_strategies/{sim_id}"

    if learning_config.get("trained_policies_load_path"):
        learning_config[
            "trained_policies_load_path"
        ] = f"{inputs_path}/{learning_config['trained_policies_load_path']}"

    if learning_config.get("learning_mode", False):
        sim_id = f"{sim_id}_{episode}"

    if learning_config.get("evaluation_mode", False):
        sim_id = f"{sim_id}_eval_{eval_episode}"

    # add forecast provider
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

        operator_id = str(market_params["operator"])
        if operator_id not in world.market_operators:
            world.add_market_operator(id=operator_id)

        world.add_market(
            market_operator_id=operator_id,
            market_config=market_config,
        )

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

    # add central RL unit oporator that handels all RL units
    if world.learning_mode == True and "Operator-RL" not in all_operators:
        all_operators = np.concatenate([all_operators, ["Operator-RL"]])

    for company_name in set(all_operators):
        world.add_unit_operator(id=str(company_name))

    # add the units to corresponsing unit operators
    add_units(
        units_df=powerplant_units,
        unit_type="power_plant",
        world=world,
        forecaster=forecaster,
    )

    add_units(
        units_df=storage_units,
        unit_type="storage",
        world=world,
        forecaster=forecaster,
    )

    add_units(
        units_df=demand_units,
        unit_type="demand",
        world=world,
        forecaster=forecaster,
    )

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
    perform_learning: bool = True,
    perform_evaluation: bool = False,
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
        perform_learning (bool, optional): A flag indicating whether learning should be performed. Defaults to True.
        perform_evaluation (bool, optional): A flag indicating whether evaluation should be performed. Defaults to False.
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
            perform_learning=True,
            perform_evaluation=False,
            episode=1,
            eval_episode=1,
            trained_policies_save_path="",
        )

    Notes:
        - The function sets up the world environment based on the provided inputs and configuration files.
        - If `perform_learning` is set to True and learning_mode is set, the function initializes the learning mode with the specified episode number.
        - If `perform_evaluation` is set to True, the function performs evaluation using the specified evaluation episode number.
        - The function utilizes the specified inputs to configure the simulation environment, including market parameters, unit operators, and forecasting data.
        - After calling this function, the world environment is prepared for further simulation and analysis.

    """
    world.loop.run_until_complete(
        load_scenario_folder_async(
            world=world,
            inputs_path=inputs_path,
            scenario=scenario,
            study_case=study_case,
            perform_learning=perform_learning,
            perform_evaluation=perform_evaluation,
            episode=episode,
            eval_episode=eval_episode,
        )
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
    world: World, inputs_path: str, scenario: str, study_case: str
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

    # remove csv path so that nothing is written while learning
    temp_csv_path = world.export_csv_path
    world.export_csv_path = ""

    buffer = ReplayBuffer(
        buffer_size=int(world.learning_config.get("replay_buffer_size", 5e5)),
        obs_dim=world.learning_role.obs_dim,
        act_dim=world.learning_role.act_dim,
        n_rl_units=len(world.learning_role.rl_strats),
        device=world.learning_role.device,
        float_type=world.learning_role.float_type,
    )
    actors_and_critics = None
    world.output_role.del_similar_runs()

    validation_interval = min(
        world.learning_role.training_episodes,
        world.learning_config.get("validation_episodes_interval", 5),
    )

    eval_episode = 1
    save_path = world.learning_config["trained_policies_save_path"]

    if Path(save_path).is_dir():
        # we are in learning mode and about to train new policies, which might overwrite existing ones
        accept = input(
            f"{save_path=} exists - should we overwrite current learnings? (y/N)"
        )
        if not accept.lower().startswith("y"):
            # stop here - do not start learning or save anything
            raise AssumeException("don't overwrite existing strategies")

    for episode in tqdm(
        range(1, world.learning_role.training_episodes + 1),
        desc="Training Episodes",
    ):
        # TODO normally, loading twice should not create issues, somehow a scheduling issue is raised currently
        if episode != 1:
            load_scenario_folder(
                world,
                inputs_path,
                scenario,
                study_case,
                perform_learning=True,
                episode=episode,
            )

        # give the newly created rl_agent the buffer that we stored from the beginning
        world.learning_role.initialize_policy(actors_and_critics=actors_and_critics)

        world.learning_role.buffer = buffer
        world.learning_role.episodes_done = episode

        if episode > world.learning_role.episodes_collecting_initial_experience:
            world.learning_role.turn_off_initial_exploration()

        world.run()

        actors_and_critics = world.learning_role.rl_algorithm.extract_policy()

        if (
            episode % validation_interval == 0
            and episode > world.learning_role.episodes_collecting_initial_experience
        ):
            # save current params in training path
            world.learning_role.rl_algorithm.save_params(directory=save_path)
            world.reset()

            # load validation run
            load_scenario_folder(
                world,
                inputs_path,
                scenario,
                study_case,
                perform_learning=False,
                perform_evaluation=True,
                eval_episode=eval_episode,
            )

            world.run()

            total_rewards = world.output_role.get_sum_reward()
            avg_reward = np.mean(total_rewards)
            # check reward improvement in validation run
            # and store best run in eval folder
            world.learning_role.compare_and_save_policies({"avg_reward": avg_reward})

            eval_episode += 1
        world.reset()

        # in load_scenario_folder_async, we initiate new container and kill old if present
        # as long as we do not skip setup container should be handled correctly
        # if enough initial experience was collected according to specifications in learning config
        # turn off initial exploration and go into full learning mode
        if episode >= world.learning_role.episodes_collecting_initial_experience:
            world.learning_role.turn_off_initial_exploration()

        # container shutdown implicitly with new initialisation
    logger.info("################")
    logger.info("Training finished, Start evaluation run")
    world.export_csv_path = temp_csv_path

    # load scenario for evaluation
    load_scenario_folder(
        world,
        inputs_path,
        scenario,
        study_case,
        perform_learning=False,
    )
