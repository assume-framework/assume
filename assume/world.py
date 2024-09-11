# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from sys import platform

import nest_asyncio
import pandas as pd
from mango import RoleAgent, create_container
from mango.container.core import Container
from mango.util.clock import ExternalClock
from mango.util.distributed_clock import DistributedClockAgent, DistributedClockManager
from mango.util.termination_detection import tasks_complete_or_sleeping
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import OperationalError
from tqdm import tqdm

from assume.common import (
    Forecaster,
    MarketConfig,
    OutputDef,
    UnitsOperator,
    WriteOutput,
    mango_codec_factory,
)
from assume.common.base import LearningConfig
from assume.common.utils import create_rrule, datetime2timestamp, timestamp2datetime
from assume.markets import MarketRole, clearing_mechanisms
from assume.strategies import LearningStrategy, bidding_strategies
from assume.units import BaseUnit, unit_types

file_handler = logging.FileHandler(filename="assume.log", mode="w+")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.INFO, handlers=handlers)
logging.getLogger("mango").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class World:
    """
    World instance with the provided address, database URI, export CSV path, log level, and distributed role settings.

    If a database URI is provided, it establishes a database connection. Additionally, it sets up various dictionaries and attributes for market operators,
    markets, unit operators, unit types, bidding strategies, and clearing mechanisms. If available, it imports learning strategies and handles any potential import errors.
    Finally, it sets up the event loop for asynchronous operations.

    Attributes:
        addr (Union[tuple[str, int], str]): The address of the world, represented as a tuple of string and int or a string.
        container (mango.Container, optional): The container for the world instance.
        distributed_role (bool, optional): A boolean indicating whether distributed roles are enabled.
        export_csv_path (str): The path for exporting CSV data.
        db (sqlalchemy.engine.base.Engine, optional): The database connection.
        market_operators (dict[str, mango.RoleAgent]): The market operators for the world instance.
        markets (dict[str, MarketConfig]): The markets for the world instance.
        unit_operators (dict[str, UnitsOperator]): The unit operators for the world instance.
        unit_types (dict[str, BaseUnit]): The unit types for the world instance.
        bidding_strategies (dict[str, type[BaseStrategy]]): The bidding strategies for the world instance.
        clearing_mechanisms (dict[str, MarketRole]): The clearing mechanisms for the world instance.
        addresses (list[str]): The addresses for the world instance.
        loop (asyncio.AbstractEventLoop): The event loop for the world instance.
        clock (ExternalClock): The external clock for the world instance.
        start (datetime.datetime): The start datetime for the simulation.
        end (datetime.datetime): The end datetime for the simulation.
        learning_config (LearningConfig): The configuration for the learning process.
        perform_evaluation (bool): A boolean indicating whether the evaluation mode is enabled.
        forecaster (Forecaster, optional): The forecaster used for custom unit types.
        learning_mode (bool): A boolean indicating whether the learning mode is enabled.
        output_agent_addr (tuple[str, str]): The address of the output agent.
        bidding_params (dict): Parameters for bidding.
        index (pandas.Series): The index for the simulation.

    Args:
        addr: The address of the world, represented as a tuple of string and int or a string.
        database_uri: The URI for the database connection.
        export_csv_path: The path for exporting CSV data.
        log_level: The logging level for the world instance.
        distributed_role: A boolean indicating whether distributed roles are enabled.
            If True - this world is a manager world which schedules events itself.
            If False - this world is a client world which receives schedules from a manager through the DistributedClock mechanism.
            If None (default) - this world is not distributed and does not use subprocesses
    """

    def __init__(
        self,
        addr: tuple[str, int] | str = "world",
        database_uri: str = "",
        export_csv_path: str = "",
        log_level: str = "INFO",
        distributed_role: bool | None = None,
    ) -> None:
        logging.getLogger("assume").setLevel(log_level)
        self.addr = addr
        self.container = None
        self.distributed_role = distributed_role

        self.export_csv_path = export_csv_path
        # intialize db connection at beginning of simulation
        if database_uri:
            if str(database_uri).startswith("sqlite:///"):
                db_path = Path(str(database_uri).replace("sqlite:///", ""))
                db_path.parent.mkdir(exist_ok=True)
            self.db = create_engine(make_url(database_uri))
            connected = False
            while not connected:
                try:
                    with self.db.connect():
                        connected = True
                        logger.info("connected to db")
                except OperationalError as e:
                    logger.error("could not connect to %s, trying again", database_uri)
                    # log error if not connection refused
                    if not e.code == "e3q8":
                        logger.error("%s", e)
                    time.sleep(2)
        else:
            self.db = None

        self.market_operators: dict[str, RoleAgent] = {}
        self.markets: dict[str, MarketConfig] = {}
        self.unit_operators: dict[str, UnitsOperator] = {}
        self.unit_types = unit_types

        self.bidding_strategies = bidding_strategies

        if "pp_learning" not in bidding_strategies:
            logger.info(
                "Learning Strategies are not available. Check that you have torch installed."
            )

        self.clearing_mechanisms: dict[str, MarketRole] = clearing_mechanisms
        self.additional_kpis: dict[str, OutputDef] = {}
        self.addresses = []
        nest_asyncio.apply()
        self.loop = asyncio.get_event_loop()
        asyncio.set_event_loop(self.loop)

    async def setup(
        self,
        start: datetime,
        end: datetime,
        simulation_id: str,
        index: pd.Series,
        save_frequency_hours: int = 24,
        bidding_params: dict = {},
        learning_config: LearningConfig = {},
        forecaster: Forecaster | None = None,
        manager_address=None,
        **kwargs: dict,
    ) -> None:
        """
        Set up the environment for the simulation, initializing various parameters and components required for the simulation run.

        Args:
            start (datetime.datetime): The start datetime for the simulation.
            end (datetime.datetime): The end datetime for the simulation.
            simulation_id (str): The unique identifier for the simulation.
            index (pandas.Series): The index for the simulation.
            save_frequency_hours (int, optional): The frequency (in hours) at which to save simulation data. Defaults to 24.
            bidding_params (dict, optional): Parameters for bidding. Defaults to an empty dictionary.
            learning_config (LearningConfig, optional): Configuration for the learning process. Defaults to an empty configuration.
            forecaster (Forecaster, optional): The forecaster used for custom unit types. Defaults to None.
            manager_address: The address of the manager.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        self.clock = ExternalClock(0)
        self.start = start
        self.end = end
        self.learning_config = learning_config
        # initiate learning if the learning mode is on and hence we want to learn new strategies
        self.perform_evaluation = self.learning_config.get("perform_evaluation", False)

        # forecaster is used only when loading custom unit types
        self.forecaster = forecaster

        self.bidding_params = bidding_params
        self.index = index

        # kill old container if exists
        if isinstance(self.container, Container) and self.container.running:
            await self.container.shutdown()

        # create new container
        container_kwargs = {}
        if self.addr == "world":
            connection_type = "external_connection"
        elif isinstance(self.addr, tuple):
            connection_type = "tcp"
        else:
            connection_type = "mqtt"
            container_kwargs["mqtt_kwargs"] = {
                "broker_addr": "localhost",
                "client_id": self.addr,
            }
            container_kwargs["mqtt_kwargs"].update(**kwargs)

        self.container = await create_container(
            connection_type=connection_type,
            codec=mango_codec_factory(),
            addr=self.addr,
            clock=self.clock,
            **container_kwargs,
        )
        self.learning_mode = self.learning_config.get("learning_mode", False)
        self.output_agent_addr = (self.addr, "export_agent_1")
        if self.distributed_role is False:
            # if distributed_role is False - we are a ChildContainer
            # and only connect to the manager_address, which can set/sync our clock
            self.clock_agent = DistributedClockAgent(self.container)
            self.output_agent_addr = (manager_address, "export_agent_1")

            def stop(fut):
                self.loop.run_until_complete(self.container.shutdown())

            # when the clock_agent is stopped, we should gracefully shutdown our container
            self.clock_agent.stopped.add_done_callback(stop)
        else:
            await self.setup_learning()
            await self.setup_output_agent(simulation_id, save_frequency_hours)
            self.clock_manager = DistributedClockManager(
                self.container, receiver_clock_addresses=self.addresses
            )

    async def setup_learning(self) -> None:
        """
        Set up the learning process for the simulation, updating bidding parameters with the learning configuration
        and initializing the reinforcement learning (RL) learning role with the specified parameters. It also sets up
        the RL agent and adds the learning role to it for further processing.
        """

        self.bidding_params.update(self.learning_config)

        if self.learning_mode or self.perform_evaluation:
            # if so, we initate the rl learning role with parameters
            from assume.reinforcement_learning.learning_role import Learning

            self.learning_role = Learning(self.learning_config)
            # separate process does not support buffer and learning
            self.learning_agent_addr = (self.addr, "learning_agent")
            rl_agent = RoleAgent(
                self.container,
                suggested_aid=self.learning_agent_addr[1],
                suspendable_tasks=False,
            )
            rl_agent.add_role(self.learning_role)

    async def setup_output_agent(
        self, simulation_id: str, save_frequency_hours: int
    ) -> None:
        """
        Set up the output agent for the simulation, creating an output role responsible for writing simulation output,
        including data storage and export settings. Depending on the platform (currently supported only on Linux),
        it adds the output agent to the container's processes, or directly adds the output role to the output agent.

        Args:
            simulation_id (str): The unique identifier for the simulation.
            save_frequency_hours (int): The frequency (in hours) at which to save simulation data.
        """

        logger.debug(
            "creating output agent db=%s export_csv_path=%s",
            self.db,
            self.export_csv_path,
        )
        self.output_role = WriteOutput(
            simulation_id=simulation_id,
            start=self.start,
            end=self.end,
            db_engine=self.db,
            export_csv_path=self.export_csv_path,
            save_frequency_hours=save_frequency_hours,
            learning_mode=self.learning_mode,
            perform_evaluation=self.perform_evaluation,
            additional_kpis=self.additional_kpis,
        )

        # mango multiprocessing is currently only supported on linux
        # with single
        if platform == "linux" and self.distributed_role is not None:
            self.addresses.append((self.addr, "clock_agent"))

            def creator(container):
                agent = RoleAgent(
                    container,
                    suggested_aid=self.output_agent_addr[1],
                    suspendable_tasks=False,
                )
                agent.add_role(self.output_role)
                DistributedClockAgent(container)

            await self.container.as_agent_process(agent_creator=creator)
        else:
            output_agent = RoleAgent(
                self.container,
                suggested_aid=self.output_agent_addr[1],
                suspendable_tasks=False,
            )
            output_agent.add_role(self.output_role)

    def add_unit_operator(self, id: str) -> None:
        """
        Add a unit operator to the simulation, creating a new role agent and applying the role of a unit operator to it.
        The unit operator is then added to the list of existing operators. Unit operator receives the output agent address
        if not in learning mode.

        Args:
            id (str): The identifier for the unit operator.
        """

        if self.unit_operators.get(id):
            raise ValueError(f"Unit operator {id} already exists")

        units_operator = UnitsOperator(available_markets=list(self.markets.values()))

        # creating a new role agent and apply the role of a units operator
        unit_operator_agent = RoleAgent(
            self.container, suggested_aid=f"{id}", suspendable_tasks=False
        )
        unit_operator_agent.add_role(units_operator)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

        # after creation of an agent - we set additional context params
        if not self.learning_mode:
            unit_operator_agent._role_context.data.update(
                {
                    "output_agent_addr": self.output_agent_addr[0],
                    "output_agent_id": self.output_agent_addr[1],
                }
            )

    def add_rl_unit_operator(self, id: str = "Operator-RL") -> None:
        """
        Add a RL unit operator to the simulation, creating a new role agent and applying the role of a unit operator to it.
        The unit operator is then added to the list of existing operators.

        The RL unit operator differs from the standard unit operator in that it is used to handle learning units. It has additional
        functions such as writing to the learning role and scheduling recurrent tasks for writing to the learning role. It also
        writes learning outputs to the output role.

        Args:
            id (str): The identifier for the unit operator.
        """

        from assume.reinforcement_learning.learning_unit_operator import RLUnitsOperator

        if self.unit_operators.get(id):
            raise ValueError(f"Unit operator {id} already exists")

        units_operator = RLUnitsOperator(available_markets=list(self.markets.values()))
        # creating a new role agent and apply the role of a units operator
        unit_operator_agent = RoleAgent(
            self.container, suggested_aid=f"{id}", suspendable_tasks=False
        )
        unit_operator_agent.add_role(units_operator)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

        unit_operator_agent._role_context.data.update(
            {
                "learning_output_agent_addr": self.output_agent_addr[0],
                "learning_output_agent_id": self.output_agent_addr[1],
            }
        )

        # after creation of an agent - we set additional context params
        if self.learning_mode:
            unit_operator_agent._role_context.data.update(
                {
                    "learning_agent_addr": self.learning_agent_addr[0],
                    "learning_agent_id": self.learning_agent_addr[1],
                }
            )
            recurrency_task = create_rrule(
                start=self.start,
                end=self.end,
                freq=self.learning_config.get("train_freq", "24h"),
            )

            units_operator.context.schedule_recurrent_task(
                units_operator.write_to_learning_role, recurrency_task
            )

        else:
            unit_operator_agent._role_context.data.update(
                {
                    "output_agent_addr": self.output_agent_addr[0],
                    "output_agent_id": self.output_agent_addr[1],
                }
            )

    async def add_units_with_operator_subprocess(self, id: str, units: list[dict]):
        """
        Adds a units operator with given ID in a separate process
        and creates and adds the given list of unit dictionaries to it
        through a creator function

        Args:
            id (str): the id of the units operator
            units (list[dict]): list of unit dictionaries forwarded to create_unit
        """
        clock_agent_name = f"clock_agent_{id}"
        self.addresses.append((self.addr, clock_agent_name))

        async def creator(container):
            # creating a new role agent and apply the role of a units operator
            units_operator = UnitsOperator(
                available_markets=list(self.markets.values())
            )
            unit_operator_agent = RoleAgent(
                container, suggested_aid=f"{id}", suspendable_tasks=False
            )
            unit_operator_agent.add_role(units_operator)
            unit_operator_agent._role_context.data.update(
                {
                    "output_agent_addr": self.output_agent_addr[0],
                    "output_agent_id": self.output_agent_addr[1],
                    "learning_output_agent_addr": self.output_agent_addr[0],
                    "learning_output_agent_id": self.output_agent_addr[1],
                }
            )
            DistributedClockAgent(container, clock_agent_name)
            for unit in units:
                await units_operator.add_unit(self.create_unit(**unit))

        await self.container.as_agent_process(agent_creator=creator)

    def create_unit(
        self,
        id: str,
        unit_type: str,
        unit_operator_id: str,
        unit_params: dict,
        forecaster: Forecaster,
    ) -> BaseUnit:
        # provided unit type does not exist yet
        unit_class: type[BaseUnit] = self.unit_types.get(unit_type)

        bidding_strategies = self._prepare_bidding_strategies(unit_params, id)
        # if we have learning strategy we need to assign the powerplant to one unit_operator handling all learning units
        unit_params["bidding_strategies"] = bidding_strategies

        if self.learning_mode:
            self._add_bidding_strategies_to_learning_role(id, bidding_strategies)

        # create unit within the unit operator its associated with
        return unit_class(
            id=id,
            unit_operator=unit_operator_id,
            index=self.index,
            forecaster=forecaster,
            **unit_params,
        )

    async def async_add_unit(
        self,
        id: str,
        unit_type: str,
        unit_operator_id: str,
        unit_params: dict,
        forecaster: Forecaster,
    ) -> None:
        """
        Asynchronously adds a unit to the simulation, checking if the unit operator exists, verifying the unit type,
        and ensuring that the unit operator does not already have a unit with the same id. It then creates bidding
        strategies for the unit and adds the unit within the associated unit operator.

        Args:
            id (str): The identifier for the unit.
            unit_type (str): The type of unit to be added.
            unit_operator_id (str): The identifier of the unit operator to which the unit will be added.
            unit_params (dict): Parameters for configuring the unit.
            forecaster (Forecaster): The forecaster used by the unit.
        """

        # check if unit operator exists
        self._validate_unit_addition(id, unit_type, unit_operator_id)

        unit = self.create_unit(
            id, unit_type, unit_operator_id, unit_params, forecaster
        )

        await self.unit_operators[unit_operator_id].add_unit(unit)

    def _add_bidding_strategies_to_learning_role(self, unit_id, bidding_strategies):
        """
        Add bidding strategies to the learning role for the specified unit.

        Args:
            unit_id (str): The identifier for the unit.
            bidding_strategies (dict[str, BaseStrategy]): The bidding strategies for the unit.
        """

        for strategy in bidding_strategies.values():
            if isinstance(strategy, LearningStrategy):
                self.learning_role.rl_strats[unit_id] = strategy

    def _prepare_bidding_strategies(self, unit_params, unit_id):
        """
        Prepare bidding strategies for the unit based on the specified parameters.

        Args:
            unit_params (dict): Parameters for configuring the unit.
            unit_id (str): The identifier for the unit.

        Returns:
            dict[str, BaseStrategy]: The bidding strategies for the unit.
        """
        bidding_strategies = {}
        for market_id, strategy in unit_params["bidding_strategies"].items():
            if not strategy:
                continue

            if strategy not in self.bidding_strategies:
                raise ValueError(
                    f"""Bidding strategy {strategy} not registered. Please check the name of
                    the bidding strategy or register the bidding strategy in the world.bidding_strategies dict."""
                )

            bidding_params = unit_params.get("bidding_params", self.bidding_params)

            bidding_strategies[market_id] = self.bidding_strategies[strategy](
                unit_id=unit_id,
                **bidding_params,
            )

        return bidding_strategies

    def _validate_unit_addition(self, id, unit_type, unit_operator_id):
        """
        Validate the addition of a unit to the simulation, checking if the unit operator and unit type exist,
        and ensuring that the unit does not already exist.

        Args:
            id (str): The identifier for the unit.
            unit_type (str): The type of the unit.
            unit_operator_id (str): The identifier of the unit operator.
        """

        if unit_operator_id not in self.unit_operators:
            raise ValueError(f"Invalid unit operator: {unit_operator_id}")

        if unit_type not in self.unit_types:
            raise ValueError(f"Invalid unit type: {unit_type}")

        if self.unit_operators[unit_operator_id].units.get(id):
            raise ValueError(f"Unit {id} already exists")

    def add_market_operator(self, id: str) -> None:
        """
        Add a market operator to the simulation by creating a new role agent for the market operator
        and setting additional context parameters. If not in learning mode and not in evaluation mode,
        it includes the output agent address and ID in the role context data dictionary.

        Args:
            id (str): The identifier for the market operator.
        """

        if self.market_operators.get(id):
            raise ValueError(f"MarketOperator {id} already exists")
        market_operator_agent = RoleAgent(
            self.container, suggested_aid=id, suspendable_tasks=False
        )
        market_operator_agent.markets = []

        # after creation of an agent - we set additional context params
        if not self.learning_mode and not self.perform_evaluation:
            market_operator_agent._role_context.data.update(
                {
                    "output_agent_addr": self.output_agent_addr[0],
                    "output_agent_id": self.output_agent_addr[1],
                }
            )
        self.market_operators[id] = market_operator_agent

    def add_market(self, market_operator_id: str, market_config: MarketConfig) -> None:
        """
        Add a market to the simulation by creating a market role based on the specified market mechanism in the market
        configuration. Then, add this role to the specified market operator and append the market configuration to the list
        of markets within the market operator. Additionally, store the market configuration in the simulation's markets dictionary.

        Args:
            market_operator_id (str): The identifier of the market operator to which the market will be added.
            market_config (MarketConfig): The configuration for the market to be added.

        Returns:
            None
        """

        if mm_class := self.clearing_mechanisms.get(market_config.market_mechanism):
            market_role = mm_class(market_config)
        else:
            raise Exception(
                f"invalid {market_config.market_mechanism=} - full version installed?"
            )

        market_operator = self.market_operators.get(market_operator_id)

        if not market_operator:
            raise Exception(f"invalid {market_operator_id=}")

        market_operator.add_role(market_role)
        market_operator.markets.append(market_config)
        self.markets[f"{market_config.market_id}"] = market_config

    async def _step(self):
        if self.distributed_role:
            # TODO find better way than sleeping
            # we need to wait, until the last step is executed correctly
            await asyncio.sleep(0.04)
        if self.distributed_role is not False:
            next_activity = await self.clock_manager.distribute_time()
        else:
            next_activity = self.clock.get_next_activity()
        if not next_activity:
            logger.info("simulation finished - no schedules left")
            return None
        delta = next_activity - self.clock.time
        self.clock.set_time(next_activity)
        await tasks_complete_or_sleeping(self.container)
        return delta

    async def async_run(self, start_ts: datetime, end_ts: datetime):
        """
        Run the simulation asynchronously, progressing the simulation time from the start timestamp to the end timestamp,
        allowing registration before the first opening. If distributed roles are enabled, broadcast the simulation time.
        Iterate through the simulation time, updating the progress bar and the simulation description. Once the simulation
        time reaches the end timestamp, close the progress bar and shut down the simulation container.

        Args:
            start_ts (datetime.datetime): The start timestamp for the simulation run.
            end_ts (datetime.datetime): The end timestamp for the simulation run.
        """
        # agent is implicit added to self.container._agents
        pbar = tqdm(total=end_ts - start_ts)

        # allow registration before first opening
        self.clock.set_time(start_ts - 1)
        if self.distributed_role is not False:
            await self.clock_manager.broadcast(self.clock.time)
        prev_delta = 0
        while self.clock.time < end_ts:
            await asyncio.sleep(0)
            delta = await self._step()
            if delta or prev_delta:
                pbar.update(delta)
                pbar.set_description(
                    f"{self.output_role.simulation_id} {timestamp2datetime(self.clock.time)}",
                    refresh=False,
                )
            else:
                self.clock.set_time(end_ts)
            prev_delta = delta
        pbar.close()
        await self.container.shutdown()

    def run(self):
        """
        Run the simulation.

        This method converts the start and end timestamps to UTC time and then runs the asynchronous simulation using
        the `async_run` method. It progresses the simulation time from the start timestamp to the end timestamp, allowing
        registration before the first opening. If distributed roles are enabled, it broadcasts the simulation time. The
        method then iterates through the simulation time, updating the progress bar and the simulation description. Once
        the simulation time reaches the end timestamp, the method closes the progress bar and shuts down the simulation
        container.
        """

        start_ts = datetime2timestamp(self.start)
        end_ts = datetime2timestamp(self.end)

        try:
            return self.loop.run_until_complete(
                self.async_run(start_ts=start_ts, end_ts=end_ts)
            )
        except KeyboardInterrupt:
            pass

    def reset(self):
        """
        Reset the market operators, markets, unit operators, and forecast providers to empty dictionaries.

        Returns:
            None
        """
        self.market_operators = {}
        self.markets = {}
        self.unit_operators = {}
        self.forecast_providers = {}

    def add_unit(
        self,
        id: str,
        unit_type: str,
        unit_operator_id: str,
        unit_params: dict,
        forecaster: Forecaster,
    ) -> None:
        """
        Add a unit to the World instance.

        This method checks if the unit operator exists, verifies the unit type, and ensures that the unit operator
        does not already have a unit with the same id. It then creates bidding strategies for the unit and creates
        the unit within the associated unit operator.

        Args:
            id (str): The identifier for the unit.
            unit_type (str): The type of the unit.
            unit_operator_id (str): The identifier of the unit operator.
            unit_params (dict): Parameters specific to the unit.
            forecaster (Forecaster): The forecaster associated with the unit.
        """

        return self.loop.run_until_complete(
            self.async_add_unit(
                id=id,
                unit_type=unit_type,
                unit_operator_id=unit_operator_id,
                unit_params=unit_params,
                forecaster=forecaster,
            )
        )
