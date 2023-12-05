# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import calendar
import logging
import sys
import time
from datetime import datetime
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
    UnitsOperator,
    WriteOutput,
    mango_codec_factory,
)
from assume.common.base import LearningConfig
from assume.markets import MarketRole, clearing_mechanisms
from assume.strategies import LearningStrategy, bidding_strategies
from assume.units import BaseUnit, Demand, PowerPlant, Storage

file_handler = logging.FileHandler(filename="assume.log", mode="w+")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.WARNING)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.INFO, handlers=handlers)
logging.getLogger("mango").setLevel(logging.WARNING)


class World:
    def __init__(
        self,
        addr: tuple[str, int] | str = "world",
        database_uri: str = "",
        export_csv_path: str = "",
        log_level: str = "INFO",
        distributed_role: bool = None,
    ):
        """
        Initialize a World instance.

        Args:
        - addr (tuple[str, int] | str): The address of the world, represented as a tuple of string and int or a string.
        - database_uri (str): The URI for the database connection.
        - export_csv_path (str): The path for exporting CSV data.
        - log_level (str): The logging level for the world instance.
        - distributed_role (bool): A boolean indicating whether distributed roles are enabled.

        Returns:
        - None

        This method initializes a World instance with the provided address, database URI,
        export CSV path, log level, and distributed role settings. It establishes a database
        connection if a database URI is provided. Additionally, it sets up various dictionaries
        and attributes for market operators, markets, unit operators, unit types, bidding strategies,
        and clearing mechanisms. If available, it imports learning strategies and handles any
        potential import errors. Finally, it sets up the event loop for asynchronous operations.
        """

        logging.getLogger("assume").setLevel(log_level)
        self.logger = logging.getLogger(__name__)
        self.addr = addr
        self.container = None
        self.distributed_role = distributed_role

        self.export_csv_path = export_csv_path
        # intialize db connection at beginning of simulation
        if database_uri:
            self.db = create_engine(make_url(database_uri))
            connected = False
            while not connected:
                try:
                    with self.db.connect():
                        connected = True
                        self.logger.info("connected to db")
                except OperationalError as e:
                    self.logger.error(
                        f"could not connect to {database_uri}, trying again"
                    )
                    # log error if not connection refused
                    if not e.code == "e3q8":
                        self.logger.error(f"{e}")
                    time.sleep(2)
        else:
            self.db = None

        self.market_operators: dict[str, RoleAgent] = {}
        self.markets: dict[str, MarketConfig] = {}
        self.unit_operators: dict[str, UnitsOperator] = {}
        self.unit_types = {
            "power_plant": PowerPlant,
            "demand": Demand,
            "storage": Storage,
        }

        self.bidding_strategies = bidding_strategies

        try:
            from assume.strategies.learning_block_strategies import RLStrategyBlocks
            from assume.strategies.learning_strategies import RLdamStrategy, RLStrategy

            self.bidding_strategies["learning"] = RLStrategy
            self.bidding_strategies["learning_dam"] = RLdamStrategy
            self.bidding_strategies["learning_blocks"] = RLStrategyBlocks

        except ImportError as e:
            self.logger.info(
                "Import of Learning Strategies failed. Check that you have all required packages installed (torch): %s",
                e,
            )
        self.clearing_mechanisms: dict[str, MarketRole] = clearing_mechanisms
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
        forecaster: Forecaster = None,
        manager_address=None,
        **kwargs,
    ):
        """
        Set up the environment for the simulation.

        Args:
        - self: The instance of the class.
        - start (datetime): The start datetime for the simulation.
        - end (datetime): The end datetime for the simulation.
        - simulation_id (str): The unique identifier for the simulation.
        - index (pd.Series): The index for the simulation.
        - save_frequency_hours (int, optional): The frequency (in hours) at which to save simulation data. Defaults to 24.
        - bidding_params (dict, optional): Parameters for bidding. Defaults to an empty dictionary.
        - learning_config (LearningConfig, optional): Configuration for the learning process. Defaults to an empty configuration.
        - forecaster (Forecaster, optional): The forecaster used for custom unit types. Defaults to None.
        - manager_address: The address of the manager.

        Other Parameters:
        - kwargs: Additional keyword arguments.

        Returns:
        - None

        This method sets up the environment for the simulation. It initializes various parameters and components
        required for the simulation run, including the clock, learning configuration, forecaster, container,
        connection type, and output agents. It also handles the setup for distributed roles if applicable.
        """

        self.clock = ExternalClock(0)
        self.start = start
        self.end = end
        self.learning_config = learning_config
        # initiate learning if the learning mode is on and hence we want to learn new strategies
        self.evaluation_mode = self.learning_config.get("evaluation_mode", False)

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
            self.clock_agent = DistributedClockAgent(self.container)
            self.output_agent_addr = (manager_address, "export_agent_1")
        else:
            await self.setup_learning()
            await self.setup_output_agent(simulation_id, save_frequency_hours)
            self.clock_manager = DistributedClockManager(
                self.container, receiver_clock_addresses=self.addresses
            )

    async def setup_learning(self):
        """
        Set up the learning process for the simulation.

        Args:
        - self: The instance of the class.

        Returns:
        - None

        This method sets up the learning process for the simulation. It updates the bidding parameters
        with the learning configuration and initializes the reinforcement learning (RL) learning role
        with the specified parameters. It also sets up the RL agent and adds the learning role to it
        for further processing.
        """

        self.bidding_params.update(self.learning_config)

        if self.learning_mode:
            # if so, we initate the rl learning role with parameters
            from assume.reinforcement_learning.learning_role import Learning

            self.learning_role = Learning(
                learning_config=self.learning_config,
                start=self.start,
                end=self.end,
            )
            # separate process does not support buffer and learning
            self.learning_agent_addr = (self.addr, "learning_agent")
            rl_agent = RoleAgent(
                self.container, suggested_aid=self.learning_agent_addr[1]
            )
            rl_agent.add_role(self.learning_role)

    async def setup_output_agent(self, simulation_id: str, save_frequency_hours: int):
        """
        Set up the output agent for the simulation.

        Args:
        - self: The instance of the class.
        - simulation_id (str): The unique identifier for the simulation.
        - save_frequency_hours (int): The frequency (in hours) at which to save simulation data.

        Returns:
        - None

        This method sets up the output agent for the simulation. It creates an output role responsible
        for writing simulation output, including data storage and export settings. Depending on the
        platform (currently supported only on Linux), it adds the output agent to the container's
        processes, or directly adds the output role to the output agent.
        """

        self.logger.debug(f"creating output agent {self.db=} {self.export_csv_path=}")
        self.output_role = WriteOutput(
            simulation_id=simulation_id,
            start=self.start,
            end=self.end,
            db_engine=self.db,
            export_csv_path=self.export_csv_path,
            save_frequency_hours=save_frequency_hours,
            learning_mode=self.learning_mode,
            evaluation_mode=self.evaluation_mode,
        )

        # mango multiprocessing is currently only supported on linux
        # with single
        if platform == "linux":
            self.addresses.append(self.addr)

            def creator(container):
                agent = RoleAgent(container, suggested_aid=self.output_agent_addr[1])
                agent.add_role(self.output_role)
                clock_agent = DistributedClockAgent(container)

            await self.container.as_agent_process(agent_creator=creator)
        else:
            output_agent = RoleAgent(
                self.container, suggested_aid=self.output_agent_addr[1]
            )
            output_agent.add_role(self.output_role)

    def add_unit_operator(
        self,
        id: str,
    ) -> None:
        """
        Add a unit operator to the simulation.

        Args:
        - self: The instance of the class.
        - id (str): The identifier for the unit operator.

        Returns:
        - None

        This method adds a unit operator to the simulation. It creates a new role agent and
        applies the role of a units operator to it. The unit operator is then added to the
        list of operators currently existing. If in learning mode, additional context parameters
        related to learning and output agents are set for the unit operator's role context.
        """

        if self.unit_operators.get(id):
            raise ValueError(f"Unit operator {id} already exists")

        units_operator = UnitsOperator(available_markets=list(self.markets.values()))
        # creating a new role agent and apply the role of a unitsoperator
        unit_operator_agent = RoleAgent(self.container, suggested_aid=f"{id}")
        unit_operator_agent.add_role(units_operator)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

        # after creation of an agent - we set additional context params
        unit_operator_agent._role_context.data_dict = {}
        if self.learning_mode:
            unit_operator_agent._role_context.data_dict = {
                "learning_output_agent_addr": self.output_agent_addr[0],
                "learning_output_agent_id": self.output_agent_addr[1],
                "learning_agent_addr": self.learning_agent_addr[0],
                "learning_agent_id": self.learning_agent_addr[1],
            }
        else:
            unit_operator_agent._role_context.data_dict = {
                "output_agent_addr": self.output_agent_addr[0],
                "output_agent_id": self.output_agent_addr[1],
                "learning_output_agent_addr": self.output_agent_addr[0],
                "learning_output_agent_id": self.output_agent_addr[1],
            }

    async def async_add_unit(
        self,
        id: str,
        unit_type: str,
        unit_operator_id: str,
        unit_params: dict,
        forecaster: Forecaster,
    ) -> None:
        """
        Asynchronously add a unit to the simulation.

        Args:
        - self: The instance of the class.
        - id (str): The identifier for the unit.
        - unit_type (str): The type of unit to be added.
        - unit_operator_id (str): The identifier of the unit operator to which the unit will be added.
        - unit_params (dict): Parameters for configuring the unit.
        - forecaster (Forecaster): The forecaster used by the unit.

        Returns:
        - None

        This method asynchronously adds a unit to the simulation. It checks if the unit operator
        exists, verifies the unit type, and ensures that the unit operator does not already have a
        unit with the same id. It then creates bidding strategies for the unit and creates the unit
        within the associated unit operator.
        """

        # check if unit operator exists
        if unit_operator_id not in self.unit_operators.keys():
            raise ValueError(f"invalid unit operator {unit_operator_id}")

        # provided unit type does not exist yet
        unit_class: type[BaseUnit] = self.unit_types.get(unit_type)
        if unit_class is None:
            raise ValueError(f"invalid unit type {unit_type}")

        # check if unit operator already has a unit with the same id
        if self.unit_operators[unit_operator_id].units.get(id):
            raise ValueError(f"Unit {id} already exists")

        bidding_strategies = {}
        for product_type, strategy in unit_params["bidding_strategies"].items():
            if not strategy:
                continue

            try:
                bidding_strategies[product_type] = self.bidding_strategies[strategy](
                    unit_id=id,
                    **self.bidding_params,
                )
                # TODO find better way to count learning agents
                if self.learning_mode and issubclass(
                    self.bidding_strategies[strategy], LearningStrategy
                ):
                    self.learning_role.rl_strats[id] = bidding_strategies[product_type]

            except KeyError as e:
                self.logger.error(
                    f"Bidding strategy {strategy} not registered, could not add {id}"
                )
                return
        unit_params["bidding_strategies"] = bidding_strategies

        # create unit within the unit operator its associated with
        unit = unit_class(
            id=id,
            unit_operator=unit_operator_id,
            index=self.index,
            forecaster=forecaster,
            **unit_params,
        )
        await self.unit_operators[unit_operator_id].add_unit(unit)

    def add_market_operator(
        self,
        id: str,
    ) -> None:
        """
        Add a market operator to the simulation.

        Args:
        - self: The instance of the class.
        - id (str): The identifier for the market operator.

        Returns:
        - None

        This method adds a market operator to the simulation. It creates a new role agent for
        the market operator and sets additional context parameters. If not in learning mode and
        not in evaluation mode, it includes output agent address and ID in the role context data
        dictionary.
        """

        if self.market_operators.get(id):
            raise ValueError(f"MarketOperator {id} already exists")
        market_operator_agent = RoleAgent(
            self.container,
            suggested_aid=id,
        )
        market_operator_agent.markets = []

        # after creation of an agent - we set additional context params
        market_operator_agent._role_context.data_dict = {}
        if not self.learning_mode and not self.evaluation_mode:
            market_operator_agent._role_context.data_dict.update(
                {
                    "output_agent_addr": self.output_agent_addr[0],
                    "output_agent_id": self.output_agent_addr[1],
                }
            )
        self.market_operators[id] = market_operator_agent

    def add_market(
        self,
        market_operator_id: str,
        market_config: MarketConfig,
    ) -> None:
        """
        Add a market to the simulation.

        Args:
        - self: The instance of the class.
        - market_operator_id (str): The identifier of the market operator to which the market will be added.
        - market_config (MarketConfig): The configuration for the market to be added.

        Returns:
        - None

        This method adds a market to the simulation. It creates a market role based on the specified
        market mechanism in the market configuration. It then adds this role to the specified market
        operator and appends the market configuration to the list of markets within the market
        operator. Additionally, it stores the market configuration in the simulation's markets dictionary.
        """

        if mm_class := self.clearing_mechanisms.get(market_config.market_mechanism):
            market_role = mm_class(market_config)
        else:
            raise Exception(f"invalid {market_config.market_mechanism=}")

        market_operator = self.market_operators.get(market_operator_id)

        if not market_operator:
            raise Exception(f"invalid {market_operator_id=}")

        market_operator.add_role(market_role)
        market_operator.markets.append(market_config)
        self.markets[f"{market_config.name}"] = market_config

    async def _step(self):
        if self.distributed_role is not False:
            next_activity = await self.clock_manager.distribute_time()
        else:
            next_activity = self.clock.get_next_activity()
        if not next_activity:
            self.logger.info("simulation finished - no schedules left")
            return None
        delta = next_activity - self.clock.time
        self.clock.set_time(next_activity)
        await tasks_complete_or_sleeping(self.container)
        return delta

    async def async_run(self, start_ts, end_ts):
        """
        Run the simulation asynchronously.

        Args:
        - self: The instance of the class.
        - start_ts: The start timestamp for the simulation run.
        - end_ts: The end timestamp for the simulation run.

        Returns:
        - None

        This method runs the simulation asynchronously. It progresses the simulation time from
        the start timestamp to the end timestamp, allowing registration before the first opening.
        If distributed roles are enabled, it broadcasts the simulation time. The method then iterates
        through the simulation time, updating the progress bar and the simulation description. Once
        the simulation time reaches the end timestamp, the method closes the progress bar and shuts
        down the simulation container.
        """

        # agent is implicit added to self.container._agents
        pbar = tqdm(total=end_ts - start_ts)

        # allow registration before first opening
        self.clock.set_time(start_ts - 1)
        if self.distributed_role is not False:
            await self.clock_manager.broadcast(self.clock.time)
        while self.clock.time < end_ts:
            await asyncio.sleep(0)
            delta = await self._step()
            if delta:
                pbar.update(delta)
                pbar.set_description(
                    f"{self.output_role.simulation_id} {datetime.utcfromtimestamp(self.clock.time)}",
                    refresh=False,
                )
            else:
                self.clock.set_time(end_ts)
        pbar.close()
        await self.container.shutdown()

    def run(self):
        """
        Run the simulation.

        Returns:
        - None

        This method converts the start and end timestamps to UTC time and then runs
        the asynchronous simulation using the `async_run` method. It progresses the simulation time from
        the start timestamp to the end timestamp, allowing registration before the first opening.
        If distributed roles are enabled, it broadcasts the simulation time. The method then iterates
        through the simulation time, updating the progress bar and the simulation description. Once
        the simulation time reaches the end timestamp, the method closes the progress bar and shuts
        down the simulation container.

        """

        start_ts = calendar.timegm(self.start.utctimetuple())
        end_ts = calendar.timegm(self.end.utctimetuple())

        try:
            return self.loop.run_until_complete(
                self.async_run(start_ts=start_ts, end_ts=end_ts)
            )
        except KeyboardInterrupt:
            pass

    def reset(self):
        """
        Reset the World instance.

        Returns:
        - None

        This method resets the market operators, markets, unit operators, and forecast
        providers to empty dictionaries.
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

        Args:
        - id (str): The identifier for the unit.
        - unit_type (str): The type of the unit.
        - unit_operator_id (str): The identifier of the unit operator.
        - unit_params (dict): Parameters specific to the unit.
        - forecaster (Forecaster): The forecaster associated with the unit.

        Returns:
        - None

        This method adds a unit to the World instance. It checks if the unit operator
        exists, verifies the unit type, and ensures that the unit operator does not already have a
        unit with the same id. It then creates bidding strategies for the unit and creates the unit
        within the associated unit operator.
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
