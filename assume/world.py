# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from mango import (
    RoleAgent,
    activate,
    addr,
    agent_composed_of,
    create_ec_container,
    create_mqtt_container,
    create_tcp_container,
)
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
from assume.common.utils import datetime2timestamp, timestamp2datetime
from assume.markets import MarketRole, clearing_mechanisms
from assume.strategies import LearningStrategy, bidding_strategies
from assume.units import BaseUnit, demand_side_technologies, unit_types

file_handler = logging.FileHandler(filename="assume.log", mode="w+")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.INFO, handlers=handlers)
logging.getLogger("mango").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class World:
    """
    Represents a simulation environment with a specified address, database connection, CSV export path,
    log level, and optional distributed role settings.

    If a database URI is provided, the World instance attempts to establish a database connection.
    It initializes key attributes for market operators, markets, unit operators, bidding strategies,
    and clearing mechanisms. Additionally, it checks for learning strategy availability and sets up an event loop.

    Attributes:
        addr (tuple[str, int] | str, optional): The address of the world, represented as a tuple (host, port) or a string.
        distributed_role (bool, optional): Defines the world’s role in distributed execution:
            - `True`: Acts as a manager world that schedules events.
            - `False`: Acts as a client world receiving schedules from a manager.
            - `None` (default): Runs independently without subprocesses.
        export_csv_path (str, optional): Path for exporting CSV data.
        log_level (str, optional): The logging level for the world instance.
        db_uri (sqlalchemy.engine.URL, optional): The processed database URI.
        db (sqlalchemy.engine.base.Engine, optional): The database connection engine.
        container (mango.Container, optional): The container for the world instance.
        loop (asyncio.AbstractEventLoop, optional): The event loop for asynchronous operations.
        clock (ExternalClock, optional): External clock instance.
        start (datetime.datetime, optional): Start datetime for the simulation.
        end (datetime.datetime, optional): End datetime for the simulation.
        market_operators (dict[str, mango.RoleAgent], optional): Market operators in the world instance.
        markets (dict[str, MarketConfig], optional): Market configurations.
        unit_operators (dict[str, UnitsOperator], optional): Unit operators.
        unit_types (dict[str, BaseUnit], optional): Available unit types.
        dst_components (dict[str, DemandSideTechnology], optional): Demand-side technologies.
        bidding_strategies (dict[str, type[BaseStrategy]], optional): Bidding strategies for the world instance.
            - If `"pp_learning"` is unavailable, learning strategies may be missing due to missing dependencies (e.g., `torch`).
        clearing_mechanisms (dict[str, MarketRole], optional): Market clearing mechanisms.
        additional_kpis (dict[str, OutputDef], optional): Additional performance indicators.
        scenario_data (dict, optional): Dictionary for scenario-specific data.
        addresses (list[str], optional): Addresses for the world instance.
        output_agent_addr (tuple[str, str], optional): Address of the output agent.
        bidding_params (dict, optional): Parameters for bidding.
        index (pandas.Series, optional): The index for simulation tracking.
        learning_config (LearningConfig, optional): Configuration for learning-based components.
        learning_mode (bool, optional): Whether learning mode is enabled.
        evaluation_mode (bool, optional): Whether evaluation mode is enabled.
        forecaster (Forecaster, optional): The forecaster used for custom unit types.

    Args:
        addr (tuple[str, int] | str, optional): The world’s address as a (host, port) tuple or a string. Defaults to `"world"`.
        database_uri (str, optional): Database URI for establishing a connection. Defaults to `""` (no database).
        export_csv_path (str, optional): Path for exporting CSV data. Defaults to `""`.
        log_level (str, optional): Logging level. Defaults to `"INFO"`.
        distributed_role (bool, optional): Defines the world’s role in distributed execution. Defaults to `None`.
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
        self.container: Container = None
        self.distributed_role = distributed_role

        self.export_csv_path = export_csv_path
        # initialize db connection at beginning of simulation
        self.db_uri = database_uri
        if database_uri:
            if str(database_uri).startswith("sqlite:///"):
                db_path = Path(str(database_uri).replace("sqlite:///", ""))
                db_path.parent.mkdir(exist_ok=True)
            self.db_uri = make_url(database_uri)
            db = create_engine(self.db_uri)
            connected = False
            attempts = 0
            max_attempts = 5

            while not connected and attempts < max_attempts:
                try:
                    with db.connect():
                        connected = True
                        logger.info("Connected to the database")
                except OperationalError as e:
                    attempts += 1
                    logger.error(
                        "Could not connect to %s, trying again (%d/%d)",
                        database_uri,
                        attempts,
                        max_attempts,
                    )
                    # log error if not connection refused
                    if not e.code == "e3q8":
                        logger.error("%s", e)
                    time.sleep(2**attempts)

            if not connected:
                raise RuntimeError(
                    f"Failed to connect to the database after {max_attempts} attempts"
                )

        self.scenario_data = {}
        self.market_operators: dict[str, RoleAgent] = {}
        self.markets: dict[str, MarketConfig] = {}
        self.unit_operators: dict[str, UnitsOperator] = {}
        self.unit_types = unit_types
        self.dst_components = demand_side_technologies

        self.bidding_strategies = bidding_strategies
        if "pp_learning" not in bidding_strategies:
            logger.info(
                "Learning Strategies are not available. Check that you have torch installed."
            )

        self.clearing_mechanisms: dict[str, MarketRole] = clearing_mechanisms
        self.additional_kpis: dict[str, OutputDef] = {}
        self.addresses = []
        # required for jupyter notebooks
        # as they already have a running loop
        import nest_asyncio

        nest_asyncio.apply()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def setup(
        self,
        start: datetime,
        end: datetime,
        simulation_id: str,
        save_frequency_hours: int = 24,
        bidding_params: dict = {},
        learning_config: LearningConfig = {},
        episode: int = 1,
        eval_episode: int = 1,
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
        self.simulation_id = simulation_id
        self.start = start
        self.end = end
        self.learning_config = learning_config
        # initiate learning if the learning mode is on and hence we want to learn new strategies
        self.learning_mode = self.learning_config.get("learning_mode", False)
        self.evaluation_mode = self.learning_config.get("evaluation_mode", False)

        # initialize a config dictionary for the scenario data if not already present
        if not self.scenario_data.get("config"):
            self.scenario_data["config"] = {}

        # make a descriptor for the tqdm progress bar
        # use simulation_id of not in learning mode; use Episode ID if in learning mode
        # and use Evaluation Episode ID if in evaluation mode
        self.simulation_desc = (
            simulation_id
            if not self.learning_mode
            else f"Training Episode {episode}"
            if not self.evaluation_mode
            else f"Evaluation Episode {eval_episode}"
        )

        # forecaster is used only when loading custom unit types
        self.forecaster = forecaster

        self.bidding_params = bidding_params

        # create new container
        container_kwargs = {"mp_method": "fork"} if sys.platform == "linux" else {}
        if self.addr == "world":
            container_func = create_ec_container
            container_kwargs.update({"addr": self.addr})
        elif isinstance(self.addr, tuple):
            container_func = create_tcp_container
            container_kwargs.update({"addr": self.addr})
        else:
            container_func = create_mqtt_container
            container_kwargs = {
                "broker_addr": "localhost",
                "client_id": self.addr,
                "inbox_topic": self.addr,
            }
            container_kwargs.update(**kwargs)

        self.container = container_func(
            codec=mango_codec_factory(),
            clock=self.clock,
            **container_kwargs,
        )

        if not self.db_uri and not self.export_csv_path:
            self.output_agent_addr = None
        else:
            self.output_agent_addr = addr(self.addr, "export_agent_1")

        if self.distributed_role is False:
            # if distributed_role is False - we are a ChildContainer
            # and only connect to the manager_address, which can set/sync our clock
            self.clock_agent = DistributedClockAgent()
            self.output_agent_addr = addr(manager_address, "export_agent_1")

            # # when the clock_agent is stopped, we should gracefully shutdown our container
            # self.clock_agent.stopped.add_done_callback(stop)
            self.container.register(self.clock_agent, suggested_aid="clock_agent")
        else:
            self.setup_learning(episode=episode, eval_episode=eval_episode)

            self.setup_output_agent(
                save_frequency_hours=save_frequency_hours,
                episode=episode,
                eval_episode=eval_episode,
            )
            self.clock_manager = DistributedClockManager(
                receiver_clock_addresses=self.addresses
            )
            self.container.register(self.clock_manager)

    def setup_learning(self, episode: int, eval_episode: int) -> None:
        """
        Set up the learning process for the simulation, updating bidding parameters with the learning configuration
        and initializing the reinforcement learning (RL) learning role with the specified parameters. It also sets up
        the RL agent and adds the learning role to it for further processing.
        """

        self.bidding_params.update(self.learning_config)

        if self.learning_mode or self.evaluation_mode:
            # if so, we initiate the rl learning role with parameters
            from assume.reinforcement_learning.learning_role import Learning

            self.learning_role = Learning(
                self.learning_config, start=self.start, end=self.end
            )

            # separate process does not support buffer and learning
            self.learning_agent_addr = addr(self.addr, "learning_agent")
            rl_agent = agent_composed_of(
                self.learning_role,
                register_in=self.container,
                suggested_aid=self.learning_agent_addr.aid,
            )
            rl_agent.suspendable_tasks = False

            self.learning_role.init_logging(
                simulation_id=self.simulation_id,
                episode=episode,
                eval_episode=eval_episode,
                db_uri=self.db_uri,
                output_agent_addr=self.output_agent_addr,
                train_start=self.start,
                freq=self.forecaster.index.freq,
            )

        else:
            self.learning_role = None
            self.learning_agent_addr = None

    def setup_output_agent(
        self,
        save_frequency_hours: int,
        episode: int,
        eval_episode: int,
    ) -> None:
        """
        Set up the output agent for the simulation, creating an output role responsible for writing simulation output,
        including data storage and export settings. Depending on the platform (currently supported only on Linux),
        it adds the output agent to the container's processes, or directly adds the output role to the output agent.

        Args:
            save_frequency_hours (int): The frequency (in hours) at which to save simulation data.
        """

        logger.debug(
            "creating output agent db=%s export_csv_path=%s",
            self.db_uri,
            self.export_csv_path,
        )
        self.output_role = WriteOutput(
            simulation_id=self.simulation_id,
            start=self.start,
            end=self.end,
            db_uri=self.db_uri,
            export_csv_path=self.export_csv_path,
            save_frequency_hours=save_frequency_hours,
            learning_mode=self.learning_mode,
            evaluation_mode=self.evaluation_mode,
            episode=episode,
            eval_episode=eval_episode,
            additional_kpis=self.additional_kpis,
            outputs_buffer_size_mb=self.scenario_data["config"].get(
                "outputs_buffer_size_mb", 300
            ),
        )
        if not self.output_agent_addr:
            return

        # mango multiprocessing is currently only supported on linux
        # with single
        if sys.platform == "linux" and self.distributed_role is not None:
            self.addresses.append(addr(self.addr, "clock_agent"))
            output_role = self.output_role
            output_aid = self.output_agent_addr.aid

            def creator(container):
                agent = agent_composed_of(
                    output_role,
                    register_in=container,
                    suggested_aid=output_aid,
                )
                agent.suspendable_tasks = False
                container.register(DistributedClockAgent(), "clock_agent")

            self.container.as_agent_process_lazy(agent_creator=creator)
        else:
            output_agent = agent_composed_of(
                self.output_role,
                register_in=self.container,
                suggested_aid=self.output_agent_addr.aid,
            )
            output_agent.suspendable_tasks = False

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
        unit_operator_agent = RoleAgent()
        unit_operator_agent.add_role(units_operator)
        self.container.register(unit_operator_agent, suggested_aid=str(id))
        unit_operator_agent.suspendable_tasks = False

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

        # after creation of an agent - we set additional context params
        if not self.learning_mode:
            unit_operator_agent._role_context.data.update(
                {
                    "output_agent_addr": self.output_agent_addr,
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
        unit_operator_agent = agent_composed_of(
            units_operator,
            register_in=self.container,
            suggested_aid=f"{id}",
        )
        unit_operator_agent.suspendable_tasks = False

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

        unit_operator_agent._role_context.data.update(
            {
                "learning_output_agent_addr": self.output_agent_addr,
            }
        )

        # after creation of an agent - we set additional context params
        if self.learning_mode:
            unit_operator_agent._role_context.data.update(
                {
                    "learning_agent_addr": self.learning_agent_addr,
                    "train_start": self.start,
                    "train_end": self.end,
                    "train_freq": self.learning_config.get("train_freq", "24h"),
                }
            )

        else:
            unit_operator_agent._role_context.data.update(
                {
                    "output_agent_addr": self.output_agent_addr,
                }
            )

    def add_units_with_operator_subprocess(self, id: str, units: list[dict]):
        """
        Adds a units operator with given ID in a separate process
        and creates and adds the given list of unit dictionaries to it
        through a creator function

        Args:
            id (str): the id of the units operator
            units (list[dict]): list of unit dictionaries forwarded to create_unit
        """
        clock_agent_name = f"clock_agent_{id}"
        markets = list(self.markets.values())
        for market in markets:
            # remove generator from rrule as it is not serializable
            if market.opening_hours._cache is not None:
                market.opening_hours._cache = None
                market.opening_hours._cache_complete = False
                market.opening_hours._cache_gen = None
        self.addresses.append(addr(self.addr, clock_agent_name))
        units_operator = UnitsOperator(available_markets=markets)

        for unit in units:
            units_operator.add_unit(self.create_unit(**unit))
        data_update_dict = {
            "output_agent_addr": self.output_agent_addr,
            "learning_output_agent_addr": self.output_agent_addr,
        }

        def creator(container):
            # creating a new role agent and apply the role of a units operator

            unit_operator_agent = agent_composed_of(
                units_operator, register_in=container, suggested_aid=str(id)
            )
            unit_operator_agent.suspendable_tasks = False
            unit_operator_agent._role_context.data.update(data_update_dict)
            container.register(DistributedClockAgent(), suggested_aid=clock_agent_name)

        self.container.as_agent_process_lazy(agent_creator=creator)

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

        # create unit within the unit operator its associated with
        return unit_class(
            id=id,
            unit_operator=unit_operator_id,
            forecaster=forecaster,
            **unit_params,
        )

    def add_learning_strategies_to_learning_role(self):
        """
        Add bidding strategies to the learning role for the specified unit.

        Args:
            unit_id (str): The identifier for the unit.
            bidding_strategies (dict[str, BaseStrategy]): The bidding strategies for the unit.
        """
        for unit in self.unit_operators["Operator-RL"].rl_units:
            for strategy in unit.bidding_strategies.values():
                if isinstance(strategy, LearningStrategy):
                    self.learning_role.rl_strats[unit.id] = strategy
                    break

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
        strategy_instances = {}  # Cache to store created instances

        # Extract bidding parameters outside the loop
        bidding_params = unit_params.get("bidding_params", self.bidding_params)

        for market_id, strategy in unit_params["bidding_strategies"].items():
            if not strategy:
                continue

            if strategy not in self.bidding_strategies:
                # raise a deprecated warning for learning_advanced_orders
                if strategy == "learning_advanced_orders":
                    logger.warning(
                        "The bidding strategy 'learning_advanced_orders' is deprecated. Please use regular 'pp_learning' instead."
                    )
                raise ValueError(
                    f"""Bidding strategy {strategy} not registered. Please check the name of
                    the bidding strategy or register the bidding strategy in the world.bidding_strategies dict."""
                )

            if strategy not in strategy_instances:
                # Create and cache the strategy instance if not already created
                strategy_instances[strategy] = self.bidding_strategies[strategy](
                    unit_id=unit_id,
                    **bidding_params,
                )

            # Use the cached instance for this market
            bidding_strategies[market_id] = strategy_instances[strategy]

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
        market_operator_agent = RoleAgent()
        self.container.register(market_operator_agent, suggested_aid=id)
        market_operator_agent.suspendable_tasks = False
        market_operator_agent.markets = []

        # after creation of an agent - we set additional context params
        if not self.learning_mode and not self.evaluation_mode:
            market_operator_agent._role_context.data.update(
                {"output_agent_addr": self.output_agent_addr}
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

    async def _step(self, container):
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
        await tasks_complete_or_sleeping(container)
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
        logger.debug("activating container")
        # agent is implicit added to self.container._agents
        async with activate(self.container) as c:
            await tasks_complete_or_sleeping(c)
            logger.debug("all agents up - starting simulation")
            pbar = tqdm(total=end_ts - start_ts)

            # allow registration before first opening
            self.clock.set_time(start_ts - 1)
            if self.distributed_role is not False:
                await self.clock_manager.broadcast(self.clock.time)
            prev_delta = 0
            while self.clock.time < end_ts:
                await asyncio.sleep(0)
                delta = await self._step(c)
                if delta or prev_delta:
                    pbar.update(delta)
                    pbar.set_description(
                        f"{self.simulation_desc} {timestamp2datetime(self.clock.time)}",
                        refresh=False,
                    )
                else:
                    self.clock.set_time(end_ts)
                prev_delta = delta
            pbar.close()

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

        # check if unit operator exists
        self._validate_unit_addition(id, unit_type, unit_operator_id)

        unit = self.create_unit(
            id, unit_type, unit_operator_id, unit_params, forecaster
        )

        self.unit_operators[unit_operator_id].add_unit(unit)
