# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
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
from mango.util.clock import AsyncioClock, ExternalClock
from mango.util.distributed_clock import DistributedClockAgent, DistributedClockManager
from mango.util.termination_detection import tasks_complete_or_sleeping
from sqlalchemy import create_engine, make_url
from sqlalchemy.exc import OperationalError
from tqdm import tqdm

from assume.common import (
    MarketConfig,
    OutputDef,
    UnitsOperator,
    WriteOutput,
    mango_codec_factory,
)
from assume.common.base import LearningConfig
from assume.common.forecast_algorithms import get_forecast_registries
from assume.common.forecaster import UnitForecaster, UnitsOperatorForecaster
from assume.common.utils import datetime2timestamp, timestamp2datetime
from assume.markets import MarketRole, clearing_mechanisms
from assume.strategies import (
    LearningStrategy,
    UnitOperatorStrategy,
    bidding_strategies,
    deprecated_bidding_strategies,
)
from assume.units import BaseUnit, demand_side_technologies, unit_types

file_handler = logging.FileHandler(filename="assume.log", mode="w+")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.INFO, handlers=handlers)
logging.getLogger("mango").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class World:
    """
    Orchestrates ASSUME simulation setup, execution, and output handling.

    ``World`` is the central runtime container for markets, operators, units, and
    simulation clocks. It coordinates the end-to-end lifecycle: initialize runtime
    infrastructure in :meth:`setup`, register market and unit entities, initialize
    forecasts, and execute the simulation loop via :meth:`run`.

    The class supports standalone execution (default) and distributed execution
    (manager/worker roles). Output can be persisted to a SQL database and/or CSV
    exports through an output agent; learning/evaluation roles are configured in
    :meth:`setup` via ``learning_dict``.

    Attributes:
        market_operators (dict[str, RoleAgent]): Registered market operator mango agents in the current world.
        markets (dict[str, MarketConfig]): Configurations of registered markets available to bidding for UnitOperators.
        unit_operators (dict[str, UnitsOperator]): Registered unit operator mango agents, responsible for formulating bids,
            based on the needs of their associated units.
        units (dict[str, BaseUnit]): All registered unit instances by id.
        bidding_strategies (dict[str, type]): Strategy registry used when creating
            unit and portfolio strategies.
        clearing_mechanisms (dict[str, type[MarketRole]]): Market mechanism
            mechanism registry used by :meth:`add_market`.

    Args:
        addr (tuple[str, int] | str, optional): Address used when creating the
            Mango container. Use ``"world"`` for local event-container execution,
            a ``(host, port)`` tuple for TCP-based execution, or a string client id
            for MQTT-based execution. Defaults to ``"world"``.
        database_uri (str, optional): SQLAlchemy database URI used by output and
            learning components. If empty, no database backend is created.
            Defaults to ``""``.
        export_csv_path (str, optional): Directory path for CSV output exports.
            If empty, CSV export is disabled. Defaults to ``""``.
        log_level (str, optional): Logging level applied to the ``assume`` logger.
            Defaults to ``"INFO"``.
        distributed_role (bool | None, optional): Distributed execution role:
            ``True`` for manager (time distribution), ``False`` for worker
            (receives distributed clock), ``None`` for standalone execution.
            Defaults to ``None``.
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
        self.units: dict[str, BaseUnit] = {}
        self.unit_types = unit_types
        self.dst_components = demand_side_technologies

        self.bidding_strategies = bidding_strategies
        if "powerplant_energy_learning" not in bidding_strategies:
            logger.info(
                "Learning Strategies are not available. Check that you have torch installed."
            )
        self.bidding_strategies.update(deprecated_bidding_strategies)

        self.clearing_mechanisms: dict[str, MarketRole] = clearing_mechanisms
        self.additional_kpis: dict[str, OutputDef] = {}
        self.addresses = []
        # required for jupyter notebooks
        # as they already have a running loop
        import nest_asyncio2

        nest_asyncio2.apply()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def setup(
        self,
        start: datetime,
        end: datetime,
        simulation_id: str,
        save_frequency_hours,
        bidding_params: dict = {},
        learning_dict: dict = {},
        episode: int = 1,
        eval_episode: int = 1,
        manager_address=None,
        real_time=False,
        **kwargs: dict,
    ) -> None:
        """
        Set up the environment for the simulation, initializing various parameters and components required for the simulation run.

        Args:
            start (datetime.datetime): The start datetime for the simulation.
            end (datetime.datetime): The end datetime for the simulation.
            simulation_id (str): The unique identifier for the simulation.
            save_frequency_hours (int): The frequency (in hours) at which to save simulation data.
            bidding_params (dict, optional): Parameters for bidding. Defaults to an empty dictionary.
            learning_dict (dict, optional): Configuration for the learning process. Defaults to an empty dictionary.
            episode (int, optional): The episode number for learning. Defaults to 1.
            eval_episode (int, optional): The episode number for evaluation. Defaults to 1.
            manager_address: The address of the manager.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """

        if real_time:
            if manager_address:
                raise Exception("Can't have manager when running realtime")
            if self.distributed_role is not None:
                raise Exception("Can't have distributed role when running realtime")
            self.clock = AsyncioClock()
        else:
            self.clock = ExternalClock(0)
        self.simulation_id = simulation_id
        self.start = start
        self.end = end

        if not learning_dict:
            self.learning_config: LearningConfig = None
        else:
            self.learning_config = LearningConfig(**learning_dict)

        # initiate learning if the learning mode is on and hence we want to learn new strategies
        self.learning_mode = learning_dict.get("learning_mode", False)
        self.evaluation_mode = learning_dict.get("evaluation_mode", False)

        # initialize a config dictionary for the scenario data if not already present
        if not self.scenario_data.get("config"):
            self.scenario_data["config"] = {}

        # make a descriptor for the tqdm progress bar
        # use simulation_id of not in learning mode; use Episode ID if in learning mode
        # and use Evaluation Episode ID if in evaluation mode
        self.simulation_desc = simulation_id

        # update simulation description when learning
        if self.learning_config:
            if self.learning_config.evaluation_mode:
                self.simulation_desc = f"Evaluation Episode {eval_episode}"
            elif self.learning_mode:
                self.simulation_desc = f"Training Episode {episode}"

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
            if self.learning_config:
                self.setup_learning(
                    episode=episode,
                    eval_episode=eval_episode,
                )

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

        from assume.reinforcement_learning.learning_role import Learning

        # create LearningConfig object
        self.learning_role = Learning(
            learning_config=self.learning_config, start=self.start, end=self.end
        )

        if self.learning_config.learning_mode or self.learning_config.evaluation_mode:
            # if so, we initiate the rl learning role with parameters
            rl_agent = agent_composed_of(
                self.learning_role,
                register_in=self.container,
                suggested_aid="learning_agent",
            )
            rl_agent.suspendable_tasks = False

            self.learning_role.init_logging(
                simulation_id=self.simulation_id,
                episode=episode,
                eval_episode=eval_episode,
                db_uri=self.db_uri,
                output_agent_addr=self.output_agent_addr,
                train_start=self.start,
            )

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

    def add_unit_operator(
        self,
        id: str,
        strategies: dict[str, UnitOperatorStrategy] = {},
        forecaster: UnitsOperatorForecaster = None,
    ) -> None:
        """
        Add a unit operator to the simulation, creating a new role agent and applying the role of a unit operator to it.
        The unit operator is then added to the list of existing operators. Unit operator receives the output agent address
        if not in learning mode.

        Args:
            id (str): The identifier for the unit operator.
            strategies (dict[str, UnitOperatorStrategy], optional): Portfolio strategies for the operator.
            forecaster (UnitsOperatorForecaster, optional): Operator-level forecaster. Defaults to None.
        """

        if self.unit_operators.get(id):
            raise ValueError(f"Unit operator {id} already exists")

        # Strategies must reference existing markets.
        for market_id in strategies.keys():
            if market_id not in list(self.markets.keys()):
                msg = (
                    f"Strategies of unit operator {id} references "
                    f"market {market_id} which is not known in world.\n"
                    f"Known markets are: {list(self.markets.keys())}.\n"
                    f"Note: Markets must be added before unit operators."
                )
                warnings.warn(msg)

        bidding_strategies = self._prepare_bidding_strategies(
            {"bidding_strategies": strategies}, id
        )

        units_operator = UnitsOperator(
            available_markets=list(self.markets.values()),
            portfolio_strategies=bidding_strategies,
            forecaster=forecaster,
        )

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

    def add_units_with_operator_subprocess(
        self,
        id: str,
        units: list[dict],
        strategies: dict[str, UnitOperatorStrategy],
        forecaster: UnitsOperatorForecaster = None,
    ):
        """
        Adds a units operator with given ID in a separate process
        and creates and adds the given list of unit dictionaries to it
        through a creator function

        Args:
            id (str): the id of the units operator
            units (list[dict]): list of unit dictionaries forwarded to create_unit
            forecaster (UnitsOperatorForecaster, optional): Operator-level forecaster. Defaults to None.
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
        units_operator = UnitsOperator(
            available_markets=markets,
            portfolio_strategies=strategies,
            forecaster=forecaster,
        )

        for unit in units:
            units_operator.add_unit(self.create_unit(**unit))
        data_update_dict = {
            "output_agent_addr": self.output_agent_addr,
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
        forecaster: UnitForecaster,
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

    def _prepare_bidding_strategies(self, unit_params, unit_id):
        """
        Prepare bidding strategies for the unit based on the specified parameters.

        Args:
            unit_params (dict): Parameters for configuring the unit.
            unit_id (str): The identifier for the unit.

        Returns:
            dict[str, BaseStrategy | UnitOperatorStrategy]: The bidding strategies for the unit.
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
                raise ValueError(
                    f"""Bidding strategy {strategy} not registered. Please check the name of
                        the bidding strategy or register the bidding strategy in the world.bidding_strategies dict."""
                )

            # remove when deprecated bidding strategies are removed
            if strategy in deprecated_bidding_strategies.keys():
                logger.warning(
                    "Bidding strategy %s is deprecated. Use the new naming instead",
                    strategy,
                )

            if strategy not in strategy_instances:
                # check if created cache has learning_strategy
                if issubclass(self.bidding_strategies[strategy], LearningStrategy):
                    # add learning role to the strategy to have access to store training data etc
                    if self.learning_config is None:
                        raise ValueError(
                            f"Learning strategy '{strategy}' requires a configured 'learning_config', but none was set. "
                            "Specify learning_config in config.yaml."
                        )
                    strategy_instances[strategy] = self.bidding_strategies[strategy](
                        unit_id=unit_id,
                        learning_role=self.learning_role,
                        **bidding_params,
                    )
                else:
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

        self._validate_unit_operator(unit_operator_id)

        if unit_type not in self.unit_types:
            raise ValueError(f"Invalid unit type: {unit_type}")

        if self.unit_operators[unit_operator_id].units.get(id):
            raise ValueError(f"Unit {id} already exists in operator {unit_operator_id}")

        if self.units.get(id):
            raise ValueError(f"Unit {id} already exists in world")

    def _validate_unit_operator(self, unit_operator_id: str):
        """
        Validate the existence of a unit operator in the simulation.

        Args:
            unit_operator_id (str): The identifier for the unit operator.
        """

        if unit_operator_id not in self.unit_operators.keys():
            raise ValueError(f"Invalid unit operator: {unit_operator_id}")

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

        if market_config.market_id in self.markets:
            raise ValueError(f"Market {market_config.market_id} already exists")

        if mm_class := self.clearing_mechanisms.get(market_config.market_mechanism):
            market_role = mm_class(market_config)
        else:
            raise Exception(
                f"invalid {market_config.market_mechanism=} - full version installed?"
            )

        market_operator = self.market_operators.get(market_operator_id)

        if not market_operator:
            raise Exception(f"invalid {market_operator_id=}")

        market_start = market_config.opening_hours[0]
        market_end = market_config.opening_hours[-1]

        if market_start < self.start or market_end > self.end:
            msg = (
                f"Market {market_config.market_id} violates world schedule. \n"
                f"Market start: {market_start}, end: {market_end}. \n)"
                f"World start: {self.start}, end: {self.end}.)"
            )
            raise ValueError(msg)

        market_operator.add_role(market_role)
        market_operator.markets.append(market_config)
        self.markets[f"{market_config.market_id}"] = market_config

    def _validate_setup(self):
        """Validate the consistency of the world configuration and fail early."""

        # For each UnitOperator: Strategies must reference existing markets.
        unit_operators = list(self.unit_operators.values())
        for operator in unit_operators:
            for market_id in operator.portfolio_strategies.keys():
                if market_id not in list(self.markets.keys()):
                    msg = (
                        f"Strategies of unit operator {operator} references"
                        f"market {market_id} which is not known in world."
                        f"Known markets are:\n{list(self.markets.keys())}."
                    )
                    raise ValueError(msg)

        # For each market: Should be referenced by a market strategy.
        from collections import defaultdict

        market_participants = defaultdict(int)
        for operator in unit_operators:
            for market_id in operator.portfolio_strategies.keys():
                market_participants[market_id] += 1

        for market_id in self.markets.keys():
            if market_participants[market_id] < 2:
                msg = f"Added market {market_id} has less than two bidding participants ({market_participants[market_id]})."
                raise ValueError(msg)

        # A Re-Dispatch market can only open if an earlier market closed.
        dispatch_markets = [
            config
            for config in self.markets.values()
            if config.market_mechanism != "redispatch"
        ]
        redispatch_markets = [
            config
            for config in self.markets.values()
            if config.market_mechanism == "redispatch"
        ]

        if len(redispatch_markets) > 0:
            if len(dispatch_markets) == 0:
                msg = "Redispatch market but no dispatch market was defined."
                raise ValueError(msg)

            # when will market result be available from dispatch market
            earliest_dispatch_closing = min(
                x.opening_hours[0] + x.opening_duration for x in dispatch_markets
            )
            # opening of redispatch market
            earliest_redispatch_opening = min(
                x.opening_hours[0] for x in redispatch_markets
            )

            if earliest_redispatch_opening < earliest_dispatch_closing:
                msg = (
                    "First redispatch market opens before first dispatch "
                    "market has closed."
                )
                raise ValueError(msg)

        # Existence of demand implies existence of generation and vice versa.
        demand_exists, generation_exists = False, False

        demand_types = [self.unit_types[x] for x in ["demand"]]
        generation_types = [
            self.unit_types[x] for x in ["power_plant", "hydrogen_plant"]
        ]

        for operator in unit_operators:
            for unit in operator.units.values():
                if type(unit) in demand_types:
                    demand_exists = True
                elif type(unit) in generation_types:
                    generation_exists = True

        if demand_exists and not generation_exists:
            msg = (
                f"Demand units but no generation units were created.\n"
                f"Known generation types are: {generation_types}.\n"
                f"This indicates an incomplete simulation setup."
            )
            warnings.warn(msg)
        elif generation_exists and not demand_exists:
            msg = (
                f"Generation units but no demand units were created.\n"
                f"Known demand types are: {demand_types}.\n"
                f"This indicates an incomplete simulation setup."
            )
            warnings.warn(msg)

    async def _step(self, container):
        """
        Executes a simulation step for the container.
        Manages distribution of time using the clock_manager.
        Waits for completion or sleeping of active tasks before returning the schedule.

        Args:
            container (mango.Container): the container which should be awaited

        Returns:
            float: the time delta since the last activity in seconds
        """
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
        self._validate_setup()

        logger.debug("activating container")
        # agent is implicit added to self.container._agents
        async with activate(self.container) as c:
            await tasks_complete_or_sleeping(c)
            logger.debug("all agents up - starting simulation")

            pbar = tqdm(total=end_ts - start_ts)

            if isinstance(self.clock, ExternalClock):
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
            else:
                # real-time mode
                while self.clock.time < end_ts:
                    time = self.clock.time
                    await asyncio.sleep(1)
                    delta = self.clock.time - time
                    pbar.update(delta)
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
        self.units = {}
        self.forecast_providers = {}

    def add_unit(
        self,
        id: str,
        unit_type: str,
        unit_operator_id: str,
        unit_params: dict,
        forecaster: UnitForecaster,
    ) -> None:
        """
        Creates a unit and adds it to the World instance.

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

        self.units[id] = unit

        self.unit_operators[unit_operator_id].add_unit(unit)

    def add_unit_instance(self, operator_id: str, unit: BaseUnit):
        """
        Add an existing unit to the World instance.

        This method checks if the unit operator exists and then assigns the provided unit instance to it.

        Args:
            operator_id (str): The identifier of the unit operator.
            unit (BaseUnit): The unit instance to be added.
        """
        self._validate_unit_operator(operator_id)
        self.unit_operators[operator_id].add_unit(unit)

    def init_forecasts(
        self,
        forecast_df: pd.DataFrame = None,
    ):
        units = self.units.values()  # make same object for cache
        markets = self.markets.values()  # make same object for cache
        registries = get_forecast_registries()
        for unit in self.units.values():
            if unit.forecaster._registries is None:
                unit.forecaster._registries = registries
            unit.forecaster.initialize(
                units,
                markets,
                forecast_df,
                unit,
            )

        # operator-level forecasters provide market-wide price / residual load
        # signals, so they initialize against all units and markets, with no
        # single initializing unit.
        for operator in self.unit_operators.values():
            if operator.forecaster is None:
                continue
            if operator.forecaster._registries is None:
                operator.forecaster._registries = registries
            operator.forecaster.initialize(
                units,
                markets,
                forecast_df,
            )

    def export(
        self,
        scenario_save_path: str | Path = "scenario_exports",
        study_case: str = "base",
    ) -> None:
        """
        Export the current world setup to a CSV-based scenario folder.

        Creates a scenario folder with config.yaml and CSV files compatible with
        the CSV loader (loader_csv.py). The exported scenario can be adjusted, loaded and
        re-run using load_scenario_folder().

        Args:
            scenario_save_path: Path where the scenario folder will be created.
                Defaults to "scenario_exports", which creates a subfolder with simulation_id under scenario_exports/.
            study_case: Name of the study case to use in config.yaml. Defaults to "base"

        Raises:
            ValueError: If world.setup() has not been called or required attributes are missing.
        """
        self._validate_export_preconditions()

        # Create path: {scenario_save_path}/{simulation_id}/
        scenario_path = Path(scenario_save_path) / self.simulation_id
        scenario_path.mkdir(parents=True, exist_ok=True)

        self._export_config(scenario_path, study_case)
        self._export_grid(scenario_path)
        self._export_units(scenario_path)
        self._export_time_series(scenario_path)

        logger.info(f"Scenario exported to {scenario_path.resolve()}")

    def _validate_export_preconditions(self) -> None:
        """Check that world has required data for export."""
        required_attrs = [
            "start",
            "end",
            "simulation_id",
            "markets",
            "units",
            "unit_operators",
        ]
        missing = [
            attr
            for attr in required_attrs
            if not hasattr(self, attr) or getattr(self, attr) is None
        ]

        if missing:
            raise ValueError(
                f"World is not properly set up for export. Missing attributes: {missing}. "
                "Please ensure world.setup() has been called before exporting."
            )

    def _export_config(self, scenario_path: Path, study_case: str) -> None:
        """Export configuration to config.yaml."""
        config = self._build_config_dict(study_case)
        config_path = scenario_path / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(
                {study_case: config},
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    def _build_config_dict(self, study_case: str) -> dict:
        """Build configuration dictionary for export."""
        config = {
            "start_date": self.start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_date": self.end.strftime("%Y-%m-%d %H:%M:%S"),
            "simulation_id": self.simulation_id,
        }

        # Only include time_step if we can infer it from units
        if time_step := self._infer_time_step_or_none():
            config["time_step"] = time_step

        # Only include seed if it was explicitly set in scenario_data
        if "seed" in self.scenario_data.get("config", {}):
            seed_value = self.scenario_data["config"]["seed"]
            config["seed"] = (
                int(seed_value)
                if isinstance(seed_value, str) and seed_value.isdigit()
                else seed_value
            )

        # Only include save_frequency_hours if it was explicitly set in scenario_data
        if "save_frequency_hours" in self.scenario_data.get("config", {}):
            save_freq = self.scenario_data["config"]["save_frequency_hours"]
            config["save_frequency_hours"] = (
                int(save_freq)
                if isinstance(save_freq, str) and save_freq.isdigit()
                else save_freq
            )

        config["markets_config"] = self._serialize_markets_config()

        # Only include bidding_strategy_params if it's not empty
        if self.bidding_params:
            config["bidding_strategy_params"] = self.bidding_params

        # Only include forecast_algorithms if it's not empty
        if forecast_algs := self._get_forecast_algorithms():
            config["forecast_algorithms"] = forecast_algs

        if self.learning_config:
            # Convert learning_config to a clean dictionary without internal attributes
            learning_dict = {}
            for key, value in self.learning_config.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    learning_dict[key] = value
            if learning_dict:
                config["learning_config"] = learning_dict

        return config

    def _infer_time_step(self) -> str:
        """Infer time step from unit forecasters as pandas frequency string, or default to '1h'."""
        if result := self._infer_time_step_or_none():
            return result
        return "1h"

    def _infer_time_step_or_none(self) -> str | None:
        """Infer time step from unit forecasters as pandas frequency string, or return None if not inferable."""
        if self.units:
            first_unit = next(iter(self.units.values()))
            if hasattr(first_unit.forecaster, "index") and first_unit.forecaster.index:
                freq = first_unit.forecaster.index.freq
                if freq:
                    # Convert timedelta to pandas frequency string
                    if isinstance(freq, timedelta):
                        total_seconds = int(freq.total_seconds())
                        if total_seconds == 3600:
                            return "1h"
                        elif total_seconds == 1800:
                            return "30min"
                        elif total_seconds == 60:
                            return "1min"
                        else:
                            return (
                                f"{total_seconds // 3600}h"
                                if total_seconds % 3600 == 0
                                else f"{total_seconds // 60}min"
                            )
                    return str(freq)
        return None

    def _timedelta_to_frequency_str(self, td: timedelta) -> str:  # TODO: move to utils?
        """Convert timedelta to pandas frequency string."""

        DURATION_FACTORS = [
            (86400, "d"),
            (3600, "h"),
            (60, "m"),
            (1, "s"),
        ]
        total_seconds = int(td.total_seconds())

        for factor, short_suffix in DURATION_FACTORS:
            if total_seconds % factor == 0:
                value = total_seconds // factor
                return f"{value}{short_suffix}"

        return str(td)

    def _serialize_markets_config(self) -> dict:
        """Serialize market configurations for export."""
        markets_config = {}
        for market_id, market_config in self.markets.items():
            market_dict = {}

            # Extract start and end from opening_hours if available
            if hasattr(market_config.opening_hours, "_dtstart"):
                market_dict["start_date"] = (
                    market_config.opening_hours._dtstart.strftime("%Y-%m-%d %H:%M:%S")
                )
            if hasattr(market_config.opening_hours, "_until"):
                market_dict["end_date"] = market_config.opening_hours._until.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

            # Only include opening_frequency if we can determine it
            if opening_freq := self._rrule_to_frequency_str(
                market_config.opening_hours
            ):
                market_dict["opening_frequency"] = opening_freq

            # Always include these core fields
            market_dict["opening_duration"] = self._timedelta_to_frequency_str(
                market_config.opening_duration
            )
            market_dict["market_mechanism"] = market_config.market_mechanism

            # Products
            if market_config.market_products:
                market_dict["products"] = [
                    {
                        "duration": self._timedelta_to_frequency_str(p.duration),
                        "count": p.count,
                        "first_delivery": self._timedelta_to_frequency_str(
                            p.first_delivery
                        ),
                    }
                    for p in market_config.market_products
                ]

            market_dict["product_type"] = market_config.product_type
            # Use values directly as they are already proper types (float | None)
            market_dict["maximum_bid_volume"] = market_config.maximum_bid_volume
            market_dict["maximum_bid_price"] = market_config.maximum_bid_price
            market_dict["minimum_bid_price"] = market_config.minimum_bid_price
            market_dict["volume_unit"] = market_config.volume_unit
            market_dict["price_unit"] = market_config.price_unit

            # Only include if not None or empty
            if market_config.additional_fields:
                market_dict["additional_fields"] = market_config.additional_fields

            # Ensure boolean is properly typed for YAML serialization
            supports_unmatched = market_config.supports_get_unmatched
            if isinstance(supports_unmatched, str):
                supports_unmatched = supports_unmatched.lower() == "true"
            market_dict["supports_get_unmatched"] = bool(supports_unmatched)

            # Only include param_dict if not empty
            if market_config.param_dict:
                # Handle grid_data separately - replace with network_path
                param_dict = {}
                has_grid_data = "grid_data" in market_config.param_dict
                for k, v in market_config.param_dict.items():
                    if k == "grid_data":
                        continue  # Skip grid_data as it's handled by _export_grid
                    # Convert to native Python types for clean YAML serialization
                    if isinstance(v, bool):
                        param_dict[k] = v
                    elif isinstance(v, (int, float)):
                        param_dict[k] = v
                    else:
                        param_dict[k] = str(v) if v is not None else v

                # Add network_path if grid_data was present
                if has_grid_data and param_dict:
                    param_dict["network_path"] = "."

                if param_dict:
                    market_dict["param_dict"] = param_dict

            # Find operator for this market
            for op_id, op in self.market_operators.items():
                if market_config in op.markets:
                    market_dict["operator"] = op_id
                    break

            markets_config[market_id] = market_dict

        return markets_config

    def _rrule_to_frequency_str(self, rrule) -> str | None:  # TODO: move to utils?
        """Convert rrule to frequency string compatible with convert_to_rrule_freq, or return None."""

        freq = rrule._freq
        interval = rrule._interval

        if freq == 4:  # HOURLY
            return f"{interval}h"
        elif freq == 3:  # DAILY
            return f"{interval}d"
        elif freq == 2:  # WEEKLY
            return f"{interval}w"
        elif freq == 1:  # MONTHLY
            return f"{interval}m"
        elif freq == 0:  # YEARLY
            return f"{interval}y"
        else:
            return None

    def _get_forecast_algorithms(self) -> dict:
        """Get forecast algorithms from units."""
        forecast_algs = {}
        for unit in self.units.values():
            if hasattr(unit.forecaster, "forecast_algorithms"):
                for key, value in unit.forecaster.forecast_algorithms.items():
                    if key not in forecast_algs:
                        forecast_algs[key] = value
        return forecast_algs

    def _export_units(self, scenario_path: Path) -> None:
        """Export all units to CSV files."""
        self._export_powerplant_units(scenario_path)
        self._export_demand_units(scenario_path)
        self._export_storage_units(scenario_path)
        self._export_exchange_units(scenario_path)
        self._export_dsm_units(scenario_path)

    def _export_powerplant_units(self, scenario_path: Path) -> None:
        """Export power plant units to CSV."""
        powerplants = [
            u for u in self.units.values() if type(u).__name__ == "PowerPlant"
        ]
        if not powerplants:
            return

        data = []
        for unit in powerplants:
            row = self._unit_to_dict(unit)
            data.append(row)

        df = pd.DataFrame(data).set_index("name")
        if not df.empty:
            df.to_csv(scenario_path / "powerplant_units.csv", index=True)

    def _export_demand_units(self, scenario_path: Path) -> None:
        """Export demand units to CSV."""
        demands = [u for u in self.units.values() if type(u).__name__ == "Demand"]
        if not demands:
            return

        data = []
        for unit in demands:
            row = self._unit_to_dict(unit)
            data.append(row)

        df = pd.DataFrame(data).set_index("name")
        if not df.empty:
            df.to_csv(scenario_path / "demand_units.csv", index=True)

    def _export_storage_units(self, scenario_path: Path) -> None:
        """Export storage units to CSV."""
        storages = [u for u in self.units.values() if type(u).__name__ == "Storage"]
        if not storages:
            return

        data = []
        for unit in storages:
            row = self._unit_to_dict(unit)
            data.append(row)

        df = pd.DataFrame(data).set_index("name")
        if not df.empty:
            df.to_csv(scenario_path / "storage_units.csv", index=True)

    def _export_exchange_units(self, scenario_path: Path) -> None:
        """Export exchange units to CSV."""
        exchanges = [u for u in self.units.values() if type(u).__name__ == "Exchange"]
        if not exchanges:
            return

        data = []
        for unit in exchanges:
            row = self._unit_to_dict(unit)
            data.append(row)

        df = pd.DataFrame(data).set_index("name")
        if not df.empty:
            df.to_csv(scenario_path / "exchange_units.csv", index=True)

    def _export_dsm_units(self, scenario_path: Path) -> None:
        """Export DSM units (Building, SteelPlant, etc.) to CSV."""
        building_units = [
            u for u in self.units.values() if type(u).__name__ == "Building"
        ]
        steel_units = [
            u for u in self.units.values() if type(u).__name__ == "SteelPlant"
        ]
        hydrogen_units = [
            u for u in self.units.values() if type(u).__name__ == "HydrogenPlant"
        ]
        steam_units = [
            u for u in self.units.values() if type(u).__name__ == "SteamPlant"
        ]

        if building_units:
            self._export_units_to_csv(
                building_units, scenario_path, "residential_dsm_units", "Building"
            )
        if steel_units:
            self._export_units_to_csv(
                steel_units, scenario_path, "industrial_dsm_units", "SteelPlant"
            )
        if hydrogen_units or steam_units:
            combined = hydrogen_units + steam_units
            # Use HydrogenPlant as the unit_type for ordering (they're similar)
            self._export_units_to_csv(
                combined, scenario_path, "industrial_dsm_units", "HydrogenPlant"
            )

    def _export_units_to_csv(
        self, units: list, scenario_path: Path, filename: str
    ) -> None:
        """Helper to export a list of units to CSV."""
        data = []
        for unit in units:
            row = self._unit_to_dict(unit)
            data.append(row)

        df = pd.DataFrame(data).set_index("name")
        if not df.empty:
            df.to_csv(scenario_path / f"{filename}.csv", index=True)

    def _unit_to_dict(self, unit: BaseUnit) -> dict:
        """Convert a unit to a dictionary for CSV export using dynamic attribute extraction."""
        # Start with as_dict() as base - gets standard attributes
        unit_dict = unit.as_dict()

        # Attributes to skip (internal/non-serializable or handled separately)
        SKIP_ATTRS = {
            "forecaster",
            "index",
            "outputs",
            "avg_op_time",
            "total_op_time",
            "bidding_strategies",
            "unit_type",
            "marginal_cost",
            "price",  # unit_type handled separately
        }

        # Dynamically add all missing attributes from unit and its bases
        for attr in dir(unit):
            if attr.startswith("_") or attr in SKIP_ATTRS:
                continue
            if attr not in unit_dict:
                try:
                    value = getattr(unit, attr)
                    # Skip callable attributes (methods) and None values
                    if callable(value) or value is None:
                        continue
                    # Format location specially as "lat,lng" string
                    if attr == "location" and value:
                        unit_dict["location"] = f"{value[0]},{value[1]}"
                    else:
                        unit_dict[attr] = value
                except (AttributeError, TypeError):
                    # Skip attributes that can't be accessed
                    pass

        # Add bidding strategies (handle separately for consistent formatting)
        for market_id, strategy in unit.bidding_strategies.items():
            unit_dict[f"bidding_{market_id}"] = (
                strategy.__class__.__name__ if strategy else ""
            )

        # Add forecast algorithms
        if hasattr(unit.forecaster, "forecast_algorithms"):
            for key, value in unit.forecaster.forecast_algorithms.items():
                unit_dict[f"forecast_{key}"] = value

        # Rename id to name for CSV format
        unit_dict["name"] = unit_dict.pop("id")

        return unit_dict

    def _export_time_series(self, scenario_path: Path) -> None:
        """Export time series data to CSV files."""
        self._export_demand_df(scenario_path)
        self._export_availability_df(scenario_path)
        self._export_fuel_prices_df(scenario_path)
        self._export_forecasts_df(scenario_path)
        self._export_exchanges_df(scenario_path)

    def _export_demand_df(self, scenario_path: Path) -> None:
        """Export demand time series."""
        demand_units = [u for u in self.units.values() if type(u).__name__ == "Demand"]
        if not demand_units:
            return

        series_dict = {}
        for unit in demand_units:
            if hasattr(unit.forecaster, "demand"):
                # Use the forecaster's index as the datetime index
                series_dict[unit.id] = -unit.forecaster.demand

        if series_dict:
            df = pd.DataFrame(
                series_dict, index=unit.forecaster.index.as_datetimeindex()
            ).rename_axis(
                "datetime"
            )  # TODO: assumption correct, that all index are the same?
            df.to_csv(scenario_path / "demand_df.csv", index=True)

    def _export_availability_df(self, scenario_path: Path) -> None:
        """Export availability time series."""
        series_dict = {}
        for unit in self.units.values():
            if hasattr(unit.forecaster, "availability"):
                series_dict[unit.id] = unit.forecaster.availability

        if series_dict:
            df = pd.DataFrame(
                series_dict, index=unit.forecaster.index.as_datetimeindex()
            ).rename_axis(
                "datetime"
            )  # TODO: assumption correct that all index are the same?
            df.to_csv(scenario_path / "availability_df.csv", index=True)

    def _export_fuel_prices_df(self, scenario_path: Path) -> None:
        """Export fuel prices time series."""
        all_fuel_prices = {}

        for unit in self.units.values():
            if hasattr(unit.forecaster, "fuel_prices") and unit.forecaster.fuel_prices:
                for fuel, series in unit.forecaster.fuel_prices.items():
                    if fuel not in all_fuel_prices:
                        all_fuel_prices[fuel] = series

        if all_fuel_prices:
            df = pd.DataFrame(
                all_fuel_prices, index=unit.forecaster.index.as_datetimeindex()
            ).rename_axis(
                "datetime"
            )  # TODO: assumption correct that all index are the same?
            df.to_csv(scenario_path / "fuel_prices_df.csv", index=True)

    def _export_forecasts_df(self, scenario_path: Path) -> None:
        """Export market forecasts (price, residual_load)."""
        all_forecasts = {}

        for unit in self.units.values():
            forecaster = unit.forecaster
            if hasattr(forecaster, "price"):
                for market_id, series in forecaster.price.items():
                    col_name = f"price_{market_id}"
                    if col_name not in all_forecasts:
                        all_forecasts[col_name] = series
            if hasattr(forecaster, "residual_load"):
                for market_id, series in forecaster.residual_load.items():
                    col_name = f"residual_load_{market_id}"
                    if col_name not in all_forecasts:
                        all_forecasts[col_name] = series

        if all_forecasts:
            df = pd.DataFrame(
                all_forecasts, index=forecaster.index.as_datetimeindex()
            ).rename_axis(
                "datetime"
            )  # TODO: assumption correct that all index are the same?
            df.to_csv(scenario_path / "forecasts_df.csv", index=True)

    def _export_exchanges_df(self, scenario_path: Path) -> None:
        """Export exchange volume time series."""
        exchange_units = [
            u for u in self.units.values() if type(u).__name__ == "Exchange"
        ]
        if not exchange_units:
            return

        series_dict = {}
        for unit in exchange_units:
            forecaster = unit.forecaster
            if (
                hasattr(forecaster, "volume_export")
                and forecaster.volume_export is not None
            ):
                series_dict[f"{unit.id}_export"] = forecaster.volume_export
            if (
                hasattr(forecaster, "volume_import")
                and forecaster.volume_import is not None
            ):
                series_dict[f"{unit.id}_import"] = forecaster.volume_import

        if series_dict:
            df = pd.DataFrame(
                series_dict, index=forecaster.index.as_datetimeindex()
            ).rename_axis("datetime")
            df.to_csv(scenario_path / "exchanges_df.csv", index=True)

    def _export_grid(self, scenario_path: Path) -> None:
        """Export grid data (buses and lines) to CSV files if available in market configurations."""
        # Look for grid_data in market configurations
        grid_data = None
        for market_config in self.markets.values():
            if (
                grid_data := market_config.param_dict.get("grid_data")
            ):  # TODO: what if there would be multiple different grid_data in different markets?
                break

        if not grid_data:
            return

        # Export only buses and lines (generators/loads/storage are already exported via _export_units)
        grid_components = [("buses", "buses.csv"), ("lines", "lines.csv")]

        for component_name, filename in grid_components:
            if (
                component_df := grid_data.get(component_name)
            ) is not None and not component_df.empty:
                component_df.to_csv(scenario_path / filename, index=True)
