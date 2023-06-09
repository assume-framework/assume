import asyncio
import calendar
import logging
import sys
import time
from datetime import datetime

import nest_asyncio
import pandas as pd
from mango import RoleAgent, create_container
from mango.container.core import Container
from mango.util.clock import ExternalClock
from mango.util.termination_detection import tasks_complete_or_sleeping
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session, sessionmaker
from tqdm import tqdm

from assume.common import (
    ForecastProvider,
    MarketConfig,
    UnitsOperator,
    WriteOutput,
    mango_codec_factory,
)
from assume.markets import MarketRole, pay_as_bid, pay_as_clear
from assume.strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
    OTCStrategy,
    RLStrategy,
    flexableCRMStorage,
    flexableEOM,
    flexableEOMStorage,
    flexableNegCRM,
    flexablePosCRM,
)
from assume.units import Demand, HeatPump, PowerPlant, Storage

file_handler = logging.FileHandler(filename="assume.log", mode="w+")
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.setLevel(logging.WARNING)
handlers = [file_handler, stdout_handler]
logging.basicConfig(level=logging.INFO, handlers=handlers)
logging.getLogger("mango").setLevel(logging.WARNING)


class World:
    def __init__(
        self,
        ifac_addr: str = "localhost",
        port: int = 9099,
        database_uri: str = "",
        export_csv_path: str = "",
        log_level: str = "INFO",
    ):
        logging.getLogger("assume").setLevel(log_level)
        self.logger = logging.getLogger(__name__)
        self.addr = (ifac_addr, port)
        self.container = None

        self.export_csv_path = export_csv_path
        # intialize db connection at beginning of simulation
        if database_uri:
            self.db = scoped_session(sessionmaker(create_engine(database_uri)))
            connected = False
            while not connected:
                try:
                    self.db.connection()
                    connected = True
                    self.logger.info("connected to db")
                except OperationalError:
                    self.logger.error(
                        f"could not connect to {database_uri}, trying again"
                    )
                    time.sleep(2)
        else:
            self.db = None

        self.market_operators: dict[str, RoleAgent] = {}
        self.markets: dict[str, MarketConfig] = {}
        self.unit_operators: dict[str, UnitsOperator] = {}
        self.forecast_providers: dict[str, ForecastProvider] = {}

        self.unit_types = {
            "power_plant": PowerPlant,
            "heatpump": HeatPump,
            "demand": Demand,
            "storage": Storage,
        }
        self.bidding_types = {
            "flexable_eom": flexableEOM,
            "flexable_pos_crm": flexablePosCRM,
            "flexable_neg_crm": flexableNegCRM,
            "flexable_crm_storage": flexableCRMStorage,
            "flexable_eom_storage": flexableEOMStorage,
            "naive": NaiveStrategy,
            "naive_neg_reserve": NaiveNegReserveStrategy,
            "naive_pos_reserve": NaivePosReserveStrategy,
            "otc_strategy": OTCStrategy,
            "rl_strategy": RLStrategy,
        }
        self.clearing_mechanisms = {
            "pay_as_clear": pay_as_clear,
            "pay_as_bid": pay_as_bid,
        }
        nest_asyncio.apply()
        self.loop = asyncio.get_event_loop()
        asyncio.set_event_loop(self.loop)

    async def setup(
        self,
        start: datetime,
        end: datetime,
        save_frequency_hours: int,
        simulation_id: str,
        bidding_params: dict,
        index: pd.Series,
        same_process: bool = True,
    ):
        self.clock = ExternalClock(0)
        self.start = start
        self.end = end
        self.bidding_params = bidding_params
        self.index = index

        # kill old container if exists
        if isinstance(self.container, Container) and self.container.running:
            self.container.shutdown()

        # create new container
        self.container = await create_container(
            addr=self.addr, clock=self.clock, codec=mango_codec_factory()
        )

        self.output_agent_addr = (self.addr, "export_agent_1")
        # Add output agent to world
        output_role = WriteOutput(
            simulation_id=simulation_id,
            start=start,
            end=end,
            db_engine=self.db,
            export_csv_path=self.export_csv_path,
            save_frequency_hours=save_frequency_hours,
        )
        same_process = True
        if same_process:
            self.output_agent = RoleAgent(
                self.container, suggested_aid=self.output_agent_addr[1]
            )
            self.output_agent.add_role(output_role)
        else:
            # this does not set the clock in output_agent correctly yet
            # see https://gitlab.com/mango-agents/mango/-/issues/59
            # but still improves performance
            def creator(container):
                agent = RoleAgent(container, suggested_aid=self.output_agent_addr[1])
                agent.add_role(output_role)

            await self.container.as_agent_process(agent_creator=creator)

    def add_unit_operator(
        self,
        id: str,
    ) -> None:
        """
        Create and add a new unit operator to the world.

        Params
        ------
        id: str or int

        """
        units_operator = UnitsOperator(available_markets=list(self.markets.values()))
        # creating a new role agent and apply the role of a unitsoperator
        unit_operator_agent = RoleAgent(self.container, suggested_aid=f"{id}")
        unit_operator_agent.add_role(units_operator)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

        # after creation of an agent - we set additional context params
        unit_operator_agent._role_context.data_dict = {
            "output_agent_addr": self.output_agent_addr[0],
            "output_agent_id": self.output_agent_addr[1],
        }

    async def add_unit(
        self,
        id: str,
        unit_type: str,
        unit_operator_id: str,
        unit_params: dict,
    ) -> None:
        """
        Create and add a new unit to the world.

        Params
        ------
        id: str
        unit_type: str
        unit_operator_id: str
        unit_params: dict

        """

        # check if unit operator exists
        if unit_operator_id not in self.unit_operators:
            raise ValueError(f"invalid unit operator {unit_operator_id}")

        # provided unit type does not exist yet
        unit_class = self.unit_types.get(unit_type)
        if unit_class is None:
            raise ValueError(f"invalid unit type {unit_type}")

        bidding_strategies = {}
        for product_type, strategy in unit_params["bidding_strategies"].items():
            if not strategy:
                continue

            try:
                bidding_strategies[product_type] = self.bidding_types[strategy](
                    **self.bidding_params
                )
            except KeyError as e:
                self.logger.error(f"Invalid bidding strategy {strategy}")
                raise e

        unit_params["bidding_strategies"] = bidding_strategies

        # create unit within the unit operator its associated with
        await self.unit_operators[unit_operator_id].add_unit(
            id=id,
            unit_class=unit_class,
            unit_params=unit_params,
            index=self.index,
        )

    def add_market_operator(
        self,
        id: str,
    ):
        """
        creates the market operator

        Params
        ------
        id = int
             market operator id is associated with the market its participating
        """
        market_operator_agent = RoleAgent(
            self.container,
            suggested_aid=id,
        )
        market_operator_agent.markets = []

        # after creation of an agent - we set additional context params
        market_operator_agent._role_context.data_dict = {
            "output_agent_addr": self.output_agent_addr[0],
            "output_agent_id": self.output_agent_addr[1],
        }
        self.market_operators[id] = market_operator_agent

    def add_market(
        self,
        market_operator_id: str,
        market_config: MarketConfig,
    ):
        """
        including the markets in the market container

        Params
        ------
        id = int
             ID of the operator
        marketconfig =
             describes the configuration of a market
        """
        if isinstance(market_config.market_mechanism, str):
            if strategy := self.clearing_mechanisms.get(market_config.market_mechanism):
                market_config.market_mechanism = strategy

            else:
                raise Exception(f"invalid strategy {market_config.market_mechanism}")

        market_operator = self.market_operators.get(market_operator_id)

        if not market_operator:
            raise Exception(f"no market operator {market_operator_id}")

        market_operator.add_role(MarketRole(market_config))
        market_operator.markets.append(market_config)
        self.markets[f"{market_config.name}"] = market_config

    async def step(self):
        next_activity = self.clock.get_next_activity()
        if not next_activity:
            self.logger.info("simulation finished - no schedules left")
            return None
        delta = next_activity - self.clock.time
        self.clock.set_time(next_activity)
        return delta

    async def run_async(self):
        """
        Run the simulation.
        """

        # agent is implicit added to self.container._agents

        start_ts = calendar.timegm(self.start.utctimetuple())
        end_ts = calendar.timegm(self.end.utctimetuple())
        pbar = tqdm(total=end_ts - start_ts)

        # allow registration before first opening
        self.clock.set_time(start_ts - 1)
        while self.clock.time < end_ts:
            await asyncio.sleep(0)
            delta = await self.step()
            if delta:
                pbar.update(delta)
                pbar.set_description(
                    f"{datetime.utcfromtimestamp(self.clock.time)}", refresh=False
                )
            else:
                self.clock.set_time(end_ts)

            await tasks_complete_or_sleeping(self.container)
        pbar.close()
        await self.container.shutdown()

        logging.info("Simulation finished")

    def run(self):
        return self.loop.run_until_complete(self.run_async())
