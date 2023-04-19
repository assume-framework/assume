# %%
import asyncio
import logging
import time

from mango import RoleAgent, create_container
from mango.util.clock import ExternalClock
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session, sessionmaker

from pathlib import Path

import pandas as pd

from .common import (
    MarketConfig,
    UnitsOperator,
    available_clearing_strategies,
    mango_codec_factory,
)
from .markets import MarketRole
from .strategies import NaiveStrategyMarkUp, NaiveStrategyNoMarkUp
from .units import Demand, PowerPlant

log = logging.getLogger(__name__)

# %%
class World:
    def __init__(
        self,
        ifac_addr: str = "0.0.0.0",
        port: int = 9099,
        database_uri: str = "",
        export_csv=False,
    ):
        self.log = logging.getLogger(__name__)
        self.addr = (ifac_addr, port)
        self.db = scoped_session(sessionmaker(create_engine(database_uri)))
        connected = False
        while not connected:
            try:
                self.db.connection()
                connected = True
                self.log.info("connected to db")
            except OperationalError:
                self.log.error(f"could not connect to {database_uri}, trying again")
            time.sleep(2)

        self.export_csv = export_csv

        self.market_operator_agents: dict[str, RoleAgent] = {}
        self.markets: dict[str, MarketConfig] = {}
        self.unit_operators: dict[str, UnitsOperator] = {}

        self.unit_types = {"power_plant": PowerPlant, "demand": Demand}
        self.bidding_types = {
            "simple": NaiveStrategyNoMarkUp,
            "markup": NaiveStrategyMarkUp,
        }
        self.available_clearing_strategies = available_clearing_strategies

    async def setup(self, start: float):
        self.clock = ExternalClock(start)
        self.container = await create_container(
            addr=self.addr, clock=self.clock, codec=mango_codec_factory()
        )

    def add_unit_operator(self, id: str) -> None:
        """
        Create and add a new unit operator to the world.

        Params
        ------
        id: str


        Returns
        -------
            None

        """
        uo = UnitsOperator(available_markets=list(self.markets.values()))
        # creating a new role agent and apply the role of a unitsoperator
        unit_operator = RoleAgent(self.container, suggested_aid=f"{id}")
        unit_operator.add_role(uo)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = uo

    def add_unit(
        self, id: str, unit_type: str, params: dict, bidding_strategy: str = ""
    ) -> None:
        """
        Create a unit based on the provided unit type and maps it to the specified unit
        operator.


        Parameters
        ----------
        id : str
            Unit id.
        unit_type : str
            Type of unit.
        params : dict
            Dict of parameters defining the unit.
        bidding_strategy : str, optional
            Bidding strategy of the created unit.

        Returns
        -------
            None
        """
        # extract unit operator id from unit parameters
        operator_id = params["unit_operator"]

        # provided unit type does not exist yet
        if unit_type not in self.unit_types.keys():
            raise Exception(f"invalid unit type {unit_type}")
        unit_class = self.unit_types[unit_type]

        if bidding_strategy not in self.bidding_types.keys():
            raise Exception(f"invalid bidding strategy {bidding_strategy}")
        bidding_strategy = self.bidding_types[bidding_strategy]()

        # create unit within the unit operator its associated with
        self.unit_operators[operator_id].add_unit(
            id, unit_class, params, bidding_strategy

        
        )

        # write unit into db
        log.info(
            f'added unit {unit_type}, fuel: {params["technology"]}'
        )
        
        df = pd.DataFrame([params])
        df['type']= unit_type
        

        if unit_type != 'demand':
        
            if self.export_csv:
                p = Path(self.export_csv)
                p.mkdir(parents=True, exist_ok=True)
                market_data_path = p.joinpath("unit_meta.csv")
                df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
            df.to_sql("unit_meta", self.db.bind, if_exists="append")

        else:
            if self.export_csv:
                p = Path(self.export_csv)
                p.mkdir(parents=True, exist_ok=True)
                market_data_path = p.joinpath("demand_meta.csv")
                df.to_csv(market_data_path, mode="a", header=not market_data_path.exists())
            df.to_sql("demand_meta", self.db.bind, if_exists="append")
        

       

    def add_market(self, market_operator_id: int, marketconfig: MarketConfig):
        """
        including the markets in the market container

        Params
        ------
        id = int
             ID of the operator
        marketconfig =
             describes the configuration of a market
        """
        if isinstance(marketconfig.market_mechanism, str):
            strategy = self.available_clearing_strategies.get(
                marketconfig.market_mechanism
            )
            if not strategy:
                raise Exception(f"invalid strategy {marketconfig.market_mechanism}")
            marketconfig.market_mechanism = strategy
        market_operator = self.market_operator_agents.get(market_operator_id)
        if not market_operator:
            raise Exception(f"no market operator {market_operator_id}")
        market_operator.add_role(MarketRole(marketconfig))
        market_operator.markets.append(marketconfig)
        self.markets[f"{market_operator_id}_{marketconfig.name}"] = marketconfig

    def add_market_operator(self, id):
        """
        creates the market operator/s

        Params
        ------
        id = int
             market operator id is associated with the market its participating
        """
        self.market_operator_agents[id] = RoleAgent(
            self.container, suggested_aid=f"{id}"
        )
        self.market_operator_agents[id].markets = []

    async def step(self):
        next_activity = self.clock.get_next_activity()
        if not next_activity:
            self.log.info("simulation finished - no schedules left")
            return None

        self.clock.set_time(next_activity)

    async def run_simulation(self, stop: float):
        # agent is implicit added to self.container._agents
        for agent in self.container._agents.values():
            # TODO add a Role which does exactly this
            agent._role_context.data.db = self.db
            agent._role_context.data.export_csv = self.export_csv

        while self.clock.time < stop:
            await asyncio.sleep(0.0001)
            await self.step()

        await self.container.shutdown()
