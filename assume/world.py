import asyncio
import logging
import time
from datetime import datetime

import nest_asyncio
import pandas as pd
import yaml
from mango import RoleAgent, create_container
from mango.util.clock import ExternalClock
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session, sessionmaker
from tqdm import tqdm

from assume.common import (
    MarketConfig,
    UnitsOperator,
    load_file,
    make_market_config,
    mango_codec_factory,
)
from assume.markets import MarketRole, pay_as_bid, pay_as_clear
from assume.strategies import NaiveStrategy, flexableEOM
from assume.units import Demand, PowerPlant

logging.basicConfig(level=logging.INFO)
logging.getLogger("mango").setLevel(logging.WARNING)
logging.getLogger("assume").setLevel(logging.INFO)


class World:
    def __init__(
        self,
        ifac_addr: str = "0.0.0.0",
        port: int = 9099,
        database_uri: str = "",
        export_csv=False,
    ):
        self.logger = logging.getLogger(__name__)
        self.addr = (ifac_addr, port)
        self.db = None
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

        self.export_csv = export_csv

        self.market_operators: dict[str, RoleAgent] = {}
        self.markets: dict[str, MarketConfig] = {}
        self.unit_operators: dict[str, UnitsOperator] = {}

        self.unit_types = {
            "power_plant": PowerPlant,
            "demand": Demand,
        }
        self.bidding_types = {
            "simple": NaiveStrategy,
            "flexable_eom": flexableEOM,
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
        start: pd.Timestamp,
    ):
        self.clock = ExternalClock(start.timestamp())
        self.container = await create_container(
            addr=self.addr, clock=self.clock, codec=mango_codec_factory()
        )

    async def async_load_scenario(
        self,
        inputs_path: str,
        scenario: str,
        study_case: str,
    ) -> None:
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
            config = config[study_case]

        self.start = pd.Timestamp(config["start_date"])
        self.end = pd.Timestamp(config["end_date"])

        self.index = pd.date_range(
            start=self.start,
            end=self.end + pd.Timedelta(hours=4),
            freq=config["time_step"],
        )

        # load the data from the csv files
        # tries to load all files, returns a warning if file does not exist
        # also attempts to resample the inputs if their resolution is higher than user specified time step
        self.logger.info("Loading input data")
        powerplants_df = load_file(path=path, config=config, file_name="powerplants")

        # storage_units_df = load_file(
        #     path=path, config=config, file_name="storage_units"
        # )

        fuel_prices_df = load_file(
            path=path, config=config, file_name="fuel_prices", index=self.index
        )

        demand_df = load_file(
            path=path, config=config, file_name="demand", index=self.index
        )

        vre_df = load_file(
            path=path, config=config, file_name="renewable_generation", index=self.index
        )

        bidding_strategies_df = load_file(
            path=path, config=config, file_name="bidding_strategies"
        )

        # cross_border_flows_df = load_file(
        #     path=path, config=config, file_name="cross_border_flows", index=self.index
        # )

        await self.setup(self.start)

        # get the market configt from the config file and add the markets
        self.logger.info("Adding markets")
        for id, market_params in config["markets_config"].items():
            market_config = make_market_config(
                id=id,
                market_params=market_params,
                world_start=self.start,
                world_end=self.end,
            )
            
            operator_id = str(market_params["operator"])
            if operator_id not in self.market_operators:
                self.add_market_operator(id=operator_id)

            self.add_market(
                market_operator_id=operator_id,
                market_config=market_config,
            )

        # add the unit operators using unique unit operator names in the powerplants csv
        self.logger.info("Adding unit operators")
        for company_name in powerplants_df.unit_operator.unique():
            self.add_unit_operator(id=str(company_name))

        # add the units to corresponsing unit operators
        # if fuel prices are provided, add them to the unit params
        # if vre generation is provided, add them to the vre units
        self.logger.info("Adding power plant units")
        for pp_name, unit_params in powerplants_df.iterrows():
            if (
                bidding_strategies_df is not None
                and pp_name in bidding_strategies_df.index
            ):
                unit_params["bidding_strategies"] = bidding_strategies_df.loc[
                    pp_name
                ].to_dict()
            else:
                self.logger.warning(
                    f"No bidding strategies specified for {pp_name}. Using default strategies."
                )
                unit_params["bidding_strategies"] = {
                    market.product_type: "simple" for market in self.markets.values()
                }

            if (
                fuel_prices_df is not None
                and unit_params["fuel_type"] in fuel_prices_df.columns
            ):
                unit_params["fuel_price"] = fuel_prices_df[unit_params["fuel_type"]]
                unit_params["co2_price"] = fuel_prices_df["co2"]

            if vre_df is not None and pp_name in vre_df.columns:
                unit_params["max_power"] = vre_df[pp_name]

            self.add_unit(
                id=pp_name,
                unit_type="power_plant",
                unit_operator_id=unit_params["unit_operator"],
                unit_params=unit_params,
            )

        # add the demand unit operators and units
        self.logger.info("Adding demand")
        for demand_name, demand in demand_df.items():
            demand_name = str(demand_name)
            self.add_unit_operator(id=demand_name)
            unit_params = {
                "technology": "inflex_demand",
                "volume": demand,
                "price": 3000.0,
            }

            if (
                bidding_strategies_df is not None
                and demand_name in bidding_strategies_df.index
            ):
                unit_params["bidding_strategies"] = bidding_strategies_df.loc[
                    demand_name
                ].to_dict()
            else:
                self.logger.warning(
                    f"No bidding strategies specified for {demand_name}. Using default strategies."
                )
                unit_params["bidding_strategies"] = {"energy": "simple"}

            self.add_unit(
                id=demand_name,
                unit_type="demand",
                unit_operator_id=demand_name,
                unit_params=unit_params,
            )

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
        unit_operator_role = RoleAgent(self.container, suggested_aid=f"{id}")
        unit_operator_role.add_role(units_operator)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

    def add_unit(
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

        # provided unit type does not exist yet
        unit_class = self.unit_types.get(unit_type)
        if unit_class is None:
            raise ValueError(f"invalid unit type {unit_type}")

        for product_type, strategy in unit_params["bidding_strategies"].items():
            try:
                unit_params["bidding_strategies"][product_type] = self.bidding_types[
                    strategy
                ]()
            except KeyError as e:
                self.logger.error(f"Invalid bidding strategy {strategy}")
                raise e

        # create unit within the unit operator its associated with
        self.unit_operators[unit_operator_id].add_unit(
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
        creates the market operator/s

        Params
        ------
        id = int
             market operator id is associated with the market its participating
        """
        self.market_operators[id] = RoleAgent(
            self.container, suggested_aid=id,
        )
        self.market_operators[id].markets = []

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
            self.clock.set_time(self.end.timestamp())
            return None
        delta = next_activity - self.clock.time
        self.clock.set_time(next_activity)
        return delta

    async def run_simulation(self):
        # agent is implicit added to self.container._agents
        for agent in self.container._agents.values():
            # TODO add a Role which does exactly this
            agent._role_context.data_dict = {
                "db": self.db,
                "export_csv": self.export_csv,
            }
        total = self.end.timestamp() - self.start.timestamp()
        pbar = tqdm(total=total)
        while self.clock.time < self.end.timestamp():
            await asyncio.sleep(0)
            delta = await self.step()
            if delta:
                pbar.update(delta)
                pbar.set_description(
                    f"{datetime.fromtimestamp(self.clock.time)}", refresh=False
                )
        pbar.close()
        await self.container.shutdown()

    def load_scenario(
        self,
        inputs_path: str,
        scenario: str,
        study_case: str,
    ):
        return self.loop.run_until_complete(
            self.async_load_scenario(
                inputs_path,
                scenario,
                study_case,
            )
        )

    def run(self):
        return self.loop.run_until_complete(self.run_simulation())
