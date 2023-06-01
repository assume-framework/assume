import asyncio
import logging
import time
from datetime import datetime

import nest_asyncio
import numpy as np
import pandas as pd
import yaml
from mango import RoleAgent, create_container
from mango.util.clock import ExternalClock
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import scoped_session, sessionmaker
from tqdm import tqdm
import os

from assume.common import (
    MarketConfig,
    UnitsOperator,
    WriteOutput,
    ForecastProvider,
    load_file,
    make_market_config,
    mango_codec_factory,
)
from assume.markets import MarketRole, pay_as_bid, pay_as_clear
from assume.strategies import (
    NaiveNegReserveStrategy,
    NaivePosReserveStrategy,
    NaiveStrategy,
    flexableEOM,
    RLStrategy,
)
from assume.units import Demand, PowerPlant

logging.basicConfig(level=logging.INFO)
logging.getLogger("mango").setLevel(logging.WARNING)
logging.getLogger("assume").setLevel(logging.INFO)


class World:
    def __init__(
        self,
        ifac_addr: str = "localhost",
        port: int = 9099,
        database_uri: str = "",
        export_csv_path: str = "",
    ):
        self.logger = logging.getLogger(__name__)
        self.addr = (ifac_addr, port)

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

        self.market_operators: dict[str, RoleAgent] = {}
        self.markets: dict[str, MarketConfig] = {}
        self.unit_operators: dict[str, UnitsOperator] = {}

        self.unit_types = {
            "power_plant": PowerPlant,
            "demand": Demand,
        }
        self.bidding_types = {
            "naive": NaiveStrategy,
            "flexable_eom": flexableEOM,
            "naive_neg_reserve": NaiveNegReserveStrategy,
            "naive_pos_reserve": NaivePosReserveStrategy,
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
        start: pd.Timestamp,
    ):
        self.clock = ExternalClock(0)
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

        fuel_prices_df = load_file(
            path=path,
            config=config,
            file_name="fuel_prices",
            index=self.index,
        )

        demand_df = load_file(
            path=path,
            config=config,
            file_name="demand_df",
            index=self.index,
        )

        vre_cf_df = load_file(
            path=path,
            config=config,
            file_name="vre_cf_df",
            index=self.index,
        )

        bidding_strategies_df = load_file(
            path=path, config=config, file_name="bidding_strategies"
        )

        bidding_strategies_df = bidding_strategies_df.fillna(0)

        # cross_border_flows_df = load_file(
        #     path=path, config=config, file_name="cross_border_flows", index=self.index,
        # )

        if powerplant_units is None or demand_units is None:
            raise ValueError("No power plant and demand units were provided!")

        await self.setup(self.start)

        # read writing properties form config
        simulation_id = study_case
        save_frequency_hours = config.get("save_frequency_hours", None)

        # Add output agent to world
        output_role = WriteOutput(
            simulation_id,
            self.start,
            self.end,
            self.db,
            self.export_csv_path,
            save_frequency_hours,
        )
        self.output_agent = RoleAgent(self.container, suggested_aid="export_agent_1")
        self.output_agent.add_role(output_role)

        # get the market config from the config file and add the markets
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

        # add making price forecast
        # forecast_role = ForecastProvider(
        #    config["markets_config"].items(),
        #    fuel_prices_df,
        #    fuel_prices_df["co2"],
        #    vre_cf_df,
        #    powerplant_units,
        #    demand_df,
        # )
        # forecast_agent = RoleAgent(self.container, suggested_aid="forecast_agent_1")
        # forecast_agent.add_role(forecast_role)

        # write forecast
        self.price_forecast = self.calculate_price_forecast(
            "EOM",
            path,
            demand_df,
            vre_cf_df,
            powerplant_units,
            fuel_prices_df,
            fuel_prices_df["co2"],
        )

        # add the unit operators using unique unit operator names in the powerplants csv
        self.logger.info("Adding unit operators")
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

        for company_name in all_operators:
            self.add_unit_operator(id=str(company_name))

        # add the units to corresponsing unit operators
        # if fuel prices are provided, add them to the unit params
        # if vre generation is provided, add them to the vre units
        # if we have RL strategy, add price forecast to unit_params
        self.logger.info("Adding power plant units")
        for unit_name, unit_params in powerplant_units.iterrows():
            if (
                bidding_strategies_df is not None
                and unit_name in bidding_strategies_df.index
            ):
                unit_params["bidding_strategies"] = bidding_strategies_df.loc[
                    unit_name
                ].to_dict()

                # check if we have RL bidding strategy
                if bidding_strategies_df.loc[unit_name].to_dict() == "RLStrategy":
                    unit_params["price_forecast"] = self.price_forecast
                    print(unit_params)

            else:
                self.logger.warning(
                    f"No bidding strategies specified for {unit_name}. Using default strategies."
                )
                unit_params["bidding_strategies"] = {
                    market.product_type: "naive" for market in self.markets.values()
                }

            if (
                fuel_prices_df is not None
                and unit_params["fuel_type"] in fuel_prices_df.columns
            ):
                unit_params["fuel_price"] = fuel_prices_df[unit_params["fuel_type"]]
                unit_params["co2_price"] = fuel_prices_df["co2"]

            if vre_cf_df is not None and unit_name in vre_cf_df.columns:
                unit_params["capacity_factor"] = vre_cf_df[unit_name]

            await self.add_unit(
                id=unit_name,
                unit_type="power_plant",
                unit_operator_id=unit_params["unit_operator"],
                unit_params=unit_params,
            )

        # add the demand unit operators and units
        self.logger.info("Adding demand")
        for unit_name, unit_params in demand_units.iterrows():
            if (
                bidding_strategies_df is not None
                and unit_name in bidding_strategies_df.index
            ):
                unit_params["bidding_strategies"] = bidding_strategies_df.loc[
                    unit_name
                ].to_dict()
            else:
                self.logger.warning(
                    f"No bidding strategies specified for {unit_name}. Using default strategies for all markets."
                )
                unit_params["bidding_strategies"] = {
                    market.product_type: "naive" for market in self.markets.values()
                }

            if demand_df is not None and unit_name in demand_df.columns:
                unit_params["volume"] = demand_df[unit_name]

            await self.add_unit(
                id=unit_name,
                unit_type="demand",
                unit_operator_id=unit_params["unit_operator"],
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
        unit_operator_agent = RoleAgent(self.container, suggested_aid=f"{id}")
        unit_operator_agent.add_role(units_operator)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = units_operator

        # after creation of an agent - we set additional context params
        unit_operator_agent._role_context.data_dict = {
            "output_agent_id": self.output_agent.aid,
            "output_agent_addr": self.output_agent.addr,
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

        # provided unit type does not exist yet
        unit_class = self.unit_types.get(unit_type)
        if unit_class is None:
            raise ValueError(f"invalid unit type {unit_type}")

        bidding_strategies = {}
        for product_type, strategy in unit_params["bidding_strategies"].items():
            if not strategy:
                continue

            try:
                bidding_strategies[product_type] = self.bidding_types[strategy]()
            except KeyError as e:
                self.logger.error(f"Invalid bidding strategy {strategy}")
                raise e

        unit_params["bidding_strategies"] = bidding_strategies

        # create unit within the unit operator its associated with
        await self.unit_operators[unit_operator_id].add_unit(
            id=id,
            unit_type=unit_type,
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
            "output_agent_id": self.output_agent.aid,
            "output_agent_addr": self.output_agent.addr,
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

    async def run_simulation(self):
        # agent is implicit added to self.container._agents
        import calendar

        start_ts = calendar.timegm(self.start.utctimetuple())
        end_ts = calendar.timegm(self.end.utctimetuple())
        pbar = tqdm(total=end_ts - start_ts)

        # allow registration before first opening
        self.clock.set_time(start_ts - 1)
        await asyncio.sleep(0.00001)
        while self.clock.time < end_ts:
            await asyncio.sleep(0.001)
            delta = await self.step()
            if delta:
                pbar.update(delta)
                pbar.set_description(
                    f"{datetime.utcfromtimestamp(self.clock.time)}", refresh=False
                )
            else:
                self.clock.set_time(end_ts)
        pbar.close()
        # for agent in self.container._agents.values():
        #    await agent.inbox.join()
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

    # calculate price forecast
    def calculate_price_forecast(
        self,
        market,
        path,
        demand,
        renewable_capacity_factors,
        registered_power_plants,
        fuel_prices,
        co2_prices,
    ):
        """
        Function that calculates the merit order price, which is given as a price forecast to the Rl agents
        Here for the entire time horizon at once
        TODO make price forecasts for all markets, not just specified for the DAM like here
        TODO consider storages?
        """

        # initialize price forecast
        price_forecast = pd.DataFrame(
            index=renewable_capacity_factors.index, columns=["mcp"]
        )

        self.demand = list(demand.values)
        forecast_file = path + "/price_forecast_EOM.csv"

        # intialize potential orders
        forecasted_orders = pd.DataFrame(columns=["mcp", "max_power"])

        if market == "EOM":
            # Check if the forecast file already exists
            if os.path.isfile(forecast_file):
                # Ask user if inputs have changed
                user_response = input(
                    "Did you change the input since the last price forecast calculation? (y/n): "
                )

                # Check user response
                if user_response.lower() == "n":
                    # If inputs haven't changed, load the forecast from the CSV file and exit
                    price_forecast = pd.read_csv(forecast_file, index_col="Timestamp")
                    self.logger.info("Price forecast loaded from the existing file.")

                    return price_forecast

            # calculate infeed of renewables and residual demand
            # check if max_power is a series or a float
            self.logger.info("Preparing market forecasts")

            for t in renewable_capacity_factors.index:
                i = 0
                # reset potential orders
                forecasted_orders = forecasted_orders.iloc[0:0]

                for unit_name, unit_params in registered_power_plants.iterrows():
                    # pp = pp[1]
                    if unit_name in renewable_capacity_factors.columns:
                        # adds availabel renewables to merrit order
                        capacity_factor = renewable_capacity_factors[unit_name]

                        max_power = capacity_factor.at[t] * unit_params["max_power"]
                        mcp = 0

                        forecasted_orders.loc[i] = {"mcp": mcp, "max_power": max_power}

                    else:
                        max_power = unit_params.max_power

                        # calculate simplified marginal costs for each power plant
                        mcp = (
                            fuel_prices[unit_params.fuel_type].at[t]
                            / unit_params["efficiency"]
                            + co2_prices.at[t]
                            * unit_params["emission_factor"]
                            / unit_params["efficiency"]
                            + unit_params["fixed_cost"]
                        )

                        forecasted_orders.loc[i] = {"mcp": mcp, "max_power": max_power}
                        i += 1

                # Sort the DataFrame by the "mcp" column
                forecasted_orders = forecasted_orders.sort_values("mcp")

                # Cumulate the "max_power" column
                forecasted_orders["cumulative_max_power"] = forecasted_orders[
                    "max_power"
                ].cumsum()

                # Find the row where "cumulative_max_power" exceeds a demand value
                filtered_forecasted_orders = forecasted_orders[
                    forecasted_orders["cumulative_max_power"]
                    > demand.at[t, "demand_DE"]
                ]
                if not filtered_forecasted_orders.empty:
                    mcp = filtered_forecasted_orders.iloc[0]["mcp"]

                else:
                    # demand cannot be supplied by power plant fleet
                    mcp = 3000

                price_forecast.at[t, "mcp"] = mcp

        else:
            raise NotImplementedError(
                "For this market the price forecast is not implemented yet"
            )

        self.logger.info("Finished market forecasts")
        price_forecast.to_csv(forecast_file, index_label="Timestamp")

        return price_forecast
