import asyncio
import logging
from datetime import datetime, timedelta

from dateutil import rrule as rr
from mango import Role, RoleAgent, create_container
from mango.util.clock import ExternalClock
from tqdm import tqdm

from .common import UnitsOperator
from .common.mango_serializer import mango_codec_factory
from .common.market_mechanisms import available_clearing_strategies
from .common.marketclasses import MarketConfig, MarketProduct
from .markets.base_market import MarketRole
from .strategies import NaiveStrategyNoMarkUp
from .units import Demand, PowerPlant
from .common.exceptions import InvalidTypeException

logger = logging.getLogger(__name__)


class World:
    def __init__(self):
        self.addr = ("localhost", 9099)

        self.market_operator_agents = {}
        self.markets = {}
        self.unit_operators = {}

        self.logger = logging.getLogger(__name__)

        self.unit_types = {"power_plant": PowerPlant, "demand": Demand}
        self.bidding_types = {"simple": NaiveStrategyNoMarkUp}
        self.available_clearing_strategies = available_clearing_strategies

    async def setup(self, start: float):
        self.clock = ExternalClock(start)
        self.container = await create_container(
            addr=self.addr, clock=self.clock, codec=mango_codec_factory()
        )
        # agent is implicit added to self.container.agents

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
        uo = UnitsOperator(available_markets=self.markets.values())
        # creating a new role agent and apply the role of a unitsoperator
        unit_operator = RoleAgent(self.container, suggested_aid=f"{id}")
        unit_operator.add_role(uo)

        # add the current unitsoperator to the list of operators currently existing
        self.unit_operators[id] = uo

    def add_unit(
        self, id: str, unit_type: str, params: dict, bidding_strategy: str = None
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
        unit_class = self.unit_types.get(unit_type)
        if unit_class is None:
            raise InvalidTypeException(
                f"unit type {unit_type} not in {self.unit_types.keys()}"
            )

        bidding_strategy_func = self.bidding_types.get(bidding_strategy)
        if bidding_strategy_func is None:
            raise InvalidTypeException(
                f"invalid bidding strategy {bidding_strategy} not in {self.bidding_types.keys()}"
            )

        # create unit within the unit operator its associated with
        self.unit_operators[operator_id].add_unit(
            id, unit_class, params, bidding_strategy_func
        )

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
            clear_strategy = self.available_clearing_strategies.get(
                marketconfig.market_mechanism
            )
            if clear_strategy is None:
                raise InvalidTypeException(
                    f"strategy {marketconfig.market_mechanism} not in {self.available_clearing_strategies.keys()}"
                )
            marketconfig.market_mechanism = clear_strategy
        market_operator = self.market_operator_agents.get(market_operator_id)
        if market_operator is None:
            raise InvalidTypeException(
                f"market operator {market_operator_id} not in {self.market_operator_agents.keys()}"
            )
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
            self.logger.info("simulation finished - no schedules left")
            return None

        self.clock.set_time(next_activity)

    async def run_simulation(self, stop: float, sleep=1e-6):
        prev = self.clock.time
        pbar = tqdm(total=stop - prev)
        while self.clock.time < stop:
            pbar.update(self.clock.time - prev)
            pbar.set_description(f"{datetime.fromtimestamp(self.clock.time)}")
            prev = self.clock.time
            await self.step()
            await asyncio.sleep(sleep)

        pbar.close()
        await self.container.shutdown()


async def main():
    world = World()
    start = datetime(2019, 1, 1).timestamp()
    end = datetime(2019, 1, 3).timestamp()
    # end = datetime(2019, 12, 31).timestamp()

    await world.setup(start)
    our_marketconfig = MarketConfig(
        "simple_dayahead_auction",
        market_products=[MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
        opening_hours=rr.rrule(
            rr.HOURLY,
            dtstart=datetime(2005, 6, 1),
            until=datetime(2030, 12, 31),
            cache=True,
        ),
        opening_duration=timedelta(hours=1),
        amount_unit="MWh",
        amount_tick=0.1,
        maximum_volume=1e9,
        price_unit="€/MW",
        market_mechanism="pay_as_clear",
    )
    world.add_market_operator(id="market")
    world.add_market(market_operator_id="market", marketconfig=our_marketconfig)
    logger.info(f"marketconfig {our_marketconfig}")

    def mein_market_clearing(market_agent: Role, market_products: list[MarketProduct]):
        # TODO example
        pass

    world.available_clearing_strategies["mein_market_clearing"] = mein_market_clearing

    # create unit operators
    for operator_id in range(1, 4):
        world.add_unit_operator(id=f"operator_{operator_id}")

    supply_params = {
        "technology": "coal",
        "node": "",
        "max_power": 500,
        "min_power": 50,
        "efficiency": 0.8,
        "fuel_type": "hard coal",
        "fuel_price": 25,
        "co2_price": 20,
        "emission_factor": 0.82,
        "unit_operator": "operator_1",
    }
    world.add_unit(
        id="unit_01",
        unit_type="power_plant",
        params=supply_params,
        bidding_strategy="simple",
    )

    world.add_unit_operator(id=99)
    demand_params = {
        "price": 999,
        "volume": -1000,
        "technology": "demand",
        "node": "",
        "unit_operator": 99,
    }
    world.add_unit(
        id=23, unit_type="demand", params=demand_params, bidding_strategy="simple"
    )
    print(world)
    await world.run_simulation(end)


# %%
if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    asyncio.run(main())
