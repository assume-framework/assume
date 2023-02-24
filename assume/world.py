# %%
import asyncio
import logging
from dateutil import rrule as rr
from datetime import datetime, timedelta
from mango import Role, RoleAgent, create_container
from mango.util.clock import ExternalClock

from assume.common.marketconfig import MarketConfig, MarketProduct
from assume.units import PowerPlant, Demand
from assume.common import UnitsOperator
from assume.markets.base_market import MarketRole
from assume.common.orders import available_clearing_strategies
from assume.strategies import NaiveStrategyNoMarkUp

# %%
class World():
    def __init__(self):
        self.clock = ExternalClock(0)
        self.addr = ("localhost", 9099)

        self.market_operator_agents =  {}
        self.markets = {}
        self.unit_operators = {}

        self.logger = logging.getLogger(__name__)

        self.unit_types = {'power_plant': PowerPlant, 'demand': Demand}
        self.bidding_types = {'simple': NaiveStrategyNoMarkUp}
        self.available_clearing_strategies = available_clearing_strategies

    async def setup(self):
        self.container = await create_container(addr=self.addr, clock=self.clock)
        # agent is implicit added to self.container.agents
            
    def add_unit_operator(self, id:str) -> None:
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
            self, id:str, unit_type:str, params:dict, bidding_strategy:str=None
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
        operator_id = params['unit_operator']

        # provided unit type does not exist yet
        if unit_type not in self.unit_types.keys():
            raise Exception(f"invalid unit type {unit_type}")
        unit_class = self.unit_types[unit_type]

        if bidding_strategy not in self.bidding_types.keys():
            raise Exception(f"invalid bidding strategy {bidding_strategy}")
        bidding_strategy = self.bidding_types[bidding_strategy]
        
        # create unit within the unit operator its associated with
        self.unit_operators[operator_id].add_unit(id, unit_class, params, bidding_strategy)

    def add_market(self, market_operator_id:int, marketconfig: MarketConfig):

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
            strategy = self.available_clearing_strategies.get(marketconfig.market_mechanism)
            if not strategy:
                raise Exception(f"invalid strategy {marketconfig.market_mechanism}")
            marketconfig.market_mechanism = strategy
        market_operator = self.market_operator_agents.get(market_operator_id)
        if not market_operator:
            raise Exception(f"no market operator {market_operator_id}")
        market_operator.add_role(MarketRole(marketconfig))
        market_operator.markets.append(marketconfig)

    def add_market_operator(self, id):

        """
        creates the market operator/s
        
        Params
        ------
        id = int
             market operator id is associated with the market its participating
        """
        self.market_operator_agents[id] = RoleAgent(self.container, suggested_aid=f"{id}")
        self.market_operator_agents[id].markets = []

    async def step(self):
        next_activity = self.clock.get_next_activity()
        if not next_activity:
            self.logger.info("simulation finished - no schedules left")
            return None

        self.clock.set_time(next_activity)
    
    async def run_simulation(self, start: float, stop: float):
        await self.setup()
        self.clock.time = start

        while self.clock.time < stop:
            await self.step()
            await asyncio.sleep(0.00001)

        await self.container.shutdown()


async def main():
    world = World()
    await world.setup()
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
        maximum_gradient=0.1,  # can only change 10% between hours - should be more generic
        amount_unit="MWh",
        amount_tick=0.1,
        price_unit="€/MW",
        market_mechanism="pay_as_clear",
    )
    world.add_market_operator(id="market")
    world.add_market(market_operator_id="market",marketconfig=our_marketconfig)


    def mein_market_clearing(market_agent: Role, market_products: list[MarketProduct]):
        # TODO example
        pass
    
    world.available_clearing_strategies['mein_market_clearing'] = mein_market_clearing
    
    # create unit operators
    for operator_id in range(1,4):
         world.add_unit_operator(id=f'operator_{operator_id}')
    
    supply_params={
        "technology": 'coal',
        "node": '',
        "max_power": 500,
        "min_power": 50,
        "efficiency": 0.8,
        "fuel_type": 'hard coal',
        "fuel_price": [25],
        "co2_price": [20],
        "emission_factor": 0.82,
        'unit_operator': 'operator_1'
        }
    world.add_unit(id='unit_01', unit_type='power_plant', params=supply_params, bidding_strategy='simple')

    world.add_unit_operator(id=99)
    demand_params = {
        'price': 999,
        'volume': -1000,
        "technology": 'demand',
        "node": '',
        'unit_operator': 99,
    }
    world.add_unit(id=23, unit_type='demand', params=demand_params, bidding_strategy='simple')
    print(world)
    start = datetime(2019, 1, 1).timestamp
    end = datetime(2019, 12, 31).timestamp
    await world.run_simulation(start, end)

# %%
if __name__ == "__main__":

    asyncio.run(main())
    