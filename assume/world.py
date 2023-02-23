# %%
import asyncio
import pandas as pd
import numpy as np
import logging
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from datetime import datetime, timedelta
from mango import RoleAgent, create_container
from mango.util.clock import ExternalClock

from assume.common.marketconfig import MarketConfig, MarketProduct
from assume.units.unitsoperator import UnitsOperatorRole
from assume.markets.base_market import MarketRole


# %%
class World():
    def __init__(self):
        self.clock = ExternalClock(0)
        self.addr = ("127.0.0.1", 5555)  

        self.markets = {}
        self.unit_operators = {}

        self.logger = logging.getLogger(__name__)

    async def setup(self):
        self.container = await create_container(addr=self.addr, clock=self.clock)
        
        for id, marketconfig in self.markets.items():
            market_agent = RoleAgent(self.container, suggested_aid=id)
            market_agent.add_role(MarketRole(marketconfig))
            

        for id, units in self.unit_operators.items():
            agent = RoleAgent(self.container, suggested_aid=id)
            agent.add_role(UnitsOperatorRole(available_markets=self.markets.values(), units=units))
            # agent is implicit added to self.container.agents

    def add_unit_operator(self, id, units: dict):
        self.unit_operators[id] = units

    def add_market(self, id, marketconfig: MarketConfig):
        self.markets[id] = marketconfig

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

# %%
if __name__ == "__main__":
    world = World()
    our_marketconfig = MarketConfig(
        "simple_dayahead_auction",
        market_products=[MarketProduct(rd(hours=+1), 1, rd(hours=1))],
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
    world.add_market('market', our_marketconfig)

    unit_dict= {
        '1': {
            'type': 'solar',
            'location': (50,10),
            'generation_power_mw': 40,
        },
        '2': {
            'type': 'coal',
            'location': (51,11),
            'generation_power_mw': 1000,
        },
        '3': {
            'type': 'wind',
            'location': (52,12),
            'generation_power_mw': 1200,
        }
    }
    
    world.add_unit_operator('agent1', unit_dict)

    start = datetime(2019, 1, 1).timestamp
    end = datetime(2019, 12, 31).timestamp
    asyncio.run(world.run_simulation(start, end))
    print(world)