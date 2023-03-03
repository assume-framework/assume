# %%
import assume
from assume.common import MarketConfig, MarketProduct
import asyncio
from dateutil import rrule as rr
from datetime import datetime, timedelta
import logging
from mango import Role

logger = logging.getLogger(__name__)

# %%
async def main():
    world = assume.World()
    start = datetime(2019, 1, 1).timestamp()
    end = datetime(2019, 1, 3).timestamp()

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
        maximum_gradient=0.1,  # can only change 10% between hours - should be more generic
        amount_unit="MWh",
        amount_tick=0.1,
        maximum_volume=1e9,
        price_unit="€/MW",
        market_mechanism="pay_as_clear",
    )
    world.add_market_operator(id="market")
    world.add_market(market_operator_id="market",marketconfig=our_marketconfig)
    logger.info(f"marketconfig {our_marketconfig}")


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
        "fuel_price": 25,
        "co2_price": 20,
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
    await world.run_simulation(end)

# %%
if __name__ == "__main__":
    logging.basicConfig(level='INFO')
    asyncio.run(main())

# %%
