import asyncio
from datetime import datetime, timedelta
from dateutil import rrule as rr
from dateutil.relativedelta import relativedelta as rd
from assume.common.marketconfig import MarketConfig, MarketProduct

from tqdm import tqdm

from mango import Role, RoleAgent, create_container
from mango.messages.message import Performatives
from mango.container.core import Container
from mango.util.clock import ExternalClock
from assume.common.bids import (
    MarketRole,
    Orderbook
)
from assume.markets.base_market import MarketRole
from assume.units.unitsoperator import UnitsOperatorRole

import logging

logger = logging.getLogger(__name__)


def aggregate_step_amount(orderbook: Orderbook):
    """
    step function with bought volume
    """
    deltas = []
    for bid in orderbook:
        if bid["only_hours"] is None:
            deltas.append(bid["start_time"], bid["volume"])
            deltas.append(bid["end_time"], -bid["volume"])
        else:
            # only_hours allows to have peak or off-peak bids
            start_hour, end_hour = bid["only_hours"]
            duration_hours = end_hour - start_hour
            if duration_hours <= 0:
                duration_hours += 24

            starts = rr.rrule(
                rr.DAILY,
                dtstart=bid["start_time"],
                byhour=start_hour,
                until=bid["end_time"],
            )
            for date in starts:
                deltas.append(date, bid["volume"])
                deltas.append(date + timedelta(hours=duration_hours), -bid["volume"])

    times = []
    aggregation = []
    for delta in sorted(deltas, key=lambda i: i[0]):
        time, volume = delta
        if len(times) > 0 and times[-1] == time:
            aggregation[-1] += volume
        else:
            times.append(time)
            aggregation.append(volume)

first_after_start = rd(days=2, hour=0)
market_products = [
    MarketProduct(rd(days=+1, hour=0), 7, first_after_start),
    MarketProduct(rd(weeks=+1, weekday=0, hour=0), 4, first_after_start),
    MarketProduct(rd(months=+1, day=1, hour=0), 9, first_after_start),
    MarketProduct(
        rr.rrule(rr.MONTHLY, bymonth=(1, 4, 7, 10), bymonthday=1, byhour=0),
        11,
        first_after_start,
    ),
    MarketProduct(rd(years=+1, yearday=1, hour=0), 10, first_after_start),
]

eex_marketconfig = MarketConfig(
    "eex_market",
    # additional_fields=["link", "offer_id"],
    opening_hours=rr.rrule(
        rr.DAILY,
        byhour=12,
        dtstart=datetime(2023, 1, 1),
        until=datetime(2023, 12, 31),
        cache=True,
    ),
    opening_duration=timedelta(hours=24),
    market_products=market_products,
    maximum_bid=9999,
    minimum_bid=-9999,
    amount_unit="MW",
    price_unit="0.01 €/MWh",
    market_mechanism="pay_as_bid",
)

simple_dayahead_auction_config = MarketConfig(
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

marketdesign = [eex_marketconfig, simple_dayahead_auction_config]


async def main(start):
    clock = ExternalClock(start_time=start.timestamp())
    addr = ("127.0.0.1", 5555)
    c = await create_container(addr=addr, clock=clock)
    # containers must know all containers, to shutdown properly
    containers: list[Container] = []
    containers.append(c)
    market = RoleAgent(c)

    for marketconfig in marketdesign:
        market.add_role(MarketRole(marketconfig))
    
    weatherforecast = weatherforecast()


 
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

    for i in range(4):
        agent = RoleAgent(c)
        agent.add_role(UnitsOperatorRole(marketdesign, price=0.05 * (i % 9), units = unit_dict, forecast = weatherforecast))
        #forecast.forecast(timedelta(days=1), self.location)
        
    #for i in range(4):
    #    agent = RoleAgent(c)
    #    agent.add_role(UnitsOperatorRole(marketdesign, price=5 * (i % 9), volume=-80))

    if isinstance(clock, ExternalClock):
        next_activity = clock.get_next_activity()
        for i in (t := tqdm(range(100))):
            await asyncio.sleep(0.0001)
            # clock.set_time(clock.time + 300)
            next_activity = clock.get_next_activity()
            if not next_activity:
                logger.info("simulation finished - no schedules left")
                break
            t.set_description(f"{datetime.fromtimestamp(next_activity)}")
            clock.set_time(next_activity)

    for c in containers:
        await c.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level="WARN")
    asyncio.run(main(datetime.now()))



for time in steps():

    for agent in agents:
        agent.step()

import SomeNewUnit
from assume.units import powerplant

operator=Operator()
operator.add_unit(SomeNewUnit)
operator.add_unit(SomeNewUnit)
operator.add_unit(SomeNewUnit)
operator.add_unit(SomeNewUnit)

world.add_operator(operator)
