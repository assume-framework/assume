import logging
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume import World
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

log = logging.getLogger(__name__)

db_uri = "postgresql://assume:assume@localhost:5432/assume"

manager_addr = ("0.0.0.0", 9099)
agent_adress = [("0.0.0.0", 9098)]
manager_addr = "manager"
agent_adress = "agent"

world = World(database_uri=db_uri, addr=agent_adress, distributed_role=False)


async def worker():
    start = datetime(2023, 10, 4)
    end = datetime(2023, 12, 5)
    index = pd.date_range(
        start=start,
        end=end + timedelta(hours=24),
        freq="H",
    )
    sim_id = "handmade_simulation"

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=sim_id,
        index=index,
        manager_address=manager_addr,
    )

    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, interval=24, dtstart=start, until=end),
            timedelta(hours=1),
            "pay_as_clear",
            [MarketProduct(timedelta(hours=1), 1, timedelta(hours=1))],
            additional_fields=["block_id", "link", "exclusive_id"],
        )
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    world.add_unit_operator("my_operator")

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    world.add_unit(
        "nuclear1",
        "power_plant",
        "my_operator",
        {
            "min_power": 200,
            "max_power": 1000,
            "bidding_strategies": {"energy": "naive"},
            "technology": "nuclear",
        },
        nuclear_forecast,
    )

    await world.clock_agent.stopped
    await world.container.shutdown()


world.loop.run_until_complete(worker())
