# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import calendar
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume import World
from assume.common.market_objects import MarketConfig, MarketProduct

log = logging.getLogger(__name__)


db_uri = os.getenv("DB_URI", "postgresql://assume:assume@localhost:5432/assume")

use_mqtt = False

if use_mqtt:
    manager_addr = "manager"
    agent_adress = "agent"
    agent_adresses = ["agent"]
    market_operator_addr = "manager"
else:
    manager_addr = ("0.0.0.0", 9099)
    agent_adress = ("0.0.0.0", 9098)
    agent_adresses = [("0.0.0.0", 9098)]
    market_operator_addr = ("0.0.0.0", 9099)

market_operator_aid = "market_operator"
broker_addr = os.getenv("MQTT_BROKER", ("0.0.0.0", 1883, 600))

start = datetime(2023, 10, 4)
end = datetime(2023, 12, 5)
index = pd.date_range(
    start=start,
    end=end + timedelta(hours=24),
    freq="H",
)
sim_id = "handmade_simulation"

marketdesign = [
    MarketConfig(
        name="EOM",
        opening_hours=rr.rrule(rr.HOURLY, interval=24, dtstart=start, until=end),
        opening_duration=timedelta(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
        additional_fields=["block_id", "link", "exclusive_id"],
    )
]


async def worker(world: World, marketdesign: list[MarketConfig], create_worker):
    if world.distributed_role:
        world.addresses.extend(agent_adresses)

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=sim_id,
        index=index,
        manager_address=manager_addr,
        broker_addr=broker_addr,
    )

    await create_worker(world, marketdesign)

    await asyncio.sleep(0)

    # wait until done if we are a worker agent
    if world.distributed_role:
        world.logger.info("sleeping 2s")
        await asyncio.sleep(2)
        world.logger.info("starting simulation")
        await world.async_run(
            start_ts=calendar.timegm(world.start.utctimetuple()),
            end_ts=calendar.timegm(world.end.utctimetuple()),
        )
    elif world.distributed_role is False:
        await world.clock_agent.stopped
        await world.container.shutdown()
