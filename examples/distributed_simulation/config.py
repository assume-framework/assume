# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import asyncio
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr
from mango import activate, addr

from assume import World
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.common.utils import datetime2timestamp

log = logging.getLogger(__name__)


db_uri = os.getenv("DB_URI", "postgresql://assume:assume@localhost:5432/assume")
use_mqtt = os.getenv("MQTT_BROKER", False)

tcp_host = os.getenv("TCP_HOST", "0.0.0.0")
tcp_port = int(os.getenv("TCP_PORT", "8089"))
if use_mqtt:
    manager_protocol_addr = "manager"
    agent_address = "agent"
    agent_addresses = [addr("agent", "clock_agent")]
else:
    manager_protocol_addr = (tcp_host, tcp_port)
    agent_address = (tcp_host, 9098)
    agent_addresses = [addr((tcp_host, 9098), "clock_agent")]

market_operator_addr = addr(manager_protocol_addr, "market_operator")
broker_addr = os.getenv("MQTT_BROKER", ("0.0.0.0", 1883, 600))

start = datetime(2019, 1, 1)
end = datetime(2019, 3, 1)
index = pd.date_range(
    start=start,
    end=end + timedelta(hours=24),
    freq="h",
)
sim_id = "handmade_simulation"

marketdesign = [
    MarketConfig(
        market_id="EOM",
        opening_hours=rr.rrule(rr.HOURLY, interval=24, dtstart=start, until=end),
        opening_duration=timedelta(hours=1),
        market_mechanism="pay_as_clear",
        market_products=[MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
        additional_fields=["block_id", "link", "exclusive_id"],
    )
]


async def worker(
    world: World,
    marketdesign: list[MarketConfig],
    create_worker,
    i=0,
    n_proc=1,
    m_agents=1,
):
    world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=sim_id,
        index=index,
        manager_address=manager_protocol_addr,
        broker_addr=broker_addr,
    )

    await create_worker(world, marketdesign, i, n_proc, m_agents)

    await asyncio.sleep(0)

    # wait until done if we are a worker agent
    if world.distributed_role:
        log.info("sleeping 2s")
        await asyncio.sleep(2)
        log.info("starting simulation from %s to %s", start, end)
        await world.async_run(
            start_ts=datetime2timestamp(world.start),
            end_ts=datetime2timestamp(world.end),
        )
    elif world.distributed_role is False:
        async with activate(world.container):
            log.info("starting worker %s of %s", i + 1, n_proc)
            await world.clock_agent.stopped
