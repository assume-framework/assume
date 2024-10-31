# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import sys

from mango import addr

from assume import MarketConfig, World
from assume.common.forecasts import NaiveForecast

from .config import (
    agent_addresses,
    db_uri,
    index,
    manager_protocol_addr,
    market_operator_addr,
    marketdesign,
    use_mqtt,
    worker,
)


async def create_worker(
    world: World,
    marketdesign: list[MarketConfig],
    i: int,
    n_proc: int,
    m_agents: int = 1,
):
    world.add_market_operator(id=market_operator_addr.aid)
    for market_config in marketdesign:
        world.add_market(market_operator_addr.aid, market_config)

    world.add_unit_operator(f"my_demand{i}")
    world.add_unit(
        f"demand{i}",
        "demand",
        f"my_demand{i}",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {market_config.market_id: "naive_eom"},
            "technology": "demand",
        },
        NaiveForecast(index, demand=1000),
    )


if __name__ == "__main__":
    world = World(
        database_uri=db_uri, addr=manager_protocol_addr, distributed_role=True
    )

    # command line option can set the agent_addresses of the distributed simulation accordingly
    # one can use this script as follows for TCP:
    # python3 -m examples.distributed_simulation.world_manager 9098 9099
    # and for MQTT:
    # python3 -m examples.distributed_simulation.world_manager agent agent2
    if len(sys.argv) > 1:
        agent_addresses = []
        for address in sys.argv[1:]:
            if not use_mqtt:
                host, port = address.split(":")
                agent_address = (host, int(port))
            else:
                agent_address = address
            agent_addresses.append(addr(agent_address, "clock_agent"))
        print("new agent addresses are", agent_addresses)
    try:
        if world.distributed_role:
            world.addresses.extend(agent_addresses)
        world.loop.run_until_complete(worker(world, marketdesign, create_worker))
    except Exception as e:
        print(e)
