# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time
from multiprocessing import Process, set_start_method

from mango import addr

from assume import World

# import common simulation config from distributed_simulation/config.py
from .config import (
    agent_address,
    agent_addresses,
    db_uri,
    manager_protocol_addr,
    marketdesign,
    tcp_host,
    use_mqtt,
    worker,
)

# import creator functions from world_agent and world_manager
from .world_agent import create_worker as create_agent
from .world_manager import create_worker as create_manager

set_start_method("spawn", force=True)


def manager():
    world = World(
        database_uri=db_uri, addr=manager_protocol_addr, distributed_role=True
    )

    if world.distributed_role:
        world.addresses.extend(agent_addresses)
    world.loop.run_until_complete(worker(world, marketdesign, create_manager))


def agent(i: int, n_proc: int, m_agent: int = 1):
    time.sleep(1)
    if use_mqtt:
        # first agent is "agent", next is "agent1"...
        if not i:
            addr = agent_address
        else:
            addr = agent_address + str(i)
    else:
        host, port = agent_address
        addr = host, port + i
    print("agent", addr)
    world = World(addr=addr, distributed_role=False)
    world.loop.run_until_complete(
        worker(world, marketdesign, create_agent, i, n_proc, m_agent)
    )


if __name__ == "__main__":
    # we are creating processes so that we do not need to open multiple terminals
    man = Process(target=manager)
    n = 1
    for i in range(1, n):
        if use_mqtt:
            agent_addresses.append(addr(agent_address + str(i), "clock_agent"))
        else:
            agent_addresses.append(addr((tcp_host, 9098 + i), "clock_agent"))
    agents = []
    for i in range(n):
        ag = Process(target=agent, args=(i, n))
        agents.append(ag)

    man.start()
    for ag in agents:
        ag.start()

    # first we are waiting for the manager to finish
    man.join()
    # then we are joining all the finished Agent Containers
    for ag in agents:
        ag.join()
