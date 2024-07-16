# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from multiprocessing import Process

from assume import World

# import common simulation config from distributed_simulation/config.py
from .config import (
    agent_adress,
    agent_adresses,
    db_uri,
    manager_addr,
    marketdesign,
    tcp_host,
    worker,
)

# import creator functions from world_agent and world_manager
from .world_agent import create_worker as create_agent
from .world_manager import create_worker as create_manager


def manager():
    world = World(database_uri=db_uri, addr=manager_addr, distributed_role=True)
    world.loop.run_until_complete(worker(world, marketdesign, create_manager))


def agent(i: int, n_proc: int, m_agent: int = 1):
    import time

    time.sleep(1)
    host, port = agent_adress
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
    for i in range(n - 1):
        agent_adresses.append(((tcp_host, 9099 + i), "clock_agent"))
    ags = []
    for i in range(n):
        ag = Process(target=agent, args=(i, n))
        ags.append(ag)

    man.start()

    import time

    time.sleep(0.1)
    for ag in ags:
        ag.start()

    # first we are waiting for the manager to finish
    man.join()
    # then we are joining all the finished Agent Containers
    for ag in ags:
        ag.join()
