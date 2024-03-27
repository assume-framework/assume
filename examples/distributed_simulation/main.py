# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from multiprocessing import Process

from .config import (
    agent_adress,
    agent_adresses,
    db_uri,
    index,
    manager_addr,
    market_operator_addr,
    market_operator_aid,
    marketdesign,
    tcp_host,
    tcp_port,
    worker,
)
from .world_agent import create_worker as create_agent
from .world_manager import create_worker as create_manager

from assume import World


def manager():
    world = World(database_uri=db_uri, addr=manager_addr, distributed_role=True)
    world.loop.run_until_complete(worker(world, marketdesign, create_manager))


def agent(i: int, n: int):
    host, port = agent_adress
    addr = host, port + i
    print("agent", addr)
    world = World(addr=addr, distributed_role=False)
    world.loop.run_until_complete(worker(world, marketdesign, create_agent, i, n))


if __name__ == "__main__":
    man = Process(target=manager)
    n = 1
    for i in range(n - 1):
        agent_adresses.append((tcp_host, 9099 + i))
    ags = []
    for i in range(n):
        ag = Process(target=agent, args=(i, n))
        ags.append(ag)

    man.start()

    import time

    time.sleep(0.1)
    for ag in ags:
        ag.start()

    man.join()
    for ag in ags:
        ag.join()
