# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from multiprocessing import Process

from assume import World

from config import (
    agent_adress,
    db_uri,
    manager_addr,
    index,
    market_operator_addr,
    market_operator_aid,
    marketdesign,
    worker,
)
from world_manager import create_worker as create_manager
from world_agent import create_worker as create_agent


def manager():
    world = World(database_uri=db_uri, addr=manager_addr, distributed_role=True)
    world.loop.run_until_complete(worker(world, marketdesign, create_manager))


def agent(i: int, n: int):
    world = World(addr=agent_adress, distributed_role=False)
    world.loop.run_until_complete(worker(world, marketdesign, create_agent, i, n))

if __name__ == "__main__":
    man = Process(target=manager)
    n = 1
    ags = []
    for i in range(n):
        ag = Process(target=agent, args=(i, n))
        ags.append(ag)
        ag.start()
    manager()

    for ag in ags:
        ag.join()