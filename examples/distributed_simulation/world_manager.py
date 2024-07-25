# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from assume import MarketConfig, World
from assume.common.forecasts import NaiveForecast

from .config import (
    db_uri,
    index,
    manager_addr,
    market_operator_aid,
    marketdesign,
    worker,
)


async def create_worker(
    world: World,
    marketdesign: list[MarketConfig],
    i: int,
    n_proc: int,
    m_agents: int = 1,
):
    world.add_market_operator(id=market_operator_aid)
    for market_config in marketdesign:
        world.add_market(market_operator_aid, market_config)

    world.add_unit_operator(f"my_operator{i}")

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
    world = World(database_uri=db_uri, addr=manager_addr, distributed_role=True)
    world.loop.run_until_complete(worker(world, marketdesign, create_worker))
