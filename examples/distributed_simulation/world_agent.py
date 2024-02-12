# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from config import (
    agent_adress,
    db_uri,
    index,
    market_operator_addr,
    market_operator_aid,
    marketdesign,
    worker,
)

from assume import MarketConfig, World
from assume.common.forecasts import NaiveForecast


async def create_worker(world: World, marketdesign: list[MarketConfig]):
    for market_config in marketdesign:
        market_config.addr = market_operator_addr
        market_config.aid = market_operator_aid
        world.markets[f"{market_config.name}"] = market_config

    world.add_unit_operator("my_demand")
    world.add_unit(
        "demand1",
        "demand",
        "my_demand",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {market_config.name: "naive"},
            "technology": "demand",
        },
        NaiveForecast(index, demand=100),
    )


world = World(database_uri=db_uri, addr=agent_adress, distributed_role=False)
world.loop.run_until_complete(worker(world, marketdesign, create_worker))
