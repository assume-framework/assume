# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from config import (
    agent_adress,
    agent_adresses,
    db_uri,
    index,
    manager_addr,
    market_operator_addr,
    market_operator_aid,
    marketdesign,
    worker,
)

from assume import MarketConfig, World
from assume.common.forecasts import NaiveForecast


async def create_worker(world: World, marketdesign: list[MarketConfig]):
    world.add_market_operator(id=market_operator_aid)
    for market_config in marketdesign:
        world.add_market(market_operator_aid, market_config)

    world.add_unit_operator("my_operator")

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    world.add_unit(
        "nuclear1",
        "power_plant",
        "my_operator",
        {
            "min_power": 200,
            "max_power": 1000,
            "bidding_strategies": {market_config.name: "naive"},
            "technology": "nuclear",
        },
        nuclear_forecast,
    )


world = World(database_uri=db_uri, addr=manager_addr, distributed_role=True)
world.loop.run_until_complete(worker(world, marketdesign, create_worker))
