# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from config import (
    agent_adress,
    index,
    market_operator_addr,
    market_operator_aid,
    marketdesign,
    worker,
)

from assume import MarketConfig, World
from assume.common.forecasts import NaiveForecast


async def create_worker(world: World, marketdesign: list[MarketConfig], i: int, n: int):
    for market_config in marketdesign:
        market_config.addr = market_operator_addr
        market_config.aid = market_operator_aid
        world.markets[f"{market_config.market_id}"] = market_config

    world.add_unit_operator(f"my_operator{i}")

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    world.add_unit(
        f"nuclear{i}",
        "power_plant",
        f"my_operator{i}",
        {
            "min_power": 200/n,
            "max_power": 1000/n,
            "bidding_strategies": {market_config.market_id: "naive_eom"},
            "technology": "nuclear",
        },
        nuclear_forecast,
    )


if __name__ == "__main__":
    world = World(addr=agent_adress, distributed_role=False)
    world.loop.run_until_complete(worker(world, marketdesign, create_worker))
