# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
from datetime import datetime, timedelta

from dateutil import rrule as rr

from assume import World
from assume.common.fast_pandas import FastIndex
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.markets.clearing_algorithms.intermediate import PayAsClearIntermediateBidder

log = logging.getLogger(__name__)


def init(world, n=1, months=1):
    start = datetime(2019, 1, 1)
    end = datetime(2019, 1, 1) + timedelta(days=30 * months)

    index = FastIndex(start, end, freq="h")
    sim_id = "twostage_simulation"

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=480,
        simulation_id=sim_id,
    )

    marketdesign = [
        MarketConfig(
            market_id="upper",
            opening_hours=rr.rrule(
                rr.HOURLY,
                interval=2,
                dtstart=start + timedelta(hours=0.5),
                until=end,
                cache=True,
            ),
            opening_duration=timedelta(hours=0.5),
            market_mechanism="pay_as_clear",
            market_products=[
                MarketProduct(timedelta(hours=2), 1, timedelta(hours=1.5))
            ],
        ),
        MarketConfig(
            market_id="lower1",
            opening_hours=rr.rrule(
                rr.HOURLY, interval=2, dtstart=start, until=end, cache=True
            ),
            opening_duration=timedelta(hours=2),
            market_mechanism="intermediate",
            market_products=[MarketProduct(timedelta(hours=2), 1, timedelta(hours=2))],
        ),
        MarketConfig(
            market_id="lower2",
            opening_hours=rr.rrule(
                rr.HOURLY, interval=2, dtstart=start, until=end, cache=True
            ),
            opening_duration=timedelta(hours=2),
            market_mechanism="intermediate",
            market_products=[MarketProduct(timedelta(hours=2), 1, timedelta(hours=2))],
        ),
    ]

    for market_config in marketdesign:
        mo_id = market_config.market_id + "_operator"

        mo = world.add_market_operator(id=mo_id)
        world.add_market(mo_id, market_config)
        if "lower" in market_config.market_id:
            mo.add_role(PayAsClearIntermediateBidder([marketdesign[0]]))
            # gebote nach oben anlegen

    world.add_unit_operator("my_demand1")
    world.add_unit(
        "demand1",
        "demand",
        "my_demand1",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {"lower1": "naive_eom"},
            "technology": "demand",
            "price": 20,
        },
        NaiveForecast(index, demand=1000),
    )

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=3, co2_price=0.1)
    i = 1
    world.add_unit_operator(f"my_operator{i}")
    world.add_unit(
        f"nuclear{i}",
        "power_plant",
        f"my_operator{i}",
        {
            "min_power": 200,
            "max_power": 800,
            "bidding_strategies": {"lower1": "naive_eom"},
            "technology": "nuclear",
        },
        nuclear_forecast,
    )

    nuclear_forecast = NaiveForecast(index, availability=1, fuel_price=4, co2_price=0.1)
    i = 2
    world.add_unit_operator(f"my_operator{i}")
    world.add_unit(
        f"nuclear{i}",
        "power_plant",
        f"my_operator{i}",
        {
            "min_power": 200,
            "max_power": 2000,
            "bidding_strategies": {"lower2": "naive_eom"},
            "technology": "nuclear",
        },
        nuclear_forecast,
    )

    world.add_unit_operator("my_demand2")
    world.add_unit(
        "demand2",
        "demand",
        "my_demand2",
        # the unit_params have no hints
        {
            "min_power": 0,
            "max_power": 1000,
            "bidding_strategies": {"lower2": "naive_eom"},
            "technology": "demand",
            "price": 20,
        },
        NaiveForecast(index, demand=1000),
    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    init(world, months=1)
    world.run()
