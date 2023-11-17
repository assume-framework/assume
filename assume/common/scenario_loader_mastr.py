# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr
from mastr_infrastructure import InfrastructureInterface

from assume import World
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

log = logging.getLogger(__name__)

db_uri = "postgresql://assume:assume@localhost:5432/assume"

world = World(database_uri=db_uri)

config = ["DEA", "DEB", "DEC", "DED", "DEE", "DEF"]
config = ["DEA27"]


async def init():
    year = 2019
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    index = pd.date_range(
        start=start,
        end=end - timedelta(hours=1),
        freq="H",
    )
    sim_id = "world_script_simulation"
    print("loading mastr scenario")

    database = os.getenv("INFRASTRUCTURE_SOURCE", "timescale.nowum.fh-aachen.de:5432")
    login = os.getenv("INFRASTRUCTURE_LOGIN", "readonly:readonly")
    infra_uri = f"postgresql://{login}@{database}"
    infra_interface = InfrastructureInterface("test", infra_uri)

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=sim_id,
        index=index,
    )
    # setup eom market

    marketdesign = [
        MarketConfig(
            "EOM",
            rr.rrule(rr.HOURLY, interval=24, dtstart=start, until=end),
            timedelta(hours=1),
            "pay_as_clear",
            [MarketProduct(timedelta(hours=1), 24, timedelta(hours=1))],
            additional_fields=["block_id", "link", "exclusive_id"],
        )
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    # for each area - add demand and generation
    for area in config:
        print(f"loading config {area}")
        demand = infra_interface.get_demand_series_in_area(area, year)
        demand = demand.resample("H").mean()
        # demand in MW
        sum_demand = demand.sum(axis=1)
        solar, wind = infra_interface.get_renewables_series_in_area(
            area,
            start,
            end,
        )

        world.add_unit_operator(f"demand{area}")
        world.add_unit(
            f"demand{area}1",
            "demand",
            f"demand{area}",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": sum_demand.max(),
                "bidding_strategies": {"energy": "naive"},
                "technology": "demand",
            },
            NaiveForecast(index, demand=sum_demand),
        )

        world.add_unit_operator(f"renewables{area}")
        world.add_unit(
            f"renewables{area}_solar",
            "power_plant",
            f"renewables{area}",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": solar.max(),
                "bidding_strategies": {"energy": "naive"},
                "technology": "solar",
            },
            NaiveForecast(
                index, availability=solar / solar.max(), fuel_price=0.1, co2_price=0
            ),
        )
        world.add_unit(
            f"renewables{area}_wind",
            "power_plant",
            f"renewables{area}",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": wind.max(),
                "bidding_strategies": {"energy": "naive"},
                "technology": "wind",
            },
            NaiveForecast(
                index, availability=wind / wind.max(), fuel_price=0.2, co2_price=0
            ),
        )

        world.add_unit_operator(f"conventional{area}")
        co2_price = 1
        fuel_prices = {
            "hard coal": 8.6,
            "lignite": 1.8,
            "oil": 22,
            "natural gas": 26,
            "biomass": 20,
            "nuclear": 1,
            "co2": 20,
        }

        for fuel_type in ["nuclear", "lignite", "hard coal", "oil", "gas"]:
            plants = infra_interface.get_power_plant_in_area(area, fuel_type)
            plants = list(plants.T.to_dict().values())
            i = 0
            for plant in plants:
                i += 1
                world.add_unit(
                    f"conventional{area}_{fuel_type}_{i}",
                    "power_plant",
                    f"conventional{area}",
                    # the unit_params have no hints
                    {
                        "min_power": plant["minPower"],
                        "max_power": plant["maxPower"],
                        "bidding_strategies": {"energy": "naive"},
                        "emission_factor": plant["chi"],
                        "efficiency": plant["eta"],
                        "technology": "wind",
                        "start_cost": plant["startCost"],
                    },
                    NaiveForecast(
                        index,
                        availability=1,
                        fuel_price=fuel_prices[fuel_type],
                        co2_price=co2_price,
                    ),
                )


world.loop.run_until_complete(init())
world.run()
