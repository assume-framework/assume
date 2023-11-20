# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import logging
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

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
config = ["DEA"]


async def init():
    year = 2019
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1) - timedelta(hours=1)
    index = pd.date_range(
        start=start,
        end=end,
        freq="H",
    )
    sim_id = "mastr_scenario"
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
            maximum_bid_volume=1e9,
            maximum_bid_price=1e9,
        )
    ]

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    co2_price = 1
    fuel_prices = {
        "hard coal": 8.6,
        "lignite": 1.8,
        "oil": 22,
        "gas": 26,
        "biomass": 20,
        "nuclear": 1,
        "co2": 20,
    }

    default_strategy = {"energy": "naive"}
    from assume.strategies.dmas_powerplant import DmasPowerplantStrategy

    world.bidding_strategies["dmas_pwp"] = DmasPowerplantStrategy
    dmas_strategy = {"energy": "dmas_pwp"}
    bidding_strategies = {
        "hard coal": default_strategy,
        "lignite": default_strategy,
        "oil": default_strategy,
        "gas": default_strategy,
        "biomass": default_strategy,
        "nuclear": default_strategy,
        "wind": default_strategy,
        "solar": default_strategy,
        "demand": default_strategy,
    }

    # for each area - add demand and generation
    for area in config:
        print(f"loading config {area} for {year}")
        config_path = Path.home() / ".assume" / f"{area}_{year}"
        if not config_path.is_dir():
            print(f"query database time series")
            demand = infra_interface.get_demand_series_in_area(area, year)
            demand = demand.resample("H").mean()
            # demand in MW
            solar, wind = infra_interface.get_renewables_series_in_area(
                area,
                start,
                end,
            )
            config_path.mkdir(parents=True, exist_ok=True)
            demand.to_csv(config_path / "demand.csv")
            solar.to_csv(config_path / "solar.csv")
            wind.to_csv(config_path / "wind.csv")
        else:
            print(f"use existing local time series")
            demand = pd.read_csv(config_path / "demand.csv", index_col=0).squeeze()
            solar = pd.read_csv(config_path / "solar.csv", index_col=0).squeeze()
            wind = pd.read_csv(config_path / "wind.csv", index_col=0).squeeze()

        sum_demand = demand.sum(axis=1)

        world.add_unit_operator(f"demand{area}")
        world.add_unit(
            f"demand{area}1",
            "demand",
            f"demand{area}",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": sum_demand.max(),
                "bidding_strategies": bidding_strategies["demand"],
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
                "bidding_strategies": bidding_strategies["solar"],
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
                "bidding_strategies": bidding_strategies["wind"],
                "technology": "wind",
            },
            NaiveForecast(
                index, availability=wind / wind.max(), fuel_price=0.2, co2_price=0
            ),
        )

        world.add_unit_operator(f"conventional{area}")

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
                        "min_power": plant["minPower"] / 1e3,  # kW -> MW
                        "max_power": plant["maxPower"] / 1e3,  # kW -> MW
                        "bidding_strategies": bidding_strategies[fuel_type],
                        "emission_factor": plant["chi"],
                        "efficiency": plant["eta"],
                        "technology": fuel_type,
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
