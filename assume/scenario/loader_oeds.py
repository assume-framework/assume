# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dateutil import rrule as rr

from assume import World
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.scenario.oeds.infrastructure import InfrastructureInterface
from assume.strategies.dmas_powerplant import DmasPowerplantStrategy


async def load_oeds_async(
    world: World,
    scenario: str,
    study_case: str,
    infra_uri: str,
    marketdesign: list[MarketConfig],
    nuts_config: list[str] = [],
):
    """
    This initializes a scenario using the open-energy-data-server
    https://github.com/NOWUM/open-energy-data-server/

    Scenarios can use any NUTS area in Germany and use any year with appropriate weather available

    Args:
        world (World): the world to add this scenario to
        scenario (str): scenario name
        study_case (str): study case name
        infra_uri (str): database uri to connect to the OEDS
        marketdesign (list[MarketConfig]): description of the market design which will be used with the scenario
        nuts_config (list[str], optional): list of NUTS areas from which the simulation data is taken. Defaults to [].
    """
    year = 2019
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1) - timedelta(hours=1)
    index = pd.date_range(
        start=start,
        end=end,
        freq="h",
    )
    sim_id = f"{scenario}_{study_case}"
    print(f"loading scenario {sim_id}")
    infra_interface = InfrastructureInterface("test", infra_uri)

    if not nuts_config:
        nuts_config = list(infra_interface.plz_nuts["nuts3"].unique())

    await world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=sim_id,
        index=index,
    )
    # setup eom market

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
    for area in nuts_config:
        print(f"loading config {area} for {year}")
        config_path = Path.home() / ".assume" / f"{area}_{year}"
        if not config_path.is_dir():
            print(f"query database time series")
            demand = infra_interface.get_demand_series_in_area(area, year)
            demand = demand.resample("h").mean()
            # demand in MW
            solar, wind = infra_interface.get_renewables_series_in_area(
                area,
                start,
                end,
            )
            try:
                config_path.mkdir(parents=True, exist_ok=True)
                demand.to_csv(config_path / "demand.csv")
                solar.to_csv(config_path / "solar.csv")
                if isinstance(wind, float):
                    print(wind, area, year)
                wind.to_csv(config_path / "wind.csv")
            except Exception:
                shutil.rmtree(config_path, ignore_errors=True)
        else:
            print(f"use existing local time series")
            demand = pd.read_csv(config_path / "demand.csv", index_col=0).squeeze()
            solar = pd.read_csv(config_path / "solar.csv", index_col=0).squeeze()
            wind = pd.read_csv(config_path / "wind.csv", index_col=0).squeeze()

        lat, lon = infra_interface.get_lat_lon_area(area)

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
                "location": (lat, lon),
                "node": area,
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
                "location": (lat, lon),
                "node": area,
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
                "location": (lat, lon),
                "node": area,
            },
            NaiveForecast(
                index, availability=wind / wind.max(), fuel_price=0.2, co2_price=0
            ),
        )

        # TODO add biomass, run_hydro and storages

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
                        "emission_factor": plant["chi"],  # [t/MWh therm]
                        "efficiency": plant["eta"],
                        "technology": fuel_type,
                        "cold_start_cost": plant["start_cost"],
                        "ramp_up": plant["ramp_up"],
                        "ramp_down": plant["ramp_down"],
                        "location": (lat, lon),
                        "node": area,
                    },
                    NaiveForecast(
                        index,
                        availability=1,
                        fuel_price=fuel_prices[fuel_type],
                        co2_price=co2_price,
                    ),
                )


def load_oeds(
    world: World,
    scenario: str,
    study_case: str,
    infra_uri: str,
    marketdesign: list[MarketConfig],
    nuts_config: list[str] = [],
):
    """
    Load a scenario from a given path.

    :param world: The world.
    :type world: World
    :param inputs_path: Path to the inputs folder.
    :type inputs_path: str
    :param scenario: Name of the scenario.
    :type scenario: str
    :param study_case: Name of the study case.
    :type study_case: str
    """
    world.loop.run_until_complete(
        load_oeds_async(
            world=world,
            scenario=scenario,
            study_case=study_case,
            infra_uri=infra_uri,
            marketdesign=marketdesign,
            nuts_config=nuts_config,
        )
    )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "world_mastr"
    study_case = "study_case"
    # FH Aachen internal server
    infra_uri = os.getenv(
        "INFRASTRUCTURE_URI",
        "postgresql://readonly:readonly@timescale.nowum.fh-aachen.de:5432",
    )

    nuts_config = ["DE1", "DEA", "DEB", "DEC", "DED", "DEE", "DEF"]
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
    load_oeds(world, scenario, study_case, infra_uri, marketdesign, nuts_config)
    world.run()
