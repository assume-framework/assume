# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil import rrule as rr

from assume import World
from assume.common.fast_pandas import FastSeries
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.scenario.oeds.infrastructure import InfrastructureInterface

logger = logging.getLogger(__name__)


def load_oeds(
    world: World,
    scenario: str,
    study_case: str,
    start: datetime,
    end: datetime,
    infra_uri: str,
    marketdesign: list[MarketConfig],
    bidding_strategies: dict[str, str],
    nuts_config: list[str] = [],
    random=True,
    entsoe_demand=True,
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
    index = pd.date_range(
        start=start,
        end=end,
        freq="h",
    )
    simulation_id = f"{scenario}_{study_case}"
    year = start.year
    logger.info(f"loading scenario {simulation_id} with {nuts_config}")
    infra_interface = InfrastructureInterface("test", infra_uri)

    if not nuts_config or nuts_config == "nuts3":
        nuts_config = list(infra_interface.plz_nuts["nuts3"].unique())
    elif nuts_config == "nuts0":
        nuts_config = ["DE"]
    elif nuts_config == "nuts1":
        nuts_config = list(infra_interface.plz_nuts["nuts1"].unique())
    elif nuts_config == "nuts2":
        nuts_config = list(infra_interface.plz_nuts["nuts2"].unique())

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=48,
        simulation_id=simulation_id,
    )
    # setup eom market

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    fuel_prices = {  # €/MWh
        # prices taken from
        # https://www.ise.fraunhofer.de/content/dam/ise/de/documents/publications/studies/DE2024_ISE_Studie_Stromgestehungskosten_Erneuerbare_Energien.pdf
        "hard coal": infra_interface.get_coal_price(start, end),  # 11.6
        "lignite": 2.3,
        "oil": infra_interface.get_oil_price(start, end),  # 22
        "gas": infra_interface.get_gas_price(start, end),  # 26
        "biomass": 20,
        "nuclear": 8,
        "co2": infra_interface.get_co2_price(start, end),  # 20€/tCO2eq
    }
    for name in fuel_prices.keys():
        if not isinstance(fuel_prices[name], float | int):
            fuel_prices[name] = (
                fuel_prices[name].reindex(index, method="nearest").values
            )

    offshore_wind = infra_interface.get_offshore_wind_series(start, end)
    if offshore_wind.max() > 0:
        world.add_unit_operator("renewables_offshore")
        world.add_unit(
            "renewables_off_wind",
            "power_plant",
            "renewables_offshore",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": offshore_wind.max(),
                "bidding_strategies": bidding_strategies["wind"],
                "technology": "wind_offshore",
                "location": (8.18, 54.4),
                "node": "DEF",
            },
            NaiveForecast(
                index,
                availability=offshore_wind / offshore_wind.max(),
                fuel_price=0.2,
                co2_price=0,
            ),
        )

    # total german demand if area is not set
    if entsoe_demand:
        demand = infra_interface.get_country_demand(start, end, "DE")
        demand = demand.resample("h").mean() / len(nuts_config)

    for area in nuts_config:
        logger.info(f"loading config {area} for {year}")
        config_path = Path.home() / ".assume" / f"{area}_{year}"
        if not config_path.is_dir():
            logger.info("query database time series")
            if area == "DE":
                # for each area - add demand and generation
                demand_save = infra_interface.get_country_demand(start, end, "DE")
                renewables = infra_interface.get_country_renewables(start, end, "DE")
                solar = renewables["solar"].resample("1h").mean()
                wind = renewables["wind_onshore"].resample("1h").mean()
            else:
                # demand from OEP is less accurate but extrapolates better
                if not entsoe_demand:
                    demand_save = infra_interface.get_demand_series_in_area(area, year)
                # TODO add battery_power as storage
                solar, wind, battery_power = (
                    infra_interface.get_renewables_series_in_area(
                        area,
                        start,
                        end,
                    )
                )

            demand_save = demand.resample("h").mean()
            # demand in MW
            if not entsoe_demand:
                demand = demand_save

            try:
                config_path.mkdir(parents=True, exist_ok=True)
                demand_save.to_csv(config_path / "demand.csv")
                solar.to_csv(config_path / "solar.csv")
                if isinstance(wind, float):
                    logger.info(wind, area, year)
                wind.to_csv(config_path / "wind.csv")
            except Exception:
                shutil.rmtree(config_path, ignore_errors=True)
        else:
            logger.info("use existing local time series")
            # demand from OEP is less accurate but extrapolates better
            if not entsoe_demand:
                demand = pd.read_csv(
                    config_path / "demand.csv", index_col=0, parse_dates=True
                ).squeeze()
            solar = pd.read_csv(
                config_path / "solar.csv", index_col=0, parse_dates=True
            ).squeeze()
            wind = pd.read_csv(
                config_path / "wind.csv", index_col=0, parse_dates=True
            ).squeeze()

        lat, lon = infra_interface.get_lat_lon_area(area)

        if isinstance(demand, pd.DataFrame):
            demand = demand.sum(axis=1)

        world.add_unit_operator(f"demand_{area}")
        world.add_unit(
            f"demand_{area}1",
            "demand",
            f"demand_{area}",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": demand.max(),
                "bidding_strategies": bidding_strategies["demand"],
                "technology": "demand",
                "location": (lat, lon),
                "node": area,
                "price": 1e3,
            },
            NaiveForecast(index, demand=demand),
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
        if wind.max() > 0:
            world.add_unit(
                f"renewables{area}_wind",
                "power_plant",
                f"renewables{area}",
                # the unit_params have no hints
                {
                    "min_power": 0,
                    "max_power": wind.max(),
                    "bidding_strategies": bidding_strategies["wind"],
                    "technology": "wind_onshore",
                    "location": (lat, lon),
                    "node": area,
                },
                NaiveForecast(
                    index, availability=wind / wind.max(), fuel_price=0.2, co2_price=0
                ),
            )

        biomass = infra_interface.get_biomass_systems_in_area(area=area)
        # somehow the maximum water power matches half of the entsoe much better
        biomass["maxPower"] /= 2

        if random:
            randomness = np.random.uniform(-20, 20)
        else:
            randomness = 0

        world.add_unit(
            f"renewables{area}_bio",
            "power_plant",
            f"renewables{area}",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": biomass["maxPower"].sum() / 1e3,  # kW -> MW
                "bidding_strategies": bidding_strategies["biomass"],
                "technology": "biomass",
                "location": (lat, lon),
                "node": area,
            },
            NaiveForecast(
                index,
                availability=1,
                fuel_price=fuel_prices["biomass"] + randomness,
                co2_price=0,
            ),
        )
        water = infra_interface.get_run_river_systems_in_area(area=area)
        # somehow the maximum water power matches half of the entsoe much better
        water["maxPower"] /= 2

        world.add_unit(
            f"renewables{area}_hydro",
            "power_plant",
            f"renewables{area}",
            # the unit_params have no hints
            {
                "min_power": 0,
                "max_power": water["maxPower"].sum() / 1e3,  # kW -> MW
                "bidding_strategies": bidding_strategies["hydro"],
                "technology": "hydro",
                "location": (lat, lon),
                "node": area,
            },
            NaiveForecast(index, availability=1, fuel_price=0.2, co2_price=0),
        )

        if True:
            storages = infra_interface.get_water_storage_systems(area)
            world.add_unit_operator(f"storage{area}")
            for storage in storages:
                world.add_unit(
                    f"storage{area}_{storage['unitID']}",
                    "storage",
                    f"storage{area}",
                    # the unit_params have no hints
                    {
                        "max_power_charge": storage["max_power_charge"] / 1e3,
                        "max_power_discharge": storage["max_power_discharge"] / 1e3,
                        "max_soc": storage["max_soc"] / 1e3,
                        "min_soc": storage["min_soc"] / 1e3,
                        "efficiency_charge": storage["efficiency_charge"],
                        "efficiency_discharge": storage["efficiency_discharge"],
                        "bidding_strategies": bidding_strategies["storage"],
                        "technology": "hydro_storage",
                        "location": (lat, lon),
                        "node": area,
                    },
                    NaiveForecast(index, availability=1, fuel_price=0.2, co2_price=0),
                )

        world.add_unit_operator(f"conventional{area}")

        for fuel_type in ["nuclear", "lignite", "hard coal", "oil", "gas"]:
            plants = infra_interface.get_power_plant_in_area(
                area,
                fuel_type,
            )
            plants = list(plants.T.to_dict().values())
            i = 0
            for plant in plants:
                i += 1
                if random:
                    randomness = np.random.uniform(-5, 5)
                else:
                    randomness = 0

                availability = 1
                if plant["endDate"] < end:
                    availability = FastSeries(index, 1)
                    availability[availability.index > end] = 0

                world.add_unit(
                    f"conventional{area}_{fuel_type}_{i}",
                    "power_plant",
                    f"conventional{area}",
                    # the unit_params have no hints
                    {
                        "min_power": plant["minPower"] / 1e3,  # kW -> MW
                        "max_power": plant["maxPower"] / 1e3,  # kW -> MW
                        "bidding_strategies": bidding_strategies[fuel_type],
                        "emission_factor": plant["chi"] * 1e3,  # [t/MWh therm]
                        "efficiency": plant["eta"],
                        "technology": fuel_type,
                        "cold_start_cost": plant["start_cost"] * 1e3,
                        "ramp_up": plant["ramp_up"],
                        "ramp_down": plant["ramp_down"],
                        "location": (lat, lon),
                        "node": area,
                    },
                    NaiveForecast(
                        index,
                        availability=availability,
                        fuel_price=fuel_prices[fuel_type] + randomness,
                        co2_price=fuel_prices["co2"],
                    ),
                )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "world_mastr"
    # FH Aachen internal server
    infra_uri = os.getenv(
        "INFRASTRUCTURE_URI",
        "postgresql://readonly:readonly@timescale.nowum.fh-aachen.de:5432/opendata",
    )

    # default_nuts_config = "DE1"
    default_nuts_config = "DE1, DEA, DEB, DEC, DED, DEE, DEF"
    nuts_config = os.getenv("NUTS_CONFIG", default_nuts_config).split(",")
    nuts_config = [n.strip() for n in nuts_config]
    nuts_config = "nuts3"
    year = 2024
    random = True
    type = "random" if random else "static"
    if isinstance(nuts_config, str):
        study_case = f"{nuts_config}_{type}_{year}"
    else:
        study_case = f"custom_{type}_{year}"
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31) - timedelta(hours=1)
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

    default_strategy = {mc.market_id: "naive_eom" for mc in marketdesign}
    default_naive_strategy = {mc.market_id: "naive_eom" for mc in marketdesign}

    bidding_strategies = {
        "hard coal": default_strategy,
        "lignite": default_strategy,
        "oil": default_strategy,
        "gas": default_strategy,
        "biomass": default_strategy,
        "hydro": default_strategy,
        "nuclear": default_strategy,
        "wind": default_naive_strategy,
        "solar": default_naive_strategy,
        "demand": default_naive_strategy,
        "storage": {mc.market_id: "flexable_eom_storage" for mc in marketdesign},
    }
    load_oeds(
        world,
        scenario,
        study_case,
        start,
        end,
        infra_uri,
        marketdesign,
        bidding_strategies,
        nuts_config,
    )

    world.run()
