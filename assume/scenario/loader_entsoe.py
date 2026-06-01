# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
from dateutil import rrule as rr

from assume import World
from assume.common.exceptions import AssumeException
from assume.common.forecaster import DemandForecaster, PowerplantForecaster, UnitForecaster
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.scenario.entsoe_helper.client import EntsoeInterface
from assume.scenario.entsoe_helper.fuel_prices import InstratFuelPrices
from assume.scenario.entsoe_helper.mappings import (
    COUNTRY_LOCATIONS,
    DEFAULT_BLOCK_SIZES_MW,
    DEFAULT_EMISSION_FACTORS,
    DEFAULT_FUEL_PRICE_RANGES,
    DEFAULT_RENEWABLE_FUEL_PRICES,
    DEFAULT_STORAGE_ADDITIONAL_COST,
    DEFAULT_STORAGE_HOURS,
    FOSSIL_BIDDING_KEYS,
    THERMAL_BIDDING_KEYS,
    block_price_factors,
    interpolate_block_prices,
    split_capacity_blocks,
)

logger = logging.getLogger(__name__)


def load_entsoe(
    world: World,
    scenario: str,
    study_case: str,
    start: datetime,
    end: datetime,
    countries: list[str],
    marketdesign: list[MarketConfig],
    bidding_strategies: dict[str, dict[str, str]],
    api_key: str | None = None,
    fuel_price_ranges: dict[str, tuple[float, float]] | None = None,
    block_sizes_mw: dict[str, float] | None = None,
    use_cache: bool = True,
    use_instrat_fuel_prices: bool = True,
    save_frequency_hours: int = 48,
):
    """
    Initialize a country-level scenario from the ENTSO-E Transparency Platform.

    Requires ``pip install 'assume-framework[entsoe]'`` and an API key from
    https://transparency.entsoe.eu/ (``ENTSOE_API_KEY`` env var or ``api_key``).

    Args:
        world (World): the world to add this scenario to
        scenario (str): scenario name
        study_case (str): study case name
        start (datetime): simulation start
        end (datetime): simulation end
        countries (list[str]): ISO country codes, e.g. ``["DE", "FR"]``
        marketdesign (list[MarketConfig]): market design for the simulation
        bidding_strategies (dict): bidding strategies per fuel or technology key
        api_key (str, optional): ENTSO-E API key
        fuel_price_ranges (dict, optional): block spread in €/MWh per fuel
        block_sizes_mw (dict, optional): block size in MW per technology
        use_cache (bool): cache API responses under ``~/.assume/entsoe``
        use_instrat_fuel_prices (bool): fetch coal, gas and CO2 from instrat.pl
        save_frequency_hours (int): database save interval
    """
    if not countries:
        countries = ["DE"]

    countries = [country.upper() for country in countries]
    index = pd.date_range(start=start, end=end, freq="h")
    simulation_id = f"{scenario}_{study_case}"
    logger.info(f"loading ENTSO-E scenario {simulation_id} with {countries}")

    api_key = api_key or os.getenv("ENTSOE_API_KEY") or os.getenv("ENTSOE")
    if not api_key:
        raise AssumeException(
            "ENTSO-E API key missing. Set ENTSOE_API_KEY or pass api_key."
        )

    entsoe = EntsoeInterface(api_key=api_key)
    fuel_price_ranges = DEFAULT_FUEL_PRICE_RANGES | (fuel_price_ranges or {})
    block_sizes_mw = DEFAULT_BLOCK_SIZES_MW | (block_sizes_mw or {})

    api_fuel_prices: dict[str, pd.Series] = {}
    if use_instrat_fuel_prices:
        api_fuel_prices = InstratFuelPrices().get_fuel_prices(
            start, end, index, use_cache=use_cache
        )

    world.setup(
        start=start,
        end=end,
        save_frequency_hours=save_frequency_hours,
        simulation_id=simulation_id,
    )

    mo_id = "market_operator"
    world.add_market_operator(id=mo_id)
    for market_config in marketdesign:
        world.add_market(mo_id, market_config)

    for country in countries:
        logger.info(f"loading ENTSO-E data for {country}")
        demand = entsoe.get_country_demand(start, end, country, use_cache=use_cache)
        demand = demand.reindex(index).ffill().bfill()
        generation = entsoe.get_country_generation(
            start, end, country, use_cache=use_cache
        )
        capacity = entsoe.get_installed_capacity(
            start, end, country, use_cache=use_cache
        )
        technologies = entsoe.aggregate_by_technology(capacity, generation)
        location = COUNTRY_LOCATIONS.get(country, (0.0, 0.0))

        world.add_unit_operator(f"demand_{country}")
        world.add_unit(
            f"demand_{country}",
            "demand",
            f"demand_{country}",
            {
                "min_power": 0,
                "max_power": -demand.max(),
                "bidding_strategies": bidding_strategies["demand"],
                "technology": "demand",
                "location": location,
                "node": country,
                "price": 1e3,
            },
            DemandForecaster(index, demand=-abs(demand)),
        )

        world.add_unit_operator(f"generation_{country}")
        for tech, tech_data in technologies.items():
            mapping = tech_data["mapping"]
            total_capacity = tech_data["capacity_mw"]
            gen_series = tech_data["generation_mw"].reindex(index, fill_value=0)

            if total_capacity <= 0:
                continue

            if mapping.unit_type == "storage":
                _add_storage_units(
                    world,
                    country,
                    tech,
                    mapping,
                    total_capacity,
                    index,
                    location,
                    bidding_strategies,
                    block_sizes_mw,
                )
            elif mapping.variable:
                _add_variable_unit(
                    world,
                    country,
                    tech,
                    mapping,
                    gen_series,
                    index,
                    location,
                    bidding_strategies,
                )
            else:
                _add_blocked_units(
                    world,
                    country,
                    tech,
                    mapping,
                    total_capacity,
                    gen_series,
                    index,
                    location,
                    bidding_strategies,
                    block_sizes_mw,
                    fuel_price_ranges,
                    api_fuel_prices,
                )

    world.init_forecasts()


def _generation_availability(gen_series, max_power):
    if max_power <= 0:
        return 0
    return (gen_series / max_power).clip(lower=0, upper=1)


def _resolve_price_range(tech, bidding_key, fuel_price_ranges):
    if bidding_key in fuel_price_ranges:
        return fuel_price_ranges[bidding_key]
    if tech in fuel_price_ranges:
        return fuel_price_ranges[tech]
    return fuel_price_ranges.get("other", (20.0, 15.0))


def _emission_factor(bidding_key: str) -> float:
    return DEFAULT_EMISSION_FACTORS.get(bidding_key, 0.0)


def _powerplant_fuel_type(bidding_key: str) -> str:
    if bidding_key in FOSSIL_BIDDING_KEYS:
        return bidding_key
    return "others"


def _build_api_fuel_prices(
    bidding_key: str,
    fuel_series: pd.Series,
    api_fuel_prices: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    fuel_type = _powerplant_fuel_type(bidding_key)
    fuel_prices = {fuel_type: fuel_series}
    if bidding_key in FOSSIL_BIDDING_KEYS:
        fuel_prices["co2"] = api_fuel_prices["co2"]
    return fuel_prices


def _add_variable_unit(
    world,
    country,
    tech,
    mapping,
    gen_series,
    index,
    location,
    bidding_strategies,
):
    peak = gen_series.max()
    if peak <= 0:
        return

    fuel_price = DEFAULT_RENEWABLE_FUEL_PRICES.get(tech, 0.2)
    world.add_unit(
        f"generation_{country}_{tech}",
        "power_plant",
        f"generation_{country}",
        {
            "min_power": 0,
            "max_power": peak,
            "bidding_strategies": bidding_strategies[mapping.bidding_key],
            "technology": tech,
            "emission_factor": _emission_factor(mapping.bidding_key),
            "location": location,
            "node": country,
        },
        PowerplantForecaster(
            index,
            availability=_generation_availability(gen_series, peak),
            fuel_prices={"others": fuel_price},
        ),
    )


def _add_blocked_units(
    world,
    country,
    tech,
    mapping,
    total_capacity,
    gen_series,
    index,
    location,
    bidding_strategies,
    block_sizes_mw,
    fuel_price_ranges,
    api_fuel_prices,
):
    peak = gen_series.max()
    if peak <= 0:
        return

    block_size = block_sizes_mw.get(
        tech, block_sizes_mw.get(mapping.bidding_key, 500.0)
    )
    blocks = split_capacity_blocks(total_capacity, block_size)
    scale = peak / total_capacity
    bid_full_capacity = mapping.bidding_key in THERMAL_BIDDING_KEYS
    price_high, price_low = _resolve_price_range(
        tech, mapping.bidding_key, fuel_price_ranges
    )

    if mapping.bidding_key in api_fuel_prices:
        factors = block_price_factors(len(blocks), price_high, price_low)
        for block_idx, (block_capacity, block_factor) in enumerate(
            zip(blocks, factors), start=1
        ):
            block_max = block_capacity if bid_full_capacity else block_capacity * scale
            share = block_capacity / total_capacity
            block_gen = gen_series * share
            availability = 1 if bid_full_capacity else _generation_availability(
                block_gen, block_max
            )
            fuel_prices = _build_api_fuel_prices(
                mapping.bidding_key,
                api_fuel_prices[mapping.bidding_key] * block_factor,
                api_fuel_prices,
            )
            world.add_unit(
                f"generation_{country}_{tech}_{block_idx}",
                "power_plant",
                f"generation_{country}",
                {
                    "min_power": 0,
                    "max_power": block_max,
                    "bidding_strategies": bidding_strategies[mapping.bidding_key],
                    "technology": tech,
                    "fuel_type": _powerplant_fuel_type(mapping.bidding_key),
                    "emission_factor": _emission_factor(mapping.bidding_key),
                    "location": location,
                    "node": country,
                },
                PowerplantForecaster(
                    index,
                    availability=availability,
                    fuel_prices=fuel_prices,
                ),
            )
        return

    block_prices = interpolate_block_prices(len(blocks), price_high, price_low)
    for block_idx, (block_capacity, block_price) in enumerate(
        zip(blocks, block_prices), start=1
    ):
        block_max = block_capacity if bid_full_capacity else block_capacity * scale
        share = block_capacity / total_capacity
        block_gen = gen_series * share
        availability = 1 if bid_full_capacity else _generation_availability(
            block_gen, block_max
        )
        fuel_type = _powerplant_fuel_type(mapping.bidding_key)
        fuel_prices = {fuel_type: block_price}
        if mapping.bidding_key in FOSSIL_BIDDING_KEYS:
            fuel_prices["co2"] = price_high

        world.add_unit(
            f"generation_{country}_{tech}_{block_idx}",
            "power_plant",
            f"generation_{country}",
            {
                "min_power": 0,
                "max_power": block_max,
                "bidding_strategies": bidding_strategies[mapping.bidding_key],
                "technology": tech,
                "fuel_type": fuel_type,
                "emission_factor": _emission_factor(mapping.bidding_key),
                "location": location,
                "node": country,
            },
            PowerplantForecaster(
                index,
                availability=availability,
                fuel_prices=fuel_prices,
            ),
        )


def _add_storage_units(
    world,
    country,
    tech,
    mapping,
    total_capacity,
    index,
    location,
    bidding_strategies,
    block_sizes_mw,
):
    if total_capacity <= 0:
        return

    block_size = block_sizes_mw.get(tech, block_sizes_mw.get("hydro_storage", 250.0))
    blocks = split_capacity_blocks(total_capacity, block_size)

    for block_idx, block_capacity in enumerate(blocks, start=1):
        world.add_unit(
            f"storage_{country}_{tech}_{block_idx}",
            "storage",
            f"generation_{country}",
            {
                "max_power_charge": -abs(block_capacity),
                "max_power_discharge": block_capacity,
                "capacity": block_capacity * DEFAULT_STORAGE_HOURS,
                "max_soc": 1.0,
                "min_soc": 0.0,
                "initial_soc": 0.5,
                "efficiency_charge": 0.85,
                "efficiency_discharge": 0.9,
                "additional_cost_charge": DEFAULT_STORAGE_ADDITIONAL_COST,
                "additional_cost_discharge": DEFAULT_STORAGE_ADDITIONAL_COST,
                "bidding_strategies": bidding_strategies[mapping.bidding_key],
                "technology": tech,
                "location": location,
                "node": country,
            },
            UnitForecaster(index, availability=1),
        )


if __name__ == "__main__":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"
    world = World(database_uri=db_uri)
    scenario = "entsoe"
    countries = os.getenv("ENTSOE_COUNTRIES", "DE").split(",")
    countries = [country.strip() for country in countries if country.strip()]
    year = int(os.getenv("ENTSOE_YEAR", "2024"))
    study_case = f"{'_'.join(countries)}_{year}"

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

    default_strategy = {mc.market_id: "powerplant_energy_naive" for mc in marketdesign}
    default_demand_strategy = {
        mc.market_id: "demand_energy_naive" for mc in marketdesign
    }
    bidding_strategies = {
        "hard coal": default_strategy,
        "lignite": default_strategy,
        "oil": default_strategy,
        "gas": default_strategy,
        "biomass": default_strategy,
        "hydro": default_strategy,
        "nuclear": default_strategy,
        "wind": default_strategy,
        "solar": default_strategy,
        "storage": {
            mc.market_id: "storage_energy_heuristic_flexable" for mc in marketdesign
        },
        "demand": default_demand_strategy,
    }

    load_entsoe(
        world,
        scenario,
        study_case,
        start,
        end,
        countries,
        marketdesign,
        bidding_strategies,
    )
    world.run()
