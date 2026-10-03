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
from assume.common.forecaster import (
    DemandForecaster,
    PowerplantForecaster,
    UnitForecaster,
)
from assume.common.market_objects import MarketConfig, MarketProduct
from assume.scenario.entsoe_helper.client import EntsoeInterface
from assume.scenario.entsoe_helper.fuel_prices import InstratFuelPrices
from assume.scenario.entsoe_helper.mappings import (
    COUNTRY_LOCATIONS,
    DEFAULT_BLOCK_SIZE_MW,
    DEFAULT_BLOCK_SIZES_MW,
    DEFAULT_CO2_PRICE_EUR_T,
    DEFAULT_EFFICIENCIES,
    DEFAULT_EMISSION_FACTORS,
    DEFAULT_FUEL_PRICE_RANGES,
    DEFAULT_RENEWABLE_FUEL_PRICES,
    FOSSIL_BIDDING_KEYS,
    block_price_factors,
    default_storage_params,
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
    efficiencies: dict[str, float] | None = None,
    storage_defaults: dict[str, float] | None = None,
    use_cache: bool = True,
    use_instrat_fuel_prices: bool = True,
    demand_proxy: str = "load",
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
        efficiencies (dict, optional): power-to-electricity efficiency
            (MWh_el per MWh fuel) per technology for thermal units. Defaults
            to ``mappings.DEFAULT_EFFICIENCIES``.
        storage_defaults (dict, optional): override the default parameters
            for storage units. Supported keys: ``hours`` (energy in MWh per
            MW of discharge power), ``additional_cost`` (€/MWh for charging
            and discharging), ``initial_soc`` (0..1), ``max_soc`` (0..1),
            ``min_soc`` (0..1), ``efficiency_charge`` (0..1) and
            ``efficiency_discharge`` (0..1).
        use_cache (bool): cache API responses under ``~/.assume/entsoe``
        use_instrat_fuel_prices (bool): fetch coal, gas and CO2 from instrat.pl
        demand_proxy (str): demand to be served, either ``"load"`` (actual
            load) or ``"generation"`` (total realised generation of the
            country). The latter includes net exports and excludes net
            imports, which matters for an island market that follows the
            realised generation profiles. Defaults to ``"load"``.
        save_frequency_hours (int): database save interval (48 h as in the CSV
            loader)
    """
    if demand_proxy not in ("load", "generation"):
        raise AssumeException(
            f"demand_proxy must be 'load' or 'generation', got {demand_proxy!r}"
        )
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
    efficiencies = DEFAULT_EFFICIENCIES | (efficiencies or {})
    storage = default_storage_params()
    storage.update(storage_defaults or {})

    api_fuel_prices: dict[str, pd.Series] = {}
    if use_instrat_fuel_prices:
        api_fuel_prices = InstratFuelPrices().get_fuel_prices(
            start, end, index, use_cache=use_cache
        )
    co2_prices = _resolve_co2_prices(index, api_fuel_prices)

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
        # fill gaps in the load series so every simulation hour has a value
        demand = demand.reindex(index).ffill().bfill()
        generation = entsoe.get_country_generation(
            start, end, country, use_cache=use_cache
        )
        capacity = entsoe.get_installed_capacity(
            start, end, country, use_cache=use_cache
        )
        technologies = entsoe.aggregate_by_technology(capacity, generation)
        if demand_proxy == "generation":
            demand = _total_generation(technologies, index)
        # countries without a known centroid get (0, 0)
        location = COUNTRY_LOCATIONS.get(country, (0.0, 0.0))

        world.add_unit_operator(f"demand_{country}")
        world.add_unit(
            f"demand_{country}",
            "demand",
            f"demand_{country}",
            {
                "min_power": 0,
                # demand units use a negative power convention in ASSUME, so
                # the bound must match the (always negative) forecaster demand
                "max_power": -abs(demand.max()),
                "bidding_strategies": bidding_strategies["demand"],
                "technology": "demand",
                "location": location,
                "node": country,
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
                    storage,
                )
            elif mapping.variable:
                _add_variable_unit(
                    world,
                    country,
                    tech,
                    mapping,
                    total_capacity,
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
                    co2_prices,
                    efficiencies,
                )

    world.init_forecasts()


def _total_generation(technologies, index):
    """Sum of the realised generation of all technologies on the hourly index."""
    total = sum(
        tech_data["generation_mw"].reindex(index).fillna(0)
        for tech_data in technologies.values()
    )
    return total.clip(lower=0)


def _generation_availability(gen_series, max_power):
    """Share of the installed capacity that was actually generated (0..1)."""
    if max_power <= 0:
        return 0
    return (gen_series / max_power).clip(lower=0, upper=1)


def _resolve_price_range(tech, bidding_key, fuel_price_ranges):
    # lookup order: bidding key, technology, then the generic "other" range
    if bidding_key in fuel_price_ranges:
        return fuel_price_ranges[bidding_key]
    if tech in fuel_price_ranges:
        return fuel_price_ranges[tech]
    return fuel_price_ranges["other"]


def _emission_factor(bidding_key: str) -> float:
    return DEFAULT_EMISSION_FACTORS.get(bidding_key, 0.0)


def _resolve_co2_prices(
    index: pd.DatetimeIndex,
    api_fuel_prices: dict[str, pd.Series],
) -> pd.Series:
    """Return a shared CO2 price series for all fossil units (€/tCO2)."""
    if "co2" in api_fuel_prices:
        return api_fuel_prices["co2"]
    return pd.Series(DEFAULT_CO2_PRICE_EUR_T, index=index, name="co2")


def _powerplant_fuel_type(bidding_key: str) -> str:
    # non-fossil technologies use the generic "others" fuel of the power plant
    if bidding_key in FOSSIL_BIDDING_KEYS:
        return bidding_key
    return "others"


def _build_fossil_fuel_prices(
    bidding_key: str,
    fuel_value: pd.Series | float,
    co2_prices: pd.Series,
) -> dict[str, pd.Series | float]:
    fuel_type = _powerplant_fuel_type(bidding_key)
    fuel_prices: dict[str, pd.Series | float] = {fuel_type: fuel_value}
    if bidding_key in FOSSIL_BIDDING_KEYS:
        fuel_prices["co2"] = co2_prices
    return fuel_prices


def _build_api_fuel_prices(
    bidding_key: str,
    fuel_series: pd.Series,
    co2_prices: pd.Series,
) -> dict[str, pd.Series]:
    return _build_fossil_fuel_prices(bidding_key, fuel_series, co2_prices)


def _add_variable_unit(
    world,
    country,
    tech,
    mapping,
    total_capacity,
    gen_series,
    index,
    location,
    bidding_strategies,
):
    if total_capacity <= 0:
        return

    fuel_price = DEFAULT_RENEWABLE_FUEL_PRICES.get(tech, 0.0)
    world.add_unit(
        f"generation_{country}_{tech}",
        "power_plant",
        f"generation_{country}",
        {
            "min_power": 0,
            "max_power": total_capacity,
            "bidding_strategies": bidding_strategies[mapping.bidding_key],
            "technology": tech,
            "emission_factor": _emission_factor(mapping.bidding_key),
            "location": location,
            "node": country,
        },
        PowerplantForecaster(
            index,
            availability=_generation_availability(gen_series, total_capacity),
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
    co2_prices,
    efficiencies,
):
    if total_capacity <= 0:
        return

    block_size = block_sizes_mw.get(
        tech, block_sizes_mw.get(mapping.bidding_key, DEFAULT_BLOCK_SIZE_MW)
    )
    blocks = split_capacity_blocks(total_capacity, block_size)
    price_high, price_low = _resolve_price_range(
        tech, mapping.bidding_key, fuel_price_ranges
    )
    # only thermal technologies with a fuel price (in EUR/MWh_thermal) get an
    # efficiency; ASSUME converts the fuel price via power / efficiency
    efficiency = efficiencies.get(mapping.bidding_key)

    def _unit_params(block_capacity: float) -> dict:
        params = {
            "min_power": 0,
            "max_power": block_capacity,
            "bidding_strategies": bidding_strategies[mapping.bidding_key],
            "technology": tech,
            "fuel_type": _powerplant_fuel_type(mapping.bidding_key),
            "emission_factor": _emission_factor(mapping.bidding_key),
            "location": location,
            "node": country,
        }
        if efficiency is not None:
            params["efficiency"] = efficiency
        return params

    # live instrat.pl series are scaled per block; otherwise each block gets a
    # fixed price interpolated between price_high and price_low
    if mapping.bidding_key in api_fuel_prices:
        factors = block_price_factors(len(blocks), price_high, price_low)
        for block_idx, (block_capacity, block_factor) in enumerate(
            zip(blocks, factors), start=1
        ):
            fuel_prices = _build_api_fuel_prices(
                mapping.bidding_key,
                api_fuel_prices[mapping.bidding_key] * block_factor,
                co2_prices,
            )
            world.add_unit(
                f"generation_{country}_{tech}_{block_idx}",
                "power_plant",
                f"generation_{country}",
                _unit_params(block_capacity),
                PowerplantForecaster(
                    index,
                    availability=1,
                    fuel_prices=fuel_prices,
                ),
            )
        return

    block_prices = interpolate_block_prices(len(blocks), price_high, price_low)
    for block_idx, (block_capacity, block_price) in enumerate(
        zip(blocks, block_prices), start=1
    ):
        fuel_prices = _build_fossil_fuel_prices(
            mapping.bidding_key, block_price, co2_prices
        )
        world.add_unit(
            f"generation_{country}_{tech}_{block_idx}",
            "power_plant",
            f"generation_{country}",
            _unit_params(block_capacity),
            PowerplantForecaster(
                index,
                availability=1,
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
    storage,
):
    if total_capacity <= 0:
        return

    block_size = block_sizes_mw.get(
        tech, block_sizes_mw.get("hydro_storage", DEFAULT_BLOCK_SIZE_MW)
    )
    blocks = split_capacity_blocks(total_capacity, block_size)

    for block_idx, block_capacity in enumerate(blocks, start=1):
        # charging power is negative in ASSUME; energy capacity = power * hours
        world.add_unit(
            f"storage_{country}_{tech}_{block_idx}",
            "storage",
            f"generation_{country}",
            {
                "max_power_charge": -abs(block_capacity),
                "max_power_discharge": block_capacity,
                "capacity": block_capacity * storage["hours"],
                "max_soc": storage["max_soc"],
                "min_soc": storage["min_soc"],
                "initial_soc": storage["initial_soc"],
                "efficiency_charge": storage["efficiency_charge"],
                "efficiency_discharge": storage["efficiency_discharge"],
                "additional_cost_charge": storage["additional_cost"],
                "additional_cost_discharge": storage["additional_cost"],
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
    end = datetime(year + 1, 1, 1) - timedelta(hours=1)
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
