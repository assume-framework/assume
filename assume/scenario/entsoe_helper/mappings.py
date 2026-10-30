# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass

# Approximate geographic centroids (latitude, longitude) of each country; only
# used as the unit location, not for any network or distance calculation.
COUNTRY_LOCATIONS: dict[str, tuple[float, float]] = {
    "AT": (47.52, 14.55),
    "BE": (50.50, 4.47),
    "BG": (42.73, 25.49),
    "CH": (46.82, 8.23),
    "CZ": (49.82, 15.47),
    "DE": (51.16, 10.45),
    "DK": (56.26, 9.50),
    "EE": (58.60, 25.01),
    "ES": (40.46, -3.75),
    "FI": (61.92, 25.75),
    "FR": (46.22, 2.21),
    "GB": (55.38, -3.44),
    "GR": (39.07, 21.82),
    "HR": (45.10, 15.20),
    "HU": (47.16, 19.50),
    "IE": (53.41, -8.24),
    "IT": (41.87, 12.57),
    "LT": (55.17, 23.88),
    "LU": (49.82, 6.13),
    "LV": (56.88, 24.60),
    "NL": (52.13, 5.29),
    "NO": (60.47, 8.47),
    "PL": (51.92, 19.15),
    "PT": (39.40, -8.22),
    "RO": (45.94, 24.97),
    "SE": (60.13, 18.64),
    "SI": (46.15, 14.99),
    "SK": (48.67, 19.70),
}


@dataclass(frozen=True)
class TechnologyMapping:
    technology: str  # ASSUME technology name of the unit
    bidding_key: str  # key into bidding_strategies and the price/efficiency defaults
    variable: bool  # weather-driven (solar/wind): one unit following actual output
    unit_type: str = "power_plant"


# Bidding keys without a fuel price series from instrat.pl (biomass, hydro, wind, ...)
# are grouped onto the key whose fallback price band best approximates their low
# marginal cost. E.g. geothermal is a dispatchable with small marginal cost (no
# fuel, similar to biomass), so it bids under the "biomass" key; instrat does not
# publish a geothermal fuel price.
PSR_TO_ASSUME: dict[str, TechnologyMapping] = {
    "Biomass": TechnologyMapping("biomass", "biomass", False),
    "Fossil Brown coal/Lignite": TechnologyMapping("lignite", "lignite", False),
    "Fossil Coal-derived gas": TechnologyMapping("gas", "gas", False),
    "Fossil Gas": TechnologyMapping("gas", "gas", False),
    "Fossil Hard coal": TechnologyMapping("hard coal", "hard coal", False),
    "Fossil Oil": TechnologyMapping("oil", "oil", False),
    "Fossil Oil shale": TechnologyMapping("oil", "oil", False),
    "Fossil Peat": TechnologyMapping("lignite", "lignite", False),
    "Geothermal": TechnologyMapping("geothermal", "biomass", False),
    "Hydro Pumped Storage": TechnologyMapping(
        "hydro_storage", "storage", False, unit_type="storage"
    ),
    # ENTSO-E renamed this production type "poundage" -> "pondage"; both spellings
    # have to stay until all transparent-data feeds use the new name.
    "Hydro Run-of-river and poundage": TechnologyMapping("hydro", "hydro", False),
    "Hydro Run-of-river and pondage": TechnologyMapping("hydro", "hydro", False),
    "Hydro Water Reservoir": TechnologyMapping("hydro", "hydro", False),
    "Marine": TechnologyMapping("marine", "hydro", False),
    "Nuclear": TechnologyMapping("nuclear", "nuclear", False),
    "Other": TechnologyMapping("other", "biomass", False),
    "Other renewable": TechnologyMapping("other_renewable", "biomass", False),
    "Solar": TechnologyMapping("solar", "solar", True),
    "Waste": TechnologyMapping("waste", "biomass", False),
    "Wind Offshore": TechnologyMapping("wind_offshore", "wind", True),
    "Wind Onshore": TechnologyMapping("wind_onshore", "wind", True),
    "Energy storage": TechnologyMapping(
        "battery_storage", "storage", False, unit_type="storage"
    ),
}

# Block size in MW used to split a technology's installed capacity into
# merit-order blocks. Thermal values are the capacity-weighted mean unit size
# (sum(P^2) / sum(P), i.e. the size of the unit a typical MW sits in) per fuel
# in examples/inputs/example_03/powerplant_units.csv (German fleet, 262 units),
# rounded to 10 MW: hard coal 602, lignite 652, gas 384, oil 198, nuclear 1432.
# They are consistent with the reference plant sizes in Kost et al., Fraunhofer
# ISE, "Stromgestehungskosten Erneuerbare Energien", Aug. 2024, section 3
# (lignite 800-1000, hard coal 600-800, CCGT 400-600, gas turbine 200 MW).
# Hydro storage is the same measure for the discharge power in
# example_03/storage_units.csv (614 MW).
DEFAULT_BLOCK_SIZES_MW: dict[str, float] = {
    "hard coal": 600.0,
    "lignite": 650.0,
    "gas": 380.0,
    "oil": 200.0,
    "nuclear": 1430.0,
    "hydro_storage": 610.0,
}
# Technologies without an entry above (e.g. hydro, biomass, batteries) have no
# unit-level data in example_03 and use the capacity-weighted mean unit size of
# its whole thermal fleet (646 MW).
DEFAULT_BLOCK_SIZE_MW = 650.0

# Fallback fuel prices, used when instrat.pl has no data for the requested
# period or is switched off. Source: Kost et al., Fraunhofer ISE,
# "Stromgestehungskosten Erneuerbare Energien", August 2024 (real 2024 EUR),
# Table 5 (fuel prices 2024) and Table 7 (CO2 price 2024: 75-90 EUR/t; the
# mid-point is used). These are the study's assumptions, not observed prices.
DEFAULT_COAL_PRICE_EUR_MWH = 11.6  # EUR/MWh thermal, hard coal
DEFAULT_GAS_PRICE_EUR_MWH = 38.0  # EUR/MWh thermal, natural gas (27.0 from 2030)
DEFAULT_CO2_PRICE_EUR_T = 82.5  # EUR/tCO2

# (price_high, price_low) in EUR/MWh for the topmost and bottom merit-order
# block. Serves two purposes:
# 1. Without instrat.pl prices, these are the fixed fuel prices of the blocks.
# 2. With instrat.pl prices, only the ratio high/low is used, as the relative
#    spread of block prices around the live base price.
# Base prices are the 2024 values of Kost et al. (Fraunhofer ISE, Aug. 2024),
# Table 5: hard coal 11.6, lignite 2.3, gas 38.0, solid biomass 23.8 EUR/MWh_th,
# uranium 8.0 EUR/MWh_th (nuclear is EUR/MWh_el: 8.0 / 0.35 efficiency from
# Table 6 = 22.9). Fossil ranges scale the base by median efficiency /
# efficiency at the 10th (high) and 90th (low) percentile of that fuel's units
# in examples/inputs/example_03/powerplant_units.csv, i.e. the marginal-cost
# spread of the example fleet. Oil is not covered by the study and keeps the
# 2019 example_03 price (25.7 EUR/MWh_th) with the same spread. Hydro has no
# fuel cost in example_03.
DEFAULT_FUEL_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "hard coal": (13.3, 10.2),
    "lignite": (2.6, 2.1),
    "gas": (52.4, 29.3),
    "oil": (26.5, 22.5),
    "biomass": (23.8, 23.8),
    "hydro": (0.0, 0.0),
    "nuclear": (22.9, 22.9),
    # fallback for bidding keys without an entry; all other production types
    # (waste, geothermal, ...) already bid under the biomass key
    "other": (23.8, 23.8),
}

# Marginal cost in EUR/MWh of weather-driven units: zero, as for all renewable
# units in example_03/powerplant_units.csv (no fuel price, additional_cost 0).
DEFAULT_RENEWABLE_FUEL_PRICES: dict[str, float] = {
    "solar": 0.0,
    "wind_onshore": 0.0,
    "wind_offshore": 0.0,
}

# Power-to-electricity efficiency (MWh_el per MWh of fuel input) for thermal
# technologies: the median efficiency per fuel in
# examples/inputs/example_03/powerplant_units.csv. ASSUME prices fuel in
# EUR/MWh_thermal and converts via power / efficiency (see
# PowerPlant.calc_marginal_cost_with_partial_eff), so without it a coal price
# would bid as EUR/MWh_e instead of EUR/MWh_th / efficiency. Nuclear is
# intentionally absent: the loader bids it as fixed electricity marginal costs,
# not as a thermal fuel price.
DEFAULT_EFFICIENCIES: dict[str, float] = {
    "hard coal": 0.40,
    "lignite": 0.37,
    "gas": 0.43,
    "oil": 0.31,
}

# Storage defaults from the pumped-hydro fleet in
# examples/inputs/example_03/storage_units.csv (25 units): hours = total
# capacity / total discharge power (5.88), efficiencies = medians (0.84 / 0.89),
# additional cost = the value shared by all units, SoC limits = 0 and 1 for all
# units. The file has no initial SoC; 0.5 is the ASSUME Storage default for it.
DEFAULT_STORAGE_HOURS = 5.9
DEFAULT_STORAGE_ADDITIONAL_COST = 0.28
DEFAULT_STORAGE_INITIAL_SOC = 0.5
DEFAULT_STORAGE_MAX_SOC = 1.0
DEFAULT_STORAGE_MIN_SOC = 0.0
DEFAULT_STORAGE_EFFICIENCY_CHARGE = 0.84
DEFAULT_STORAGE_EFFICIENCY_DISCHARGE = 0.89


def default_storage_params() -> dict[str, float]:
    """Return the default storage unit parameters (see constants above)."""
    return {
        "hours": DEFAULT_STORAGE_HOURS,
        "additional_cost": DEFAULT_STORAGE_ADDITIONAL_COST,
        "initial_soc": DEFAULT_STORAGE_INITIAL_SOC,
        "max_soc": DEFAULT_STORAGE_MAX_SOC,
        "min_soc": DEFAULT_STORAGE_MIN_SOC,
        "efficiency_charge": DEFAULT_STORAGE_EFFICIENCY_CHARGE,
        "efficiency_discharge": DEFAULT_STORAGE_EFFICIENCY_DISCHARGE,
    }


# fossil keys pay the CO2 price; other keys get no CO2 cost
FOSSIL_BIDDING_KEYS = {"hard coal", "lignite", "oil", "gas"}
THERMAL_BIDDING_KEYS = FOSSIL_BIDDING_KEYS | {"nuclear"}

# tCO2 per MWh of fuel input (thermal): the emission_factor per fuel in
# examples/inputs/example_03/powerplant_units.csv, zero for non-fossil units. ASSUME converts to
# tCO2/MWh_e via power / efficiency in the marginal cost calculation.
DEFAULT_EMISSION_FACTORS: dict[str, float] = {
    "hard coal": 0.335,
    "lignite": 0.406,
    "gas": 0.201,
    "oil": 0.776,
    "nuclear": 0.0,
    "biomass": 0.0,
    "hydro": 0.0,
    "wind": 0.0,
    "solar": 0.0,
    "storage": 0.0,
}


def split_capacity_blocks(capacity_mw: float, block_size_mw: float) -> list[float]:
    """Split capacity into full blocks plus one smaller remainder block."""
    if capacity_mw <= 0:
        return []
    if block_size_mw <= 0:
        return [capacity_mw]

    full_blocks = int(capacity_mw // block_size_mw)
    blocks = [block_size_mw] * full_blocks
    remainder = capacity_mw % block_size_mw
    if remainder > 0:
        blocks.append(remainder)
    if not blocks:
        blocks = [capacity_mw]
    return blocks


def interpolate_block_prices(
    n_blocks: int,
    price_high: float,
    price_low: float,
) -> list[float]:
    """Return prices falling linearly from price_high (first block) to price_low."""
    if n_blocks <= 0:
        return []
    if n_blocks == 1:
        return [price_high]
    step = (price_high - price_low) / (n_blocks - 1)
    return [price_high - i * step for i in range(n_blocks)]


def block_price_factors(
    n_blocks: int, price_high: float, price_low: float
) -> list[float]:
    """Return multipliers for a base price series, highest block first."""
    block_prices = interpolate_block_prices(n_blocks, price_high, price_low)
    # the mid-point is the reference, so the average block bids at the base price
    base = (price_high + price_low) / 2
    if base <= 0:
        return [1.0] * n_blocks
    return [price / base for price in block_prices]
