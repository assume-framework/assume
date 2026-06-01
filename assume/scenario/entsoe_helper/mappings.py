# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass

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

VALIDATED_PSR_FROM_API = {
    "Biomass",
    "Fossil Brown coal/Lignite",
    "Fossil Coal-derived gas",
    "Fossil Gas",
    "Fossil Hard coal",
    "Fossil Oil",
    "Fossil Oil shale",
    "Fossil Peat",
    "Geothermal",
    "Hydro Pumped Storage",
    "Hydro Run-of-river and poundage",
    "Hydro Water Reservoir",
    "Marine",
    "Nuclear",
    "Other",
    "Other renewable",
    "Solar",
    "Waste",
    "Wind Offshore",
    "Wind Onshore",
}


@dataclass(frozen=True)
class TechnologyMapping:
    technology: str
    bidding_key: str
    variable: bool
    unit_type: str = "power_plant"


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
    "Hydro Run-of-river and poundage": TechnologyMapping("hydro", "hydro", False),
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

DEFAULT_BLOCK_SIZES_MW: dict[str, float] = {
    "hard coal": 500.0,
    "lignite": 500.0,
    "gas": 400.0,
    "oil": 200.0,
    "nuclear": 1000.0,
    "hydro": 300.0,
    "biomass": 200.0,
    "geothermal": 100.0,
    "marine": 100.0,
    "other": 200.0,
    "other_renewable": 200.0,
    "waste": 200.0,
    "hydro_storage": 250.0,
    "battery_storage": 50.0,
}

# spread around the instrat base price for block interpolation
DEFAULT_FUEL_PRICE_RANGES: dict[str, tuple[float, float]] = {
    "hard coal": (13.0, 10.0),
    "lignite": (3.0, 2.0),
    "gas": (32.0, 22.0),
    "oil": (25.0, 18.0),
    "biomass": (22.0, 18.0),
    "hydro": (0.4, 0.1),
    "geothermal": (15.0, 10.0),
    "marine": (0.4, 0.1),
    "other": (25.0, 15.0),
    "other_renewable": (15.0, 8.0),
    "waste": (22.0, 18.0),
    "nuclear": (9.0, 7.0),
}

DEFAULT_RENEWABLE_FUEL_PRICES: dict[str, float] = {
    "solar": 0.1,
    "wind_onshore": 0.2,
    "wind_offshore": 0.2,
}

DEFAULT_STORAGE_HOURS = 8.0
DEFAULT_STORAGE_ADDITIONAL_COST = 0.28

FOSSIL_BIDDING_KEYS = {"hard coal", "lignite", "oil", "gas"}
THERMAL_BIDDING_KEYS = FOSSIL_BIDDING_KEYS | {"nuclear"}

# tCO2/MWh_el, aligned with examples/inputs/example_03/powerplant_units.csv
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
    base = (price_high + price_low) / 2
    if base <= 0:
        return [1.0] * n_blocks
    return [price / base for price in block_prices]
