# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Populate the aFRR (CRM) reserve-demand columns of ``demand_df.csv`` from the real
German ENTSO-E 2021-2025 average profiles.

Source data (outside this repo), produced by the pypsa_de_zenodo ENTSO-E pipeline:
  - aFRR capacity: ``afrr_capacity_DE_2021_2025_average_profile_2025.csv``
        4380 rows = 365 days x 6 four-hour blocks x 2 directions (UP/DOWN).
        We use ``average_awarded_to_german_bsps_MW``.
  - aFRR energy:   ``afrr_activated_energy_DE_2021_2025_average_profile_2025.csv``
        70080 rows = 365 days x 96 quarter-hours x 2 directions (UP/DOWN).
        We use ``average_activated_quantity_MW``.

Mapping onto the 2030 simulation (hourly, 8760 rows):
  - capacity: repeat each 4h-block value 4x -> 8760 hourly values;
  - energy:   average each hour's four 15-min values -> 8760 hourly values;
  - both source years and 2030 are 365-day years, so the 8760-length series is
    assigned positionally (day-of-year + hour-of-day aligned). Direction UP -> the
    ``_pos`` column, DOWN -> the ``_neg`` column. Values stay positive MW, matching
    the other demand columns.

Re-run this script whenever the source profiles change:
    python build_reserve_demand.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Location of the ENTSO-E average-profile CSVs (adjust if the data moves).
ENTSOE_DIR = Path(r"C:\Users\khm\Models\pypsa_de_zenodo\data\ENTSOE")
CAPACITY_CSV = (
    ENTSOE_DIR / "aFRR capacity" / "afrr_capacity_DE_2021_2025_average_profile_2025.csv"
)
ENERGY_CSV = (
    ENTSOE_DIR
    / "aFRR energy"
    / "afrr_activated_energy_DE_2021_2025_average_profile_2025.csv"
)

DEMAND_DF = Path(__file__).parent / "demand_df.csv"

# maps demand_df reserve column -> (source, direction)
COLUMNS = {
    "demand_CRM_capacity_pos": ("capacity", "UP"),
    "demand_CRM_capacity_neg": ("capacity", "DOWN"),
    "demand_CRM_energy_pos": ("energy", "UP"),
    "demand_CRM_energy_neg": ("energy", "DOWN"),
}


def _capacity_series(cap: pd.DataFrame, direction: str) -> np.ndarray:
    """4h awarded-capacity blocks -> hourly (repeat each block value 4x)."""
    block = cap[cap["direction"] == direction].sort_values("delivery_start_utc")
    return np.repeat(block["average_awarded_to_german_bsps_MW"].to_numpy(), 4)


def _energy_series(en: pd.DataFrame, direction: str) -> np.ndarray:
    """15-min activated energy -> hourly mean MW (= the hourly MWh procured)."""
    q = en[en["direction"] == direction].sort_values("delivery_start_utc")
    return q["average_activated_quantity_MW"].to_numpy().reshape(-1, 4).mean(axis=1)


def main() -> None:
    cap = pd.read_csv(CAPACITY_CSV)
    en = pd.read_csv(ENERGY_CSV)

    demand_df = pd.read_csv(DEMAND_DF, index_col=0)
    n = len(demand_df)

    for column, (source, direction) in COLUMNS.items():
        if source == "capacity":
            series = _capacity_series(cap, direction)
        else:
            series = _energy_series(en, direction)

        if len(series) != n:
            raise ValueError(
                f"{column}: built {len(series)} hourly values but demand_df has {n} rows"
            )
        demand_df[column] = series

    demand_df.to_csv(DEMAND_DF)

    stats = demand_df[list(COLUMNS)].agg(["min", "mean", "max"]).round(1)
    print(f"Updated {DEMAND_DF} ({n} rows)")
    print(stats.to_string())


if __name__ == "__main__":
    main()
