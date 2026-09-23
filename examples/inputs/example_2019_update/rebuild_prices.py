# SPDX-FileCopyrightText: ASSUME Developers
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Rebuild the researched fuel prices offline from the recorded source values.

Run from any directory. Only this scenario's fuel_prices_df.csv is written.
The original example_03 supplies the retained fuel and CO2 assumptions.
"""

from pathlib import Path

import pandas as pd


def main():
    folder = Path(__file__).resolve().parent
    original = pd.read_csv(
        folder.parent / "example_03/fuel_prices_df.csv", index_col=0, parse_dates=True
    )
    monthly = pd.read_csv(folder / "price_sources/gas_monthly_2019.csv").set_index("month")
    annual = pd.read_csv(folder / "price_sources/annual_benchmarks_2019.csv").set_index("fuel")

    # Gas exchange quotations use gross calorific value; the scenario adopts
    # net calorific value for thermal efficiency and fuel input (AGEB: 0.9024).
    # Each observed monthly mean is held constant, without inventing daily prices.
    original["natural gas"] = (
        original.index.month.map(monthly["natural_gas_eur_per_mwh_hhv"]).to_numpy()
        / 0.9024
    )
    # A tonne of coal equivalent is an energy unit, NOT a physical tonne of coal.
    mwh_per_tce = 29.3076 / 3.6
    for fuel in ("hard coal", "oil"):
        original[fuel] = annual.at[fuel, "price"] / mwh_per_tce

    # Boundary values serve simulation initialization / the terminal state only.
    index = pd.date_range("2018-12-31 11:00", "2020-01-01 00:45", freq="15min")
    prices = original.reindex(index).ffill().bfill()
    prices.index.name = "datetime"
    prices.to_csv(folder / "fuel_prices_df.csv")


if __name__ == "__main__":
    main()
