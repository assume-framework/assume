# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plot the aggregated load procured by the EVs on EOM, one line per tariff case.

Usage: python plot_ev_procured_load.py [test_case_prefix] [output.png]
Default prefix: test_case_5 (folders <prefix>_<tariff>_ev).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "inputs" / "compare_ev_and_building" / "output_data"
SCENARIO_DIR = Path(__file__).parent / "inputs" / "compare_ev_and_building" / "scenario_data" / "study_cases"
TARIFFS = ["no_tariff", "flat_tariff", "tou_tariff", "capacity_tariff"]


def load_procured(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path, parse_dates=["start_time"])
    df = df[df["unit_id"].astype(str).str.contains("EV", na=False)]
    df = df[df["market_id"] == "EOM"]
    # EV demand is a negative accepted volume; report the load as positive MW
    return -df.groupby("start_time")["accepted_volume"].sum().sort_index()


def load_eom_price(prefix: str) -> pd.Series:
    # the EOM price forecast is identical for all tariff cases
    path = SCENARIO_DIR / f"{prefix}_ev" / "forecasts_df.csv"
    return pd.read_csv(path, index_col="datetime", parse_dates=True)["price_EOM"]


def plot_ev_procured_load(prefix: str = "test_case_5", output: str | None = None):
    fig, ax = plt.subplots(figsize=(10, 4))
    styles = ["-", "--", ":", "-."]
    widths = [3.0, 2.2, 1.6, 1.2]  # thinner on top so overlapping lines stay visible
    price = load_eom_price(prefix)
    for tariff, ls, lw in zip(TARIFFS, styles, widths):
        path = OUTPUT_DIR / f"{prefix}_{tariff}_ev" / "market_orders.csv"
        if not path.exists():
            continue
        load = load_procured(path)
        ax.plot(load.index, load, label=tariff, lw=lw, ls=ls, drawstyle="steps-post")
    ax.set_ylabel("Procured EV load [MW]", fontweight="bold")
    ax.grid(alpha=0.3)

    # EOM price on a secondary axis with ticks on the right
    ax2 = ax.twinx()
    ax2.plot(
        price.index,
        price,
        color="gray",
        lw=1,
        alpha=0.8,
        label="EOM price",
    )
    ax2.set_ylabel("EOM price [EUR/MWh]", fontweight="bold")
    # draw the load lines on top of the price line
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=5, fontsize=8)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)  # leave room for the legend
    ax.set_xlim(left=price.index[0] + pd.Timedelta(hours=3))  # cut the first 3 hours
    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    plot_ev_procured_load(
        sys.argv[1] if len(sys.argv) > 1 else "test_case_5",
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
