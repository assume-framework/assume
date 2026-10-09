# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plot accepted EV volumes from market_orders.csv.

Upper half: accepted volume on EOM (positive).
Lower half: accepted volume on GridTariff (drawn downwards).

Usage: python plot_ev_market_orders.py path/to/market_orders.csv [output.png]
"""

import sys

import matplotlib.pyplot as plt
import pandas as pd


def plot_ev_market_orders(csv_path: str, output: str | None = None):
    df = pd.read_csv(csv_path, parse_dates=["start_time"])
    df = df[df["unit_id"].astype(str).str.contains("EV", na=False)]

    # sum accepted volume per timestep and market; absolute values so both halves are positive magnitudes
    vol = (
        df[df["market_id"].isin(["EOM", "GridTariff"])]
        .groupby(["start_time", "market_id"])["accepted_volume"]
        .sum()
        .abs()
        .unstack("market_id")
        .reindex(columns=["EOM", "GridTariff"])
        .fillna(0)
        .sort_index()
    )

    if len(vol) > 1:
        width = (vol.index[1] - vol.index[0]) * 0.9
    else:
        width = pd.Timedelta(minutes=54)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(vol.index, vol["EOM"], width=width, align="edge", label="EOM", color="tab:blue")
    ax.bar(
        vol.index,
        -vol["GridTariff"],
        width=width,
        align="edge",
        label="GridTariff",
        color="tab:orange",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Accepted volume [MW]")
    ax.set_title("Accepted volume of EV units: EOM (up) vs. GridTariff (down)")
    ax.legend()
    # show lower-half ticks as positive magnitudes
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels([f"{abs(t):g}" for t in ax.get_yticks()])
    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    plot_ev_market_orders(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
