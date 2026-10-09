# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plot aggregated EV costs per tariff case.

One small subplot per tariff case, each with three curves:
EOM costs (EOM accepted_volume * accepted_price), GridTariff costs
(GridTariff accepted_price * EOM accepted_volume), their sum and the
optimizer's expected total (planned EOM volume * forecast EOM price + GridTariff price).

Usage: python plot_ev_market_costs.py [test_case_prefix] [output.png]
Default prefix: test_case_5 (folders <prefix>_<tariff>_ev).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "inputs" / "compare_ev_and_building" / "output_data"
SCENARIO_DIR = Path(__file__).parent / "inputs" / "compare_ev_and_building" / "scenario_data" / "study_cases"
TARIFFS = ["no_tariff", "flat_tariff", "tou_tariff", "capacity_tariff"]


def load_costs(csv_path: Path, forecast_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["start_time"])
    df = df[df["unit_id"].astype(str).str.contains("EV", na=False)]
    key = ["start_time", "unit_id"]

    eom = df[df["market_id"] == "EOM"].groupby(key)[["accepted_volume", "accepted_price"]].sum()
    tariff_price = df[df["market_id"] == "GridTariff"].groupby(key)["accepted_price"].sum()

    # grid tariff costs: GridTariff accepted_price * EOM accepted_volume
    per_unit = pd.DataFrame(
        {
            "EOM": eom["accepted_volume"] * eom["accepted_price"],
            "GridTariff": tariff_price * eom["accepted_volume"],
        }
    ).fillna(0)
    costs = per_unit.groupby("start_time").sum().sort_index()
    costs["Sum"] = costs["EOM"] + costs["GridTariff"]

    # Optimizer expectation: planned volume (= EOM bid volume) valued with the
    # EOM price forecast and the cleared GridTariff price.
    planned = df[df["market_id"] == "EOM"].groupby(key)["volume"].sum()
    forecast = pd.read_csv(forecast_path, index_col="datetime", parse_dates=True)["price_EOM"]
    start = planned.index.get_level_values("start_time")
    expected = planned * forecast.reindex(start).to_numpy() + planned * tariff_price.reindex(
        planned.index
    ).fillna(0)
    costs["Expected"] = expected.groupby("start_time").sum().reindex(costs.index).fillna(0)
    return costs

def plot_ev_market_costs(prefix: str = "test_case_5", output: str | None = None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for ax, tariff in zip(axes.flat, TARIFFS):
        path = OUTPUT_DIR / f"{prefix}_{tariff}_ev" / "market_orders.csv"
        if not path.exists():
            ax.set_title(f"{tariff} (missing)", fontsize=9)
            continue
        forecast_path = SCENARIO_DIR / f"{prefix}_ev" / "forecasts_df.csv"
        costs = load_costs(path, forecast_path)
        ax.plot(costs.index, costs["EOM"], label="EOM", lw=1)
        ax.plot(costs.index, costs["GridTariff"], label="GridTariff", lw=1)
        ax.plot(costs.index, costs["Sum"], label="Sum", lw=1.2, color="black", ls="--")
        """ax.plot(costs.index, costs["Expected"], label="Sum (optimizer expected)", lw=1.2, color="tab:red", ls=":")"""
        ax.set_title(tariff, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel("EV costs [€]", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    fig.autofmt_xdate()
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    plot_ev_market_costs(
        sys.argv[1] if len(sys.argv) > 1 else "test_case_5",
        sys.argv[2] if len(sys.argv) > 2 else None,
    )
