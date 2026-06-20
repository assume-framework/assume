# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Solve quadratic MPEC with fixed storage dispatch for every learning unit
(solar + wind), then UC re-solve for each.

Pipeline:
  1. Load exported data from ``mpec_input_data/`` (produced by
     ``12_eval_futur_markets_data.ipynb``).
  2. Optionally filter to a date range (set ``SOLVE_START`` / ``SOLVE_END``).
  3. Fold storage dispatch into demand so storage is exogenous.
  4. For each learning unit: solve MPEC -> UC re-solve -> collect prices.
  5. Save CSVs + plot to ``results/``.
"""

import matplotlib

matplotlib.use("Agg")

import time

import matplotlib.pyplot as plt
import pandas as pd
from bilevel_opt import find_optimal_dispatch_quadratic_fixed_storage
from uc_problem import solve_uc_problem

# ---------------------------------------------------------------------------
# Configuration: set to None to solve all exported dates
# ---------------------------------------------------------------------------
SOLVE_START = "2013-01-28"  # e.g. "2013-01-28", or None for all
SOLVE_END = "2013-01-28"  # e.g. "2013-01-28", or None for all

out_dir = "mpec_input_data"

gens_df = pd.read_csv(f"{out_dir}/gens_df.csv", index_col=0)
k_values_df = pd.read_csv(
    f"{out_dir}/k_values_df.csv", index_col=0, parse_dates=True
).fillna(0.0)
availability_df = pd.read_csv(
    f"{out_dir}/availability_df.csv", index_col=0, parse_dates=True
)
demand_df = pd.read_csv(f"{out_dir}/demand_df.csv", index_col=0, parse_dates=True)
for col in demand_df.columns:
    if col.startswith("volume"):
        demand_df[col] = -demand_df[col]
        demand_df[col] = demand_df[col].clip(lower=0.01)

# Optional date-range filter
if SOLVE_START and SOLVE_END:
    mask = (k_values_df.index >= SOLVE_START) & (k_values_df.index <= pd.Timestamp(SOLVE_END) + pd.Timedelta(days=1))
    k_values_df = k_values_df[mask]
    demand_df = demand_df[mask]
    availability_df = availability_df[mask]
    print(f"Filtered to {SOLVE_START} -- {SOLVE_END}: {len(demand_df)} timesteps")

dispatch_df = pd.read_csv(f"{out_dir}/dispatch_df.csv", index_col=0)
dispatch_df["time"] = pd.to_datetime(dispatch_df["time"])
storage_units = dispatch_df[dispatch_df["soc"].notna()]["unit"].unique()
storage_dispatch = dispatch_df[dispatch_df["unit"].isin(storage_units)]
storage_pivot = storage_dispatch.pivot_table(
    index="time", columns="unit", values="power", aggfunc="first"
).fillna(0.0)
net_storage = storage_pivot.sum(axis=1)

k_times = k_values_df.index
net_storage_aligned = net_storage.reindex(k_times).fillna(0.0)
net_storage_aligned.index = range(len(k_times))

gens_df_idx = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df.copy()
demand_df_r = demand_df.reset_index(drop=True).copy()
k_values_r = k_values_df.reset_index(drop=True).copy()
avail_r = availability_df.reset_index(drop=True).copy()
mc_df = pd.DataFrame(
    {gen: gens_df_idx.at[gen, "mc"] for gen in gens_df_idx.index},
    index=demand_df_r.index,
)

# Fold storage dispatch into demand (subtract from volume_1 at 3000 €/MWh price)
# net_storage > 0 = discharge (reduces demand), < 0 = charge (increases demand)
demand_df_r["volume_1"] = demand_df_r["volume_1"] - net_storage_aligned.values
demand_df_r["volume_1"] = demand_df_r["volume_1"].clip(lower=0.01)
print(
    f"Adjusted demand by storage dispatch: min={net_storage_aligned.min():.0f}, max={net_storage_aligned.max():.0f} MW"
)

learning_units = [u for u in gens_df_idx.index if "solar" in u or "wind" in u]
print(f"Solving MPEC + UC for {len(learning_units)} learning units")

res_lambda = {}
res_lambda_hat = {}
res_uc_mcp = {}

for i, opt_gen in enumerate(learning_units):
    mc_val = mc_df[opt_gen].iloc[0]
    k_max = max(int(3100 / mc_val), 10) if mc_val > 0 else 300000
    print(
        f"\n[{i + 1}/{len(learning_units)}] {opt_gen} (mc={mc_val:.4f}, k_max={k_max})"
    )

    t0 = time.time()
    main_df, supp_df, k_vals = find_optimal_dispatch_quadratic_fixed_storage(
        gens_df=gens_df_idx,
        k_values_df=k_values_r,
        availabilities_df=avail_r,
        demand_df=demand_df_r,
        k_max=k_max,
        opt_gen=opt_gen,
        fixed_storage_dispatch=None,
        big_w=10000,
        time_limit=120,
        big_M=10e6,
        demand_bids=10,
        mc_df=mc_df,
    )
    t_mpec = time.time() - t0

    if main_df is None:
        print(f"  MPEC INFEASIBLE ({t_mpec:.1f}s)")
        continue

    res_lambda[opt_gen] = main_df["mcp"].astype(float)
    res_lambda_hat[opt_gen] = main_df["mcp_hat"].astype(float)

    # UC re-solve: update k_values with MPEC result, then solve UC
    k_values_uc = k_values_r.copy()
    k_series = pd.to_numeric(k_vals["k"], errors="coerce")
    fallback = pd.to_numeric(k_values_uc[opt_gen], errors="coerce").fillna(1.0)
    k_values_uc[opt_gen] = k_series.fillna(fallback).astype(float).values
    for col in k_values_uc.columns:
        k_values_uc[col] = pd.to_numeric(k_values_uc[col], errors="coerce").fillna(1.0)

    t0 = time.time()
    uc_main, uc_supp = solve_uc_problem(
        gens_df=gens_df_idx,
        demand_df=demand_df_r,
        k_values_df=k_values_uc,
        availabilities_df=avail_r,
        demand_bids=10,
        mc_df=mc_df,
        fixed_storage_dispatch=None,
    )
    t_uc = time.time() - t0

    res_uc_mcp[opt_gen] = uc_main["mcp"].astype(float)
    print(f"  OK (MPEC {t_mpec:.1f}s, UC {t_uc:.1f}s)")
    print(
        f"    lambda_: {res_lambda[opt_gen].min():.1f} to {res_lambda[opt_gen].max():.1f}"
    )
    print(
        f"    lambda_hat: {res_lambda_hat[opt_gen].min():.1f} to {res_lambda_hat[opt_gen].max():.1f}"
    )
    print(
        f"    UC MCP: {res_uc_mcp[opt_gen].min():.1f} to {res_uc_mcp[opt_gen].max():.1f}"
    )

print(f"\n{'=' * 60}")
print(f"Solved {len(res_lambda)}/{len(learning_units)} successfully")

if not res_lambda:
    print("No results to plot.")
    exit()

# Save results to CSV
all_lambda = pd.DataFrame(res_lambda)
all_lambda_hat = pd.DataFrame(res_lambda_hat)
all_uc = pd.DataFrame(res_uc_mcp)
all_lambda.to_csv("results/results_lambda_storageindemand.csv")
all_lambda_hat.to_csv("results/results_lambda_hat_storageindemand.csv")
all_uc.to_csv("results/results_uc_mcp_storageindemand.csv")
print(
    "Saved results to results/ directory"
)

# --- PLOT ---
fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

titles = [
    "lambda_ (MPEC non-hat dual)",
    "lambda_hat (MPEC hat dual / KKT relaxation)",
    "UC Re-Solve MCP (actual market price)",
]
data_dicts = [res_lambda, res_lambda_hat, res_uc_mcp]

for ax, title, data in zip(axes, titles, data_dicts):
    for unit_name, series in data.items():
        ls = "-" if "wind" in unit_name else "--"
        ax.plot(
            series.index,
            series.values,
            label=unit_name,
            alpha=0.7,
            lw=1.2,
            linestyle=ls,
        )
    ax.set_ylabel("EUR/MWh")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=2)

axes[-1].set_xlabel("Hour")
plt.tight_layout()
plt.savefig("results/mcp_all_learning_units_storageindemand.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: results/mcp_all_learning_units_storageindemand.png")
