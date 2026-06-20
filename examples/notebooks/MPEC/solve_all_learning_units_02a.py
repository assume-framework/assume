# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Solve quadratic MPEC with no storage for every generator in
example_02a_elasticDemand_EE_Ramp, then UC re-solve for each.

This scenario has:
  - No storage units (no folding into demand needed)
  - 10 elastic demand bids (isoelastic demand model)
  - Ramping constraints on conventional generators

Pipeline:
  1. Load exported data from ``mpec_input_data_02a/`` (produced by
     ``12_eval_02a_elasticDemand_EE_Ramp.ipynb``).
  2. Optionally filter to a date range (set ``SOLVE_START`` / ``SOLVE_END``).
  3. For each generator: solve MPEC -> UC re-solve -> collect prices.
  4. Save CSVs + plot to ``results_02a/``.
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
SOLVE_START = "2019-03-02"  # e.g. "2019-03-02", or None for all
SOLVE_END = "2019-03-02"  # e.g. "2019-03-02", or None for all
MIN_MC_THRESHOLD = 0.1  # skip units with mc below this (renewables)

out_dir = "mpec_input_data_02a"

gens_df = pd.read_csv(f"{out_dir}/gens_df.csv", index_col=0)
k_values_df = pd.read_csv(
    f"{out_dir}/k_values_df.csv", index_col=0, parse_dates=True
).fillna(0.0)
availability_df = pd.read_csv(
    f"{out_dir}/availability_df.csv", index_col=0, parse_dates=True
)
demand_df = pd.read_csv(f"{out_dir}/demand_df.csv", index_col=0, parse_dates=True)

# Make demand volumes positive
for col in demand_df.columns:
    if col.startswith("volume"):
        demand_df[col] = demand_df[col].abs().clip(lower=0.01)

# Count demand bids
n_demand_bids = len([c for c in demand_df.columns if c.startswith("volume_")])
print(f"Demand bids per timestep: {n_demand_bids}")

# Optional date-range filter
if SOLVE_START and SOLVE_END:
    mask = (k_values_df.index >= SOLVE_START) & (
        k_values_df.index <= pd.Timestamp(SOLVE_END) + pd.Timedelta(days=1)
    )
    k_values_df = k_values_df[mask]
    demand_df = demand_df[mask]
    availability_df = availability_df[mask]
    print(f"Filtered to {SOLVE_START} -- {SOLVE_END}: {len(demand_df)} timesteps")

dispatch_df = pd.read_csv(f"{out_dir}/dispatch_df.csv", index_col=0)
dispatch_df["time"] = pd.to_datetime(dispatch_df["time"])

gens_df_idx = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df.copy()
gen_names = list(gens_df_idx.index)

demand_df_r = demand_df.reset_index(drop=True).copy()
# Drop date column if present
demand_df_r = demand_df_r.drop(columns=["date"], errors="ignore")

k_values_r = k_values_df.reset_index(drop=True).copy()
avail_r = availability_df.reset_index(drop=True).copy()

# Filter to generator columns only
k_values_r = k_values_r[[c for c in k_values_r.columns if c in gen_names]]
avail_r = avail_r[[c for c in avail_r.columns if c in gen_names]]

mc_df = pd.DataFrame(
    {gen: gens_df_idx.at[gen, "mc"] for gen in gen_names},
    index=demand_df_r.index,
)

# Select units to solve (skip renewables with near-zero mc)
solve_units = [
    u for u in gen_names if gens_df_idx.at[u, "mc"] >= MIN_MC_THRESHOLD
]
print(f"Solving MPEC + UC for {len(solve_units)} units (skipping renewables)")
print(f"Ramp constraints:")
for u in gen_names:
    print(f"  {u}: r_up={gens_df_idx.at[u, 'r_up']:.0f}, r_down={gens_df_idx.at[u, 'r_down']:.0f}, g_max={gens_df_idx.at[u, 'g_max']:.0f}")

res_lambda = {}
res_lambda_hat = {}
res_uc_mcp = {}

for i, opt_gen in enumerate(solve_units):
    mc_val = mc_df[opt_gen].iloc[0]
    k_max = max(int(3100 / mc_val), 10) if mc_val > 0 else 300000
    print(
        f"\n[{i + 1}/{len(solve_units)}] {opt_gen} (mc={mc_val:.4f}, k_max={k_max})"
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
        time_limit=600,
        big_M=10e6,
        demand_bids=n_demand_bids,
        mc_df=mc_df,
    )
    t_mpec = time.time() - t0

    if main_df is None:
        print(f"  MPEC INFEASIBLE ({t_mpec:.1f}s)")
        continue

    res_lambda[opt_gen] = main_df["mcp"].astype(float)
    res_lambda_hat[opt_gen] = main_df["mcp_hat"].astype(float)

    # UC re-solve: update k_values with MPEC result
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
        demand_bids=n_demand_bids,
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
print(f"Solved {len(res_lambda)}/{len(solve_units)} successfully")

if not res_lambda:
    print("No results to save.")
    exit()

# Save results
import os

os.makedirs("results_02a", exist_ok=True)

all_lambda = pd.DataFrame(res_lambda)
all_lambda_hat = pd.DataFrame(res_lambda_hat)
all_uc = pd.DataFrame(res_uc_mcp)
all_lambda.to_csv("results_02a/results_lambda.csv")
all_lambda_hat.to_csv("results_02a/results_lambda_hat.csv")
all_uc.to_csv("results_02a/results_uc_mcp.csv")
print("Saved results to results_02a/")

# --- PLOT ---
fig, axes = plt.subplots(3, 1, figsize=(14, 15), sharex=True)

titles = [
    "lambda_ (MPEC non-hat dual)",
    "lambda_hat (MPEC hat dual / KKT relaxation)",
    "UC Re-Solve MCP (market price with strategic bidding)",
]
data_dicts = [res_lambda, res_lambda_hat, res_uc_mcp]

for ax, title, data in zip(axes, titles, data_dicts):
    for unit_name, series in data.items():
        ax.plot(
            series.index, series.values,
            label=unit_name, alpha=0.7, lw=1.2,
        )
    ax.set_ylabel("EUR/MWh")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

axes[-1].set_xlabel("Hour")
plt.tight_layout()
plt.savefig("results_02a/mcp_all_units.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: results_02a/mcp_all_units.png")
