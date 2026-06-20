"""Test whether tightening big_M from 10e6 to 10000 speeds up solve."""

import time
import pandas as pd
import numpy as np

from bilevel_opt import find_optimal_dispatch_quadratic
from uc_problem import solve_uc_problem

# ---------- load data ----------
data_dir = "mpec_input_data_02a"

gens_df = pd.read_csv(f"{data_dir}/gens_df.csv", index_col=0)
k_values_df = pd.read_csv(f"{data_dir}/k_values_df.csv", index_col=0, parse_dates=True)
k_values_df = k_values_df.fillna(0.0)
demand_df = pd.read_csv(f"{data_dir}/demand_df.csv", index_col=0, parse_dates=True)
availability_df = pd.read_csv(f"{data_dir}/availability_df.csv", index_col=0, parse_dates=True)

# ---------- filter to 2019-03-16 ----------
date_mask_k = (k_values_df.index >= "2019-03-16") & (k_values_df.index < "2019-03-17")
date_mask_d = (demand_df.index >= "2019-03-16") & (demand_df.index < "2019-03-17")
date_mask_a = (availability_df.index >= "2019-03-16") & (availability_df.index < "2019-03-17")

k_values_df = k_values_df.loc[date_mask_k].reset_index(drop=True)
demand_df = demand_df.loc[date_mask_d].reset_index(drop=True)
availability_df = availability_df.loc[date_mask_a].reset_index(drop=True)

# drop non-generator columns (like 'date')
gen_cols = [c for c in gens_df["unit"].values if c in k_values_df.columns]
k_values_df = k_values_df[gen_cols]
avail_cols = [c for c in gens_df["unit"].values if c in availability_df.columns]
availability_df = availability_df[avail_cols]

# ---------- make demand volumes positive ----------
vol_cols = [c for c in demand_df.columns if c.startswith("volume_")]
demand_df[vol_cols] = demand_df[vol_cols].abs().clip(lower=0.01)

# drop 'date' column from demand if present
if "date" in demand_df.columns:
    demand_df = demand_df.drop(columns=["date"])

# ---------- auto-detect n_demand_bids ----------
n_demand_bids = len(vol_cols)
print(f"n_demand_bids: {n_demand_bids}")

# ---------- pick unit pp_3 ----------
gens_df_indexed = gens_df.set_index("unit")
opt_gen = "pp_3"
k_max = gens_df_indexed.loc[opt_gen, "mc"]  # mc=31.15 
k_max = 4
mc = gens_df_indexed.loc[opt_gen, "mc"]
print(f"opt_gen: {opt_gen}, mc: {mc}, k_max: {k_max}")

# ---------- build mc_df ----------
mc_df = pd.DataFrame(
    {gen: gens_df_indexed.at[gen, "mc"] for gen in gens_df_indexed.index},
    index=demand_df.index,
)

print(f"demand_df shape: {demand_df.shape}")
print(f"k_values_df shape: {k_values_df.shape}")
print(f"availability_df shape: {availability_df.shape}")
print(f"gens_df shape: {gens_df.shape}")
print(f"Timesteps: {len(demand_df)}")
print()

# ---------- solve ----------
print("=" * 60)
print("Solving with big_M=9000 (tight) ...")
print("=" * 60)

t0 = time.time()
result = find_optimal_dispatch_quadratic(
    gens_df=gens_df,
    k_values_df=k_values_df,
    availabilities_df=availability_df,
    demand_df=demand_df,
    k_max=k_max,
    opt_gen=opt_gen,
    big_w=1,
    time_limit=6000,
    big_M=9000,
    demand_bids=n_demand_bids,
    mc_df=mc_df,
)
elapsed = time.time() - t0

main_df, supp_df, k_values = result

print()
print("=" * 60)
print(f"Solve time: {elapsed:.2f} seconds")

if main_df is not None:
    print("Status: SOLUTION FOUND")
    # mcp (lambda) range
    if "mcp" in main_df.columns:
        print(f"MCP range: [{main_df['mcp'].min():.2f}, {main_df['mcp'].max():.2f}]")
    elif "lambda" in main_df.columns:
        print(f"Lambda range: [{main_df['lambda'].min():.2f}, {main_df['lambda'].max():.2f}]")
    # k range
    if k_values is not None:
        if isinstance(k_values, pd.DataFrame):
            k_col = k_values[opt_gen] if opt_gen in k_values.columns else k_values.iloc[:, 0]
        elif isinstance(k_values, pd.Series):
            k_col = k_values
        else:
            k_col = pd.Series(k_values)
        print(f"k range: [{k_col.min():.4f}, {k_col.max():.4f}]")
    print(f"\nmain_df columns: {list(main_df.columns)}")
    print(main_df.head())
else:
    print("Status: INFEASIBLE / NO SOLUTION")
    
if main_df is not None:
    k_values_uc = k_values_df.copy()
    k_series = pd.to_numeric(k_values["k"], errors="coerce")
    fallback = pd.to_numeric(k_values_uc[opt_gen], errors="coerce").fillna(1.0)
    k_values_uc[opt_gen] = k_series.fillna(fallback).astype(float).values
    for col in k_values_uc.columns:
        k_values_uc[col] = pd.to_numeric(k_values_uc[col], errors="coerce").fillna(1.0)

    t0 = time.time()
    uc_main, uc_supp = solve_uc_problem(
        gens_df=gens_df,
        demand_df=demand_df,
        k_values_df=k_values_uc,
        availabilities_df=availability_df,
        demand_bids=n_demand_bids,
        mc_df=mc_df,
        fixed_storage_dispatch=None,
    )
    t_uc = time.time() - t0
    print(f"UC solved in {t_uc:.1f}s")
    print(f"  UC MCP: {uc_main['mcp'].astype(float).min():.1f} to {uc_main['mcp'].astype(float).max():.1f}")

print("=" * 60)
