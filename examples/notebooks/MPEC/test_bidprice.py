# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test bid-price MPEC with fixed storage dispatch from MADRL learning.

Runs the full pipeline: MPEC → UC re-solve → profit calculation.
Storage dispatch is taken as-is from dispatch_df.csv and enters the
energy balance as a fixed parameter.
"""

import time

import pandas as pd
from utils import run_MPEC_bidprice

out_dir = "mpec_input_data"

# -------------------------------------------------------------------------
# 1. Load generator & market data
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 2. Load and prepare fixed storage dispatch
# -------------------------------------------------------------------------
dispatch_df = pd.read_csv(f"{out_dir}/dispatch_df.csv", index_col=0)
dispatch_df["time"] = pd.to_datetime(dispatch_df["time"])

# Identify storage units (those with soc column populated)
storage_units_in_dispatch = dispatch_df[dispatch_df["soc"].notna()]["unit"].unique()
storage_dispatch = dispatch_df[dispatch_df["unit"].isin(storage_units_in_dispatch)]

# Pivot: one column per storage, rows = time, values = power
storage_pivot = storage_dispatch.pivot_table(
    index="time", columns="unit", values="power", aggfunc="first"
).fillna(0.0)

# Net storage dispatch: sum across all storages per hour
# positive = net discharge (generation), negative = net charge (demand)
net_storage = storage_pivot.sum(axis=1)

print("=== Fixed storage dispatch ===")
print(f"Storage units: {len(storage_units_in_dispatch)}")
print(f"Time range: {net_storage.index.min()} to {net_storage.index.max()}")
n_charge = (net_storage < 0).sum()
n_discharge = (net_storage > 0).sum()
n_idle = (net_storage == 0).sum()
print(f"Hours charging: {n_charge}, discharging: {n_discharge}, idle: {n_idle}")
print(
    f"Max charge: {net_storage.min():.0f} MW, max discharge: {net_storage.max():.0f} MW"
)

# -------------------------------------------------------------------------
# 3. Align timeframes
# -------------------------------------------------------------------------
k_times = pd.read_csv(f"{out_dir}/k_values_df.csv", index_col=0, parse_dates=True).index

print(f"\nMPEC timeframe: {k_times[0]} to {k_times[-1]} ({len(k_times)} hours)")

net_storage_aligned = net_storage.reindex(k_times).fillna(0.0)
net_storage_aligned.index = range(len(k_times))

print("Storage dispatch in MPEC window:")
for t_int, t_dt in zip(range(len(k_times)), k_times):
    val = net_storage_aligned.at[t_int]
    tag = "GEN" if val > 0 else ("LOAD" if val < 0 else "---")
    print(f"  {t_dt} (t={t_int}): {val:8.1f} MW  [{tag}]")

# -------------------------------------------------------------------------
# 4. Run full MPEC pipeline (MPEC → UC re-solve → profits)
# -------------------------------------------------------------------------
opt_name = "wind_onshore_01"

print(f"\n{'=' * 60}")
print(f"Running MPEC pipeline: {len(k_times)} timesteps")
print(f"Strategic gen: {opt_name}")
print(f"{'=' * 60}")

t0 = time.time()
profits_1, profits_2, uc_main_df, uc_supp_df = run_MPEC_bidprice(
    opt_gen=opt_name,
    gens_df=gens_df,
    demand_df=demand_df,
    k_values_df=k_values_df,
    availability_df=availability_df,
    big_w=10000,
    fixed_storage_dispatch=net_storage_aligned,
    bid_price_max=3100,
    demand_bids=10,
    time_limit=600,
    big_M=10000,
)
elapsed = time.time() - t0

print(f"\n{'=' * 60}")
print(f"DONE in {elapsed:.1f}s")
print(f"{'=' * 60}")

print("\nUC results (accurate market prices):")
print(uc_main_df[["demand", "mcp"]].to_string())

print("\nProfits from MPEC solution (profits_1):")
print(profits_1[[opt_name]].to_string())

print("\nProfits from UC re-solve (profits_2):")
print(profits_2[[opt_name]].to_string())

print(f"\nTotal profit {opt_name}:")
print(f"  MPEC:      {profits_1[opt_name].sum():.2f} €")
print(f"  UC re-solve: {profits_2[opt_name].sum():.2f} €")
