# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test bid-price MPEC with fixed storage dispatch from MADRL learning.

Storage dispatch is taken as-is from dispatch_df.csv and enters the
energy balance as a fixed parameter (net positive = generation, net
negative = demand). No storage optimization variables needed.
"""

import time

import pandas as pd
from bilevel_opt_bidprice import find_optimal_dispatch_bidprice

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

demand_df = pd.read_csv(
    f"{out_dir}/demand_df.csv", index_col=0, parse_dates=True
)
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
print(f"Max charge: {net_storage.min():.0f} MW, max discharge: {net_storage.max():.0f} MW")

# -------------------------------------------------------------------------
# 3. Align timeframes
# -------------------------------------------------------------------------
# The MPEC data (demand, k_values, availability) defines the timeframe.
mpec_times = demand_df.index  # integer index 0..N-1

# Get datetime index from k_values to map to dispatch times
k_times = pd.read_csv(
    f"{out_dir}/k_values_df.csv", index_col=0, parse_dates=True
).index

print(f"\nMPEC timeframe: {k_times[0]} to {k_times[-1]} ({len(k_times)} hours)")

# Get net storage for the MPEC timeframe, re-indexed to integer
net_storage_aligned = net_storage.reindex(k_times).fillna(0.0)
net_storage_aligned.index = range(len(k_times))

print(f"Storage dispatch in MPEC window:")
for t_int, t_dt in zip(range(len(k_times)), k_times):
    val = net_storage_aligned.at[t_int]
    tag = "GEN" if val > 0 else ("LOAD" if val < 0 else "---")
    print(f"  {t_dt} (t={t_int}): {val:8.1f} MW  [{tag}]")

# -------------------------------------------------------------------------
# 4. Run MPEC
# -------------------------------------------------------------------------
opt_name = "wind_onshore_01"

print(f"\n{'='*60}")
print(f"Running MPEC: {len(mpec_times)} timesteps, fixed storage dispatch")
print(f"Strategic gen: {opt_name}")
print(f"{'='*60}")

t0 = time.time()
result = find_optimal_dispatch_bidprice(
    gens_df=gens_df,
    k_values_df=k_values_df.reset_index(drop=True),
    availabilities_df=availability_df.reset_index(drop=True),
    demand_df=demand_df.reset_index(drop=True),
    opt_gen=opt_name,
    fixed_storage_dispatch=net_storage_aligned,
    bid_price_max=3100,
    big_w=10000,
    big_M=10000,
    time_limit=600,
    demand_bids=10,
)
elapsed = time.time() - t0

if result[0] is not None:
    main_df, _, bid_prices = result
    print(f"\n{'='*60}")
    print(f"SUCCESS in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\nResults:")
    print(main_df[["demand", "mcp", "mcp_hat"]].to_string())
    print(f"\nBid prices:")
    print(bid_prices.to_string())
else:
    print(f"\n>>> FAILED in {elapsed:.1f}s")
