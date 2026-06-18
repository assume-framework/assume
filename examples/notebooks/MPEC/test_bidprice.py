# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test bid-price reformulation with SoC-feasibility filter for storages."""
import pandas as pd
import numpy as np
import time

from bilevel_opt_bidprice import find_optimal_dispatch_bidprice_with_storage

out_dir = "mpec_input_data"

gens_df = pd.read_csv(f"{out_dir}/gens_df.csv", index_col=0)
storage_units = pd.read_csv(f"{out_dir}/storage_units.csv", index_col=0)
storage_units["initial_soc"] = storage_units["initial_soc"].fillna(0.5)
storage_units["initial_soc"] = storage_units["initial_soc"].clip(lower=0.5)

k_values_df_short = pd.read_csv(
    f"{out_dir}/k_values_df.csv", index_col=0, parse_dates=True
).fillna(0.0)
storage_k_values_df_short = pd.read_csv(
    f"{out_dir}/storage_k_values_df.csv", index_col=0, parse_dates=True
)
availability_short = pd.read_csv(
    f"{out_dir}/availability_df.csv", index_col=0, parse_dates=True
)
demand_df_short = pd.read_csv(
    f"{out_dir}/demand_df.csv", index_col=0, parse_dates=True
)

for col in demand_df_short.columns:
    if col.startswith("volume"):
        demand_df_short[col] = -demand_df_short[col]
        demand_df_short[col] = demand_df_short[col].clip(lower=0.01)

opt_name = "wind_onshore_01"
n_timesteps = len(demand_df_short)

# Filter storages: only keep those whose SoC can survive full-horizon discharge
# The KKT may force max discharge + max charge simultaneously.
# Net SoC drain per step = max_discharge/eff_discharge - eff_charge*max_charge
# If net drain > 0, check if initial_soc * capacity survives n_timesteps
print("=== Storage SoC feasibility check ===")
feasible_storages = []
for s in storage_units.index:
    cap = storage_units.at[s, "capacity"]
    p_dis = storage_units.at[s, "max_power_discharge"]
    p_ch = abs(storage_units.at[s, "max_power_charge"])
    eff_ch = storage_units.at[s, "efficiency_charge"]
    eff_dis = storage_units.at[s, "efficiency_discharge"]
    soc0 = storage_units.at[s, "initial_soc"] * cap

    net_drain = p_dis / eff_dis - eff_ch * p_ch
    if net_drain > 0:
        max_steps = soc0 / net_drain
        if max_steps < n_timesteps:
            print(f"  SKIP {s}: cap={cap:.0f}, net_drain={net_drain:.1f}/h, survives {max_steps:.1f}h < {n_timesteps}h")
            continue
    feasible_storages.append(s)

print(f"\n  Kept {len(feasible_storages)}/{len(storage_units)} storages")
test_storages = storage_units.loc[feasible_storages]
test_storage_k = storage_k_values_df_short[feasible_storages]

print(f"\n{'='*60}")
print(f"Running MPEC with {len(test_storages)} storages, time_limit=600s")
print(f"{'='*60}")

t0 = time.time()
result = find_optimal_dispatch_bidprice_with_storage(
    gens_df=gens_df,
    storage_df=test_storages,
    k_values_df=k_values_df_short.reset_index(drop=True),
    storage_k_values_df=test_storage_k.reset_index(drop=True),
    availabilities_df=availability_short.reset_index(drop=True),
    demand_df=demand_df_short.reset_index(drop=True),
    opt_gen=opt_name,
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
    print(f"\nMCP:\n{main_df[['demand', 'mcp', 'mcp_hat']].head(10)}")
    print(f"\nBid prices:\n{bid_prices.head(10)}")
else:
    print(f"\n>>> FAILED in {elapsed:.1f}s")
