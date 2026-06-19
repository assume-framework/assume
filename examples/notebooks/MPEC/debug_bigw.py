# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import time

import pandas as pd

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

from bilevel_opt_bidprice import find_optimal_dispatch_bidprice

gens_df_idx = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df.copy()
demand_df_r = demand_df.reset_index(drop=True).copy()
k_values_r = k_values_df.reset_index(drop=True).copy()
avail_r = availability_df.reset_index(drop=True).copy()

mc_df = pd.DataFrame(
    {gen: gens_df_idx.at[gen, "mc"] for gen in gens_df_idx.index},
    index=demand_df_r.index,
)

fixed_sd = net_storage_aligned.copy()
fixed_sd.index = demand_df_r.index

opt_gen = "wind_onshore_01"

for big_w in [10_000, 100_000, 1_000_000, 10_000_000]:
    print(f"\n{'=' * 70}")
    print(f"big_w = {big_w:,.0f}")
    print(f"{'=' * 70}")

    t0 = time.time()
    main_df, supp_df, bid_prices = find_optimal_dispatch_bidprice(
        gens_df=gens_df_idx,
        k_values_df=k_values_r,
        availabilities_df=avail_r,
        demand_df=demand_df_r,
        opt_gen=opt_gen,
        fixed_storage_dispatch=fixed_sd,
        bid_price_max=3100,
        big_w=big_w,
        time_limit=120,
        big_M=10000,
        demand_bids=10,
        mc_df=mc_df,
    )
    elapsed = time.time() - t0

    if main_df is None:
        print(f"  INFEASIBLE / FAILED ({elapsed:.1f}s)")
        continue

    print(f"  Solved in {elapsed:.1f}s")

    hat_gap = abs(main_df["mcp_hat"].astype(float) - main_df["mcp"].astype(float))
    print(
        f"  lambda_hat range: {main_df['mcp_hat'].astype(float).min():.2f} - {main_df['mcp_hat'].astype(float).max():.2f}"
    )
    print(
        f"  lambda_   range: {main_df['mcp'].astype(float).min():.2f} - {main_df['mcp'].astype(float).max():.2f}"
    )
    print(f"  |hat - lambda| mean: {hat_gap.mean():.2f}, max: {hat_gap.max():.2f}")
    print(
        f"  demand range: {main_df['demand'].astype(float).min():.0f} - {main_df['demand'].astype(float).max():.0f}"
    )
    print(
        f"  bid_price range: {bid_prices['bid_price'].astype(float).min():.2f} - {bid_prices['bid_price'].astype(float).max():.2f}"
    )

    gen_col = f"gen_{opt_gen}"
    gen_vals = main_df[gen_col].astype(float)
    mcp_vals = main_df["mcp"].astype(float)
    profit_mpec = (gen_vals * (mcp_vals - mc_df[opt_gen])).sum()
    print(f"  MPEC profit (lambda_): {profit_mpec:,.0f} EUR")

    hat_vals = main_df["mcp_hat"].astype(float)
    profit_hat = (gen_vals * (hat_vals - mc_df[opt_gen])).sum()
    print(f"  MPEC profit (lambda_hat): {profit_hat:,.0f} EUR")
