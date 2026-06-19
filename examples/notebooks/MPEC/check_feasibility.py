# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd

out_dir = "mpec_input_data"
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

k_values_df = pd.read_csv(
    f"{out_dir}/k_values_df.csv", index_col=0, parse_dates=True
).fillna(0.0)
k_times = k_values_df.index
net_storage_aligned = net_storage.reindex(k_times).fillna(0.0)

gens_df = pd.read_csv(f"{out_dir}/gens_df.csv", index_col=0)
if "unit" in gens_df.columns:
    gens_df = gens_df.set_index("unit")
avail_df = pd.read_csv(f"{out_dir}/availability_df.csv", index_col=0, parse_dates=True)

total_demand_vol = sum(
    demand_df[f"volume_{n}"].iloc[0]
    for n in range(1, 11)
    if f"volume_{n}" in demand_df.columns
)
total_gen_cap_t0 = sum(
    gens_df.at[g, "g_max"] * avail_df.iloc[0].get(g, 1.0) for g in gens_df.index
)
print(f"Total demand volume (t=0): {total_demand_vol:.0f} MW")
print(f"Total gen capacity (t=0): {total_gen_cap_t0:.0f} MW")
print(
    f"Net storage range: {net_storage_aligned.min():.0f} to {net_storage_aligned.max():.0f} MW"
)
print()
for i, (t_dt, ns) in enumerate(zip(k_times, net_storage_aligned)):
    avail_cap = sum(
        gens_df.at[g, "g_max"] * avail_df.iloc[i].get(g, 1.0) for g in gens_df.index
    )
    need = total_demand_vol + ns
    print(
        f"t={i}: storage={ns:8.0f}, demand_vol={total_demand_vol:8.0f}, need_gen={need:8.0f}, avail_cap={avail_cap:8.0f}, feasible={'YES' if avail_cap >= need else 'NO'}"
    )
