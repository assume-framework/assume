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

print("=== All demand bids (first row) ===")
for n in range(1, 11):
    vc = f"volume_{n}"
    pc = f"price_{n}"
    if vc in demand_df.columns:
        print(
            f"  bid {n}: volume={demand_df[vc].iloc[0]:.2f} MW, price={demand_df[pc].iloc[0]:.2f} EUR/MWh"
        )

total_vol = sum(
    demand_df[f"volume_{n}"].iloc[0]
    for n in range(1, 11)
    if f"volume_{n}" in demand_df.columns
)
print(f"\nTotal demand volume = {total_vol:.2f} MW")

# Check k_values for all gens
k_values_df = pd.read_csv(
    f"{out_dir}/k_values_df.csv", index_col=0, parse_dates=True
).fillna(0.0)
gens_df = pd.read_csv(f"{out_dir}/gens_df.csv", index_col=0)
if "unit" in gens_df.columns:
    gens_df = gens_df.set_index("unit")

print("\n=== k_values and offer prices at t=0 (sorted by offer price) ===")
t0 = k_values_df.index[0]
offers = []
for gen in gens_df.index:
    if gen in k_values_df.columns:
        k = k_values_df.at[t0, gen]
        mc = gens_df.at[gen, "mc"]
        gmax = gens_df.at[gen, "g_max"]
        offer = k * mc
        offers.append((gen, k, mc, offer, gmax))
offers.sort(key=lambda x: x[3])
for gen, k, mc, offer, gmax in offers[:20]:
    print(
        f"  {gen:30s}: k={k:12.2f}, mc={mc:8.4f}, offer={offer:10.4f} EUR/MWh, g_max={gmax:.0f}"
    )
print("  ...")
for gen, k, mc, offer, gmax in offers[-5:]:
    print(
        f"  {gen:30s}: k={k:12.2f}, mc={mc:8.4f}, offer={offer:10.4f} EUR/MWh, g_max={gmax:.0f}"
    )

total_cap = sum(gens_df.at[gen, "g_max"] for gen in gens_df.index)
print(f"\nTotal generation capacity: {total_cap:.0f} MW")

# Show how much capacity is available below common price thresholds
avail_df = pd.read_csv(f"{out_dir}/availability_df.csv", index_col=0, parse_dates=True)
t0_avail = avail_df.iloc[0]
for threshold in [0, 10, 25, 50, 90, 100, 200, 500, 3000]:
    cap = sum(
        gens_df.at[gen, "g_max"] * t0_avail.get(gen, 1.0)
        for gen, k, mc, offer, gmax in offers
        if offer <= threshold
    )
    print(f"  Available capacity at offer <= {threshold:5d}: {cap:10.1f} MW")
