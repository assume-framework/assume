"""Debug why coupled lambda is infeasible — compute IIS."""
import pandas as pd, numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

data_dir = 'mpec_input_data_02a'
gens_df = pd.read_csv(f'{data_dir}/gens_df.csv', index_col=0)
demand_df = pd.read_csv(f'{data_dir}/demand_df.csv', index_col=0, parse_dates=True)
k_values_df = pd.read_csv(f'{data_dir}/k_values_df.csv', index_col=0, parse_dates=True).fillna(1.0)
availability_df = pd.read_csv(f'{data_dir}/availability_df.csv', index_col=0, parse_dates=True)

mask = (k_values_df.index >= '2019-03-16') & (k_values_df.index < '2019-03-17')
k_values_df = k_values_df[mask]; demand_df = demand_df[mask]; availability_df = availability_df[mask]
demand_df = demand_df.reset_index(drop=True)
k_values_df = k_values_df.reset_index(drop=True).drop(columns=['date'], errors='ignore')
availability_df = availability_df.reset_index(drop=True)

vol_cols = [c for c in demand_df.columns if c.startswith('volume_')]
demand_df[vol_cols] = demand_df[vol_cols].abs().clip(lower=0.01)
demand_df = demand_df.drop(columns=['date'], errors='ignore')
n_demand_bids = len(vol_cols)
gens_df = gens_df.set_index('unit') if 'unit' in gens_df.columns else gens_df
gen_names = list(gens_df.index)
k_values_df = k_values_df[[c for c in k_values_df.columns if c in gen_names]]
availability_df = availability_df[[c for c in availability_df.columns if c in gen_names]]
mc_df = pd.DataFrame({g: gens_df.at[g, 'mc'] for g in gen_names}, index=demand_df.index)

# Check demand bid prices
price_cols = [c for c in demand_df.columns if c.startswith('price_')]
print("Demand bid prices (t=0):")
for c in price_cols:
    v = [c2 for c2 in vol_cols if c2.split('_')[1] == c.split('_')[1]][0]
    print(f"  {c}: {demand_df.at[0, c]:.2f} EUR/MWh, {v}: {demand_df.at[0, v]:.2f} MW")

print(f"\nGenerators:")
for g in gen_names:
    print(f"  {g}: mc={gens_df.at[g,'mc']:.2f}, g_max={gens_df.at[g,'g_max']:.0f}")

# Minimal test: just t=0, just the hat-KKT + non-hat dual with single lambda
# to find what's conflicting
print("\n--- Minimal feasibility test (t=0 only, fixed k=1, fixed dispatch) ---")

# First compute the competitive equilibrium dispatch
# Sort by mc, dispatch cheapest first
sorted_gens = gens_df.sort_values('mc')
total_demand = 9310
remaining = total_demand
dispatch = {}
for g in sorted_gens.index:
    gen_max = gens_df.at[g, 'g_max'] * availability_df.at[0, g]
    dispatch[g] = min(remaining, gen_max)
    remaining -= dispatch[g]
    if remaining <= 0:
        break

print("\nCompetitive dispatch:")
for g, gval in dispatch.items():
    if gval > 0:
        print(f"  {g}: {gval:.0f} MW (mc={gens_df.at[g,'mc']:.2f})")

# The MCP should be the mc of the marginal unit
marginal_gen = [g for g,v in dispatch.items() if 0 < v < gens_df.at[g,'g_max']]
if marginal_gen:
    print(f"\nMarginal gen: {marginal_gen[0]}, MCP = {gens_df.at[marginal_gen[0],'mc']:.2f}")
else:
    print("\nNo marginal gen found (all at capacity or zero)")

# Check which demand bids are active at MCP=85.71
mcp_competitive = 85.71
print(f"\nDemand bids at MCP={mcp_competitive}:")
for c in price_cols:
    n = c.split('_')[1]
    p = demand_df.at[0, c]
    v = demand_df.at[0, f'volume_{n}']
    if p >= mcp_competitive:
        print(f"  Bid {n}: price={p:.2f} >= MCP, fully filled ({v:.2f} MW)")
    else:
        print(f"  Bid {n}: price={p:.2f} < MCP, NOT filled")

# Now check if the hat-KKT demand stationarity can be satisfied
# For each bid n: lambda_ = price_n - nu_max_hat + nu_min_hat
# With complementarity: if d=vol_n, nu_min_hat=0; if d=0, nu_max_hat=0; if 0<d<vol, both=0
print(f"\nHat-KKT demand stationarity check (lambda={mcp_competitive}):")
for c in price_cols:
    n = c.split('_')[1]
    p = demand_df.at[0, c]
    v = demand_df.at[0, f'volume_{n}']
    if p >= mcp_competitive:
        # d = vol_n, so nu_min_hat=0, nu_max_hat = price_n - lambda_
        nu_max_hat_needed = p - mcp_competitive
        print(f"  Bid {n}: nu_max_hat = {nu_max_hat_needed:.2f} (bound 3500) {'OK' if nu_max_hat_needed <= 3500 else 'EXCEEDS BOUND'}")
    else:
        # d = 0, so nu_max_hat=0, nu_min_hat = lambda_ - price_n
        nu_min_hat_needed = mcp_competitive - p
        print(f"  Bid {n}: nu_min_hat = {nu_min_hat_needed:.2f} (bound 3500) {'OK' if nu_min_hat_needed <= 3500 else 'EXCEEDS BOUND'}")

# Check non-hat demand dual feasibility
print(f"\nNon-hat demand dual feasibility (lambda={mcp_competitive}):")
for c in price_cols:
    n = c.split('_')[1]
    p = demand_df.at[0, c]
    slack = mcp_competitive - p  # lambda - price_n; need lambda + nu_max >= price_n, i.e. nu_max >= price_n - lambda
    if slack >= 0:
        print(f"  Bid {n}: lambda >= price ({mcp_competitive} >= {p}), nu_max can be 0. OK")
    else:
        nu_max_needed = -slack
        print(f"  Bid {n}: nu_max >= {nu_max_needed:.2f} (bound 3500) {'OK' if nu_max_needed <= 3500 else 'EXCEEDS BOUND'}")

# Check hat-KKT gen stationarity
print(f"\nHat-KKT gen stationarity check (lambda={mcp_competitive}, k=1):")
for g in gen_names:
    mc = gens_df.at[g, 'mc']
    rhs = mcp_competitive - mc  # Need mu_max_hat - mu_min_hat = lambda - k*mc
    d_val = dispatch.get(g, 0)
    g_max = gens_df.at[g, 'g_max'] * availability_df.at[0, g]
    if d_val >= g_max - 0.01:
        # at capacity: mu_max_hat > 0, mu_min_hat = 0 (complementarity)
        print(f"  {g}: at capacity, mu_max_hat = {rhs:.2f} {'OK' if rhs >= 0 else 'NEGATIVE!'}")
    elif d_val <= 0.01:
        # at zero: mu_min_hat > 0, mu_max_hat = 0 (but mu_max depends on u)
        # Actually if u=0, g_max*u=0 and g=0, so g=g_max*u, mu_max_hat can be nonzero
        print(f"  {g}: at zero, need mu_min_hat - mu_max_hat = {-rhs:.2f}")
    else:
        # marginal: mu_max_hat = 0, mu_min_hat = 0 (neither at bound)
        # But this requires rhs = 0, i.e. mc = lambda
        print(f"  {g}: marginal, need rhs=0 but rhs = {rhs:.2f} {'OK' if abs(rhs) < 0.01 else 'NONZERO - check pi terms'}")
