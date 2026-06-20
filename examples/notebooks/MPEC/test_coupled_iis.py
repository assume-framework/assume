"""Compute IIS for coupled lambda infeasibility."""
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

# Simplified: just t=0, fix k=1 for pp_3, write LP and compute IIS
opt_gen = 'pp_3'
big_M = 3500

import gurobipy as gp

m = gp.Model("coupled_test")
m.setParam('OutputFlag', 0)

T = [0]  # just one timestep
G = list(gens_df.index)
N = list(range(1, n_demand_bids + 1))

# Variables
g = m.addVars(G, T, lb=0, name="g")
for i in G:
    for t in T:
        g[i,t].ub = gens_df.at[i, 'g_max']

d = m.addVars([(t,n) for t in T for n in N], lb=0, name="d")
for t in T:
    for n in N:
        d[t,n].ub = demand_df.at[t, f'volume_{n}']

u = m.addVars(G, T, vtype=gp.GRB.BINARY, name="u")
k = m.addVars(T, lb=1, ub=99, name="k")
lam = m.addVars(T, lb=-500, ub=3100, name="lambda")

# Non-hat duals
mu_max = m.addVars(G, T, lb=0, ub=3500, name="mu_max")
mu_min = m.addVars(G, T, lb=0, ub=3500, name="mu_min")
nu_max = m.addVars([(t,n) for t in T for n in N], lb=0, ub=3500, name="nu_max")
pi_u = m.addVars(G, T, lb=0, ub=3500, name="pi_u")
pi_d = m.addVars(G, T, lb=0, ub=3500, name="pi_d")

# Hat duals (no separate lambda_hat)
mu_max_hat = m.addVars(G, T, lb=0, ub=3500, name="mu_max_hat")
mu_min_hat = m.addVars(G, T, lb=0, ub=3500, name="mu_min_hat")
nu_max_hat = m.addVars([(t,n) for t in T for n in N], lb=0, ub=3500, name="nu_max_hat")
nu_min_hat = m.addVars([(t,n) for t in T for n in N], lb=0, ub=3500, name="nu_min_hat")
pi_u_hat = m.addVars(G, T, lb=0, ub=3500, name="pi_u_hat")
pi_d_hat = m.addVars(G, T, lb=0, ub=3500, name="pi_d_hat")

# Complementarity binaries
z_mu = m.addVars(G, T, vtype=gp.GRB.BINARY, name="z_mu")
z_nu_max = m.addVars([(t,n) for t in T for n in N], vtype=gp.GRB.BINARY, name="z_nu_max")
z_nu_min = m.addVars([(t,n) for t in T for n in N], vtype=gp.GRB.BINARY, name="z_nu_min")
z_pi_u = m.addVars(G, T, vtype=gp.GRB.BINARY, name="z_pi_u")
z_pi_d = m.addVars(G, T, vtype=gp.GRB.BINARY, name="z_pi_d")

t = 0
# Energy balance
m.addConstr(gp.quicksum(d[t,n] for n in N) == gp.quicksum(g[i,t] for i in G), "balance")

# Gen capacity
for i in G:
    avail = availability_df.at[t, i]
    m.addConstr(g[i,t] <= gens_df.at[i,'g_max'] * avail * u[i,t], f"g_max_{i}")

# Demand capacity
for n in N:
    m.addConstr(d[t,n] <= demand_df.at[t, f'volume_{n}'], f"d_max_{n}")

# Ramp constraints (trivial for t=0 with g_0=0)
for i in G:
    m.addConstr(g[i,t] - gens_df.at[i,'g_0'] <= gens_df.at[i,'r_up'], f"ramp_up_{i}")
    m.addConstr(gens_df.at[i,'g_0'] - g[i,t] <= gens_df.at[i,'r_down'], f"ramp_down_{i}")

# Non-hat gen dual feasibility
for i in G:
    k_term = k[t] if i == opt_gen else k_values_df.at[t,i]
    m.addConstr(k_term * mc_df.at[t,i] - lam[t] + mu_max[i,t] - mu_min[i,t]
                + pi_u[i,t] - pi_d[i,t] == 0, f"gen_dual_{i}")

# Non-hat demand dual feasibility
for n in N:
    m.addConstr(-demand_df.at[t, f'price_{n}'] + lam[t] + nu_max[t,n] >= 0, f"demand_dual_{n}")

# Hat-KKT gen stationarity (uses lam, NOT separate lambda_hat)
for i in G:
    k_term = k[t] if i == opt_gen else k_values_df.at[t,i]
    m.addConstr(k_term * mc_df.at[t,i] - lam[t] + mu_max_hat[i,t] - mu_min_hat[i,t]
                + pi_u_hat[i,t] - pi_d_hat[i,t] == 0, f"kkt_gen_{i}")

# Hat-KKT demand stationarity (uses lam)
for n in N:
    m.addConstr(-demand_df.at[t, f'price_{n}'] + lam[t] + nu_max_hat[t,n] - nu_min_hat[t,n] == 0, f"kkt_demand_{n}")

# COMPLEMENTARITY
for i in G:
    avail = availability_df.at[t, i]
    gmax_i = gens_df.at[i,'g_max']
    # mu_max_hat complementarity
    m.addConstr(g[i,t] - gmax_i * avail * u[i,t] <= gmax_i * (1 - z_mu[i,t]), f"mu_comp1_{i}")
    m.addConstr(g[i,t] - gmax_i * avail * u[i,t] >= -gmax_i * (1 - z_mu[i,t]), f"mu_comp2_{i}")
    m.addConstr(mu_max_hat[i,t] <= big_M * z_mu[i,t], f"mu_comp3_{i}")

    # pi_u_hat complementarity
    m.addConstr(g[i,t] - gens_df.at[i,'g_0'] - gens_df.at[i,'r_up'] <= gmax_i * (1 - z_pi_u[i,t]), f"piu_comp1_{i}")
    m.addConstr(g[i,t] - gens_df.at[i,'g_0'] - gens_df.at[i,'r_up'] >= -gmax_i * (1 - z_pi_u[i,t]), f"piu_comp2_{i}")
    m.addConstr(pi_u_hat[i,t] <= big_M * z_pi_u[i,t], f"piu_comp3_{i}")

    # pi_d_hat complementarity
    m.addConstr(gens_df.at[i,'g_0'] - g[i,t] - gens_df.at[i,'r_down'] <= gmax_i * (1 - z_pi_d[i,t]), f"pid_comp1_{i}")
    m.addConstr(gens_df.at[i,'g_0'] - g[i,t] - gens_df.at[i,'r_down'] >= -gmax_i * (1 - z_pi_d[i,t]), f"pid_comp2_{i}")
    m.addConstr(pi_d_hat[i,t] <= big_M * z_pi_d[i,t], f"pid_comp3_{i}")

for n in N:
    vol_n = demand_df.at[t, f'volume_{n}']
    # nu_max_hat complementarity
    m.addConstr(d[t,n] - vol_n <= vol_n * (1 - z_nu_max[t,n]), f"nu_max_comp1_{n}")
    m.addConstr(d[t,n] - vol_n >= -vol_n * (1 - z_nu_max[t,n]), f"nu_max_comp2_{n}")
    m.addConstr(nu_max_hat[t,n] <= big_M * z_nu_max[t,n], f"nu_max_comp3_{n}")

    # nu_min_hat complementarity
    m.addConstr(d[t,n] <= vol_n * (1 - z_nu_min[t,n]), f"nu_min_comp1_{n}")
    m.addConstr(nu_min_hat[t,n] <= big_M * z_nu_min[t,n], f"nu_min_comp3_{n}")

m.setObjective(0, gp.GRB.MAXIMIZE)
m.optimize()

print(f"Status: {m.Status}")
if m.Status == gp.GRB.INFEASIBLE:
    print("\nComputing IIS...")
    m.computeIIS()
    print("\nIIS Constraints:")
    for c in m.getConstrs():
        if c.IISConstr:
            print(f"  {c.ConstrName}: {c.Sense} {c.RHS}")
    print("\nIIS Variable bounds:")
    for v in m.getVars():
        if v.IISLB:
            print(f"  {v.VarName} LB={v.LB}")
        if v.IISUB:
            print(f"  {v.VarName} UB={v.UB}")
elif m.Status == gp.GRB.OPTIMAL:
    print("FEASIBLE!")
    print(f"lambda = {lam[0].X:.2f}")
    print(f"k = {k[0].X:.4f}")
    for i in G:
        if g[i,0].X > 0.1:
            print(f"g[{i}] = {g[i,0].X:.1f}, u={u[i,0].X:.0f}")
