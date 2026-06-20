"""
Test: Separate big_M per constraint type.
Same formulation as find_optimal_dispatch_quadratic, but uses tight big_M
values tailored to each complementarity constraint instead of one global big_M.

Minimum valid big_M per constraint:
  mu_max_hat <= M_dual * z          : M_dual >= max(lambda_hat) = 300
  nu_max_hat <= M_nu * z            : M_nu >= max(price) = 3000
  nu_min_hat <= M_dual * z          : M_dual >= max(lambda_hat) = 300
  d <= M_demand * (1-z)             : M_demand >= max(volume) = 6973
  ramp slack >= -M_ramp * (1-z)     : M_ramp >= max(g_max + r_up) = 10000
  pi_u/d_hat <= M_dual * z          : M_dual >= 300
"""
import time
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ---------- load data ----------
data_dir = "mpec_input_data_02a"
gens_df = pd.read_csv(f"{data_dir}/gens_df.csv", index_col=0)
k_values_df = pd.read_csv(f"{data_dir}/k_values_df.csv", index_col=0, parse_dates=True).fillna(0.0)
demand_df = pd.read_csv(f"{data_dir}/demand_df.csv", index_col=0, parse_dates=True)
availability_df = pd.read_csv(f"{data_dir}/availability_df.csv", index_col=0, parse_dates=True)

date_mask_k = (k_values_df.index >= "2019-03-16") & (k_values_df.index < "2019-03-17")
date_mask_d = (demand_df.index >= "2019-03-16") & (demand_df.index < "2019-03-17")
date_mask_a = (availability_df.index >= "2019-03-16") & (availability_df.index < "2019-03-17")
k_values_df = k_values_df.loc[date_mask_k].reset_index(drop=True)
demand_df = demand_df.loc[date_mask_d].reset_index(drop=True)
availability_df = availability_df.loc[date_mask_a].reset_index(drop=True)

gen_cols = [c for c in gens_df["unit"].values if c in k_values_df.columns]
k_values_df = k_values_df[gen_cols]
avail_cols = [c for c in gens_df["unit"].values if c in availability_df.columns]
availability_df = availability_df[avail_cols]

vol_cols = [c for c in demand_df.columns if c.startswith("volume_")]
demand_df[vol_cols] = demand_df[vol_cols].abs().clip(lower=0.01)
if "date" in demand_df.columns:
    demand_df = demand_df.drop(columns=["date"])

n_demand_bids = len(vol_cols)
gens_df = gens_df.set_index("unit")
mc_df = pd.DataFrame({gen: gens_df.at[gen, "mc"] for gen in gens_df.index}, index=demand_df.index)

opt_gen = "pp_3"
k_max = 4
big_w = 10000
availabilities_df = availability_df

# ---------- Compute tight per-constraint big_M ----------
max_lambda = 300
max_price = max(demand_df.at[0, f"price_{n}"] for n in range(1, n_demand_bids + 1))
max_volume = max(abs(demand_df[f"volume_{n}"]).max() for n in range(1, n_demand_bids + 1))
max_gmax = gens_df["g_max"].max()

# Per-unit ramp big_M: need g_max_i + r_up_i for the worst case
ramp_bigM = {}
for i in gens_df.index:
    ramp_bigM[i] = gens_df.at[i, "g_max"] + max(gens_df.at[i, "r_up"], gens_df.at[i, "r_down"])

M_dual = max_lambda + 200   # 500: generous bound for mu/pi duals
M_nu_max = max_price + 100  # 3100: for nu_max_hat (price - lambda_hat)
M_nu_min = max_lambda + 100 # 400: for nu_min_hat (lambda_hat - price)
M_demand = max_volume + 100 # ~7073: for d <= M*(1-z)

print(f"Tight big_M values:")
print(f"  M_dual (mu/pi): {M_dual}")
print(f"  M_nu_max:       {M_nu_max}")
print(f"  M_nu_min:       {M_nu_min}")
print(f"  M_demand:       {M_demand}")
for i in gens_df.index:
    print(f"  M_ramp[{i}]: {ramp_bigM[i]}")

# ---------- Build model ----------
model = pyo.ConcreteModel()
model.time = pyo.Set(initialize=demand_df.index)
model.gens = pyo.Set(initialize=gens_df.index)
model.demand_bids = pyo.Set(initialize=np.arange(1, n_demand_bids + 1))

# Primary variables
model.g = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
model.d = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)
model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
model.k = pyo.Var(model.time, bounds=(1, k_max), within=pyo.NonNegativeReals)
model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(0, 300))
model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

# Non-hat duals (with explicit bounds for LP tightness)
model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))
model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))
model.nu_max = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, M_nu_max))
model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))
model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))
model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))
model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual * max_gmax))

# Hat duals (with explicit bounds)
model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(0, 300))
model.mu_max_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))
model.mu_min_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))
model.nu_max_hat = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, M_nu_max))
model.nu_min_hat = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, M_nu_min))
model.pi_u_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))
model.pi_d_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, M_dual))

# Binaries
model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

# ==================== OBJECTIVE (identical) ====================
def primary_objective_rule(model):
    return sum(
        model.lambda_hat[t] * model.g[opt_gen, t]
        - mc_df.at[t, opt_gen] * model.g[opt_gen, t]
        - model.c_up[opt_gen, t] - model.c_down[opt_gen, t]
        for t in model.time)

def duality_gap_part_1_rule(model):
    expr = sum(
        (mc_df.at[t, gen] * model.k[t] * model.g[gen, t] + model.c_up[gen, t] + model.c_down[gen, t])
        if gen == opt_gen else
        (k_values_df.at[t, gen] * mc_df.at[t, gen] * model.g[gen, t] + model.c_up[gen, t] + model.c_down[gen, t])
        for gen in model.gens for t in model.time)
    expr -= sum(demand_df.at[t, f"price_{n}"] * model.d[t, n]
                for t in model.time for n in model.demand_bids)
    return expr

def duality_gap_part_2_rule(model):
    expr = -sum(model.nu_max[t, n] * demand_df.at[t, f"volume_{n}"]
                for t in model.time for n in model.demand_bids)
    expr -= sum(model.pi_u[i, t] * gens_df.at[i, "r_up"] for i in model.gens for t in model.time)
    expr -= sum(model.pi_d[i, t] * gens_df.at[i, "r_down"] for i in model.gens for t in model.time)
    expr -= sum(model.pi_u[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)
    expr += sum(model.pi_d[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)
    expr -= sum(model.sigma_u[i, 0] * gens_df.at[i, "k_up"] * gens_df.at[i, "u_0"] for i in model.gens)
    expr += sum(model.sigma_d[i, 0] * gens_df.at[i, "k_down"] * gens_df.at[i, "u_0"] for i in model.gens)
    expr -= sum(model.psi_max[i, t] for i in model.gens for t in model.time)
    return expr

model.objective = pyo.Objective(
    expr=lambda m: primary_objective_rule(m) - big_w * (duality_gap_part_1_rule(m) - duality_gap_part_2_rule(m)),
    sense=pyo.maximize)

# ==================== CONSTRAINTS (all identical) ====================
model.balance = pyo.Constraint(model.time,
    rule=lambda m, t: sum(m.d[t, n] for n in m.demand_bids) - sum(m.g[i, t] for i in m.gens) == 0)
model.g_max = pyo.Constraint(model.gens, model.time,
    rule=lambda m, i, t: m.g[i, t] <= gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * m.u[i, t])
model.d_max = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: m.d[t, n] <= demand_df.at[t, f"volume_{n}"])

def ru_max_rule(m, i, t):
    if t == 0: return m.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
    return m.g[i, t] - m.g[i, t - 1] <= gens_df.at[i, "r_up"]
model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_max_rule)

def rd_max_rule(m, i, t):
    if t == 0: return gens_df.at[i, "g_0"] - m.g[i, t] <= gens_df.at[i, "r_down"]
    return m.g[i, t - 1] - m.g[i, t] <= gens_df.at[i, "r_down"]
model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_max_rule)

def start_up_cost_rule(m, i, t):
    if t == 0: return m.c_up[i, t] >= (m.u[i, t] - gens_df.at[i, "u_0"]) * gens_df.at[i, "k_up"]
    return m.c_up[i, t] >= (m.u[i, t] - m.u[i, t - 1]) * gens_df.at[i, "k_up"]
model.start_up_cost = pyo.Constraint(model.gens, model.time, rule=start_up_cost_rule)

def shut_down_cost_rule(m, i, t):
    if t == 0: return m.c_down[i, t] >= (gens_df.at[i, "u_0"] - m.u[i, t]) * gens_df.at[i, "k_down"]
    return m.c_down[i, t] >= (m.u[i, t - 1] - m.u[i, t]) * gens_df.at[i, "k_down"]
model.shut_down_cost = pyo.Constraint(model.gens, model.time, rule=shut_down_cost_rule)

# Dual constraints
def gen_dual_rule(m, i, t):
    k_term = m.k[t] if i == opt_gen else k_values_df.at[t, i]
    pi_u_next = 0 if t == m.time.at(-1) else m.pi_u[i, t + 1]
    pi_d_next = 0 if t == m.time.at(-1) else m.pi_d[i, t + 1]
    return (k_term * mc_df.at[t, i] - m.lambda_[t] + m.mu_max[i, t] - m.mu_min[i, t]
            + m.pi_u[i, t] - pi_u_next - m.pi_d[i, t] + pi_d_next == 0)
model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual_rule)

def status_dual_rule(m, i, t):
    if t != m.time.at(-1):
        return (-m.mu_max[i, t] * gens_df.at[i, "g_max"] * availabilities_df.at[t, i]
                + (m.sigma_u[i, t] - m.sigma_u[i, t + 1]) * gens_df.at[i, "k_up"]
                - (m.sigma_d[i, t] - m.sigma_d[i, t + 1]) * gens_df.at[i, "k_down"]
                + m.psi_max[i, t] >= 0)
    return (-m.mu_max[i, t] * gens_df.at[i, "g_max"] * availabilities_df.at[t, i]
            + m.sigma_u[i, t] * gens_df.at[i, "k_up"]
            - m.sigma_d[i, t] * gens_df.at[i, "k_down"]
            + m.psi_max[i, t] >= 0)
model.status_dual = pyo.Constraint(model.gens, model.time, rule=status_dual_rule)

model.demand_dual = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: -demand_df.at[t, f"price_{n}"] + m.lambda_[t] + m.nu_max[t, n] >= 0)

# KKT conditions
def kkt_gen_rule(m, i, t):
    k_term = m.k[t] if i == opt_gen else k_values_df.at[t, i]
    pi_u_hat_next = 0 if t == m.time.at(-1) else m.pi_u_hat[i, t + 1]
    pi_d_hat_next = 0 if t == m.time.at(-1) else m.pi_d_hat[i, t + 1]
    return (k_term * mc_df.at[t, i] - m.lambda_hat[t] + m.mu_max_hat[i, t] - m.mu_min_hat[i, t]
            + m.pi_u_hat[i, t] - pi_u_hat_next - m.pi_d_hat[i, t] + pi_d_hat_next == 0)
model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen_rule)

model.kkt_demand = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: -demand_df.at[t, f"price_{n}"] + m.lambda_hat[t] + m.nu_max_hat[t, n] - m.nu_min_hat[t, n] == 0)

# ==================== COMPLEMENTARITY — with per-constraint big_M ====================

# mu_max_hat: uses max(g_max) for primal slack, M_dual for dual bound
model.mu_max_hat_binary_constr_1 = pyo.Constraint(model.gens, model.time,
    rule=lambda m, i, t: m.g[i, t] - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * m.u[i, t]
                          <= max(gens_df["g_max"]) * (1 - m.mu_max_hat_binary[i, t]))
model.mu_max_hat_binary_constr_2 = pyo.Constraint(model.gens, model.time,
    rule=lambda m, i, t: m.g[i, t] - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * m.u[i, t]
                          >= -max(gens_df["g_max"]) * (1 - m.mu_max_hat_binary[i, t]))
# KEY: use M_dual instead of global big_M
model.mu_max_hat_binary_constr_3 = pyo.Constraint(model.gens, model.time,
    rule=lambda m, i, t: m.mu_max_hat[i, t] <= M_dual * m.mu_max_hat_binary[i, t])

# nu_max_hat: uses max(volume) for primal slack, M_nu_max for dual bound
model.nu_max_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
model.nu_max_hat_binary_constr_1 = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: m.d[t, n] - demand_df.at[t, f"volume_{n}"]
                          <= max(demand_df[f"volume_{n}"]) * (1 - m.nu_max_hat_binary[t, n]))
model.nu_max_hat_binary_constr_2 = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: m.d[t, n] - demand_df.at[t, f"volume_{n}"]
                          >= -max(demand_df[f"volume_{n}"]) * (1 - m.nu_max_hat_binary[t, n]))
# KEY: use M_nu_max (=3100) instead of 10e6
model.nu_max_hat_binary_constr_3 = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: m.nu_max_hat[t, n] <= M_nu_max * m.nu_max_hat_binary[t, n])

# nu_min_hat: uses M_demand for primal, M_nu_min for dual
model.nu_min_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
# KEY: use M_demand (~7073) instead of global big_M for d <= M*(1-z)
model.nu_min_hat_binary_constr_1 = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: m.d[t, n] <= M_demand * (1 - m.nu_min_hat_binary[t, n]))
# KEY: use M_nu_min (=400) instead of global big_M
model.nu_min_hat_binary_constr_3 = pyo.Constraint(model.time, model.demand_bids,
    rule=lambda m, t, n: m.nu_min_hat[t, n] <= M_nu_min * m.nu_min_hat_binary[t, n])

# pi_u_hat: uses PER-UNIT ramp big_M for primal slack, M_dual for dual
model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
def pi_u_hat_rule_1(m, i, t):
    M_ramp_i = ramp_bigM[i]
    if t == 0:
        return m.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[i, "r_up"] <= M_ramp_i * (1 - m.pi_u_hat_binary[i, t])
    return m.g[i, t] - m.g[i, t - 1] - gens_df.at[i, "r_up"] <= M_ramp_i * (1 - m.pi_u_hat_binary[i, t])
def pi_u_hat_rule_2(m, i, t):
    M_ramp_i = ramp_bigM[i]
    if t == 0:
        return m.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[i, "r_up"] >= -M_ramp_i * (1 - m.pi_u_hat_binary[i, t])
    return m.g[i, t] - m.g[i, t - 1] - gens_df.at[i, "r_up"] >= -M_ramp_i * (1 - m.pi_u_hat_binary[i, t])
model.pi_u_hat_binary_constr_1 = pyo.Constraint(model.gens, model.time, rule=pi_u_hat_rule_1)
model.pi_u_hat_binary_constr_2 = pyo.Constraint(model.gens, model.time, rule=pi_u_hat_rule_2)
# KEY: use M_dual (=500) instead of 10e6 for dual bound
model.pi_u_hat_binary_constr_3 = pyo.Constraint(model.gens, model.time,
    rule=lambda m, i, t: m.pi_u_hat[i, t] <= M_dual * m.pi_u_hat_binary[i, t])

# pi_d_hat: same logic
model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
def pi_d_hat_rule_1(m, i, t):
    M_ramp_i = ramp_bigM[i]
    if t == 0:
        return gens_df.at[i, "g_0"] - m.g[i, t] - gens_df.at[i, "r_down"] <= M_ramp_i * (1 - m.pi_d_hat_binary[i, t])
    return m.g[i, t - 1] - m.g[i, t] - gens_df.at[i, "r_down"] <= M_ramp_i * (1 - m.pi_d_hat_binary[i, t])
def pi_d_hat_rule_2(m, i, t):
    M_ramp_i = ramp_bigM[i]
    if t == 0:
        return gens_df.at[i, "g_0"] - m.g[i, t] - gens_df.at[i, "r_down"] >= -M_ramp_i * (1 - m.pi_d_hat_binary[i, t])
    return m.g[i, t - 1] - m.g[i, t] - gens_df.at[i, "r_down"] >= -M_ramp_i * (1 - m.pi_d_hat_binary[i, t])
model.pi_d_hat_binary_constr_1 = pyo.Constraint(model.gens, model.time, rule=pi_d_hat_rule_1)
model.pi_d_hat_binary_constr_2 = pyo.Constraint(model.gens, model.time, rule=pi_d_hat_rule_2)
model.pi_d_hat_binary_constr_3 = pyo.Constraint(model.gens, model.time,
    rule=lambda m, i, t: m.pi_d_hat[i, t] <= M_dual * m.pi_d_hat_binary[i, t])

# ==================== SOLVE ====================
print(f"\nopt_gen={opt_gen}, k_max={k_max}, big_w={big_w}")
print(f"Timesteps: {len(demand_df)}, Demand bids: {n_demand_bids}")
print()

instance = model.create_instance()
solver = SolverFactory("gurobi")
solver.options["NonConvex"] = 2
t0 = time.time()
results = solver.solve(instance, options={
    "LogToConsole": 1, "TimeLimit": 3600, "MIPGap": 0.05,
    "DualReductions": 0, "MIPFocus": 1, "Presolve": 2, "Cuts": 2,
}, tee=True)
elapsed = time.time() - t0

tc = results.solver.termination_condition
print(f"\nTermination: {tc}")

if tc in (pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded):
    print(f"INFEASIBLE ({elapsed:.1f}s)")
else:
    ti = demand_df.index
    mcp = np.array([float(instance.lambda_[t].value) for t in ti])
    mcp_hat = np.array([float(instance.lambda_hat[t].value) for t in ti])
    k_vals = np.array([float(instance.k[t].value) for t in ti])
    gen_vals = np.array([float(instance.g[opt_gen, t].value) for t in ti])
    demand_vals = np.array([sum(float(instance.d[t, n].value) for n in instance.demand_bids) for t in ti])
    print(f"\nTime: {elapsed:.1f}s")
    print(f"MCP (lambda_):   min={mcp.min():.2f}, max={mcp.max():.2f}, mean={mcp.mean():.2f}")
    print(f"MCP_hat:         min={mcp_hat.min():.2f}, max={mcp_hat.max():.2f}, mean={mcp_hat.mean():.2f}")
    print(f"K:               min={k_vals.min():.4f}, max={k_vals.max():.4f}")
    print(f"Gen {opt_gen}:   min={gen_vals.min():.0f}, max={gen_vals.max():.0f}")
    print(f"Demand:          min={demand_vals.min():.0f}, max={demand_vals.max():.0f}")
    profit = sum((mcp_hat - gens_df.at[opt_gen, 'mc']) * gen_vals)
    print(f"Profit (hat): {profit:.0f} EUR/day")
