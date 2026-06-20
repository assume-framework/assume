"""
Test: Original formulation with explicit variable bounds on dual variables.
NO formulation changes — same constraints and objective as find_optimal_dispatch_quadratic.
Only adds tighter .bounds on Var declarations to improve LP relaxation quality.

big_M stays at 10000 (minimum valid for pp_7 ramp complementarity).
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

# ---------- config ----------
opt_gen = "pp_3"
k_max = 4
big_w = 10000


def solve_with_bounds(big_M, dual_ub, nu_max_ub, psi_ub, time_limit, gurobi_opts):
    """Exact same formulation as find_optimal_dispatch_quadratic, but with explicit variable bounds."""
    availabilities_df = availability_df

    model = pyo.ConcreteModel()
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, n_demand_bids + 1))

    # === PRIMARY VARIABLES (same as original) ===
    model.g = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)
    model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.k = pyo.Var(model.time, bounds=(1, k_max), within=pyo.NonNegativeReals)
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(0, 300))
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # === DUAL VARIABLES — with explicit upper bounds (ONLY change vs original) ===
    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.nu_max = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, nu_max_ub))
    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, psi_ub))

    # hat duals — with bounds
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(0, 300))
    model.mu_max_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.mu_min_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.nu_max_hat = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, nu_max_ub))
    model.nu_min_hat = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.pi_u_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))
    model.pi_d_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, dual_ub))

    # binaries
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # ==================== OBJECTIVE (identical to original) ====================
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

    # ==================== CONSTRAINTS (all identical to original) ====================
    model.balance = pyo.Constraint(model.time,
        rule=lambda m, t: sum(m.d[t, n] for n in m.demand_bids) - sum(m.g[i, t] for i in m.gens) == 0)

    model.g_max = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.g[i, t] <= gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * m.u[i, t])

    model.d_max = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] <= demand_df.at[t, f"volume_{n}"])

    def ru_max_rule(m, i, t):
        if t == 0:
            return m.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
        return m.g[i, t] - m.g[i, t - 1] <= gens_df.at[i, "r_up"]
    model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_max_rule)

    def rd_max_rule(m, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - m.g[i, t] <= gens_df.at[i, "r_down"]
        return m.g[i, t - 1] - m.g[i, t] <= gens_df.at[i, "r_down"]
    model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_max_rule)

    def start_up_cost_rule(m, i, t):
        if t == 0:
            return m.c_up[i, t] >= (m.u[i, t] - gens_df.at[i, "u_0"]) * gens_df.at[i, "k_up"]
        return m.c_up[i, t] >= (m.u[i, t] - m.u[i, t - 1]) * gens_df.at[i, "k_up"]
    model.start_up_cost = pyo.Constraint(model.gens, model.time, rule=start_up_cost_rule)

    def shut_down_cost_rule(m, i, t):
        if t == 0:
            return m.c_down[i, t] >= (gens_df.at[i, "u_0"] - m.u[i, t]) * gens_df.at[i, "k_down"]
        return m.c_down[i, t] >= (m.u[i, t - 1] - m.u[i, t]) * gens_df.at[i, "k_down"]
    model.shut_down_cost = pyo.Constraint(model.gens, model.time, rule=shut_down_cost_rule)

    # --- Dual constraints (identical) ---
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

    # --- KKT conditions (identical) ---
    def kkt_gen_rule(m, i, t):
        k_term = m.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_hat_next = 0 if t == m.time.at(-1) else m.pi_u_hat[i, t + 1]
        pi_d_hat_next = 0 if t == m.time.at(-1) else m.pi_d_hat[i, t + 1]
        return (k_term * mc_df.at[t, i] - m.lambda_hat[t] + m.mu_max_hat[i, t] - m.mu_min_hat[i, t]
                + m.pi_u_hat[i, t] - pi_u_hat_next - m.pi_d_hat[i, t] + pi_d_hat_next == 0)
    model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen_rule)

    model.kkt_demand = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: -demand_df.at[t, f"price_{n}"] + m.lambda_hat[t] + m.nu_max_hat[t, n] - m.nu_min_hat[t, n] == 0)

    # --- Complementarity (identical, using big_M) ---
    model.mu_max_hat_binary_constr_1 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.g[i, t] - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * m.u[i, t]
                              <= max(gens_df["g_max"]) * (1 - m.mu_max_hat_binary[i, t]))
    model.mu_max_hat_binary_constr_2 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.g[i, t] - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * m.u[i, t]
                              >= -max(gens_df["g_max"]) * (1 - m.mu_max_hat_binary[i, t]))
    model.mu_max_hat_binary_constr_3 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.mu_max_hat[i, t] <= big_M * m.mu_max_hat_binary[i, t])

    model.nu_max_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.nu_max_hat_binary_constr_1 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] - demand_df.at[t, f"volume_{n}"]
                              <= max(demand_df[f"volume_{n}"]) * (1 - m.nu_max_hat_binary[t, n]))
    model.nu_max_hat_binary_constr_2 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] - demand_df.at[t, f"volume_{n}"]
                              >= -max(demand_df[f"volume_{n}"]) * (1 - m.nu_max_hat_binary[t, n]))
    model.nu_max_hat_binary_constr_3 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.nu_max_hat[t, n] <= big_M * m.nu_max_hat_binary[t, n])

    model.nu_min_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.nu_min_hat_binary_constr_1 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.d[t, n] <= big_M * (1 - m.nu_min_hat_binary[t, n]))
    model.nu_min_hat_binary_constr_3 = pyo.Constraint(model.time, model.demand_bids,
        rule=lambda m, t, n: m.nu_min_hat[t, n] <= big_M * m.nu_min_hat_binary[t, n])

    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    def pi_u_hat_binary_rule_1(m, i, t):
        if t == 0:
            return m.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[i, "r_up"] <= big_M * (1 - m.pi_u_hat_binary[i, t])
        return m.g[i, t] - m.g[i, t - 1] - gens_df.at[i, "r_up"] <= big_M * (1 - m.pi_u_hat_binary[i, t])
    def pi_u_hat_binary_rule_2(m, i, t):
        if t == 0:
            return m.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[i, "r_up"] >= -big_M * (1 - m.pi_u_hat_binary[i, t])
        return m.g[i, t] - m.g[i, t - 1] - gens_df.at[i, "r_up"] >= -big_M * (1 - m.pi_u_hat_binary[i, t])
    model.pi_u_hat_binary_constr_1 = pyo.Constraint(model.gens, model.time, rule=pi_u_hat_binary_rule_1)
    model.pi_u_hat_binary_constr_2 = pyo.Constraint(model.gens, model.time, rule=pi_u_hat_binary_rule_2)
    model.pi_u_hat_binary_constr_3 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.pi_u_hat[i, t] <= big_M * m.pi_u_hat_binary[i, t])

    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    def pi_d_hat_binary_rule_1(m, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - m.g[i, t] - gens_df.at[i, "r_down"] <= big_M * (1 - m.pi_d_hat_binary[i, t])
        return m.g[i, t - 1] - m.g[i, t] - gens_df.at[i, "r_down"] <= big_M * (1 - m.pi_d_hat_binary[i, t])
    def pi_d_hat_binary_rule_2(m, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - m.g[i, t] - gens_df.at[i, "r_down"] >= -big_M * (1 - m.pi_d_hat_binary[i, t])
        return m.g[i, t - 1] - m.g[i, t] - gens_df.at[i, "r_down"] >= -big_M * (1 - m.pi_d_hat_binary[i, t])
    model.pi_d_hat_binary_constr_1 = pyo.Constraint(model.gens, model.time, rule=pi_d_hat_binary_rule_1)
    model.pi_d_hat_binary_constr_2 = pyo.Constraint(model.gens, model.time, rule=pi_d_hat_binary_rule_2)
    model.pi_d_hat_binary_constr_3 = pyo.Constraint(model.gens, model.time,
        rule=lambda m, i, t: m.pi_d_hat[i, t] <= big_M * m.pi_d_hat_binary[i, t])

    # === SOLVE ===
    instance = model.create_instance()
    solver = SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    results = solver.solve(instance, options=gurobi_opts, tee=True)

    tc = results.solver.termination_condition
    print(f"\nTermination: {tc}")

    if tc in (pyo.TerminationCondition.infeasible, pyo.TerminationCondition.infeasibleOrUnbounded):
        return None

    ti = demand_df.index
    mcp = [float(instance.lambda_[t].value) for t in ti]
    mcp_hat = [float(instance.lambda_hat[t].value) for t in ti]
    k_vals = [float(instance.k[t].value) for t in ti]
    gen_vals = [float(instance.g[opt_gen, t].value) for t in ti]
    demand_vals = [sum(float(instance.d[t, n].value) for n in instance.demand_bids) for t in ti]
    return {'mcp': mcp, 'mcp_hat': mcp_hat, 'k': k_vals,
            f'gen_{opt_gen}': gen_vals, 'demand': demand_vals}


# ===================== RUN EXPERIMENTS =====================
import sys

configs = {
    "A": {
        "desc": "big_M=10000, dual_ub=500, nu_max_ub=3100, psi_ub=500000",
        "big_M": 10000, "dual_ub": 500, "nu_max_ub": 3100, "psi_ub": 500000,
        "time_limit": 1800,
        "gurobi": {"LogToConsole": 1, "TimeLimit": 1800, "MIPGap": 0.05,
                   "DualReductions": 0, "MIPFocus": 1, "Presolve": 2, "Cuts": 2},
    },
    "B": {
        "desc": "big_M=10000, dual_ub=1000, nu_max_ub=3100, psi_ub=1000000",
        "big_M": 10000, "dual_ub": 1000, "nu_max_ub": 3100, "psi_ub": 1000000,
        "time_limit": 1800,
        "gurobi": {"LogToConsole": 1, "TimeLimit": 1800, "MIPGap": 0.05,
                   "DualReductions": 0, "MIPFocus": 1, "Presolve": 2, "Cuts": 2},
    },
    "C": {
        "desc": "big_M=10000, dual_ub=300, nu_max_ub=3100, psi_ub=300000, aggressive gurobi",
        "big_M": 10000, "dual_ub": 300, "nu_max_ub": 3100, "psi_ub": 300000,
        "time_limit": 1800,
        "gurobi": {"LogToConsole": 1, "TimeLimit": 1800, "MIPGap": 0.10,
                   "DualReductions": 0, "MIPFocus": 1, "Presolve": 2, "Cuts": 3,
                   "Heuristics": 0.3},
    },
}

# Select config from command line: python test_with_var_bounds.py A
config_key = sys.argv[1] if len(sys.argv) > 1 else "A"
cfg = configs[config_key]

print(f"{'=' * 70}")
print(f"Config {config_key}: {cfg['desc']}")
print(f"opt_gen={opt_gen}, k_max={k_max}, big_w={big_w}")
print(f"{'=' * 70}")

t0 = time.time()
res = solve_with_bounds(
    big_M=cfg["big_M"], dual_ub=cfg["dual_ub"], nu_max_ub=cfg["nu_max_ub"],
    psi_ub=cfg["psi_ub"], time_limit=cfg["time_limit"], gurobi_opts=cfg["gurobi"])
elapsed = time.time() - t0

if res:
    import numpy as np
    mcp = np.array(res['mcp'])
    mcp_hat = np.array(res['mcp_hat'])
    k = np.array(res['k'])
    g = np.array(res[f'gen_{opt_gen}'])
    d = np.array(res['demand'])
    print(f"\nTime: {elapsed:.1f}s")
    print(f"MCP (lambda_):   min={mcp.min():.2f}, max={mcp.max():.2f}, mean={mcp.mean():.2f}")
    print(f"MCP_hat:         min={mcp_hat.min():.2f}, max={mcp_hat.max():.2f}, mean={mcp_hat.mean():.2f}")
    print(f"K:               min={k.min():.4f}, max={k.max():.4f}, mean={k.mean():.4f}")
    print(f"Gen {opt_gen}:   min={g.min():.0f}, max={g.max():.0f}")
    print(f"Demand:          min={d.min():.0f}, max={d.max():.0f}")
    profit = sum((mcp_hat - gens_df.at[opt_gen, 'mc']) * g)
    print(f"Profit (hat): {profit:.0f} EUR/day")
else:
    print(f"\nINFEASIBLE ({elapsed:.1f}s)")
