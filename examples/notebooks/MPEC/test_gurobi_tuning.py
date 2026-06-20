# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Test Gurobi solver tuning parameters on the MPEC problem."""

import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def find_optimal_dispatch_quadratic_fixed_storage_tuned(
    gens_df,
    k_values_df,
    availabilities_df,
    demand_df,
    k_max,
    opt_gen,
    fixed_storage_dispatch=None,
    big_w=10000,
    time_limit=600,
    big_M=10000,
    demand_bids=1,
    mc_df=None,
):
    """Quadratic MPEC with aggressive Gurobi tuning.

    Copied from bilevel_opt.py find_optimal_dispatch_quadratic_fixed_storage
    with modified solver options for performance testing.
    """
    gens_df = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df
    if mc_df is None:
        mc_df = pd.DataFrame(
            {gen: gens_df.at[gen, "mc"] for gen in gens_df.index},
            index=demand_df.index,
        )
    if fixed_storage_dispatch is None:
        fixed_storage_dispatch = pd.Series(0.0, index=demand_df.index)

    model = pyo.ConcreteModel()

    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, demand_bids + 1))

    model.g = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)
    model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.k = pyo.Var(model.time, bounds=(1, k_max), within=pyo.NonNegativeReals)
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3100))
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)
    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3100))
    model.mu_max_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max_hat = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals
    )
    model.nu_min_hat = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals
    )
    model.pi_u_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # --- OBJECTIVE ---
    def primary_objective_rule(model):
        return sum(
            model.lambda_hat[t] * model.g[opt_gen, t]
            - mc_df.at[t, opt_gen] * model.g[opt_gen, t]
            - model.c_up[opt_gen, t]
            - model.c_down[opt_gen, t]
            for t in model.time
        )

    def duality_gap_part_1_rule(model):
        expr = sum(
            (
                (
                    mc_df.at[t, gen] * model.k[t] * model.g[gen, t]
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )
                if gen == opt_gen
                else (
                    k_values_df.at[t, gen] * mc_df.at[t, gen] * model.g[gen, t]
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )
            )
            for gen in model.gens
            for t in model.time
        )
        expr -= sum(
            demand_df.at[t, f"price_{n}"] * model.d[t, n]
            for t in model.time
            for n in model.demand_bids
        )
        return expr

    def duality_gap_part_2_rule(model):
        expr = -sum(
            model.nu_max[t, n] * demand_df.at[t, f"volume_{n}"]
            for t in model.time
            for n in model.demand_bids
        )
        expr -= sum(
            model.pi_u[i, t] * gens_df.at[i, "r_up"]
            for i in model.gens
            for t in model.time
        )
        expr -= sum(
            model.pi_d[i, t] * gens_df.at[i, "r_down"]
            for i in model.gens
            for t in model.time
        )
        expr -= sum(model.pi_u[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)
        expr += sum(model.pi_d[i, 0] * gens_df.at[i, "g_0"] for i in model.gens)
        expr -= sum(
            model.sigma_u[i, 0] * gens_df.at[i, "k_up"] * gens_df.at[i, "u_0"]
            for i in model.gens
        )
        expr += sum(
            model.sigma_d[i, 0] * gens_df.at[i, "k_down"] * gens_df.at[i, "u_0"]
            for i in model.gens
        )
        expr -= sum(model.psi_max[i, t] for i in model.gens for t in model.time)
        expr += sum(
            model.lambda_[t] * float(fixed_storage_dispatch.at[t]) for t in model.time
        )
        return expr

    def final_objective_rule(model):
        return primary_objective_rule(model) - big_w * (
            duality_gap_part_1_rule(model) - duality_gap_part_2_rule(model)
        )

    model.objective = pyo.Objective(expr=final_objective_rule, sense=pyo.maximize)

    # --- PRIMAL CONSTRAINTS ---
    def balance_rule(model, t):
        net_storage = float(fixed_storage_dispatch.at[t])
        return (
            sum(model.d[t, n] for n in model.demand_bids)
            - sum(model.g[i, t] for i in model.gens)
            - net_storage
            == 0
        )

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    def g_max_rule(model, i, t):
        return (
            model.g[i, t]
            <= gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        )

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    def d_max_rule(model, t, n):
        return model.d[t, n] <= demand_df.at[t, f"volume_{n}"]

    model.d_max = pyo.Constraint(model.time, model.demand_bids, rule=d_max_rule)

    def ru_max_rule(model, i, t):
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
        else:
            return model.g[i, t] - model.g[i, t - 1] <= gens_df.at[i, "r_up"]

    model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_max_rule)

    def rd_max_rule(model, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] <= gens_df.at[i, "r_down"]
        else:
            return model.g[i, t - 1] - model.g[i, t] <= gens_df.at[i, "r_down"]

    model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_max_rule)

    def start_up_cost_rule(model, i, t):
        if t == 0:
            return (
                model.c_up[i, t]
                >= (model.u[i, t] - gens_df.at[i, "u_0"]) * gens_df.at[i, "k_up"]
            )
        else:
            return (
                model.c_up[i, t]
                >= (model.u[i, t] - model.u[i, t - 1]) * gens_df.at[i, "k_up"]
            )

    model.start_up_cost = pyo.Constraint(
        model.gens, model.time, rule=start_up_cost_rule
    )

    def shut_down_cost_rule(model, i, t):
        if t == 0:
            return (
                model.c_down[i, t]
                >= (gens_df.at[i, "u_0"] - model.u[i, t]) * gens_df.at[i, "k_down"]
            )
        else:
            return (
                model.c_down[i, t]
                >= (model.u[i, t - 1] - model.u[i, t]) * gens_df.at[i, "k_down"]
            )

    model.shut_down_cost = pyo.Constraint(
        model.gens, model.time, rule=shut_down_cost_rule
    )

    # --- DUAL FEASIBILITY (non-hat) ---
    def gen_dual_rule(model, i, t):
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_next = 0 if t == model.time.at(-1) else model.pi_u[i, t + 1]
        pi_d_next = 0 if t == model.time.at(-1) else model.pi_d[i, t + 1]
        return (
            k_term * mc_df.at[t, i]
            - model.lambda_[t] * availabilities_df.at[t, i]
            + model.mu_max[i, t]
            - model.mu_min[i, t]
            + model.pi_u[i, t]
            - pi_u_next
            - model.pi_d[i, t]
            + pi_d_next
            == 0
        )

    model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual_rule)

    def status_dual_rule(model, i, t):
        if t != model.time.at(-1):
            return (
                -model.mu_max[i, t]
                * gens_df.at[i, "g_max"]
                * availabilities_df.at[t, i]
                + (model.sigma_u[i, t] - model.sigma_u[i, t + 1])
                * gens_df.at[i, "k_up"]
                - (model.sigma_d[i, t] - model.sigma_d[i, t + 1])
                * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= 0
            )
        else:
            return (
                -model.mu_max[i, t]
                * gens_df.at[i, "g_max"]
                * availabilities_df.at[t, i]
                + model.sigma_u[i, t] * gens_df.at[i, "k_up"]
                - model.sigma_d[i, t] * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= 0
            )

    model.status_dual = pyo.Constraint(model.gens, model.time, rule=status_dual_rule)

    def demand_dual_rule(model, t, n):
        return (
            -demand_df.at[t, f"price_{n}"] + model.lambda_[t] + model.nu_max[t, n] >= 0
        )

    model.demand_dual = pyo.Constraint(
        model.time, model.demand_bids, rule=demand_dual_rule
    )

    # --- KKT STATIONARITY (hat) ---
    def kkt_gen_rule(model, i, t):
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_hat_next = 0 if t == model.time.at(-1) else model.pi_u_hat[i, t + 1]
        pi_d_hat_next = 0 if t == model.time.at(-1) else model.pi_d_hat[i, t + 1]
        return (
            k_term * mc_df.at[t, i]
            - model.lambda_hat[t]
            + model.mu_max_hat[i, t]
            - model.mu_min_hat[i, t]
            + model.pi_u_hat[i, t]
            - pi_u_hat_next
            - model.pi_d_hat[i, t]
            + pi_d_hat_next
            == 0
        )

    model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen_rule)

    def kkt_demand_rule(model, t, n):
        return (
            -demand_df.at[t, f"price_{n}"]
            + model.lambda_hat[t]
            + model.nu_max_hat[t, n]
            - model.nu_min_hat[t, n]
            == 0
        )

    model.kkt_demand = pyo.Constraint(
        model.time, model.demand_bids, rule=kkt_demand_rule
    )

    # --- COMPLEMENTARITY (Big-M) ---
    def mu_max_hat_binary_rule_1(model, i, t):
        return (
            model.g[i, t]
            - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        ) <= max(gens_df["g_max"]) * (1 - model.mu_max_hat_binary[i, t])

    def mu_max_hat_binary_rule_2(model, i, t):
        return (
            model.g[i, t]
            - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        ) >= -max(gens_df["g_max"]) * (1 - model.mu_max_hat_binary[i, t])

    def mu_max_hat_binary_rule_3(model, i, t):
        return model.mu_max_hat[i, t] <= big_M * model.mu_max_hat_binary[i, t]

    model.mu_max_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_1
    )
    model.mu_max_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_2
    )
    model.mu_max_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=mu_max_hat_binary_rule_3
    )

    model.nu_max_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)

    def nu_max_hat_binary_rule_1(model, t, n):
        return model.d[t, n] - demand_df.at[t, f"volume_{n}"] <= max(
            demand_df[f"volume_{n}"]
        ) * (1 - model.nu_max_hat_binary[t, n])

    def nu_max_hat_binary_rule_2(model, t, n):
        return model.d[t, n] - demand_df.at[t, f"volume_{n}"] >= -max(
            demand_df[f"volume_{n}"]
        ) * (1 - model.nu_max_hat_binary[t, n])

    def nu_max_hat_binary_rule_3(model, t, n):
        return model.nu_max_hat[t, n] <= big_M * model.nu_max_hat_binary[t, n]

    model.nu_max_hat_binary_constr_1 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_max_hat_binary_rule_1
    )
    model.nu_max_hat_binary_constr_2 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_max_hat_binary_rule_2
    )
    model.nu_max_hat_binary_constr_3 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_max_hat_binary_rule_3
    )

    model.nu_min_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)

    def nu_min_hat_binary_rule_1(model, t, n):
        return model.d[t, n] <= demand_df.at[t, f"volume_{n}"] * (
            1 - model.nu_min_hat_binary[t, n]
        )

    def nu_min_hat_binary_rule_3(model, t, n):
        return model.nu_min_hat[t, n] <= big_M * model.nu_min_hat_binary[t, n]

    model.nu_min_hat_binary_constr_1 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_min_hat_binary_rule_1
    )
    model.nu_min_hat_binary_constr_3 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_min_hat_binary_rule_3
    )

    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_u_hat_binary_rule_1(model, i, t):
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[
                i, "r_up"
            ] <= big_M * (1 - model.pi_u_hat_binary[i, t])
        else:
            return model.g[i, t] - model.g[i, t - 1] - gens_df.at[
                i, "r_up"
            ] <= big_M * (1 - model.pi_u_hat_binary[i, t])

    def pi_u_hat_binary_rule_2(model, i, t):
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[
                i, "r_up"
            ] >= -big_M * (1 - model.pi_u_hat_binary[i, t])
        else:
            return model.g[i, t] - model.g[i, t - 1] - gens_df.at[
                i, "r_up"
            ] >= -big_M * (1 - model.pi_u_hat_binary[i, t])

    def pi_u_hat_binary_rule_3(model, i, t):
        return model.pi_u_hat[i, t] <= big_M * model.pi_u_hat_binary[i, t]

    model.pi_u_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_1
    )
    model.pi_u_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_2
    )
    model.pi_u_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=pi_u_hat_binary_rule_3
    )

    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_d_hat_binary_rule_1(model, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] <= big_M * (1 - model.pi_d_hat_binary[i, t])
        else:
            return model.g[i, t - 1] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] <= big_M * (1 - model.pi_d_hat_binary[i, t])

    def pi_d_hat_binary_rule_2(model, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] >= -big_M * (1 - model.pi_d_hat_binary[i, t])
        else:
            return model.g[i, t - 1] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] >= -big_M * (1 - model.pi_d_hat_binary[i, t])

    def pi_d_hat_binary_rule_3(model, i, t):
        return model.pi_d_hat[i, t] <= big_M * model.pi_d_hat_binary[i, t]

    model.pi_d_hat_binary_constr_1 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_1
    )
    model.pi_d_hat_binary_constr_2 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_2
    )
    model.pi_d_hat_binary_constr_3 = pyo.Constraint(
        model.gens, model.time, rule=pi_d_hat_binary_rule_3
    )

    # --- SOLVE with tuned parameters ---
    instance = model.create_instance()
    solver = SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    options = {
        "LogToConsole": 1,
        "TimeLimit": time_limit,
        "MIPGap": 0.05,          # relaxed from 0.03
        "MIPFocus": 1,           # focus on finding feasible solutions
        "Heuristics": 0.2,       # more heuristic effort
        "Presolve": 2,           # aggressive presolve
        "Cuts": 2,               # aggressive cuts
        "DualReductions": 0,
    }

    wall_start = time.perf_counter()
    results = solver.solve(instance, options=options, tee=True)
    wall_end = time.perf_counter()

    wall_time = wall_end - wall_start

    print(f"\n{'='*60}")
    print(f"Solver status:  {results.solver.status}")
    print(f"Termination:    {results.solver.termination_condition}")
    print(f"Wall-clock time: {wall_time:.2f} s")
    print(f"{'='*60}")

    if results.solver.termination_condition in (
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    ):
        print("INFEASIBLE - no results to show.")
        return None, None, None

    # --- EXTRACT ---
    time_index = demand_df.index
    generation_df = pd.DataFrame(
        index=time_index, columns=[f"gen_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in time_index:
            generation_df.at[t, f"gen_{gen}"] = instance.g[gen, t].value

    cleared_demand = pd.DataFrame(index=time_index, columns=["demand"])
    for t in time_index:
        cleared_demand.at[t, "demand"] = sum(
            instance.d[t, n].value for n in instance.demand_bids
        )

    mcp = pd.DataFrame(index=time_index, columns=["mcp"])
    mcp_hat = pd.DataFrame(index=time_index, columns=["mcp_hat"])
    for t in time_index:
        mcp.at[t, "mcp"] = instance.lambda_[t].value
        mcp_hat.at[t, "mcp_hat"] = instance.lambda_hat[t].value

    main_df = pd.concat([generation_df, cleared_demand, mcp, mcp_hat], axis=1)

    supp_df_parts = []
    for prefix, var in [("start_up", instance.c_up), ("shut_down", instance.c_down)]:
        df = pd.DataFrame(
            index=time_index, columns=[f"{prefix}_{gen}" for gen in gens_df.index]
        )
        for gen in gens_df.index:
            for t in time_index:
                df.at[t, f"{prefix}_{gen}"] = var[gen, t].value
        supp_df_parts.append(df)
    supp_df = pd.concat(supp_df_parts, axis=1)

    k_values = pd.DataFrame(index=time_index, columns=["k"])
    for t in time_index:
        k_values.at[t, "k"] = instance.k[t].value

    return main_df, supp_df, k_values


# ============================================================
# Main: load data, prepare, and run
# ============================================================
if __name__ == "__main__":
    data_dir = "mpec_input_data_02a"

    # --- Load ---
    gens_df = pd.read_csv(f"{data_dir}/gens_df.csv", index_col=0)
    demand_df = pd.read_csv(f"{data_dir}/demand_df.csv", parse_dates=["datetime"])
    demand_df = demand_df.set_index("datetime")
    k_values_df = pd.read_csv(f"{data_dir}/k_values_df.csv", parse_dates=["time"])
    k_values_df = k_values_df.set_index("time")
    k_values_df = k_values_df.drop(columns=["date"], errors="ignore")
    k_values_df = k_values_df.fillna(0.0)
    availability_df = pd.read_csv(
        f"{data_dir}/availability_df.csv", parse_dates=["time"]
    )
    availability_df = availability_df.set_index("time")

    # --- Filter to 2019-03-16 ---
    date_filter = (demand_df.index >= "2019-03-16") & (demand_df.index < "2019-03-17")
    demand_df = demand_df.loc[date_filter].copy()
    demand_df = demand_df.drop(columns=["date"], errors="ignore")

    k_date_filter = (k_values_df.index >= "2019-03-16") & (
        k_values_df.index < "2019-03-17"
    )
    k_values_df = k_values_df.loc[k_date_filter].copy()

    avail_date_filter = (availability_df.index >= "2019-03-16") & (
        availability_df.index < "2019-03-17"
    )
    availability_df = availability_df.loc[avail_date_filter].copy()

    # Reset indices
    demand_df = demand_df.reset_index(drop=True)
    k_values_df = k_values_df.reset_index(drop=True)
    availability_df = availability_df.reset_index(drop=True)

    # Filter to generator columns only
    gen_units = gens_df["unit"].tolist()
    k_values_df = k_values_df[[c for c in k_values_df.columns if c in gen_units]]
    availability_df = availability_df[
        [c for c in availability_df.columns if c in gen_units]
    ]

    # --- Make demand volumes positive ---
    vol_cols = [c for c in demand_df.columns if c.startswith("volume_")]
    demand_df[vol_cols] = demand_df[vol_cols].abs().clip(lower=0.01)

    # --- Auto-detect n_demand_bids ---
    n_demand_bids = len(vol_cols)

    # --- Pick unit pp_3 ---
    opt_gen = "pp_3"
    gens_row = gens_df[gens_df["unit"] == opt_gen].iloc[0]
    k_max = gens_row["mc"]  # use mc as k_max proxy (99 per task description)

    # Override k_max to 99 as specified
    k_max = 99

    # --- Build mc_df ---
    gens_df_indexed = gens_df.set_index("unit")
    mc_df = pd.DataFrame(
        {gen: gens_df_indexed.at[gen, "mc"] for gen in gens_df_indexed.index},
        index=demand_df.index,
    )

    print(f"Data loaded for 2019-03-16:")
    print(f"  Timesteps: {len(demand_df)}")
    print(f"  Generators: {list(k_values_df.columns)}")
    print(f"  Demand bids: {n_demand_bids}")
    print(f"  Opt gen: {opt_gen} (mc={gens_df_indexed.at[opt_gen, 'mc']}, k_max={k_max})")
    print(f"  big_M = 10000 (tight)")
    print(f"  Tuned params: MIPGap=0.05, MIPFocus=1, Heuristics=0.2, Presolve=2, Cuts=2")
    print()

    # --- Run ---
    main_df, supp_df, k_values = find_optimal_dispatch_quadratic_fixed_storage_tuned(
        gens_df=gens_df,
        k_values_df=k_values_df,
        availabilities_df=availability_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_M=10000,
        demand_bids=n_demand_bids,
        mc_df=mc_df,
    )

    if main_df is not None:
        print("\n--- Key Results ---")
        print(f"\nGeneration for {opt_gen}:")
        print(main_df[f"gen_{opt_gen}"].to_string())
        print(f"\nMarket clearing price (lambda_hat):")
        print(main_df["mcp_hat"].to_string())
        print(f"\nBid multiplier k:")
        print(k_values["k"].to_string())
        print(f"\nTotal demand cleared:")
        print(main_df["demand"].to_string())

        # Summary stats
        gen_vals = main_df[f"gen_{opt_gen}"].astype(float)
        mcp_vals = main_df["mcp_hat"].astype(float)
        k_vals = k_values["k"].astype(float)
        print(f"\n--- Summary ---")
        print(f"  {opt_gen} avg generation: {gen_vals.mean():.2f} MW")
        print(f"  Avg MCP (hat): {mcp_vals.mean():.2f}")
        print(f"  Avg k: {k_vals.mean():.4f}")
        print(f"  k range: [{k_vals.min():.4f}, {k_vals.max():.4f}]")
    else:
        print("\nSolve failed (infeasible).")
