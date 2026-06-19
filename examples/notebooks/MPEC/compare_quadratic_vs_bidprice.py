# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Compare quadratic k-multiplier MPEC vs bid-price MPEC,
both with fixed storage dispatch.
"""

import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def find_optimal_dispatch_quadratic_fixed_storage(
    gens_df,
    k_values_df,
    availabilities_df,
    demand_df,
    k_max,
    opt_gen,
    fixed_storage_dispatch=None,
    big_w=10000,
    time_limit=600,
    big_M=10e6,
    demand_bids=1,
    mc_df=None,
):
    """Quadratic MPEC (k-multiplier) with fixed storage dispatch in balance."""
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
            - model.lambda_[t]
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

    # --- SOLVE ---
    instance = model.create_instance()
    solver = SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    options = {
        "LogToConsole": 1,
        "TimeLimit": time_limit,
        "MIPGap": 0.03,
        "DualReductions": 0,
    }
    results = solver.solve(instance, options=options, tee=True)

    print(f"\nSolver status: {results.solver.status}")
    print(f"Termination: {results.solver.termination_condition}")

    if results.solver.termination_condition in (
        pyo.TerminationCondition.infeasible,
        pyo.TerminationCondition.infeasibleOrUnbounded,
    ):
        instance.write(
            "debug_quadratic.lp", io_options={"symbolic_solver_labels": True}
        )
        print("Wrote debug_quadratic.lp")
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


# =========================================================================
# MAIN: load data and run both formulations
# =========================================================================
if __name__ == "__main__":
    from bilevel_opt_bidprice import find_optimal_dispatch_bidprice

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

    gens_df_idx = (
        gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df.copy()
    )
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

    # --- RUN 1: Quadratic k-multiplier MPEC ---
    print("=" * 70)
    mc_opt_gen = mc_df[opt_gen].iloc[0]
    k_max = int(3100 / mc_opt_gen) if mc_opt_gen > 0 else 300000
    print(f"QUADRATIC K-MULTIPLIER MPEC (k bounded 1..{k_max}, mc={mc_opt_gen})")
    print("=" * 70)
    t0 = time.time()
    q_main, q_supp, q_kvals = find_optimal_dispatch_quadratic_fixed_storage(
        gens_df=gens_df_idx,
        k_values_df=k_values_r,
        availabilities_df=avail_r,
        demand_df=demand_df_r,
        k_max=k_max,
        opt_gen=opt_gen,
        fixed_storage_dispatch=fixed_sd,
        big_w=10000,
        time_limit=600,
        big_M=10e6,
        demand_bids=10,
        mc_df=mc_df,
    )
    t_quad = time.time() - t0

    # --- RUN 2: Bid-price MPEC ---
    print("\n" + "=" * 70)
    print("BID-PRICE MPEC (bid_price bounded -500..3100)")
    print("=" * 70)
    t0 = time.time()
    b_main, b_supp, b_bidprices = find_optimal_dispatch_bidprice(
        gens_df=gens_df_idx,
        k_values_df=k_values_r,
        availabilities_df=avail_r,
        demand_df=demand_df_r,
        opt_gen=opt_gen,
        fixed_storage_dispatch=fixed_sd,
        bid_price_max=3100,
        big_w=10000,
        time_limit=600,
        big_M=10000,
        demand_bids=10,
        mc_df=mc_df,
    )
    t_bid = time.time() - t0

    # --- COMPARISON ---
    print("\n" + "=" * 70)
    print(f"COMPARISON  (quadratic: {t_quad:.1f}s, bidprice: {t_bid:.1f}s)")
    print("=" * 70)

    gen_col = f"gen_{opt_gen}"
    mc_opt = mc_df[opt_gen].iloc[0]

    print(f"\n{'t':>3} | {'--- QUADRATIC ---':^42} | {'--- BID-PRICE ---':^42}")
    print(
        f"{'':>3} | {'demand':>10} {'mcp':>8} {'mcp_hat':>8} {'gen':>8} {'k':>6} | {'demand':>10} {'mcp':>8} {'mcp_hat':>8} {'gen':>8} {'bid':>8}"
    )
    print("-" * 105)
    for t in q_main.index:
        qd = float(q_main.at[t, "demand"])
        qm = float(q_main.at[t, "mcp"])
        qh = float(q_main.at[t, "mcp_hat"])
        qg = float(q_main.at[t, gen_col])
        qk = float(q_kvals.at[t, "k"])

        bd = float(b_main.at[t, "demand"])
        bm = float(b_main.at[t, "mcp"])
        bh = float(b_main.at[t, "mcp_hat"])
        bg = float(b_main.at[t, gen_col])
        bb = float(b_bidprices.at[t, "bid_price"])

        print(
            f"{t:3d} | {qd:10.0f} {qm:8.2f} {qh:8.2f} {qg:8.2f} {qk:6.2f}"
            f" | {bd:10.0f} {bm:8.2f} {bh:8.2f} {bg:8.2f} {bb:8.2f}"
        )

    q_hat_gap = abs(q_main["mcp_hat"].astype(float) - q_main["mcp"].astype(float))
    b_hat_gap = abs(b_main["mcp_hat"].astype(float) - b_main["mcp"].astype(float))
    print(
        f"\n|lambda_hat - lambda_| mean:  quadratic={q_hat_gap.mean():.2f}  bidprice={b_hat_gap.mean():.2f}"
    )
    print(
        f"|lambda_hat - lambda_| max:   quadratic={q_hat_gap.max():.2f}  bidprice={b_hat_gap.max():.2f}"
    )

    q_profit = (
        q_main[gen_col].astype(float) * (q_main["mcp"].astype(float) - mc_opt)
    ).sum()
    b_profit = (
        b_main[gen_col].astype(float) * (b_main["mcp"].astype(float) - mc_opt)
    ).sum()
    q_profit_hat = (
        q_main[gen_col].astype(float) * (q_main["mcp_hat"].astype(float) - mc_opt)
    ).sum()
    b_profit_hat = (
        b_main[gen_col].astype(float) * (b_main["mcp_hat"].astype(float) - mc_opt)
    ).sum()

    print(
        f"\nProfit (lambda_):    quadratic={q_profit:>12,.0f} EUR   bidprice={b_profit:>12,.0f} EUR"
    )
    print(
        f"Profit (lambda_hat): quadratic={q_profit_hat:>12,.0f} EUR   bidprice={b_profit_hat:>12,.0f} EUR"
    )
