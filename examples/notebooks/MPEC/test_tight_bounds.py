# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Test script: tighten variable bounds on dual variables + generation + demand,
and reduce big_M to see if Gurobi solves faster.
"""

import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def find_optimal_dispatch_tight_bounds(
    gens_df,
    k_values_df,
    availabilities_df,
    demand_df,
    k_max,
    opt_gen,
    fixed_storage_dispatch=None,
    big_w=10000,
    time_limit=600,
    big_M=3500,
    demand_bids=1,
    mc_df=None,
):
    """MPEC with tightened variable bounds on duals, generation, and demand."""
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

    # --- TIGHTENED PRIMAL VARIABLES ---
    # Generation bounded per-unit by g_max
    model.g = pyo.Var(
        model.gens,
        model.time,
        within=pyo.NonNegativeReals,
        bounds=lambda m, i, t: (0, gens_df.at[i, "g_max"]),
    )
    # Demand bounded per-bid by volume
    model.d = pyo.Var(
        model.time,
        model.demand_bids,
        within=pyo.NonNegativeReals,
        bounds=lambda m, t, n: (0, demand_df.at[t, f"volume_{n}"]),
    )
    model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.k = pyo.Var(model.time, bounds=(1, k_max), within=pyo.NonNegativeReals)
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3100))
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # --- TIGHTENED DUAL VARIABLES ---
    model.mu_max = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.mu_min = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.nu_max = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.pi_u = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.pi_d = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.psi_max = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 5000)
    )

    # --- TIGHTENED HAT DUALS ---
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 3100))
    model.mu_max_hat = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.mu_min_hat = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.nu_max_hat = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.nu_min_hat = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.pi_u_hat = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )
    model.pi_d_hat = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals, bounds=(0, 3500)
    )

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
    # mu_max_hat: g - g_max*avail*u complementary to mu_max_hat
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

    # nu_max_hat: d - volume complementary to nu_max_hat
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

    # nu_min_hat: d >= 0 complementary to nu_min_hat
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

    # pi_u_hat: ramp-up slack complementary to pi_u_hat
    # USE per-unit g_max instead of big_M for primal ramp slack forcing
    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_u_hat_binary_rule_1(model, i, t):
        ramp_bound = gens_df.at[i, "g_max"]  # max possible ramp change
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[
                i, "r_up"
            ] <= ramp_bound * (1 - model.pi_u_hat_binary[i, t])
        else:
            return model.g[i, t] - model.g[i, t - 1] - gens_df.at[
                i, "r_up"
            ] <= ramp_bound * (1 - model.pi_u_hat_binary[i, t])

    def pi_u_hat_binary_rule_2(model, i, t):
        ramp_bound = gens_df.at[i, "g_max"]  # max possible ramp change
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] - gens_df.at[
                i, "r_up"
            ] >= -ramp_bound * (1 - model.pi_u_hat_binary[i, t])
        else:
            return model.g[i, t] - model.g[i, t - 1] - gens_df.at[
                i, "r_up"
            ] >= -ramp_bound * (1 - model.pi_u_hat_binary[i, t])

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

    # pi_d_hat: ramp-down slack complementary to pi_d_hat
    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    def pi_d_hat_binary_rule_1(model, i, t):
        ramp_bound = gens_df.at[i, "g_max"]  # max possible ramp change
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] <= ramp_bound * (1 - model.pi_d_hat_binary[i, t])
        else:
            return model.g[i, t - 1] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] <= ramp_bound * (1 - model.pi_d_hat_binary[i, t])

    def pi_d_hat_binary_rule_2(model, i, t):
        ramp_bound = gens_df.at[i, "g_max"]  # max possible ramp change
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] >= -ramp_bound * (1 - model.pi_d_hat_binary[i, t])
        else:
            return model.g[i, t - 1] - model.g[i, t] - gens_df.at[
                i, "r_down"
            ] >= -ramp_bound * (1 - model.pi_d_hat_binary[i, t])

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
        "MIPGap": 0.05,
        "MIPFocus": 1,
        "Presolve": 2,
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
            "debug_tight_bounds.lp", io_options={"symbolic_solver_labels": True}
        )
        print("Wrote debug_tight_bounds.lp")
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


# =============================================================================
# Main: load data and run
# =============================================================================
if __name__ == "__main__":
    import os

    data_dir = os.path.join(os.path.dirname(__file__), "mpec_input_data_02a")

    # Load data
    gens_df = pd.read_csv(os.path.join(data_dir, "gens_df.csv"), index_col=0)
    demand_df = pd.read_csv(
        os.path.join(data_dir, "demand_df.csv"), index_col="datetime"
    )
    k_values_df = pd.read_csv(
        os.path.join(data_dir, "k_values_df.csv"), index_col="time"
    )
    availability_df = pd.read_csv(
        os.path.join(data_dir, "availability_df.csv"), index_col="time"
    )

    # Filter to 2019-03-16
    demand_df = demand_df[
        (demand_df.index >= "2019-03-16") & (demand_df.index < "2019-03-17")
    ].copy()
    k_values_df = k_values_df[
        (k_values_df.index >= "2019-03-16") & (k_values_df.index < "2019-03-17")
    ].copy()
    availability_df = availability_df[
        (availability_df.index >= "2019-03-16")
        & (availability_df.index < "2019-03-17")
    ].copy()

    # Reset indices to 0-based integer
    demand_df = demand_df.reset_index(drop=True)
    k_values_df = k_values_df.reset_index(drop=True)
    availability_df = availability_df.reset_index(drop=True)

    # Make volumes positive
    vol_cols = [c for c in demand_df.columns if c.startswith("volume_")]
    demand_df[vol_cols] = demand_df[vol_cols].abs().clip(lower=0.01)

    # Auto-detect n_demand_bids
    n_demand_bids = len(vol_cols)
    print(f"Detected {n_demand_bids} demand bids")

    # Drop date column if present
    if "date" in demand_df.columns:
        demand_df = demand_df.drop(columns=["date"])
    if "date" in k_values_df.columns:
        k_values_df = k_values_df.drop(columns=["date"])

    # Setup gens_df
    gens_df = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df

    # Strategic unit
    opt_gen = "pp_3"
    k_max = 99

    print(f"Strategic unit: {opt_gen}")
    print(f"  mc = {gens_df.at[opt_gen, 'mc']:.2f}")
    print(f"  k_max = {k_max}")
    print(f"  g_max = {gens_df.at[opt_gen, 'g_max']}")

    # Filter k_values and availability to generator columns only
    gen_cols = [c for c in k_values_df.columns if c in gens_df.index]
    k_values_df = k_values_df[gen_cols]
    avail_cols = [c for c in availability_df.columns if c in gens_df.index]
    availability_df = availability_df[avail_cols]

    # Build mc_df
    mc_df = pd.DataFrame(
        {gen: gens_df.at[gen, "mc"] for gen in gens_df.index},
        index=demand_df.index,
    )

    print(f"\nTimesteps: {len(demand_df)}")
    print(f"Generators: {list(gens_df.index)}")
    print(f"big_M = 3500 (tightened from 10e6)")
    print(f"Dual bounds: 3500, psi_max bound: 5000")
    print(f"Generation bounds: per-unit g_max")
    print(f"Demand bounds: per-bid volume")
    print(f"Ramp complementarity: per-unit g_max instead of big_M")
    print("=" * 60)

    t_start = time.time()
    main_df, supp_df, k_values = find_optimal_dispatch_tight_bounds(
        gens_df=gens_df,
        k_values_df=k_values_df,
        availabilities_df=availability_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        fixed_storage_dispatch=None,
        big_w=10000,
        time_limit=600,
        big_M=3500,
        demand_bids=n_demand_bids,
        mc_df=mc_df,
    )
    t_end = time.time()

    print("=" * 60)
    print(f"Total solve time (wall clock): {t_end - t_start:.1f} seconds")

    if main_df is not None:
        print(f"\nMCP range: {main_df['mcp'].min():.2f} to {main_df['mcp'].max():.2f}")
        print(
            f"MCP_hat range: {main_df['mcp_hat'].min():.2f} to {main_df['mcp_hat'].max():.2f}"
        )
        print(f"K range: {k_values['k'].min():.4f} to {k_values['k'].max():.4f}")
        print(f"\nFirst 5 rows of main_df:")
        print(main_df.head())
        print(f"\nK values:")
        print(k_values)
    else:
        print("\nModel was infeasible or unbounded!")
