# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# %%
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def find_optimal_dispatch_linearized(
    gens_df,
    k_values_df,
    demand_df,
    k_max,
    opt_gen,
    big_w=10000,
    time_limit=60,
    print_results=False,
    K=5,  # number of discrete binary steps considered in the linearisation
    big_M=10e6,
):
    model = pyo.ConcreteModel()

    # sets
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)

    # primary variables
    model.g = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Power output of producer 𝑖 at period 𝑡 (MW) — lower-level primal variable
    model.d = pyo.Var(
        model.time, within=pyo.NonNegativeReals
    )  # satisfied demand at period 𝑡 (MW)
    model.c_up = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Start-up cost of producer 𝑖 at period 𝑡 (€)
    model.c_down = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Shut-down cost of producer 𝑖 at period 𝑡 (€)
    model.k = pyo.Var(
        model.time, bounds=(1, k_max), within=pyo.NonNegativeReals
    )  # Bidding decision at timestep t as a multiplier of the marginal costs — upper-level decision
    model.lambda_ = pyo.Var(
        model.time, within=pyo.Reals, bounds=(-500, 200)
    )  # Market clearing price — lower-level dual variable
    model.u = pyo.Var(
        model.gens, model.time, within=pyo.Binary
    )  # Binary UC status of producer 𝑖 at period 𝑡 (𝑢 = 1 if it is on, 𝑢 = 0 if it is off)

    # secondary variables
    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max = pyo.Var(model.time, within=pyo.NonNegativeReals)

    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))

    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # duals of LP relaxation
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(0, 200))
    model.mu_max_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max_hat = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.nu_min_hat = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.pi_u_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d_hat = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # binary expansion variables
    model.g_binary = pyo.Var(model.time, range(K), within=pyo.Binary)
    model.z_lambda = pyo.Var(model.time, range(K), within=pyo.NonNegativeReals)
    model.z_k = pyo.Var(model.time, range(K), within=pyo.NonNegativeReals)
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    model.M = pyo.Param(initialize=max(gens_df["mc"]) * k_max)
    delta = [gens_df.at[gen, "g_max"] / (pow(2, K) - 1) for gen in gens_df.index]

    # ---------------------------------------------------------------------------
    # binary expansion constraints
    def g_binary_rule(model, t):
        return model.g[opt_gen, t] <= delta[opt_gen] * sum(
            pow(2, k) * model.g_binary[t, k] for k in range(K)
        )

    model.g_binary_constr = pyo.Constraint(model.time, rule=g_binary_rule)

    def binary_expansion_1_constr_1_max_rule(model, t, n):
        return model.lambda_hat[t] - model.z_lambda[t, n] <= model.M * (
            1 - model.g_binary[t, n]
        )

    def binary_expansion_1_constr_1_min_rule(model, t, n):
        return model.lambda_hat[t] - model.z_lambda[t, n] >= 0

    def binary_expansion_1_constr_2_rule(model, t, n):
        return model.z_lambda[t, n] <= model.M * model.g_binary[t, n]

    model.binary_expansion_1_constr_1_max = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_1_constr_1_max_rule
    )
    model.binary_expansion_1_constr_1_min = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_1_constr_1_min_rule
    )
    model.binary_expansion_1_constr_2 = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_1_constr_2_rule
    )

    def binary_expansion_2_constr_1_max_rule(model, t, n):
        return model.k[t] - model.z_k[t, n] <= model.M * (1 - model.g_binary[t, n])

    def binary_expansion_2_constr_1_min_rule(model, t, n):
        return model.k[t] - model.z_k[t, n] >= 0

    def binary_expansion_2_constr_2_rule(model, t, n):
        return model.z_k[t, n] <= model.M * model.g_binary[t, n]

    model.binary_expansion_2_constr_1_max = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_2_constr_1_max_rule
    )
    model.binary_expansion_2_constr_1_min = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_2_constr_1_min_rule
    )
    model.binary_expansion_2_constr_2 = pyo.Constraint(
        model.time, range(K), rule=binary_expansion_2_constr_2_rule
    )

    # ---------------------------------------------------------------------------
    # objective rules
    def primary_objective_rule(model):
        # This is the strategic producer’s revenue based on the market clearing price (approximated via z_lambda) and actual production, minus their marginal cost and startup/shutdown costs.
        # (7a) (1st line)
        return sum(
            delta[opt_gen] * sum(pow(2, n) * model.z_lambda[t, n] for n in range(K))
            - gens_df.at[opt_gen, "mc"] * model.g[opt_gen, t]
            - model.c_up[opt_gen, t]
            - model.c_down[opt_gen, t]
            for t in model.time
        )

    def duality_gap_part_1_rule(model):
        # (7a) (2nd line)
        # Lower-level primal cost, basically system costs
        expr = sum(
            (
                (
                    gens_df.at[gen, "mc"]
                    * delta[gen]
                    * sum(pow(2, n) * model.z_k[t, n] for n in range(K))
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )
                if gen == opt_gen
                else (
                    k_values_df.at[t, gen] * gens_df.at[gen, "mc"] * model.g[gen, t]
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )
            )
            for gen in model.gens
            for t in model.time
        )
        expr -= sum(demand_df.at[t, "price"] * model.d[t] for t in model.time)
        return expr

    def duality_gap_part_2_rule(model):
        # (7a) (3rd line)
        # objective value of the dual problem, using the current values of the dual variables
        expr = -sum(model.nu_max[t] * demand_df.at[t, "volume"] for t in model.time)

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

        return expr

    def final_objective_rule(model):
        # (7a) overall
        # summing up the primal objective and the duality gap (difference between the primal cost and dual value, which should be zero at optimality)
        return primary_objective_rule(model) - big_w * (
            duality_gap_part_1_rule(model) - duality_gap_part_2_rule(model)
        )

    model.objective = pyo.Objective(expr=final_objective_rule, sense=pyo.maximize)

    # ---------------------------------------------------------------------------
    # constraints
    # energy balance constraint
    def balance_rule(model, t):
        # (7d)
        return model.d[t] - sum(model.g[i, t] for i in model.gens) == 0

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    # max generation constraint
    def g_max_rule(model, i, t):
        # (7e)
        return model.g[i, t] <= gens_df.at[i, "g_max"] * model.u[i, t]

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    # max demand constraint
    def d_max_rule(model, t):
        # (7f)
        return model.d[t] <= demand_df.at[t, "volume"]

    model.d_max = pyo.Constraint(model.time, rule=d_max_rule)

    # max ramp up constraint
    def ru_max_rule(model, i, t):
        # 	(7h)
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
        else:
            return model.g[i, t] - model.g[i, t - 1] <= gens_df.at[i, "r_up"]

    model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_max_rule)

    # max ramp down constraint
    def rd_max_rule(model, i, t):
        # (7i)
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] <= gens_df.at[i, "r_down"]
        else:
            return model.g[i, t - 1] - model.g[i, t] <= gens_df.at[i, "r_down"]

    model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_max_rule)

    # start up cost constraint
    def start_up_cost_rule(model, i, t):
        # (7m)
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

    # shut down cost constraint
    def shut_down_cost_rule(model, i, t):
        # (7n)
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

    # dual constraints
    def gen_dual_rule(model, i, t):
        # (7aa) – Stationarity with respect to generation variable 𝑔𝑖,𝑡
        # Conditional parts based on `i` and `t`
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_next_term = 0 if t == model.time.at(-1) else model.pi_u[i, t + 1]
        pi_d_next_term = 0 if t == model.time.at(-1) else model.pi_d[i, t + 1]

        # Combined expression
        return (
            k_term * gens_df.at[i, "mc"]
            - model.lambda_[t]
            + model.mu_max[i, t]
            - model.mu_min[i, t]
            + model.pi_u[i, t]
            - pi_u_next_term
            - model.pi_d[i, t]
            + pi_d_next_term
            == 0
        )

    model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual_rule)

    def status_dual_rule(model, i, t):
        # (7ab) – Stationarity w.r.t. the binary unit commitment variable
        if t != model.time.at(-1):
            return (
                -model.mu_max[i, t] * gens_df.at[i, "g_max"]
                + (model.sigma_u[i, t] - model.sigma_u[i, t + 1])
                * gens_df.at[i, "k_up"]
                - (model.sigma_d[i, t] - model.sigma_d[i, t + 1])
                * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= 0
            )
        else:
            return (
                -model.mu_max[i, t] * gens_df.at[i, "g_max"]
                + model.sigma_u[i, t] * gens_df.at[i, "k_up"]
                - model.sigma_d[i, t] * gens_df.at[i, "k_down"]
                + model.psi_max[i, t]
                >= 0
            )

    model.status_dual = pyo.Constraint(model.gens, model.time, rule=status_dual_rule)

    def demand_dual_rule(model, t):
        # (7ac) – Stationarity w.r.t. demand variable 𝑑𝑡
        return -demand_df.at[t, "price"] + model.lambda_[t] + model.nu_max[t] >= 0

    model.demand_dual = pyo.Constraint(model.time, rule=demand_dual_rule)

    # KKT conditions
    # Stationarity conditions
    def kkt_gen_rule(model, i, t):
        # (7ae) – Stationarity (relaxed KKT) w.r.t. the generation variable 𝑔𝑖,𝑡
        # Conditional parts based on `i` and `t`
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_hat_next_term = 0 if t == model.time.at(-1) else model.pi_u_hat[i, t + 1]
        pi_d_hat_next_term = 0 if t == model.time.at(-1) else model.pi_d_hat[i, t + 1]

        # Combined expression
        return (
            k_term * gens_df.at[i, "mc"]
            - model.lambda_hat[t]
            + model.mu_max_hat[i, t]
            - model.mu_min_hat[i, t]
            + model.pi_u_hat[i, t]
            - pi_u_hat_next_term
            - model.pi_d_hat[i, t]
            + pi_d_hat_next_term
            == 0
        )

    model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen_rule)

    def kkt_demand_rule(model, t):
        # (7af) – Stationarity (relaxed KKT) w.r.t. the demand variable 𝑑𝑡
        return (
            -demand_df.at[t, "price"]
            + model.lambda_hat[t]
            + model.nu_max_hat[t]
            - model.nu_min_hat[t]
            == 0
        )

    model.kkt_demand = pyo.Constraint(model.time, rule=kkt_demand_rule)

    # Complementary slackness conditions
    # for generation and demand upper bounds
    # (7t)–(7y), Relaxed or reformulated versions of (7ad)–(7ah), MILP-friendly using binaries + big-M
    def mu_max_hat_binary_rule_1(model, i, t):
        return (model.g[i, t] - gens_df.at[i, "g_max"] * model.u[i, t]) <= max(
            gens_df["g_max"]
        ) * (1 - model.mu_max_hat_binary[i, t])

    def mu_max_hat_binary_rule_2(model, i, t):
        return (model.g[i, t] - gens_df.at[i, "g_max"] * model.u[i, t]) >= -max(
            gens_df["g_max"]
        ) * (1 - model.mu_max_hat_binary[i, t])

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

    model.nu_max_hat_binary = pyo.Var(model.time, within=pyo.Binary)

    def nu_max_hat_binary_rule_1(model, t):
        return model.d[t] - demand_df.at[t, "volume"] <= max(demand_df["volume"]) * (
            1 - model.nu_max_hat_binary[t]
        )

    def nu_max_hat_binary_rule_2(model, t):
        return model.d[t] - demand_df.at[t, "volume"] >= -max(demand_df["volume"]) * (
            1 - model.nu_max_hat_binary[t]
        )

    def nu_max_hat_binary_rule_3(model, t):
        return model.nu_max_hat[t] <= big_M * model.nu_max_hat_binary[t]

    model.nu_max_hat_binary_constr_1 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_1
    )
    model.nu_max_hat_binary_constr_2 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_2
    )
    model.nu_max_hat_binary_constr_3 = pyo.Constraint(
        model.time, rule=nu_max_hat_binary_rule_3
    )

    model.nu_min_hat_binary = pyo.Var(model.time, within=pyo.Binary)

    def nu_min_hat_binary_rule_1(model, t):
        return model.d[t] <= big_M * (1 - model.nu_min_hat_binary[t])

    def nu_min_hat_binary_rule_3(model, t):
        return model.nu_min_hat[t] <= big_M * model.nu_min_hat_binary[t]

    model.nu_min_hat_binary_constr_1 = pyo.Constraint(
        model.time, rule=nu_min_hat_binary_rule_1
    )

    model.nu_min_hat_binary_constr_3 = pyo.Constraint(
        model.time, rule=nu_min_hat_binary_rule_3
    )

    # Complementary slackness conditions
    # for ramp-up and ramp-down residuals
    # (??)–(??), Relaxed or reformulated versions of (7ai)–(7aj), MILP-friendly using binaries + big-M
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

    # ---------------------------------------------------------------------------
    # solve
    instance = model.create_instance()

    solver = SolverFactory("gurobi")
    options = {
        "LogToConsole": print_results,
        "TimeLimit": time_limit,
        "MIPGap": 0.03,
        # "MIPFocus": 3,
    }

    results = solver.solve(instance, options=options, tee=print_results)

    # check if solver exited due to time limit
    if results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
        print("Solver did not converge to an optimal solution")

    # ---------------------------------------------------------------------------
    # extract results
    generation_df = pd.DataFrame(
        index=demand_df.index, columns=[f"gen_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            generation_df.at[t, f"gen_{gen}"] = instance.g[gen, t].value

    demand_df = pd.DataFrame(index=demand_df.index, columns=["demand"])
    for t in demand_df.index:
        demand_df.at[t, "demand"] = instance.d[t].value

    mcp = pd.DataFrame(index=demand_df.index, columns=["mcp"])
    for t in demand_df.index:
        mcp.at[t, "mcp"] = instance.lambda_[t].value

    mcp_hat = pd.DataFrame(index=demand_df.index, columns=["mcp_hat"])
    for t in demand_df.index:
        mcp_hat.at[t, "mcp_hat"] = instance.lambda_hat[t].value

    main_df = pd.concat([generation_df, demand_df, mcp, mcp_hat], axis=1)

    start_up_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"start_up_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            start_up_cost.at[t, f"start_up_{gen}"] = instance.c_up[gen, t].value

    shut_down_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"shut_down_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            shut_down_cost.at[t, f"shut_down_{gen}"] = instance.c_down[gen, t].value

    supp_df = pd.concat([start_up_cost, shut_down_cost], axis=1)

    k_values = pd.DataFrame(index=demand_df.index, columns=["k"])
    for t in demand_df.index:
        k_values.at[t, "k"] = instance.k[t].value

    return main_df, supp_df, k_values


def find_optimal_dispatch_quadratic(
    gens_df,
    k_values_df,
    availabilities_df,
    demand_df,
    k_max,
    opt_gen,
    big_w=10000,
    time_limit=60,
    print_results=False,
    big_M=10e6,
    demand_bids=1,
):
    model = pyo.ConcreteModel()

    # sets
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, demand_bids + 1))

    # primary variables
    model.g = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Power output of producer 𝑖 at period 𝑡 (MW) — lower-level primal variable
    model.d = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals
    )  # satisfied demand at period 𝑡, from the multiple demand-bids n (MW)
    model.c_up = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Start-up cost of producer 𝑖 at period 𝑡 (€)
    model.c_down = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Shut-down cost of producer 𝑖 at period 𝑡 (€)
    model.k = pyo.Var(
        model.time, bounds=(1, k_max), within=pyo.NonNegativeReals
    )  # Bidding decision at timestep t as a multiplier of the marginal costs — upper-level decision
    model.lambda_ = pyo.Var(
        model.time, within=pyo.Reals, bounds=(-500, 200)
    )  # Market clearing price — lower-level dual variable
    model.u = pyo.Var(
        model.gens, model.time, within=pyo.Binary
    )  # Binary UC status of producer 𝑖 at period 𝑡 (𝑢 = 1 if it is on, 𝑢 = 0 if it is off)

    # secondary variables
    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)

    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))

    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # duals of LP relaxation
    # Note: ohne Hut = die „echten“ KKT-Duals, mit Komplementarität, evtl. nichtlinear.; mit Hut = linearisierte / relaxierte Variablen, die dieselben Bedingungen abbilden, aber in einem LP/MILP-freundlichen Format.
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(0, 200))
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

    # binary expansion variables
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # ---------------------------------------------------------------------------
    # objective rules
    def primary_objective_rule(model):
        # This is the strategic producer’s revenue based on the market clearing price and actual production, minus their marginal cost and startup/shutdown costs.
        # (7a) (1st line)
        return sum(
            model.lambda_hat[t] * model.g[opt_gen, t]
            - gens_df.at[opt_gen, "mc"] * model.g[opt_gen, t]
            - model.c_up[opt_gen, t]
            - model.c_down[opt_gen, t]
            for t in model.time
        )

    def duality_gap_part_1_rule(model):
        # (7a) (2nd line)
        # Lower-level primal cost, basically system costs
        expr = sum(
            (
                (
                    gens_df.at[gen, "mc"] * model.k[t]
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )
                if gen == opt_gen
                else (
                    k_values_df.at[t, gen] * gens_df.at[gen, "mc"] * model.g[gen, t]
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
        # (7a) (3rd line)
        # objective value of the dual problem, using the current values of the dual variables
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

        return expr

    def final_objective_rule(model):
        # (7a) overall
        # summing up the primal objective and the duality gap (difference between the primal cost and dual value, which should be zero at optimality)
        return primary_objective_rule(model) - big_w * (
            duality_gap_part_1_rule(model) - duality_gap_part_2_rule(model)
        )

    model.objective = pyo.Objective(expr=final_objective_rule, sense=pyo.maximize)

    # ---------------------------------------------------------------------------
    # constraints
    # energy balance constraint
    def balance_rule(model, t):
        # (7d)
        return (
            sum(model.d[t, n] for n in model.demand_bids)
            - sum(model.g[i, t] for i in model.gens)
            == 0
        )

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    # max generation constraint
    def g_max_rule(model, i, t):
        # (7e)
        return (
            model.g[i, t]
            <= gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        )

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    # max demand constraint
    def d_max_rule(model, t, n):
        # (7f)
        return model.d[t, n] <= demand_df.at[t, f"volume_{n}"]

    model.d_max = pyo.Constraint(model.time, model.demand_bids, rule=d_max_rule)

    # max ramp up constraint
    def ru_max_rule(model, i, t):
        # 	(7h)
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
        else:
            return model.g[i, t] - model.g[i, t - 1] <= gens_df.at[i, "r_up"]

    model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_max_rule)

    # max ramp down constraint
    def rd_max_rule(model, i, t):
        # (7i)
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] <= gens_df.at[i, "r_down"]
        else:
            return model.g[i, t - 1] - model.g[i, t] <= gens_df.at[i, "r_down"]

    model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_max_rule)

    # start up cost constraint
    def start_up_cost_rule(model, i, t):
        # (7m)
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

    # shut down cost constraint
    def shut_down_cost_rule(model, i, t):
        # (7n)
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

    # dual constraints
    def gen_dual_rule(model, i, t):
        # (7aa) – Stationarity with respect to generation variable 𝑔𝑖,𝑡
        # Conditional parts based on `i` and `t`
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_next_term = 0 if t == model.time.at(-1) else model.pi_u[i, t + 1]
        pi_d_next_term = 0 if t == model.time.at(-1) else model.pi_d[i, t + 1]

        # Combined expression
        return (
            k_term * gens_df.at[i, "mc"]
            - model.lambda_[t]
            + model.mu_max[i, t]
            - model.mu_min[i, t]
            + model.pi_u[i, t]
            - pi_u_next_term
            - model.pi_d[i, t]
            + pi_d_next_term
            == 0
        )

    model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual_rule)

    def status_dual_rule(model, i, t):
        # (7ab) – Stationarity w.r.t. the binary unit commitment variable
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
        # (7ac) – Stationarity w.r.t. demand variable 𝑑𝑡
        return (
            -demand_df.at[t, f"price_{n}"] + model.lambda_[t] + model.nu_max[t, n] >= 0
        )

    model.demand_dual = pyo.Constraint(
        model.time, model.demand_bids, rule=demand_dual_rule
    )

    # KKT conditions
    # Stationarity conditions
    def kkt_gen_rule(model, i, t):
        # (7ae) – Stationarity (relaxed KKT) w.r.t. the generation variable 𝑔𝑖,𝑡
        # Conditional parts based on `i` and `t`
        k_term = model.k[t] if i == opt_gen else k_values_df.at[t, i]
        pi_u_hat_next_term = 0 if t == model.time.at(-1) else model.pi_u_hat[i, t + 1]
        pi_d_hat_next_term = 0 if t == model.time.at(-1) else model.pi_d_hat[i, t + 1]

        # Combined expression
        return (
            k_term * gens_df.at[i, "mc"]
            - model.lambda_hat[t]
            + model.mu_max_hat[i, t]
            - model.mu_min_hat[i, t]
            + model.pi_u_hat[i, t]
            - pi_u_hat_next_term
            - model.pi_d_hat[i, t]
            + pi_d_hat_next_term
            == 0
        )

    model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen_rule)

    def kkt_demand_rule(model, t, n):
        # (7af) – Stationarity (relaxed KKT) w.r.t. the demand variable 𝑑𝑡
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

    # Complementary slackness conditions
    # for generation and demand upper bounds
    # (7t)–(7y), Relaxed or reformulated versions of (7ad)–(7ah), MILP-friendly using binaries + big-M
    def mu_max_hat_binary_rule_1(model, i, t):
        return (
            model.g[i, t]
            - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        ) <= max(gens_df["g_max"] * availabilities_df.at[t, i]) * (
            1 - model.mu_max_hat_binary[i, t]
        )

    def mu_max_hat_binary_rule_2(model, i, t):
        return (
            model.g[i, t]
            - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        ) >= -max(gens_df["g_max"] * availabilities_df.at[t, i]) * (
            1 - model.mu_max_hat_binary[i, t]
        )

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
        return model.d[t, n] <= big_M * (1 - model.nu_min_hat_binary[t, n])

    def nu_min_hat_binary_rule_3(model, t, n):
        return model.nu_min_hat[t, n] <= big_M * model.nu_min_hat_binary[t, n]

    model.nu_min_hat_binary_constr_1 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_min_hat_binary_rule_1
    )

    model.nu_min_hat_binary_constr_3 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_min_hat_binary_rule_3
    )

    # Complementary slackness conditions
    # for ramp-up and ramp-down residuals
    # (??)–(??), Relaxed or reformulated versions of (7ai)–(7aj), MILP-friendly using binaries + big-M
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

    # ---------------------------------------------------------------------------
    # solve
    instance = model.create_instance()

    solver = SolverFactory("gurobi")
    options = {
        "LogToConsole": print_results,
        "TimeLimit": time_limit,
        "MIPGap": 0.03,
        # "MIPFocus": 3,
    }

    solver.options["NonConvex"] = 2
    results = solver.solve(instance, options=options, tee=print_results)

    # check if solver exited due to time limit
    if results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
        print("Solver did not converge to an optimal solution")

    # ---------------------------------------------------------------------------
    # extract results
    generation_df = pd.DataFrame(
        index=demand_df.index, columns=[f"gen_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            generation_df.at[t, f"gen_{gen}"] = instance.g[gen, t].value

    demand_df = pd.DataFrame(index=demand_df.index, columns=["demand"])
    for t in demand_df.index:
        demand_df.at[t, "demand"] = sum(
            instance.d[t, n].value for n in instance.demand_bids
        )

    mcp = pd.DataFrame(index=demand_df.index, columns=["mcp"])
    for t in demand_df.index:
        mcp.at[t, "mcp"] = instance.lambda_[t].value

    mcp_hat = pd.DataFrame(index=demand_df.index, columns=["mcp_hat"])
    for t in demand_df.index:
        mcp_hat.at[t, "mcp_hat"] = instance.lambda_hat[t].value

    main_df = pd.concat([generation_df, demand_df, mcp, mcp_hat], axis=1)

    start_up_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"start_up_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            start_up_cost.at[t, f"start_up_{gen}"] = instance.c_up[gen, t].value

    shut_down_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"shut_down_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            shut_down_cost.at[t, f"shut_down_{gen}"] = instance.c_down[gen, t].value

    supp_df = pd.concat([start_up_cost, shut_down_cost], axis=1)

    k_values = pd.DataFrame(index=demand_df.index, columns=["k"])
    for t in demand_df.index:
        k_values.at[t, "k"] = instance.k[t].value

    return main_df, supp_df, k_values


def find_optimal_dispatch_storage_leader_quadratic(
    gens_df,
    storage_df,
    demand_df,
    k_values_df,
    storage_k_values_df,
    availabilities_df,
    opt_storage,
    k_max,
    big_w=10000,
    time_limit=60,
    demand_bids=1,
    big_M=10e6,
    print_results=False,
):
    """
    Monolithic quadratic MPEC with one strategic storage leader.

    Hybrid paper-aligned formulation:
    - Lower level follows the paper simplification for storage: only charging/discharging
      power enters the follower market-clearing/KKT block.
    - SoC dynamics and SoC bounds are NOT part of the lower-level KKT.
    - To keep the strategic storage physically feasible, SoC is retained only for the
      strategic storage opt_storage as upper-level feasibility constraints.

    This means:
    - non-strategic storages: represented in the follower only through p_charge/p_discharge
      and power bounds, consistent with the paper-style simplification.
    - strategic storage: additionally has SoC state constraints, but these are not dualized
      and do not appear in the KKT system.
    """
    if opt_storage not in storage_k_values_df.columns:
        raise ValueError(
            f"opt_storage '{opt_storage}' not found in storage_k_values_df columns"
        )

    model = pyo.ConcreteModel()

    # -------------------------------------------------------------------------
    # SETS
    # -------------------------------------------------------------------------
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.storages = pyo.Set(initialize=storage_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, demand_bids + 1))

    # -------------------------------------------------------------------------
    # PRIMAL LOWER-LEVEL VARIABLES
    # -------------------------------------------------------------------------
    model.g = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)
    model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # Storage power variables remain in the lower-level market clearing.
    model.p_charge = pyo.Var(model.storages, model.time, within=pyo.NonNegativeReals)
    model.p_discharge = pyo.Var(model.storages, model.time, within=pyo.NonNegativeReals)

    # Upper-level-only SoC state for the strategic storage.
    # This is NOT part of the lower-level KKT.
    model.soc = pyo.Var(model.time, within=pyo.NonNegativeReals)

    # -------------------------------------------------------------------------
    # UPPER-LEVEL LEADER DECISION
    # -------------------------------------------------------------------------
    model.k = pyo.Var(model.time, bounds=(1, k_max), within=pyo.NonNegativeReals)

    # -------------------------------------------------------------------------
    # DUAL VARIABLES FOR THE ECONOMIC LOWER LEVEL
    # -------------------------------------------------------------------------
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, 200))
    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)

    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # Storage power-bound duals only.
    model.alpha_charge_max = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )
    model.alpha_discharge_max = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )

    # -------------------------------------------------------------------------
    # HAT DUALS FOR THE RELAXED / REFORMULATED KKT SYSTEM
    # -------------------------------------------------------------------------
    model.lambda_hat = pyo.Var(model.time, within=pyo.Reals, bounds=(0, 200))
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

    model.alpha_charge_max_hat = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )
    model.alpha_discharge_max_hat = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )

    # -------------------------------------------------------------------------
    # BINARIES FOR COMPLEMENTARITY LINEARIZATION
    # -------------------------------------------------------------------------
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.nu_max_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.nu_min_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.alpha_charge_max_hat_binary = pyo.Var(
        model.storages, model.time, within=pyo.Binary
    )
    model.alpha_discharge_max_hat_binary = pyo.Var(
        model.storages, model.time, within=pyo.Binary
    )

    def storage_bid_charge(model, s, t):
        # optionaler separater Kosteneintrag für Laden
        mc_charge = (
            storage_df.at[s, "additional_cost_charge"]
            if "additional_cost_charge" in storage_df.columns
            else storage_df.at[s, "mc"]
        )
        base_k = storage_k_values_df.at[t, s]
        return model.k[t] * mc_charge if s == opt_storage else base_k * mc_charge

    def storage_bid_discharge(model, s, t):
        # optionaler separater Kosteneintrag für Entladen
        mc_discharge = (
            storage_df.at[s, "additional_cost_discharge"]
            if "additional_cost_discharge" in storage_df.columns
            else storage_df.at[s, "mc"]
        )
        base_k = storage_k_values_df.at[t, s]
        return model.k[t] * mc_discharge if s == opt_storage else base_k * mc_discharge

    # -------------------------------------------------------------------------
    # OBJECTIVE
    # -------------------------------------------------------------------------
    def primary_objective_rule(model):
        return sum(
            model.lambda_hat[t]
            * (model.p_discharge[opt_storage, t] - model.p_charge[opt_storage, t])
            for t in model.time
        )

    def duality_gap_part_1_rule(model):
        # Lower-level Wohlfahrt (19, Wang) (1st and 2nd line)
        expr = sum(
            gens_df.at[gen, "mc"] * model.g[gen, t]
            + model.c_up[gen, t]
            + model.c_down[gen, t]
            for gen in model.gens
            for t in model.time
        )

        expr += sum(
            storage_bid_discharge(model, s, t) * model.p_discharge[s, t]
            - storage_bid_charge(model, s, t) * model.p_charge[s, t]
            for s in model.storages
            for t in model.time
        )

        expr -= sum(
            demand_df.at[t, f"price_{n}"] * model.d[t, n]
            for t in model.time
            for n in model.demand_bids
        )

        return expr

    def duality_gap_part_2_rule(model):
        # (7a) (3rd line)
        # objective value of the dual problem, using the current values of the dual variables
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

        # adding this according to (19, Wang) to account for storage power bounds in the dual objective
        # somehow they do not seem to add this in the paper which really confuses me
        # or they do so (34, Wang) not entirely sure
        expr -= sum(
            model.alpha_charge_max[s, t] * abs(storage_df.at[s, "max_power_charge"])
            for s in model.storages
            for t in model.time
        )
        expr -= sum(
            model.alpha_discharge_max[s, t] * storage_df.at[s, "max_power_discharge"]
            for s in model.storages
            for t in model.time
        )

        return expr

    def final_objective_rule(model):
        return primary_objective_rule(model) - big_w * (
            duality_gap_part_1_rule(model) - duality_gap_part_2_rule(model)
        )

    model.objective = pyo.Objective(
        expr=final_objective_rule(model), sense=pyo.maximize
    )

    # -------------------------------------------------------------------------
    # PRIMAL LOWER-LEVEL CONSTRAINTS
    # -------------------------------------------------------------------------
    # Almost everything according to Nick just added some constraints where indicated
    def balance_rule(model, t):
        return (
            sum(model.d[t, n] for n in model.demand_bids)
            # add charge to demand
            + sum(model.p_charge[s, t] for s in model.storages)
            - sum(model.g[i, t] for i in model.gens)
            # add discharge to generation
            - sum(model.p_discharge[s, t] for s in model.storages)
            == 0
        )

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    # max generation constraint
    def g_max_rule(model, i, t):
        # (7e)
        return (
            model.g[i, t]
            <= gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        )

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    # max demand constraint
    def d_max_rule(model, t, n):
        # (7f)
        return model.d[t, n] <= demand_df.at[t, f"volume_{n}"]

    model.d_max = pyo.Constraint(model.time, model.demand_bids, rule=d_max_rule)

    # max ramp up constraint
    def ru_max_rule(model, i, t):
        # 	(7h)
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
        else:
            return model.g[i, t] - model.g[i, t - 1] <= gens_df.at[i, "r_up"]

    model.ru_max = pyo.Constraint(model.gens, model.time, rule=ru_max_rule)

    # max ramp down constraint
    def rd_max_rule(model, i, t):
        # (7i)
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] <= gens_df.at[i, "r_down"]
        else:
            return model.g[i, t - 1] - model.g[i, t] <= gens_df.at[i, "r_down"]

    model.rd_max = pyo.Constraint(model.gens, model.time, rule=rd_max_rule)

    # start up cost constraint
    def start_up_cost_rule(model, i, t):
        # (7m)
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

    # shut down cost constraint
    def shut_down_cost_rule(model, i, t):
        # (7n)
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

    # add storage constraints according to Wang
    # (24, Wang)
    def charge_max_rule(model, s, t):
        return model.p_charge[s, t] <= abs(storage_df.at[s, "max_power_charge"])

    model.charge_max = pyo.Constraint(model.storages, model.time, rule=charge_max_rule)

    # (25, Wang)
    def discharge_max_rule(model, s, t):
        return model.p_discharge[s, t] <= storage_df.at[s, "max_power_discharge"]

    model.discharge_max = pyo.Constraint(
        model.storages, model.time, rule=discharge_max_rule
    )

    # -------------------------------------------------------------------------
    # UPPER-LEVEL FEASIBILITY CONSTRAINTS FOR THE STRATEGIC STORAGE ONLY
    # -------------------------------------------------------------------------
    def soc_rule(model, t):
        # SoC dynamics only for the strategic storage.
        # This is NOT dualized and not part of the lower-level KKT.
        if t == 0:
            # (12, Wang)
            return (
                model.soc[t]
                - storage_df.at[opt_storage, "efficiency_charge"]
                * model.p_charge[opt_storage, t]
                + model.p_discharge[opt_storage, t]
                / storage_df.at[opt_storage, "efficiency_discharge"]
                == storage_df.at[opt_storage, "initial_soc"]
                * storage_df.at[opt_storage, "capacity"]
            )
        # (13, Wang)
        return (
            model.soc[t]
            - model.soc[t - 1]
            - storage_df.at[opt_storage, "efficiency_charge"]
            * model.p_charge[opt_storage, t]
            + model.p_discharge[opt_storage, t]
            / storage_df.at[opt_storage, "efficiency_discharge"]
            == 0
        )

    model.soc_coupling = pyo.Constraint(model.time, rule=soc_rule)

    def soc_max_rule(model, t):
        # (14, Wang) without minimum SoC
        return model.soc[t] <= storage_df.at[opt_storage, "capacity"]

    model.soc_max = pyo.Constraint(model.time, rule=soc_max_rule)

    # -------------------------------------------------------------------------
    # DUAL FEASIBILITY / STATIONARITY OF THE ECONOMIC LOWER LEVEL
    # -------------------------------------------------------------------------
    def gen_dual_rule(model, i, t):
        # (7aa) – Stationarity with respect to generation variable 𝑔𝑖,𝑡
        # Conditional parts based on `i` and `t`, now without opt_gen differentiation because we only have storage_opt now
        pi_u_next_term = 0 if t == model.time.at(-1) else model.pi_u[i, t + 1]
        pi_d_next_term = 0 if t == model.time.at(-1) else model.pi_d[i, t + 1]
        return (
            gens_df.at[i, "mc"] * k_values_df.at[t, i]
            - model.lambda_[t]
            + model.mu_max[i, t]
            - model.mu_min[i, t]
            + model.pi_u[i, t]
            - pi_u_next_term
            - model.pi_d[i, t]
            + pi_d_next_term
            == 0
        )

    model.gen_dual = pyo.Constraint(model.gens, model.time, rule=gen_dual_rule)

    def status_dual_rule(model, i, t):
        # (7ab) – Stationarity w.r.t. the binary unit commitment variable
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
        # (7ac) – Stationarity w.r.t. demand variable 𝑑𝑡
        return (
            -demand_df.at[t, f"price_{n}"] + model.lambda_[t] + model.nu_max[t, n] >= 0
        )

    model.demand_dual = pyo.Constraint(
        model.time, model.demand_bids, rule=demand_dual_rule
    )

    def storage_charge_dual_rule(model, s, t):
        # (37, Wang) dual feasibility
        return (
            -storage_bid_charge(model, s, t)
            + model.lambda_[t]
            + model.alpha_charge_max[s, t]
            >= 0
        )

    model.storage_charge_dual = pyo.Constraint(
        model.storages, model.time, rule=storage_charge_dual_rule
    )

    def storage_discharge_dual_rule(model, s, t):
        # (38, Wang) dual feasibility
        return (
            storage_bid_discharge(model, s, t)
            - model.lambda_[t]
            + model.alpha_discharge_max[s, t]
            >= 0
        )

    model.storage_discharge_dual = pyo.Constraint(
        model.storages, model.time, rule=storage_discharge_dual_rule
    )

    # -------------------------------------------------------------------------
    # RELAXED KKT STATIONARITY BLOCK (HAT SYSTEM)
    # -------------------------------------------------------------------------
    def kkt_gen_rule(model, i, t):
        # (7ae) – Stationarity (relaxed KKT) w.r.t. the generation variable 𝑔𝑖,𝑡
        # Conditional parts based on `i` and `t`, now without opt_gen differentiation because we only have storage_opt now
        pi_u_hat_next_term = 0 if t == model.time.at(-1) else model.pi_u_hat[i, t + 1]
        pi_d_hat_next_term = 0 if t == model.time.at(-1) else model.pi_d_hat[i, t + 1]
        return (
            gens_df.at[i, "mc"]
            - model.lambda_hat[t]
            + model.mu_max_hat[i, t]
            - model.mu_min_hat[i, t]
            + model.pi_u_hat[i, t]
            - pi_u_hat_next_term
            - model.pi_d_hat[i, t]
            + pi_d_hat_next_term
            == 0
        )

    model.kkt_gen = pyo.Constraint(model.gens, model.time, rule=kkt_gen_rule)

    def kkt_demand_rule(model, t, n):
        # (7af) – Stationarity (relaxed KKT) w.r.t. the demand variable 𝑑𝑡
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

    def kkt_storage_charge_rule(model, s, t):
        return (
            -storage_bid_charge(model, s, t)
            + model.lambda_hat[t]
            + model.alpha_charge_max_hat[s, t]
            >= 0
        )

    model.kkt_storage_charge = pyo.Constraint(
        model.storages, model.time, rule=kkt_storage_charge_rule
    )

    def kkt_storage_discharge_rule(model, s, t):
        return (
            storage_bid_discharge(model, s, t)
            - model.lambda_hat[t]
            + model.alpha_discharge_max_hat[s, t]
            >= 0
        )

    model.kkt_storage_discharge = pyo.Constraint(
        model.storages, model.time, rule=kkt_storage_discharge_rule
    )

    # -------------------------------------------------------------------------
    # COMPLEMENTARITY LINEARIZATION VIA BIG-M AND BINARIES
    # -------------------------------------------------------------------------
    # Complementary slackness conditions
    # for generation and demand upper bounds
    # (7t)–(7y), Relaxed or reformulated versions of (7ad)–(7ah), MILP-friendly using binaries + big-M
    def mu_max_hat_binary_rule_1(model, i, t):
        return (
            model.g[i, t]
            - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        ) <= max(gens_df["g_max"] * availabilities_df.at[t, i]) * (
            1 - model.mu_max_hat_binary[i, t]
        )

    def mu_max_hat_binary_rule_2(model, i, t):
        return (
            model.g[i, t]
            - gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        ) >= -max(gens_df["g_max"] * availabilities_df.at[t, i]) * (
            1 - model.mu_max_hat_binary[i, t]
        )

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
        return model.d[t, n] <= big_M * (1 - model.nu_min_hat_binary[t, n])

    def nu_min_hat_binary_rule_3(model, t, n):
        return model.nu_min_hat[t, n] <= big_M * model.nu_min_hat_binary[t, n]

    model.nu_min_hat_binary_constr_1 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_min_hat_binary_rule_1
    )

    model.nu_min_hat_binary_constr_3 = pyo.Constraint(
        model.time, model.demand_bids, rule=nu_min_hat_binary_rule_3
    )

    # Complementary slackness conditions
    # for ramp-up and ramp-down residuals
    # (??)–(??), Relaxed or reformulated versions of (7ai)–(7aj), MILP-friendly using binaries + big-M
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

    # Complementary slackness conditions for storage power bounds
    # reformulation of equation (50-51, Wang), MILP-friendly using binaries + big-M
    def alpha_charge_hat_binary_rule_1(model, s, t):
        return model.p_charge[s, t] - abs(
            storage_df.at[s, "max_power_charge"]
        ) <= big_M * (1 - model.alpha_charge_max_hat_binary[s, t])

    def alpha_charge_hat_binary_rule_2(model, s, t):
        return model.p_charge[s, t] - abs(
            storage_df.at[s, "max_power_charge"]
        ) >= -big_M * (1 - model.alpha_charge_max_hat_binary[s, t])

    def alpha_charge_hat_binary_rule_3(model, s, t):
        return (
            model.alpha_charge_max_hat[s, t]
            <= big_M * model.alpha_charge_max_hat_binary[s, t]
        )

    model.alpha_charge_hat_binary_constr_1 = pyo.Constraint(
        model.storages, model.time, rule=alpha_charge_hat_binary_rule_1
    )
    model.alpha_charge_hat_binary_constr_2 = pyo.Constraint(
        model.storages, model.time, rule=alpha_charge_hat_binary_rule_2
    )
    model.alpha_charge_hat_binary_constr_3 = pyo.Constraint(
        model.storages, model.time, rule=alpha_charge_hat_binary_rule_3
    )

    # reformulation of equation (48 - 49, Wang), MILP-friendly using binaries + big-M
    def alpha_discharge_hat_binary_rule_1(model, s, t):
        return model.p_discharge[s, t] - storage_df.at[
            s, "max_power_discharge"
        ] <= big_M * (1 - model.alpha_discharge_max_hat_binary[s, t])

    def alpha_discharge_hat_binary_rule_2(model, s, t):
        return model.p_discharge[s, t] - storage_df.at[
            s, "max_power_discharge"
        ] >= -big_M * (1 - model.alpha_discharge_max_hat_binary[s, t])

    def alpha_discharge_hat_binary_rule_3(model, s, t):
        return (
            model.alpha_discharge_max_hat[s, t]
            <= big_M * model.alpha_discharge_max_hat_binary[s, t]
        )

    model.alpha_discharge_hat_binary_constr_1 = pyo.Constraint(
        model.storages, model.time, rule=alpha_discharge_hat_binary_rule_1
    )
    model.alpha_discharge_hat_binary_constr_2 = pyo.Constraint(
        model.storages, model.time, rule=alpha_discharge_hat_binary_rule_2
    )
    model.alpha_discharge_hat_binary_constr_3 = pyo.Constraint(
        model.storages, model.time, rule=alpha_discharge_hat_binary_rule_3
    )

    # -------------------------------------------------------------------------
    # SOLVE
    # -------------------------------------------------------------------------
    instance = model.create_instance()
    solver = SolverFactory("gurobi")
    options = {
        "LogToConsole": print_results,
        "TimeLimit": time_limit,
        "MIPGap": 0.03,
    }
    solver.options["NonConvex"] = 2
    results = solver.solve(instance, options=options, tee=print_results)

    if results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
        print("Solver did not converge to an optimal solution")

    # -------------------------------------------------------------------------
    # RESULT EXTRACTION
    # -------------------------------------------------------------------------
    generation_df = pd.DataFrame(
        index=demand_df.index, columns=[f"gen_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            generation_df.at[t, f"gen_{gen}"] = instance.g[gen, t].value

    charge_df = pd.DataFrame(
        index=demand_df.index, columns=[f"charge_{s}" for s in storage_df.index]
    )
    discharge_df = pd.DataFrame(
        index=demand_df.index, columns=[f"discharge_{s}" for s in storage_df.index]
    )
    for s in storage_df.index:
        for t in demand_df.index:
            charge_df.at[t, f"charge_{s}"] = instance.p_charge[s, t].value
            discharge_df.at[t, f"discharge_{s}"] = instance.p_discharge[s, t].value

    soc_df = pd.DataFrame(index=demand_df.index, columns=[f"soc_{opt_storage}"])
    for t in demand_df.index:
        soc_df.at[t, f"soc_{opt_storage}"] = instance.soc[t].value

    cleared_demand_df = pd.DataFrame(index=demand_df.index, columns=["demand"])
    for t in demand_df.index:
        cleared_demand_df.at[t, "demand"] = sum(
            instance.d[t, n].value for n in instance.demand_bids
        )

    mcp = pd.DataFrame(index=demand_df.index, columns=["mcp"])
    mcp_hat = pd.DataFrame(index=demand_df.index, columns=["mcp_hat"])
    for t in demand_df.index:
        mcp.at[t, "mcp"] = instance.lambda_[t].value
        mcp_hat.at[t, "mcp_hat"] = instance.lambda_hat[t].value

    main_df = pd.concat(
        [
            generation_df,
            charge_df,
            discharge_df,
            soc_df,
            cleared_demand_df,
            mcp,
            mcp_hat,
        ],
        axis=1,
    )

    start_up_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"start_up_{gen}" for gen in gens_df.index]
    )
    shut_down_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"shut_down_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            start_up_cost.at[t, f"start_up_{gen}"] = instance.c_up[gen, t].value
            shut_down_cost.at[t, f"shut_down_{gen}"] = instance.c_down[gen, t].value

    supp_df = pd.concat([start_up_cost, shut_down_cost], axis=1)

    k_values = pd.DataFrame(index=demand_df.index, columns=["k_storage"])
    for t in demand_df.index:
        k_values.at[t, "k_storage"] = instance.k[t].value

    return main_df, supp_df, k_values
