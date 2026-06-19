# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bid-price reformulation of the MPEC.

Instead of model.k (multiplier on mc, needs k_max=300000 for renewables),
the leader decision is model.bid_price in €/MWh directly (bounded -500..bid_price_max).
This eliminates the huge k*mc products and keeps all coefficients well-scaled.

Non-strategic generators also get pre-computed bid prices (k_values * mc)
so the entire formulation works in €/MWh space.

Storage dispatch is taken as fixed input (realized MADRL dispatch) and
enters the balance as a parameter — no storage optimization variables needed.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def find_optimal_dispatch_bidprice(
    gens_df,
    k_values_df,
    availabilities_df,
    demand_df,
    opt_gen,
    fixed_storage_dispatch=None,
    bid_price_max=3000,
    big_w=10000,
    time_limit=600,
    print_results=False,
    big_M=10000,
    demand_bids=1,
    mc_df=None,
):
    """
    Quadratic MPEC with one strategic generator leader.

    Bid-price reformulation: the leader's decision variable is the bid price
    in €/MWh directly (not a multiplier k on marginal cost).

    Storage dispatch is treated as a fixed exogenous parameter in the energy
    balance (realized dispatch from MADRL learning), not optimized.

    Args:
        gens_df: Generator DataFrame (index=unit names)
        k_values_df: k multipliers per generator per timestep
        availabilities_df: Availability factors per generator per timestep
        demand_df: Demand bids with price_N and volume_N columns
        opt_gen: Name of the strategic generator
        fixed_storage_dispatch: Series indexed by timestep with net storage
            power in MW. Positive = net discharge (generation), negative =
            net charge (demand). None or all-zero = no storage effect.
        bid_price_max: Upper bound on the strategic generator's bid price in €/MWh
    """
    gens_df = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df
    if mc_df is None:
        mc_df = pd.DataFrame(
            {gen: gens_df.at[gen, "mc"] for gen in gens_df.index},
            index=demand_df.index,
        )

    # Pre-compute bid prices for non-strategic generators: k * mc (in €/MWh)
    gen_bid_prices_df = pd.DataFrame(index=demand_df.index)
    for gen in gens_df.index:
        if gen == opt_gen:
            continue
        gen_bid_prices_df[gen] = k_values_df[gen] * mc_df[gen]

    # Prepare fixed storage dispatch lookup
    if fixed_storage_dispatch is None:
        fixed_storage_dispatch = pd.Series(0.0, index=demand_df.index)

    model = pyo.ConcreteModel()

    # -------------------------------------------------------------------------
    # SETS
    # -------------------------------------------------------------------------
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, demand_bids + 1))

    # -------------------------------------------------------------------------
    # PRIMAL LOWER-LEVEL VARIABLES
    # -------------------------------------------------------------------------
    model.g = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.d = pyo.Var(model.time, model.demand_bids, within=pyo.NonNegativeReals)
    model.c_up = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.u = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # -------------------------------------------------------------------------
    # UPPER-LEVEL: bid price in €/MWh (NOT a multiplier)
    # -------------------------------------------------------------------------
    model.bid_price = pyo.Var(
        model.time, within=pyo.Reals, bounds=(-500, bid_price_max)
    )
    model.lambda_ = pyo.Var(model.time, within=pyo.Reals, bounds=(-500, bid_price_max))

    # -------------------------------------------------------------------------
    # DUAL VARIABLES
    # -------------------------------------------------------------------------
    model.mu_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.mu_min = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.nu_max = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals
    )

    model.pi_u = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.pi_d = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)
    model.sigma_u = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.sigma_d = pyo.Var(model.gens, model.time, bounds=(0, 1))
    model.psi_max = pyo.Var(model.gens, model.time, within=pyo.NonNegativeReals)

    # -------------------------------------------------------------------------
    # HAT DUALS
    # -------------------------------------------------------------------------
    model.lambda_hat = pyo.Var(
        model.time, within=pyo.Reals, bounds=(-500, bid_price_max)
    )
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

    # -------------------------------------------------------------------------
    # BINARIES FOR COMPLEMENTARITY
    # -------------------------------------------------------------------------
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.nu_max_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.nu_min_hat_binary = pyo.Var(model.time, model.demand_bids, within=pyo.Binary)
    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)

    # Helper: bid price for any generator at time t (€/MWh)
    def gen_bid(gen, t):
        if gen == opt_gen:
            return model.bid_price[t]
        return gen_bid_prices_df.at[t, gen]

    # -------------------------------------------------------------------------
    # OBJECTIVE
    # -------------------------------------------------------------------------
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
            gen_bid(gen, t) * model.g[gen, t]
            + model.c_up[gen, t]
            + model.c_down[gen, t]
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
    def balance_rule(model, t):
        # fixed_storage_dispatch > 0 = net discharge (acts as generation)
        # fixed_storage_dispatch < 0 = net charge (acts as demand)
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

    # -------------------------------------------------------------------------
    # DUAL FEASIBILITY / STATIONARITY (all in €/MWh)
    # -------------------------------------------------------------------------
    def gen_dual_rule(model, i, t):
        bid = gen_bid(i, t)
        pi_u_next = 0 if t == model.time.at(-1) else model.pi_u[i, t + 1]
        pi_d_next = 0 if t == model.time.at(-1) else model.pi_d[i, t + 1]
        return (
            bid
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

    # -------------------------------------------------------------------------
    # RELAXED KKT STATIONARITY (HAT SYSTEM)
    # -------------------------------------------------------------------------
    def kkt_gen_rule(model, i, t):
        bid = gen_bid(i, t)
        pi_u_hat_next = 0 if t == model.time.at(-1) else model.pi_u_hat[i, t + 1]
        pi_d_hat_next = 0 if t == model.time.at(-1) else model.pi_d_hat[i, t + 1]
        return (
            bid
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

    # -------------------------------------------------------------------------
    # COMPLEMENTARITY LINEARIZATION
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # SOLVE
    # -------------------------------------------------------------------------
    instance = model.create_instance()
    solver = SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    solver.options["NumericFocus"] = 1
    solver.options["DualReductions"] = 0
    options = {
        "LogToConsole": 1,
        "TimeLimit": time_limit,
        "MIPGap": 0.03,
    }
    results = solver.solve(instance, options=options, tee=True)

    print(f"\nSolver status: {results.solver.status}")
    print(f"Termination: {results.solver.termination_condition}")

    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        instance.write("debug_bidprice.lp", io_options={"symbolic_solver_labels": True})
        print("Model written to debug_bidprice.lp")
        try:
            import gurobipy as gp

            m = gp.read("debug_bidprice.lp")
            m.computeIIS()
            m.write("debug_bidprice.ilp")
            print("\n=== IIS Constraints ===")
            for c in m.getConstrs():
                if c.IISConstr:
                    print(f"  {c.ConstrName}")
            print("\n=== IIS Variable Bounds ===")
            for v in m.getVars():
                if v.IISLB or v.IISUB:
                    print(f"  {v.VarName} (lb={v.IISLB}, ub={v.IISUB})")
        except Exception as e:
            print(f"IIS computation failed: {e}")
        return None, None, None

    if results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
        print("WARNING: Solver hit time limit, solution may be suboptimal")

    # -------------------------------------------------------------------------
    # RESULT EXTRACTION
    # -------------------------------------------------------------------------
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

    main_df = pd.concat(
        [generation_df, cleared_demand, mcp, mcp_hat],
        axis=1,
    )

    bid_prices = pd.DataFrame(index=time_index, columns=["bid_price"])
    for t in time_index:
        bid_prices.at[t, "bid_price"] = instance.bid_price[t].value

    return main_df, None, bid_prices
