# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

import pandas as pd

# %%
# Imports
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# %%
def solve_uc_problem(gens_df, demand_df, k_values_df):
    model = pyo.ConcreteModel()

    # sets
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)

    # primary problem variables
    model.g = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Power output of producer 𝑖 at period 𝑡 (MW)
    model.d = pyo.Var(
        model.time, within=pyo.NonNegativeReals
    )  # satisfied demand at period 𝑡 (MW)
    model.c_up = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Start-up cost of producer 𝑖 at period 𝑡 (€)
    model.c_down = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Shut-down cost of producer 𝑖 at period 𝑡 (€)
    model.u = pyo.Var(
        model.gens, model.time, within=pyo.Binary
    )  # Binary UC status of producer 𝑖 at period 𝑡 (𝑢 = 1 if it is on, 𝑢 = 0 if it is off)

    # primary problem objective
    def objective_rule(model):
        expr = 0
        for gen in model.gens:
            for t in model.time:
                expr += (
                    k_values_df.at[t, gen] * gens_df.at[gen, "mc"] * model.g[gen, t]
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )

        for t in model.time:
            expr -= demand_df.at[t, "price"] * model.d[t]

        return expr

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # energy balance constraint
    def balance_rule(model, t):
        return model.d[t] - sum(model.g[i, t] for i in model.gens) == 0

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    # max generation constraint
    def g_max_rule(model, i, t):
        return model.g[i, t] <= gens_df.at[i, "g_max"] * model.u[i, t]

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    # max demand constraint
    # TODO: Wieso erlauben wir hier demand kleiner als der input demand?
    def d_max_rule(model, t):
        return model.d[t] <= demand_df.at[t, "volume"]

    model.d_max = pyo.Constraint(model.time, rule=d_max_rule)

    # ramp up constraint
    def ramp_up_rule(model, i, t):
        if t == 0:
            return model.g[i, t] - gens_df.at[i, "g_0"] <= gens_df.at[i, "r_up"]
        else:
            return model.g[i, t] - model.g[i, t - 1] <= gens_df.at[i, "r_up"]

    model.ramp_up = pyo.Constraint(model.gens, model.time, rule=ramp_up_rule)

    # ramp down constraint
    def ramp_down_rule(model, i, t):
        if t == 0:
            return gens_df.at[i, "g_0"] - model.g[i, t] <= gens_df.at[i, "r_down"]
        else:
            return model.g[i, t - 1] - model.g[i, t] <= gens_df.at[i, "r_down"]

    model.ramp_down = pyo.Constraint(model.gens, model.time, rule=ramp_down_rule)

    # start up cost constraint
    def start_up_rule(model, i, t):
        if t == 0:
            return (
                model.c_up[i, t]
                - (model.u[i, t] - gens_df.at[i, "u_0"]) * gens_df.at[i, "k_up"]
                >= 0
            )
        else:
            return (
                model.c_up[i, t]
                - (model.u[i, t] - model.u[i, t - 1]) * gens_df.at[i, "k_up"]
                >= 0
            )

    model.start_up = pyo.Constraint(model.gens, model.time, rule=start_up_rule)

    # shut down cost constraint
    def shut_down_rule(model, i, t):
        if t == 0:
            return (
                model.c_down[i, t]
                - (gens_df.at[i, "u_0"] - model.u[i, t]) * gens_df.at[i, "k_down"]
                >= 0
            )
        else:
            return (
                model.c_down[i, t]
                - (model.u[i, t - 1] - model.u[i, t]) * gens_df.at[i, "k_down"]
                >= 0
            )

    model.shut_down = pyo.Constraint(model.gens, model.time, rule=shut_down_rule)

    instance = model.create_instance()

    solver = SolverFactory("gurobi")
    solver.solve(instance, tee=False)

    # make new instance with fixed u
    instance_fixed_u = model.create_instance()
    instance_fixed_u.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    for gen in gens_df.index:
        for t in demand_df.index:
            instance_fixed_u.u[gen, t].fix(instance.u[gen, t].value)

    solver.solve(instance_fixed_u, tee=False)

    # get price as dual variable of balance constraint
    prices = pd.DataFrame(columns=["mcp"], index=demand_df.index, data=0.0)
    for t in demand_df.index:
        prices.loc[t, "mcp"] = -instance_fixed_u.dual[instance_fixed_u.balance[t]]

    # get generation and demand
    generation = pd.DataFrame(
        columns=[f"gen_{gen}" for gen in gens_df.index],
        index=demand_df.index,
        data=0.0,
    )

    demand = pd.DataFrame(columns=["demand"], index=demand_df.index, data=0.0)

    for gen in gens_df.index:
        for t in demand_df.index:
            generation.loc[t, f"gen_{gen}"] = instance_fixed_u.g[gen, t].value

    for t in demand_df.index:
        demand.loc[t, "demand"] = instance_fixed_u.d[t].value

    main_df = pd.concat([generation, demand, prices], axis=1)

    start_up_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"start_up_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            start_up_cost.at[t, f"start_up_{gen}"] = instance_fixed_u.c_up[gen, t].value

    shut_down_cost = pd.DataFrame(
        index=demand_df.index, columns=[f"shut_down_{gen}" for gen in gens_df.index]
    )
    for gen in gens_df.index:
        for t in demand_df.index:
            shut_down_cost.at[t, f"shut_down_{gen}"] = instance_fixed_u.c_down[
                gen, t
            ].value

    supp_df = pd.concat([start_up_cost, shut_down_cost], axis=1)

    return main_df, supp_df


# %%
if __name__ == "__main__":
    start = pd.to_datetime("2019-03-01 00:00")
    end = pd.to_datetime("2019-03-02 00:00")

    # generators
    gens_df = pd.read_csv("inputs/gens.csv", index_col=0)

    # 24 hours of demand first increasing and then decreasing
    demand_df = pd.read_csv("inputs/demand.csv", index_col=0)
    demand_df.index = pd.to_datetime(demand_df.index)
    demand_df = demand_df.loc[start:end]

    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = pd.read_csv("outputs/k_values_df.csv", index_col=0)
    k_values_df.columns = k_values_df.columns.astype(int)
