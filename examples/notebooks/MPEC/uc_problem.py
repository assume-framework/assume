# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""

Created by: Nick Harder (nick.harder94@gmail.com)
Created on August, 21th, 2023

"""

import numpy as np
import pandas as pd

# %%
# Imports
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


# %%
def solve_uc_problem(gens_df, demand_df, k_values_df, availabilities_df, demand_bids=1, mc_df=None):
    gens_df = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df
    if mc_df is None:
        mc_df = pd.DataFrame(
            {gen: gens_df.at[gen, "mc"] for gen in gens_df.index}, index=demand_df.index
        )

    model = pyo.ConcreteModel()

    # sets
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, demand_bids + 1))

    # primary problem variables
    model.g = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Power output of producer 𝑖 at period 𝑡 (MW)
    model.d = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals
    )  # satisfied demand at period 𝑡, from the multiple demand-bids n (MW)
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
                    k_values_df.at[t, gen] * mc_df.at[t, gen] * model.g[gen, t]
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )

        for t in model.time:
            for n in model.demand_bids:
                expr -= demand_df.at[t, f"price_{n}"] * model.d[t, n]

        return expr

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # energy balance constraint
    def balance_rule(model, t):
        return (
            sum(model.d[t, n] for n in model.demand_bids)
            - sum(model.g[i, t] for i in model.gens)
            == 0
        )

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    # max generation constraint
    def g_max_rule(model, i, t):
        return (
            model.g[i, t]
            <= gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        )

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    # max demand constraint
    def d_max_rule(model, t, n):
        return model.d[t, n] <= demand_df.at[t, f"volume_{n}"]

    model.d_max = pyo.Constraint(model.time, model.demand_bids, rule=d_max_rule)

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

    # ---------------------------------------------------------------------------
    # solve
    # Comment: Because duals (shadow prices) are not well-defined for mixed-integer problems — so we solve the second time as a pure LP, after fixing all binaries.

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

    # ---------------------------------------------------------------------------
    # extract results

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
        demand.loc[t, "demand"] = sum(
            instance_fixed_u.d[t, n].value for n in instance_fixed_u.demand_bids
        )

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


def solve_uc_problem_with_storage(
    gens_df,
    storage_df,
    demand_df,
    k_values_df,
    storage_k_values_df,
    availabilities_df,
    demand_bids=1,
    mc_df=None,
):
    """
    Draft UC problem with storage units.

    This keeps the original UC formulation as intact as possible and extends it by
    storage charge/discharge variables and a state-of-charge coupling over time.

    Expected storage_df columns
    ---------------------------
    - capacity
    - initial_soc
    - max_power_charge
    - max_power_discharge
    - efficiency_charge
    - efficiency_discharge
        - storage bid prices are provided via `storage_k_values_df`
            with the same index structure as `demand_df` and one column per storage

    Notes
    -----
        - Storage units bid time-varying prices via `storage_k_values_df.at[t, s]`.
        - The same price is used for charging and discharging.
    - `initial_soc` is interpreted as a fraction of capacity.
    - To keep the formulation simple and linear, there is no binary variable that
      forbids simultaneous charging and discharging.
    """
    gens_df = gens_df.set_index("unit") if "unit" in gens_df.columns else gens_df
    if mc_df is None:
        mc_df = pd.DataFrame(
            {gen: gens_df.at[gen, "mc"] for gen in gens_df.index}, index=demand_df.index
        )

    model = pyo.ConcreteModel()

    # sets
    model.time = pyo.Set(initialize=demand_df.index)
    model.gens = pyo.Set(initialize=gens_df.index)
    model.storages = pyo.Set(initialize=storage_df.index)
    model.demand_bids = pyo.Set(initialize=np.arange(1, demand_bids + 1))

    # primary problem variables
    model.g = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Power output of producer i at period t (MW)
    model.d = pyo.Var(
        model.time, model.demand_bids, within=pyo.NonNegativeReals
    )  # satisfied demand at period t, from the multiple demand-bids n (MW)
    model.c_up = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Start-up cost of producer i at period t (€)
    model.c_down = pyo.Var(
        model.gens, model.time, within=pyo.NonNegativeReals
    )  # Shut-down cost of producer i at period t (€)
    model.u = pyo.Var(
        model.gens, model.time, within=pyo.Binary
    )  # Binary UC status of producer i at period t

    # storage variables
    model.p_charge = pyo.Var(model.storages, model.time, within=pyo.NonNegativeReals)
    model.p_discharge = pyo.Var(model.storages, model.time, within=pyo.NonNegativeReals)
    model.soc = pyo.Var(model.storages, model.time, within=pyo.NonNegativeReals)

    # primary problem objective
    def objective_rule(model):
        expr = 0
        for gen in model.gens:
            for t in model.time:
                expr += (
                    k_values_df.at[t, gen] * mc_df.at[t, gen] * model.g[gen, t]
                    + model.c_up[gen, t]
                    + model.c_down[gen, t]
                )

        for s in model.storages:
            for t in model.time:
                expr += storage_k_values_df.at[t, s] * model.p_discharge[s, t]
                expr -= storage_k_values_df.at[t, s] * model.p_charge[s, t]

        for t in model.time:
            for n in model.demand_bids:
                expr -= demand_df.at[t, f"price_{n}"] * model.d[t, n]

        return expr

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # energy balance constraint
    def balance_rule(model, t):
        return (
            sum(model.d[t, n] for n in model.demand_bids)
            + sum(model.p_charge[s, t] for s in model.storages)
            - sum(model.g[i, t] for i in model.gens)
            - sum(model.p_discharge[s, t] for s in model.storages)
            == 0
        )

    model.balance = pyo.Constraint(model.time, rule=balance_rule)

    # max generation constraint
    def g_max_rule(model, i, t):
        return (
            model.g[i, t]
            <= gens_df.at[i, "g_max"] * availabilities_df.at[t, i] * model.u[i, t]
        )

    model.g_max = pyo.Constraint(model.gens, model.time, rule=g_max_rule)

    # max demand constraint
    def d_max_rule(model, t, n):
        return model.d[t, n] <= demand_df.at[t, f"volume_{n}"]

    model.d_max = pyo.Constraint(model.time, model.demand_bids, rule=d_max_rule)

    # storage power limits
    def charge_max_rule(model, s, t):
        return model.p_charge[s, t] <= abs(storage_df.at[s, "max_power_charge"])

    model.charge_max = pyo.Constraint(model.storages, model.time, rule=charge_max_rule)

    def discharge_max_rule(model, s, t):
        return model.p_discharge[s, t] <= storage_df.at[s, "max_power_discharge"]

    model.discharge_max = pyo.Constraint(
        model.storages, model.time, rule=discharge_max_rule
    )

    # storage state of charge coupling
    def soc_rule(model, s, t):
        if t == 0:
            previous_soc = (
                storage_df.at[s, "initial_soc"] * storage_df.at[s, "capacity"]
            )
        else:
            previous_soc = model.soc[s, t - 1]

        return model.soc[s, t] == (
            previous_soc
            + storage_df.at[s, "efficiency_charge"] * model.p_charge[s, t]
            - model.p_discharge[s, t] / storage_df.at[s, "efficiency_discharge"]
        )

    model.soc_coupling = pyo.Constraint(model.storages, model.time, rule=soc_rule)

    def soc_max_rule(model, s, t):
        return model.soc[s, t] <= storage_df.at[s, "capacity"]

    model.soc_max = pyo.Constraint(model.storages, model.time, rule=soc_max_rule)

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

    # ---------------------------------------------------------------------------
    # solve
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

    # ---------------------------------------------------------------------------
    # extract results

    prices = pd.DataFrame(columns=["mcp"], index=demand_df.index, data=0.0)
    for t in demand_df.index:
        prices.loc[t, "mcp"] = -instance_fixed_u.dual[instance_fixed_u.balance[t]]

    generation = pd.DataFrame(
        columns=[f"gen_{gen}" for gen in gens_df.index],
        index=demand_df.index,
        data=0.0,
    )
    demand = pd.DataFrame(columns=["demand"], index=demand_df.index, data=0.0)
    charge = pd.DataFrame(
        columns=[f"charge_{s}" for s in storage_df.index],
        index=demand_df.index,
        data=0.0,
    )
    discharge = pd.DataFrame(
        columns=[f"discharge_{s}" for s in storage_df.index],
        index=demand_df.index,
        data=0.0,
    )
    soc = pd.DataFrame(
        columns=[f"soc_{s}" for s in storage_df.index],
        index=demand_df.index,
        data=0.0,
    )

    for gen in gens_df.index:
        for t in demand_df.index:
            generation.loc[t, f"gen_{gen}"] = instance_fixed_u.g[gen, t].value

    for t in demand_df.index:
        demand.loc[t, "demand"] = sum(
            instance_fixed_u.d[t, n].value for n in instance_fixed_u.demand_bids
        )

    for s in storage_df.index:
        for t in demand_df.index:
            charge.loc[t, f"charge_{s}"] = instance_fixed_u.p_charge[s, t].value
            discharge.loc[t, f"discharge_{s}"] = instance_fixed_u.p_discharge[
                s, t
            ].value
            soc.loc[t, f"soc_{s}"] = instance_fixed_u.soc[s, t].value

    main_df = pd.concat([generation, charge, discharge, soc, demand, prices], axis=1)

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
