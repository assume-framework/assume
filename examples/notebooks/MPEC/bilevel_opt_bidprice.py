# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Bid-price reformulation of the MPEC with storage.

Instead of model.k (multiplier on mc, needs k_max=300000 for renewables),
the leader decision is model.bid_price in €/MWh directly (bounded 0..bid_price_max).
This eliminates the huge k*mc products and keeps all coefficients well-scaled.

Non-strategic generators also get pre-computed bid prices (k_values * mc)
so the entire formulation works in €/MWh space.
"""

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def find_optimal_dispatch_bidprice_with_storage(
    gens_df,
    storage_df,
    k_values_df,
    storage_k_values_df,
    availabilities_df,
    demand_df,
    opt_gen,
    bid_price_max=3000,
    big_w=10000,
    time_limit=600,
    print_results=False,
    big_M=10000,
    demand_bids=1,
    mc_df=None,
):
    """
    Quadratic MPEC with one strategic generator leader and non-strategic
    storages in the lower level.

    Bid-price reformulation: the leader's decision variable is the bid price
    in €/MWh directly (not a multiplier k on marginal cost). This avoids
    extreme coefficient ranges when mc is near zero (renewables).

    Args:
        bid_price_max: upper bound on the strategic generator's bid price in €/MWh
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

    model.p_charge = pyo.Var(model.storages, model.time, within=pyo.NonNegativeReals)
    model.p_discharge = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )
    model.soc = pyo.Var(model.storages, model.time, within=pyo.NonNegativeReals)

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

    model.alpha_charge_max = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )
    model.alpha_discharge_max = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )

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

    model.alpha_charge_max_hat = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )
    model.alpha_discharge_max_hat = pyo.Var(
        model.storages, model.time, within=pyo.NonNegativeReals
    )

    # -------------------------------------------------------------------------
    # BINARIES FOR COMPLEMENTARITY
    # -------------------------------------------------------------------------
    model.mu_max_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.nu_max_hat_binary = pyo.Var(
        model.time, model.demand_bids, within=pyo.Binary
    )
    model.nu_min_hat_binary = pyo.Var(
        model.time, model.demand_bids, within=pyo.Binary
    )
    model.pi_u_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.pi_d_hat_binary = pyo.Var(model.gens, model.time, within=pyo.Binary)
    model.alpha_charge_max_hat_binary = pyo.Var(
        model.storages, model.time, within=pyo.Binary
    )
    model.alpha_discharge_max_hat_binary = pyo.Var(
        model.storages, model.time, within=pyo.Binary
    )

    # -------------------------------------------------------------------------
    # STORAGE BID HELPERS (pre-computed, in €/MWh)
    # -------------------------------------------------------------------------
    def storage_bid_charge(s, t):
        mc_charge = (
            storage_df.at[s, "additional_cost_charge"]
            if "additional_cost_charge" in storage_df.columns
            else storage_df.at[s, "mc"]
        )
        return storage_k_values_df.at[t, s] * mc_charge

    def storage_bid_discharge(s, t):
        mc_discharge = (
            storage_df.at[s, "additional_cost_discharge"]
            if "additional_cost_discharge" in storage_df.columns
            else storage_df.at[s, "mc"]
        )
        return storage_k_values_df.at[t, s] * mc_discharge

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
        # Primal lower-level objective in €/MWh space
        expr = sum(
            gen_bid(gen, t) * model.g[gen, t]
            + model.c_up[gen, t]
            + model.c_down[gen, t]
            for gen in model.gens
            for t in model.time
        )

        expr += sum(
            storage_bid_discharge(s, t) * model.p_discharge[s, t]
            - storage_bid_charge(s, t) * model.p_charge[s, t]
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
    def balance_rule(model, t):
        return (
            sum(model.d[t, n] for n in model.demand_bids)
            + sum(model.p_charge[s, t] for s in model.storages)
            - sum(model.g[i, t] for i in model.gens)
            - sum(model.p_discharge[s, t] for s in model.storages)
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

    def charge_max_rule(model, s, t):
        return model.p_charge[s, t] <= abs(storage_df.at[s, "max_power_charge"])

    model.charge_max = pyo.Constraint(
        model.storages, model.time, rule=charge_max_rule
    )

    def discharge_max_rule(model, s, t):
        return model.p_discharge[s, t] <= storage_df.at[s, "max_power_discharge"]

    model.discharge_max = pyo.Constraint(
        model.storages, model.time, rule=discharge_max_rule
    )

    # -------------------------------------------------------------------------
    # NON-DUALIZED SOC SIDE CONSTRAINTS
    # -------------------------------------------------------------------------
    def soc_rule(model, s, t):
        if t == 0:
            return (
                model.soc[s, t]
                - storage_df.at[s, "efficiency_charge"] * model.p_charge[s, t]
                + model.p_discharge[s, t]
                / storage_df.at[s, "efficiency_discharge"]
                == storage_df.at[s, "initial_soc"] * storage_df.at[s, "capacity"]
            )
        return (
            model.soc[s, t]
            - model.soc[s, t - 1]
            - storage_df.at[s, "efficiency_charge"] * model.p_charge[s, t]
            + model.p_discharge[s, t]
            / storage_df.at[s, "efficiency_discharge"]
            == 0
        )

    model.soc_coupling = pyo.Constraint(
        model.storages, model.time, rule=soc_rule
    )

    def soc_max_rule(model, s, t):
        return model.soc[s, t] <= storage_df.at[s, "capacity"]

    model.soc_max = pyo.Constraint(model.storages, model.time, rule=soc_max_rule)

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

    model.status_dual = pyo.Constraint(
        model.gens, model.time, rule=status_dual_rule
    )

    def demand_dual_rule(model, t, n):
        return (
            -demand_df.at[t, f"price_{n}"]
            + model.lambda_[t]
            + model.nu_max[t, n]
            >= 0
        )

    model.demand_dual = pyo.Constraint(
        model.time, model.demand_bids, rule=demand_dual_rule
    )

    def storage_charge_dual_rule(model, s, t):
        return (
            -storage_bid_charge(s, t)
            + model.lambda_[t]
            + model.alpha_charge_max[s, t]
            >= 0
        )

    model.storage_charge_dual = pyo.Constraint(
        model.storages, model.time, rule=storage_charge_dual_rule
    )

    def storage_discharge_dual_rule(model, s, t):
        return (
            storage_bid_discharge(s, t)
            - model.lambda_[t]
            + model.alpha_discharge_max[s, t]
            >= 0
        )

    model.storage_discharge_dual = pyo.Constraint(
        model.storages, model.time, rule=storage_discharge_dual_rule
    )

    # -------------------------------------------------------------------------
    # RELAXED KKT STATIONARITY (HAT SYSTEM)
    # -------------------------------------------------------------------------
    def kkt_gen_rule(model, i, t):
        bid = gen_bid(i, t)
        pi_u_hat_next = (
            0 if t == model.time.at(-1) else model.pi_u_hat[i, t + 1]
        )
        pi_d_hat_next = (
            0 if t == model.time.at(-1) else model.pi_d_hat[i, t + 1]
        )
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

    def kkt_storage_charge_rule(model, s, t):
        return (
            -storage_bid_charge(s, t)
            + model.lambda_hat[t]
            + model.alpha_charge_max_hat[s, t]
            >= 0
        )

    model.kkt_storage_charge = pyo.Constraint(
        model.storages, model.time, rule=kkt_storage_charge_rule
    )

    def kkt_storage_discharge_rule(model, s, t):
        return (
            storage_bid_discharge(s, t)
            - model.lambda_hat[t]
            + model.alpha_discharge_max_hat[s, t]
            >= 0
        )

    model.kkt_storage_discharge = pyo.Constraint(
        model.storages, model.time, rule=kkt_storage_discharge_rule
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

    charge_df = pd.DataFrame(
        index=time_index, columns=[f"charge_{s}" for s in storage_df.index]
    )
    discharge_df = pd.DataFrame(
        index=time_index, columns=[f"discharge_{s}" for s in storage_df.index]
    )
    soc_df = pd.DataFrame(
        index=time_index, columns=[f"soc_{s}" for s in storage_df.index]
    )
    for s in storage_df.index:
        for t in time_index:
            charge_df.at[t, f"charge_{s}"] = instance.p_charge[s, t].value
            discharge_df.at[t, f"discharge_{s}"] = instance.p_discharge[
                s, t
            ].value
            soc_df.at[t, f"soc_{s}"] = instance.soc[s, t].value

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
        [generation_df, charge_df, discharge_df, soc_df, cleared_demand, mcp, mcp_hat],
        axis=1,
    )

    bid_prices = pd.DataFrame(index=time_index, columns=["bid_price"])
    for t in time_index:
        bid_prices.at[t, "bid_price"] = instance.bid_price[t].value

    return main_df, None, bid_prices
