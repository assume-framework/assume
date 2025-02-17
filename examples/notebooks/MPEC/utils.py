import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


def calculate_profits(main_df, gens_df, price_column="mcp", supp_df=None):
    profits = pd.DataFrame(index=main_df.index, columns=gens_df.index)
    for gen in gens_df.index:
        profits[gen] = (
            main_df[f"gen_{gen}"] * (main_df[price_column] - gens_df.at[gen, "mc"])
        )
        if supp_df is not None:
            profits[gen] -= (supp_df[f"start_up_{gen}"] + supp_df[f"shut_down_{gen}"])

    return profits


def calculate_uplift(main_df, gens_df, gen_unit, profits, price_column="mcp"):
    model = pyo.ConcreteModel()
    # sets
    model.time = pyo.Set(initialize=main_df.index)

    # primary variables
    model.g = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.c_up = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.c_down = pyo.Var(model.time, within=pyo.NonNegativeReals)
    model.u = pyo.Var(model.time, within=pyo.Binary)

    def objective_rule(model):
        return sum(
            main_df.at[t, price_column] * model.g[t]
            - gens_df.at[gen_unit, "mc"] * model.g[t]
            - model.c_up[t]
            - model.c_down[t]
            for t in model.time
        )

    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

    # max generation constraint
    def g_max_rule(model, t):
        return model.g[t] <= gens_df.at[gen_unit, "g_max"] * model.u[t]

    model.g_max = pyo.Constraint(model.time, rule=g_max_rule)

    # min generation constraint
    def g_min_rule(model, t):
        return model.g[t] >= gens_df.at[gen_unit, "g_min"] * model.u[t]

    model.g_min = pyo.Constraint(model.time, rule=g_min_rule)

    # max ramp up constraint
    def ru_max_rule(model, t):
        if t == 0:
            return (
                model.g[t] - gens_df.at[gen_unit, "g_0"] <= gens_df.at[gen_unit, "r_up"]
            )
        else:
            return model.g[t] - model.g[t - 1] <= gens_df.at[gen_unit, "r_up"]

    model.ru_max = pyo.Constraint(model.time, rule=ru_max_rule)

    # max ramp down constraint
    def rd_max_rule(model, t):
        if t == 0:
            return (
                gens_df.at[gen_unit, "g_0"] - model.g[t]
                <= gens_df.at[gen_unit, "r_down"]
            )
        else:
            return model.g[t - 1] - model.g[t] <= gens_df.at[gen_unit, "r_down"]

    model.rd_max = pyo.Constraint(model.time, rule=rd_max_rule)

    # start up cost constraint
    def start_up_cost_rule(model, t):
        if t == 0:
            return (
                model.c_up[t]
                >= (model.u[t] - gens_df.at[gen_unit, "u_0"])
                * gens_df.at[gen_unit, "k_up"]
            )
        else:
            return (
                model.c_up[t]
                >= (model.u[t] - model.u[t - 1]) * gens_df.at[gen_unit, "k_up"]
            )

    model.start_up_cost = pyo.Constraint(model.time, rule=start_up_cost_rule)

    # shut down cost constraint
    def shut_down_cost_rule(model, t):
        if t == 0:
            return (
                model.c_down[t]
                >= (gens_df.at[gen_unit, "u_0"] - model.u[t])
                * gens_df.at[gen_unit, "k_down"]
            )
        else:
            return (
                model.c_down[t]
                >= (model.u[t - 1] - model.u[t]) * gens_df.at[gen_unit, "k_down"]
            )

    model.shut_down_cost = pyo.Constraint(model.time, rule=shut_down_cost_rule)

    # solve model
    solver = SolverFactory("gurobi")
    instance = model.create_instance()
    solver.solve(instance, tee=False)

    # calculate uplift
    uplift = max(instance.objective() - profits, 0)

    generation_df = pd.DataFrame(index=main_df.index, columns=[f"gen_{gen_unit}"])
    for t in main_df.index:
        generation_df.at[t, f"gen_{gen_unit}"] = instance.g[t].value

    start_up_cost = pd.DataFrame(index=main_df.index, columns=[f"start_up_{gen_unit}"])
    for t in main_df.index:
        start_up_cost.at[t, f"start_up_{gen_unit}"] = instance.c_up[t].value

    shut_down_cost = pd.DataFrame(
        index=main_df.index, columns=[f"shut_down_{gen_unit}"]
    )
    for t in main_df.index:
        shut_down_cost.at[t, f"shut_down_{gen_unit}"] = instance.c_down[t].value

    uplift_df = pd.concat([generation_df, start_up_cost, shut_down_cost], axis=1)

    return uplift, uplift_df


def calculate_rl_profit(gens_df, demand_df, market_orders, start, end):
    rl_profits = pd.DataFrame(index=demand_df.index, columns=gens_df.index, data=0.0)
    for opt_gen in gens_df.index:
        rl_unit_orders = market_orders[market_orders["unit_id"] == f"Unit_{opt_gen}"]
        rl_unit_orders = rl_unit_orders.loc[start:end]
        rl_unit_orders = rl_unit_orders.reset_index(drop=False)
        marginal_cost = gens_df.at[opt_gen, "mc"]
        rl_profits[opt_gen] = rl_unit_orders["accepted_volume"] * (
            rl_unit_orders["accepted_price"] - marginal_cost
        )

        # iterate over all rows and subtract start up and shut down costs if the unit turned on or off
        for t in range(1, len(rl_unit_orders)):
            if t == 1:
                if (
                    rl_unit_orders.at[t, "accepted_volume"] > 0
                    and gens_df.at[opt_gen, "u_0"] == 0
                ):
                    rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_up"]
                elif (
                    rl_unit_orders.at[t, "accepted_volume"] == 0
                    and gens_df.at[opt_gen, "u_0"] > 0
                ):
                    rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_down"]
            elif (
                rl_unit_orders.at[t, "accepted_volume"] == 0
                and rl_unit_orders.at[t - 1, "accepted_volume"] > 0
            ):
                rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_down"]
            elif (
                rl_unit_orders.at[t, "accepted_volume"] > 0
                and rl_unit_orders.at[t - 1, "accepted_volume"] == 0
            ):
                rl_profits[opt_gen][t] -= gens_df.at[opt_gen, "k_up"]

    return rl_profits


def calculate_mc(powerplant, fuel_prices):
    fuel = powerplant["fuel"]
    fuel_cost = fuel_prices[fuel].iloc[0] / powerplant["efficiency"]
    co2_cost = fuel_prices["co2"].iloc[0] * powerplant["ef"] / powerplant["efficiency"]
    variable_cost = powerplant["variable_cost"]

    mc = fuel_cost + co2_cost + variable_cost

    return mc
