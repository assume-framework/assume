# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import yaml
from matplotlib import pyplot as plt
from pyomo.opt import SolverFactory

from examples.notebooks.MPEC.bilevel_opt import (
    find_optimal_dispatch_quadratic,
)
from examples.notebooks.MPEC.uc_problem import solve_uc_problem


def calculate_profits(main_df, gens_df, price_column="mcp", supp_df=None):
    profits = pd.DataFrame(index=main_df.index, columns=gens_df.index)
    for gen in gens_df.index:
        profits[gen] = main_df[f"gen_{gen}"] * (
            main_df[price_column] - gens_df.at[gen, "mc"]
        )
        if supp_df is not None:
            profits[gen] -= supp_df[f"start_up_{gen}"] + supp_df[f"shut_down_{gen}"]

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


def load_config(inputs_dir, scenario, study_case=""):
    """

    Load the config file from the given directory.

    Args:
        inputs_dir (str): The directory containing the config file.
        scenario (str): The scenario name.
        study_case (str): The study case name. If left empty, the entire config file is loaded.

    Returns:
        dict: The loaded config file.
    """
    config_path = os.path.join(inputs_dir, scenario, "config.yaml")
    with open(config_path) as file:
        config = yaml.safe_load(file)
    if study_case != "":
        config = config[study_case]
    return config


def store_config(config, inputs_dir, scenario):
    """
    Store the config file in the given directory.

    Args:
        config (dict): The config file to store.
        scenario (str): The scenario name.
        inputs_dir (str): The directory to store the config file.
    """

    class NoAliasDumper(yaml.SafeDumper):
        def ignore_aliases(self, _):
            return True

    # Store the config file in the given directory
    config_path = os.path.join(inputs_dir, scenario, "config.yaml")
    with open(config_path, "w") as file:
        yaml.dump(
            config,
            file,
            Dumper=NoAliasDumper,
            default_flow_style=False,
            sort_keys=False,
        )


def get_no_of_val_episodes(inputs_dir, scenario, study_case):
    # Read and calculate the number of validation episodes from the config file
    config = load_config(inputs_dir, scenario, study_case)
    learning_config = config["learning_config"]
    no_of_val_episodes = (
        learning_config["training_episodes"]
        - learning_config["episodes_collecting_initial_experience"]
    ) // learning_config.get("validation_episodes_interval", 5)
    return no_of_val_episodes


def retrieve_best_episode_actions(inputs_dir, scenario, study_case, db):
    """
    Retrieve the actions of the best episode based on the average reward of each episode.
    """
    simulation = f"{scenario}_{study_case}"
    # Get the number of validation episodes
    no_of_val_episodes = get_no_of_val_episodes(inputs_dir, scenario, study_case)
    # Get the average reward for each episode in order to determine the best episode.
    reward_df = pd.DataFrame(
        columns=["avg_reward"], index=range(1, no_of_val_episodes + 1)
    )
    for episode in range(1, no_of_val_episodes + 1):
        query = f"SELECT AVG(reward) as avg_reward FROM rl_params where simulation = '{simulation}' and episode ='{episode}' and evaluation_mode = true"
        reward_df.at[episode, "avg_reward"] = pd.read_sql(query, db).values[0][0]

    # Use the episode with the best reward to get the respective actions
    best_episode = reward_df["avg_reward"].idxmax()
    print(
        f"Best episode: {best_episode} found with an average reward of {reward_df.at[best_episode, 'avg_reward']:.3f}"
    )

    query = f"SELECT datetime as dt, unit, actions_0 FROM rl_params where simulation = '{simulation}' and episode ='{best_episode}' and evaluation_mode = true"

    actions_df = pd.read_sql(query, db)
    actions_df.index = pd.to_datetime(actions_df["dt"])
    actions_df.drop(columns=["dt"], inplace=True)

    return actions_df


def sample_seasonal_weeks(datetime_index):
    """
    Sample one random complete week from each season.

    Args:
        datetime_index (pd.DatetimeIndex): DatetimeIndex of the DataFrame

    Returns:
        pd.DatetimeIndex: Combined index of four sampled weeks (one per season)
    """
    import random

    # Define seasons by month numbers
    seasons = {
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Fall": [9, 10, 11],
        "Winter": [12, 1, 2],
    }

    sampled_dates = []

    for season, months in seasons.items():
        # Get seasonal data indices
        seasonal_idx = datetime_index[datetime_index.month.isin(months)]

        # Find complete weeks within season
        complete_weeks = []
        for week in seasonal_idx.isocalendar().week.unique():
            week_idx = datetime_index[datetime_index.isocalendar().week == week]
            # Check if week is complete (168 hours) and fully within season
            if len(week_idx) == 168 and all(
                month in months for month in week_idx.month.unique()
            ):
                complete_weeks.append(week)

        if complete_weeks:
            # Set a seed for reproducibility
            random.seed(42)

            random_week = random.choice(complete_weeks)
            week_idx = datetime_index[datetime_index.isocalendar().week == random_week]
            sampled_dates.extend([d.date() for d in week_idx])

        print(f"{season} complete weeks: {complete_weeks}")

    return sorted(list(set(sampled_dates)))


def create_gens_df(pp_units, dispatch_df):
    gens_df = pp_units.copy()

    # Transform gen_df into the format that is expected by the optimization problem
    # g_max	mc	u_0	g_0	r_up	r_down	k_up	k_down
    gens_df = gens_df.reset_index()
    gens_df = gens_df.rename(columns={"max_power": "g_max", "min_power": "u_0"})
    gens_df["r_up"] = gens_df["g_max"]  # ramping up constraints
    gens_df["r_down"] = gens_df["g_max"]  # ramping down constraints
    gens_df["k_up"] = 0  # start up costs
    gens_df["k_down"] = 0  # shut down costs
    gens_df["g_0"] = 0  # start with no power output

    # get average mc from dispatch_df per unit name
    mc = dispatch_df.groupby("unit")["marginal_cost"].mean()

    # based on name and unit column join mc into gens_df
    gens_df = gens_df.merge(mc, left_on="name", right_on="unit", how="left")
    gens_df = gens_df.rename(columns={"marginal_cost": "mc"})
    return gens_df


def create_storage_df(storage_units, dispatch_df):
    storage_df = storage_units.copy()

    # check if max_power_charge and max_power_discharge columns are the same value otherwise throw error that this wont work
    assert all(
        storage_df["max_power_charge"] == storage_df["max_power_discharge"]
    ), "max_power_charge and max_power_discharge must be the same value for this to work"
    storage_df["g_max"] = storage_df["max_power_discharge"]
    storage_df["u_0"] = 0  # storage units always can produce power
    storage_df["g_0"] = 0  # start with no power output
    storage_df["r_up"] = storage_df["g_max"]  # ramping up constraints
    storage_df["r_down"] = storage_df["g_max"]  # ramping down constraints
    storage_df["k_up"] = 0  # start up costs
    storage_df["k_down"] = 0  # shut down costs

    # get average mc from dispatch_df per unit name
    mc = dispatch_df.groupby("unit")["marginal_cost"].mean()

    # based on name and unit column join mc into storage_df
    storage_df = storage_df.merge(mc, left_on="name", right_on="unit", how="left")
    storage_df = storage_df.rename(columns={"marginal_cost": "mc"})
    return storage_df


def join_demand_market_orders(demand_df, market_orders_df):
    """
    Join demand_df and market_orders_df, handling multiple demand_EOM bid_ids if present.
    If only demand_EOM_1 (or demand_EOM) is present, keep old behavior.
    If multiple demand_EOM_X exist (e.g., demand_EOM_1, demand_EOM_2), merge all corresponding price/volume pairs.

    Returns:
        demand_df with columns ["volume", "price"] (single or multi-bid as appropriate)
    """
    demand_df = demand_df.copy()
    # Find all demand_EOM bid_ids
    demand_bids = market_orders_df[market_orders_df["unit_id"] == "demand_EOM"]
    unique_bids = demand_bids["bid_id"].unique()

    # If only demand_EOM or demand_EOM_1, keep old behavior
    if len(unique_bids) == 1 and (unique_bids[0] in ["demand_EOM", "demand_EOM_1"]):
        demand_df["price"] = demand_bids["price"].values
        demand_df = demand_df.drop(columns=["date"])
        demand_df.index.name = "datetime"
        demand_df.columns = ["volume_1", "price_1"]
        demand_df.index = pd.to_datetime(demand_df.index)
        demand_df["date"] = demand_df.index.date
        return demand_df

    # If multiple demand_EOM_X bids, merge all
    # Pivot market_orders_df to get all demand_EOM_X price/volume columns
    demand_bids = demand_bids.copy()
    demand_bids["bid_num"] = demand_bids["bid_id"].str.extract(r"(\d+)$").fillna("1")
    demand_bids["bid_num"] = demand_bids["bid_num"].astype(int)
    demand_bids = demand_bids.sort_values("bid_num")

    # For each bid, create price_X and volume_X columns
    merged = demand_df.copy()
    merged = merged.drop(columns=["date"])
    merged.index.name = "datetime"
    merged.index = pd.to_datetime(merged.index)

    for bid_num in sorted(demand_bids["bid_num"].unique()):
        bid_mask = demand_bids["bid_num"] == bid_num
        price_col = f"price_{bid_num}"
        volume_col = f"volume_{bid_num}"
        merged[price_col] = demand_bids.loc[bid_mask, "price"].values
        merged[volume_col] = (
            demand_bids.loc[bid_mask, "volume"].values * -1
        )  # make volume positive

    merged["date"] = merged.index.date
    return merged


def obtain_k_values(k_df, gens_df):
    mc_mapping = dict(zip(gens_df["name"], gens_df["mc"]))
    k_df["gens_df_mc"] = k_df["unit_id"].map(mc_mapping)

    # transformed actions into k_values, one per generator
    k_df["k"] = k_df["price"] / k_df["gens_df_mc"]

    # replace inf with 0
    k_df["k"] = k_df["k"].replace(np.inf, 0)

    # delete rows where unit_id is none
    k_df = k_df[k_df["unit_id"].notna()]

    k_values_df = k_df.pivot(index="time", columns="unit_id", values="k")
    # k_values_df.reset_index(inplace=True)

    # sort columns to match the order of the columns in the gens_df
    k_values_df = k_values_df[gens_df["name"].values]
    k_values_df["date"] = k_values_df.index.date
    return k_values_df


def run_MPEC(
    opt_gen,
    gens_df,
    demand_df,
    k_values_df,
    availability_df,
    k_max,
    big_w,
    demand_bids=1,
):
    """
    Run the MPEC optimization for the given unit and return the profits before and after the optimization.

    Args:
        opt_gen (str): The unit to optimize.
        index (pd.Index): The index of the time series data.
        gens_df (pd.DataFrame): The generator data.
        demand_df (pd.DataFrame): The demand data.
        k_values_df (pd.DataFrame): The k-values data.
        availability_df (pd.DataFrame): The availability data [0,1].
        k_max (float): The maximum k-value.
        big_w (float): The big W value.

    Returns:
        tuple: The profits before and after the optimization.
    """
    print("We now optimize the decison for unit index", gens_df.index[opt_gen])

    demand_df = demand_df.copy(deep=True)
    # reset index to start at 0
    demand_df = demand_df.reset_index(drop=True)

    k_values_df = k_values_df.copy(deep=True)
    # rename columns to match index of gens_df
    k_values_df.columns = gens_df.index
    k_values_df.reset_index(inplace=True)

    availability_df = availability_df.copy(deep=True)
    availability_df.columns = gens_df.index
    availability_df.reset_index(inplace=True)

    gens_df = gens_df.copy(deep=True)

    main_df, supp_df, k_values = find_optimal_dispatch_quadratic(
        gens_df=gens_df,
        k_values_df=k_values_df,
        availabilities_df=availability_df,
        demand_df=demand_df,
        k_max=k_max,
        opt_gen=opt_gen,
        big_w=big_w,
        time_limit=3600,
        print_results=True,
        K=5,
        big_M=10e6,
        demand_bids=demand_bids,
    )

    # calculate actual market clearing prices
    k_values_df_2 = k_values_df.copy()
    k_values_df_2[opt_gen] = k_values

    updated_main_df_2, updated_supp_df_2 = solve_uc_problem(
        gens_df, demand_df, k_values_df_2, availability_df, demand_bids=demand_bids
    )

    # Calculate profits
    profits_1 = calculate_profits(main_df=main_df, supp_df=supp_df, gens_df=gens_df)
    profits_2 = calculate_profits(
        main_df=updated_main_df_2, supp_df=updated_supp_df_2, gens_df=gens_df
    )

    return profits_1, profits_2, updated_main_df_2, updated_supp_df_2


def plot_sample_distribution(sample_df, rest_df):
    colors = list(["green"] * len(rest_df)) + list(["blue"] * len(sample_df))

    # Scatter matrix
    fig = pd.plotting.scatter_matrix(
        pd.concat([rest_df, sample_df], sort=False),
        c=colors,
        figsize=(7, 7),
        range_padding=0.2,
        hist_kwds={"bins": 20},  # Generic histogram configuration
        s=30,
        alpha=0.5,
    )

    # Customize histogram colors for each diagonal
    hist_colors = ["green", "blue"]
    for i, ax in enumerate(fig.diagonal()):
        ax.hist(
            [rest_df.iloc[:, i], sample_df.iloc[:, i]],
            bins=20,
            color=hist_colors,
            stacked=True,
            alpha=0.7,
        )

    # Show plot
    plt.show()


def plot_profit_comparison(df_rl, df_mpec, bound=-10):
    # Filter out zero value columns
    df_rl = df_rl.loc[:, (df_rl != 0).any(axis=0)]
    df_mpec = df_mpec.loc[:, (df_mpec != 0).any(axis=0)]

    df_rl_mean = df_rl.mean(axis=0)
    # Calculate percentage deviation
    percent_deviation = ((df_rl - df_mpec) / df_rl_mean) * 100

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # TODO: comment in when values fixed and kernel approximation works, currently too many zero and similar values exist.
    # Create violin plot
    # parts = ax.violinplot(
    #     [percent_deviation[col].values for col in percent_deviation.columns],
    #     showmeans=False,
    #     showmedians=False,
    #     showextrema=False,
    # )

    # Customize violin plot colors
    # for pc in parts["bodies"]:
    #     pc.set_facecolor("lightblue")
    #     pc.set_alpha(0.7)

    # Add box plot inside violin plot
    ax.boxplot(
        [percent_deviation[col].values for col in percent_deviation.columns],
        positions=range(1, len(percent_deviation.columns) + 1),
        widths=0.2,
        showfliers=True,
        notch=True,
    )

    # Add horizontal lines and colored regions
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.axhline(y=bound, color="black", linestyle="--", alpha=0.7, linewidth=1.5)

    # Create background colors for different regions
    plt.axhspan(
        0,
        max(percent_deviation.max()) + 10,
        color="lightgreen",
        alpha=0.3,
        label="RL profit > MPEC profit",
    )
    plt.axhspan(
        bound,
        0,
        color="yellow",
        alpha=0.3,
        label="RL profit < MPEC profit but in bounds",
    )
    plt.axhspan(
        min(percent_deviation.min()) - 5,
        bound,
        color="lightcoral",
        alpha=0.3,
        label="RL profit < MPEC profit outside bounds",
    )

    # Customize plot
    ax.set_xlabel("Power Plant Units")
    ax.set_ylabel("Deviation (%)\n(RL - MPEC) / mean(RL)")
    ax.set_title("Profit Deviation Distribution (Combined Violin and Box Plot)")
    ax.grid(True, alpha=0.3)

    # Set x-ticks
    ax.set_xticks(range(1, len(percent_deviation.columns) + 1))
    ax.set_xticklabels(percent_deviation.columns)

    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, labels, loc="upper right")

    # Adjust layout
    plt.tight_layout()

    return fig
