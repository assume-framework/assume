# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Minimalbeispiel für Assume
- 2 nodes
- wind + oil at node 1, gas and coal at node 2
- hourly EOM with pay as clear
- redispatch market pay as bid opening subsequent to EOM
- 2 demand timeseries representing germany divided by 10 demand
- simulation of 1 month with 1h timesteps
- study_cases base_case, learning_oil, learning_gas, learning_coal with learning switched on for oil, gas, coal respectively
"""

#
#!pip install assume-framework -U
#

import logging

import pandas as pd

from assume import World

# import the main World class and the load_scenario_folder functions from assume
from assume.scenario.loader_csv import load_scenario_folder, run_learning
from assume.strategies.learning_strategies import RLStrategy

log = logging.getLogger(__name__)

# Define paths for input and output data
# csv_path = r"C:\Users\Gunter\Code\EEM_paper\outputs"
# input_path = r"C:\Users\Gunter\Code\EEM_paper\inputs"
csv_path = "outputs"
input_path = "inputs"
scenario = "2_nodes"

# powerplant units according to example in Hirth and Schlecht (2024)
num_wind = 1
num_diesel = 5
num_gas = 25
num_coal = 20
num_all = num_wind + num_diesel + num_gas + num_coal

for study_case in [
    #"base_case",
    #"learning_gas",
    #"learning_diesel",
    #"learning_coal",
    #"learning_all3",
    #"learning_single_gas",
    #"learning_single_coal",
    "learning_single_diesel",
]:
    # study_case="base_case"
    # study_case = "learning_coal"
    # study_case = "learning_gas"
    # study_case = "learning_oil"
    # study_case = "learning_all3"
    # study_case = "learning_coal"

    if study_case == "base_case":
        eom_bidding_list = ["naive_eom"] * num_all
        redispatch_bidding_list = ["naive_redispatch"] * num_all
    elif study_case == "learning_coal":
        eom_bidding_list = ["naive_eom"] * (num_all - num_coal) + [
            "redispatch_learning"
        ] * num_coal
        redispatch_bidding_list = ["naive_redispatch"] * (num_all - num_coal) + [
            "redispatch_learning"
        ] * num_coal
    elif study_case == "learning_gas":
        eom_bidding_list = (
            ["naive_eom"] * (num_wind + num_diesel)
            + ["redispatch_learning"] * num_gas
            + ["naive_eom"] * num_coal
        )
        redispatch_bidding_list = (
            ["naive_redispatch"] * (num_wind + num_diesel)
            + ["redispatch_learning"] * num_gas
            + ["naive_redispatch"] * num_coal
        )
    elif study_case == "learning_diesel":
        eom_bidding_list = (
            ["naive_eom"] * num_wind
            + ["redispatch_learning"] * num_diesel
            + ["naive_eom"] * (num_gas + num_coal)
        )
        redispatch_bidding_list = (
            ["naive_redispatch"] * num_wind
            + ["redispatch_learning"] * num_diesel
            + ["naive_redispatch"] * (num_gas + num_coal)
        )
    elif study_case == "learning_all3":
        eom_bidding_list = ["naive_eom"] * num_wind + ["redispatch_learning"] * (
            num_all - num_wind
        )
        redispatch_bidding_list = ["naive_redispatch"] * num_wind + [
            "redispatch_learning"
        ] * (num_all - num_wind)
    elif study_case == "learning_single_gas":
        eom_bidding_list = (
            ["naive_eom"] * num_all
        )
        redispatch_bidding_list = (
            ["naive_redispatch"] * num_all
        )
        # change bidding behaviour for one gas unit (e.g. the fifth unit (unit 4) with MC=44 €)
        # unit 4 does not lern anything, profit = 0. when sigma=0.1 und initial experience = 20 bietet es immer -100€ und bekommt 49€ von EOM
        eom_bidding_list[num_wind + num_diesel + 4] = "redispatch_learning"
        redispatch_bidding_list[num_wind + num_diesel + 4] = "redispatch_learning"
    elif study_case == "learning_single_coal":
        eom_bidding_list = (
            ["naive_eom"] * num_all
        )
        redispatch_bidding_list = (
            ["naive_redispatch"] * num_all
        )
        # change bidding behaviour for one coal unit (e.g. the fifteenth unit (unit 14) with MC=34 €)
        # unit 14 does not leran overbidding in current setup. it always bids -100€ and gets payment of 49 from EOM...
        # try with unit that is more on the edge: unit 19 with mc=39
        eom_bidding_list[num_wind + num_diesel + num_gas + 14] = "redispatch_learning"
        redispatch_bidding_list[num_wind + num_diesel + num_gas + 14] = "redispatch_learning"
    
    
    elif study_case == "learning_single_diesel":
        eom_bidding_list = (
            ["naive_eom"] * num_all
        )
        redispatch_bidding_list = (
            ["naive_redispatch"] * num_all
        )
        # change bidding behaviour for one diesel unit (e.g. the 2nd unit (unit 1) with MC=66 €)
        eom_bidding_list[num_wind + 1] = "redispatch_learning"
        redispatch_bidding_list[num_wind + 1] = "redispatch_learning"

    mc_wind_list = [0 for i in range(num_wind)]
    mc_diesel_list = [65 + i for i in range(num_diesel)]
    mc_gas_list = [40 + i for i in range(num_gas)]
    mc_coal_list = [20 + i for i in range(num_coal)]
    # Create the data
    powerplant_units_data = {
        "name": [f"Wind {i}" for i in range(num_wind)]
        + [f"Diesel {i}" for i in range(num_diesel)]
        + [f"Gas {i}" for i in range(num_gas)]
        + [f"Coal {i}" for i in range(num_coal)],
        "technology": ["lignite"] * num_wind # hier vorübergehend technology lignite gemacht um 20 gw 0 marginal cost zu haben und keine Schwankungen durch Wind generation
        + ["diesel"] * num_diesel
        + ["natural gas"] * num_gas
        + ["lignite"] * num_coal,
        "node": ["north"] * num_wind
        + ["north"] * num_diesel
        + ["south"] * num_gas
        + ["north"] * num_coal,
        "bidding_EOM": eom_bidding_list,
        "bidding_redispatch": redispatch_bidding_list,
        "fuel_type": ["renewable"] * num_wind
        + ["diesel"] * num_diesel
        + ["natural gas"] * num_gas
        + ["lignite"] * num_coal,
        "max_power": [20000.0] * num_wind + [1000.0] * (num_all - num_wind),
        "min_power": [0] * num_all,
        "additional_cost": mc_wind_list + mc_diesel_list + mc_gas_list + mc_coal_list,
        "unit_operator": ["wind operator"] * num_wind
        + ["diesel operator"] * num_diesel
        + ["gas operator"] * num_gas
        + ["coal operator"] * num_coal,
        "emission_factor": [0] * num_all,
        "efficiency": [1] * num_wind
        + [0.3] * num_diesel
        + [0.4] * num_gas
        + [0.5] * num_coal,
    }
    #
    # Convert to DataFrame and save as CSV
    powerplant_units_df = pd.DataFrame(powerplant_units_data)
    powerplant_units_df.to_csv(
        f"{input_path}/{scenario}/powerplant_units.csv", index=False
    )

    demand_units_data = {
        "name": ["demand_south", "demand_north"],
        "technology": ["inflex_demand", "inflex_demand"],
        "bidding_EOM": ["naive_eom", "naive_eom"],
        "bidding_redispatch": ["naive_redispatch", "naive_redispatch"],
        "node": ["south", "north"],
        "max_power": [100000, 100000],
        "min_power": [0, 0],
        "unit_operator": ["demand operator south", "demand operator north"],
    }

    # Convert to DataFrame and save as CSV
    demand_units_df = pd.DataFrame(demand_units_data)
    demand_units_df.to_csv(f"{input_path}/{scenario}/demand_units.csv", index=False)

    # read demand data for germany and split into two regions
    german_demand_data = pd.read_csv(
        f"{input_path}/example_01f/demand_df.csv", index_col="datetime"
    )
    # demand_data = pd.DataFrame(
    # {
    # "demand_north": german_demand_data.loc["2019-01-01 00:00":"2019-02-01 23:45", "demand_EOM"] / 20,
    # "demand_south": german_demand_data.loc["2019-01-01 00:00":"2019-02-01 23:45", "demand_EOM"] / 10,
    # }
    # )
    demand_data = pd.DataFrame(
        index=german_demand_data.loc["2019-01-01 00:00":"2019-02-01 23:45"].index
    )
    demand_data["demand_north"] = [0] * len(demand_data)
    demand_data["demand_south"] = [50000] * len(demand_data)
    demand_data.index = pd.to_datetime(demand_data.index)
    demand_data.to_csv(f"{input_path}/{scenario}/demand_df.csv", index=True)

    # define the database uri. In this case we are using a local sqlite database
    # db_uri = "sqlite:///local_db/assume_db.db"
    db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # create world instance
    world = World(database_uri=db_uri, export_csv_path=csv_path)
    world.bidding_strategies["pp_learning"] = RLStrategy
    # load scenario by providing the world instance
    # the path to the inputs folder and the scenario name (subfolder in inputs)
    # and the study case name (which config to use for the simulation)
    load_scenario_folder(
        world,
        inputs_path=input_path,
        scenario=scenario,
        study_case=study_case,
    )

    if world.learning_config.get("learning_mode", False):
        run_learning(
            world,
            inputs_path=input_path,
            scenario=scenario,
            study_case=study_case,
        )

    # run the simulation
    world.run()
    #
