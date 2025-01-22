'''
Minimalbeispiel für Assume
- 2 nodes
- wind + oil at node 1, gas and coal at node 2
- hourly EOM with pay as clear
- redispatch market pay as bid opening subsequent to EOM
- 2 demand timeseries representing germany divided by 10 demand
- simulation of 1 month with 1h timesteps
- study_cases base_case, learning_oil, learning_gas, learning_coal with learning switched on for oil, gas, coal respectively
'''

#
#!pip install assume-framework -U
#

import logging
import os
from datetime import datetime, timedelta
import numpy as np
import yaml

import pandas as pd
from dateutil import rrule as rr

from assume import World
from assume.common.forecasts import NaiveForecast
from assume.common.market_objects import MarketConfig, MarketProduct

log = logging.getLogger(__name__)

# import the main World class and the load_scenario_folder functions from assume
from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning
from assume.strategies.learning_strategies import RLStrategy
#from assume.reinforcement_learning.learning_utils import NormalActionNoise
#from assume.reinforcement_learning.algorithms import actor_architecture_aliases


# Define paths for input and output data
#csv_path = r"C:\Users\Gunter\Code\EEM_paper\outputs"
#input_path = r"C:\Users\Gunter\Code\EEM_paper\inputs"
csv_path = "outputs"
input_path = "inputs"
scenario = "2_nodes"

for study_case in ["base_case", "learning_oil", "learning_gas", "learning_coal", "learning_all3"]:
#study_case="base_case"
#study_case = "learning_coal"
#study_case = "learning_gas"
#study_case = "learning_oil"
#study_case = "learning_all3"
#study_case = "learning_coal"

    if study_case == "base_case":
        eom_bidding_list = ["naive_eom"] * 4
        redispatch_bidding_list = ["naive_redispatch"] * 4
    elif study_case == "learning_coal":
        eom_bidding_list = ["naive_eom"] * 3 + ["pp_learning"]
        redispatch_bidding_list = ["naive_redispatch"] * 3 + ["redispatch_learning"]
    elif study_case == "learning_gas":
        eom_bidding_list = ["naive_eom"] * 2 + ["pp_learning"] + ["naive_eom"]
        redispatch_bidding_list = ["naive_redispatch"] * 2 + ["redispatch_learning"] + ["naive_redispatch"]
    elif study_case == "learning_oil":
        eom_bidding_list = ["naive_eom"] + ["pp_learning"] + ["naive_eom"] * 2
        redispatch_bidding_list = ["naive_redispatch"] + ["redispatch_learning"] + ["naive_redispatch"] * 2
    elif study_case == "learning_all3":
        eom_bidding_list = ["naive_eom"] + ["pp_learning"] * 3
        redispatch_bidding_list = ["naive_redispatch"] + ["redispatch_learning"] * 3
    elif study_case == "test_redispatch":
        eom_bidding_list = ["naive_eom"] * 3 + ["pp_learning"]
        redispatch_bidding_list = ["naive_redispatch"] * 3 + ["redispatch_learning"]

    # as long as two markets learning is not implemented: choose naive redispatch for all
    #redispatch_bidding_list = ["naive_redispatch"] * 4

    # Create the data
    powerplant_units_data = {
        "name": ["Wind", "Oil", "Gas", "Coal"],
        "technology": ["wind", "oil", "natural gas", "lignite"],
        "node": ["north", "north", "south", "south"],
        "bidding_EOM": eom_bidding_list,
        "bidding_Redispatch": redispatch_bidding_list,
        "fuel_type": ["renewable", "oil", "natural gas", "lignite"],
        "max_power": [15000.0, 2000.0, 3000.0, 3000.0],
        "min_power": [0, 0, 0, 0],
        "additional_cost": [0, 120, 80, 50],
        "unit_operator": ["wind operator", "oil operator", "gas operator", "coal operator"],
        "emission_factor": [0, 0, 0, 0],
        "efficiency": [1, 0.3, 0.4, 0.5],
    }
    #
    # Convert to DataFrame and save as CSV
    powerplant_units_df = pd.DataFrame(powerplant_units_data)
    powerplant_units_df.to_csv(f"{input_path}/{scenario}/powerplant_units.csv", index=False)

    demand_units_data = {
        "name": ["demand_north", "demand_south"],
        "technology": ["inflex_demand", "inflex_demand"],
        "bidding_EOM": ["naive_eom", "naive_eom"],
        "bidding_Redispatch": ["naive_redispatch", "naive_redispatch"],
        "node": ["north", "south"],
        "max_power": [100000, 100000],
        "min_power": [0, 0],
        "unit_operator": ["demand operator north", "demand operator south"],
    }

    # Convert to DataFrame and save as CSV
    demand_units_df = pd.DataFrame(demand_units_data)
    demand_units_df.to_csv(f"{input_path}/{scenario}/demand_units.csv", index=False)

    # read demand data for germany and split into two regions
    german_demand_data = pd.read_csv(f"{input_path}/example_01f/demand_df.csv", index_col='datetime')
    demand_data = pd.DataFrame(
        {
            "demand_north": german_demand_data.loc["2019-01-01 00:00":"2019-02-01 23:45", "demand_EOM"] / 20,
            "demand_south": german_demand_data.loc["2019-01-01 00:00":"2019-02-01 23:45", "demand_EOM"] / 20,
        }
    )
    demand_data.index = pd.to_datetime(demand_data.index)
    demand_data.to_csv(f"{input_path}/{scenario}/demand_df.csv", index=True)



    # define the database uri. In this case we are using a local sqlite database
    #db_uri = "sqlite:///local_db/assume_db.db"
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
            inputs_path="inputs",
            scenario=scenario,
            study_case=study_case,
        )

    # run the simulation
    world.run()
    # 
