# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# %%
import logging
import os
import random as rd

import numpy as np
import torch as th

from assume import World, load_scenario_folder, run_learning

np.random.seed(0)
rd.seed(0)
th.manual_seed(0)
th.use_deterministic_algorithms(True)

log = logging.getLogger(__name__)

csv_path = "./examples/outputs"
os.makedirs(csv_path, exist_ok=True)

os.makedirs("./examples/local_db", exist_ok=True)

availabe_examples = {
    "2020_rule_SB": {
        # merit order clearing for 24 hours
        "scenario": "2020_SB",
        "study_case": "dam",
    },
    "2020_rule_BB": {
        # complex clearing with block bids
        "scenario": "2020_BB",
        "study_case": "dam",
    },
    "2020_rule_LB": {
        # complex clearing with linked bids
        "scenario": "2020_LB",
        "study_case": "dam",
    },
    "2020_rule_tiny": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2020_LB",
        "study_case": "tiny",
    },
    "2020_RL_SB": {
        # merit order clearing with RL strategies
        "scenario": "2020_RL",
        "study_case": "dam_SB",
    },
    "2020_RL_BB": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_BB",
    },
    "2020_RL_tiny": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2020_RL",
        "study_case": "tiny",
    },
    "2037_RL_SB": {
        # merit order clearing with RL strategies
        "scenario": "2037_RL",
        "study_case": "dam_SB",
    },
    "2037_RL_BB": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_BB",
    },
    "2037_rule_SB": {
        # merit order clearing for 24 hours
        "scenario": "2037_SB",
        "study_case": "dam",
    },
    "2037_rule_BB": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2037_BB",
        "study_case": "dam",
    },
    "2037_rule_LB": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2037_LB",
        "study_case": "dam",
    },
    "2020_RL_1": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_LB_winter",
    },
    "2020_RL_2": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_LB_summer",
    },
    "2020_RL_3": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_BB_winter",
    },
    "2020_RL_4": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_BB_summer",
    },
    "2020_RL_5": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_SB_winter",
    },
    "2020_RL_6": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_SB_summer",
    },
    "2037_RL_1": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_LB_winter",
    },
    "2037_RL_2": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_LB_summer",
    },
    "2037_RL_3": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_BB_winter",
    },
    "2037_RL_4": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_BB_summer",
    },
    "2037_RL_5": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_SB_winter",
    },
    "2037_RL_6": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_SB_summer",
    },
}

# %%
if __name__ == "__main__":
    examples = [
        "2020_RL_1",
    ]

    data_format = "timescale"  # "local_db" or "timescale"

    for example in examples:
        # delete examples/inputs/2020_RL/forecasts_df.csv
        if os.path.exists(
            f"examples/inputs/{availabe_examples[example]['scenario']}/forecasts_df.csv"
        ):
            os.remove(
                f"examples/inputs/{availabe_examples[example]['scenario']}/forecasts_df.csv"
            )

        if data_format == "local_db":
            db_uri = f"sqlite:///./examples/local_db/assume_db_{example}.db"
        elif data_format == "timescale":
            db_uri = "postgresql://assume:assume@localhost:5432/assume"

            world = World(database_uri=db_uri, export_csv_path=csv_path)
            load_scenario_folder(
                world,
                inputs_path="examples/inputs",
                scenario=availabe_examples[example]["scenario"],
                study_case=availabe_examples[example]["study_case"],
            )
            if world.learning_config.get("learning_mode", False):
                # run learning if learning mode is enabled
                run_learning(
                    world,
                    inputs_path="examples/inputs",
                    scenario=availabe_examples[example]["scenario"],
                    study_case=availabe_examples[example]["study_case"],
                )

            world.run()