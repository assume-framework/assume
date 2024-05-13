# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# %%
import logging
import os
import shutil

from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning
from assume.common.utils import rename_study_case

log = logging.getLogger(__name__)

csv_path = ""

os.makedirs("./examples/local_db", exist_ok=True)

available_examples = {
    "small": {"scenario": "example_01a", "study_case": "base"},
    "small_dam": {"scenario": "example_01a", "study_case": "dam"},
    "small_with_opt_clearing": {
        "scenario": "example_01a",
        "study_case": "dam_with_complex_clearing",
    },
    "small_with_BB_and_LB": {
        "scenario": "example_01c",
        "study_case": "dam_with_complex_opt_clearing",
    },
    "small_with_vre": {"scenario": "example_01b", "study_case": "base"},
    "small_with_vre_and_storage": {
        "scenario": "example_01c",
        "study_case": "eom_only",
    },
    "small_with_vre_and_storage_and_complex_clearing": {
        "scenario": "example_01c",
        "study_case": "dam_with_complex_opt_clearing",
    },
    "small_with_crm": {
        "scenario": "example_01c",
        "study_case": "eom_and_crm",
    },
    "small_with_redispatch": {
        "scenario": "example_01d",
        "study_case": "base",
    },
    "small_with_nodal_clearing": {
        "scenario": "example_01d",
        "study_case": "nodal_case",
    },
    "large_2019_eom": {"scenario": "example_02", "study_case": "base_case_2019"},
    "large_2019_eom_crm": {
        "scenario": "example_02",
        "study_case": "eom_crm_case_2019",
    },
    "large_2019_day_ahead": {
        "scenario": "example_02",
        "study_case": "dam_case_2019",
    },
    "small_learning_1": {
        "scenario": "example_02a", 
        "study_case": "base"
    },
    "small_learning_1_lstm": {
        "scenario": "example_02a",
        "study_case": "base_lstm"
    },
    "small_learning_2": {
        "scenario": "example_02b", 
        "study_case": "base"
    }    ,
    "small_learning_3": {
        "scenario": "example_02c", 
        "study_case": "dam"
        },

    #Cases resembling Harder et al. 2023
    "harder_case1": {
        "scenario": "example_02a", 
        "study_case": "harder"
    },
    "harder_case1_lstm": {
        "scenario": "example_02a",
        "study_case": "harder_lstm"
    },
    "harder_case2": {
        "scenario": "example_02b", 
        "study_case": "harder"
    },
    "harder_case2_lstm": {
        "scenario": "example_02b", 
        "study_case": "harder_lstm"
    },
}


# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """
    data_format = "timescale"  # "local_db" or "timescale"
    examples = ["harder_case2_lstm", "harder_case2"]
    inputs_path = "examples/inputs"
    no_runs = 1 # later: no_runs = 10 for assessing robustness of model training

    for example in examples:
        # temporarily store original config file to restore it after runs
        scenario=available_examples[example]["scenario"]
        study_case=available_examples[example]["study_case"]

        config_path=f"{inputs_path}/{scenario}/config.yaml"
        tmp_config_path=f"{inputs_path}/{scenario}/tmp_config.yaml"

        shutil.copyfile(config_path, tmp_config_path)

        # simulate the same example [no_runs] times
        for run in range(1, no_runs + 1):
            set.seed(run)

            if data_format == "local_db":
                db_uri = f"sqlite:///./examples/local_db/assume_db_{example}.db"
            elif data_format == "timescale":
                db_uri = "postgresql://assume:assume@localhost:5432/assume"

            # edit config["study_case"] to include run number
            study_case_run=f"{study_case}_run_{run}"
            rename_study_case(config_path, study_case, study_case_run)

            # create world
            world = World(database_uri=db_uri, export_csv_path=csv_path)

            # load scenario
            load_scenario_folder(
                world,
                inputs_path=inputs_path,
                scenario=scenario,
                study_case=study_case_run,
            )

            if world.learning_config.get("learning_mode", False):
                # run learning if learning mode is enabled
                run_learning(
                    world,
                    inputs_path=inputs_path,
                    scenario=scenario,
                    study_case=study_case_run,
                )

            world.run()

            # Restore original config file, not only change back study_case name
            shutil.copyfile(tmp_config_path, config_path) 

    os.remove(tmp_config_path)     