# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# %%
import logging
import os

from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning

log = logging.getLogger(__name__)

csv_path = ""

os.makedirs("./examples/local_db", exist_ok=True)

availabe_examples = {
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
    "small_learning_2": {"scenario": "example_02b", "study_case": "base"},
    "small_learning_3": {"scenario": "example_02c", "study_case": "dam"},
}


# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """
    data_format = "timescale"  # "local_db" or "timescale"
    example = "small_learning_1"

    if data_format == "local_db":
        db_uri = f"sqlite:///./examples/local_db/assume_db_{example}.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # load scenario
    # TODO: loop over examples
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
