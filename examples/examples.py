# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# %%
import logging
import os

from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning

log = logging.getLogger(__name__)

csv_path = "/nimble/home/par19744/assume/examples"

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
    "small_with_zonal_clearing": {
        "scenario": "example_01d",
        "study_case": "zonal_case",
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
    "small_learning_1": {"scenario": "example_02a", "study_case": "base"},
    "small_learning_2": {"scenario": "example_02b", "study_case": "base"},
    "small_learning_3": {"scenario": "example_02c", "study_case": "dam"},
    "learning_with_complex_bids": {
        "scenario": "example_02e",
        "study_case": "base",
    },
    "redispatch_validation_2019": {"scenario": "example_04a", "study_case": "base"},
    "redispatch_validation_2023": {"scenario": "example_04b", "study_case": "base"},
    
}

# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """
    data_format = "local_db"  # "local_db" or "timescale"
    example = "redispatch_validation_2023"

    if data_format == "local_db":
        db_uri = f"sqlite:///./examples/local_db/assume_db_{example}.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario=availabe_examples[example]["scenario"],
        study_case=availabe_examples[example]["study_case"],
    )

    # to add custom bidding strategies, you need to import them
    # and add them to the world as follows:
    # from custom_bidding_strategy import CustomBiddingStrategy
    # world.bidding_strategies["custom_bidding_strategy"] = CustomBiddingStrategy

    # to add a custom unit type, you need to import it
    # and add it to the world as follows:
    # from custom_unit import CustomUnit
    # world.unit_types["custom_unit"] = CustomUnit

    # next you need to load and add the custom units to the scenario
    # from assume import load_custom_units
    # load_custom_units(
    #     world,
    #     inputs_path="examples/inputs",
    #     scenario=availabe_examples[example]["scenario"],
    #     file_name="custom_units",
    #     unit_type="custom_unit",
    # )

    if world.learning_config.get("learning_mode", False):
        # run learning if learning mode is enabled
        run_learning(
            world,
            inputs_path="examples/inputs",
            scenario=availabe_examples[example]["scenario"],
            study_case=availabe_examples[example]["study_case"],
        )

    world.run()
