# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# %%
import logging
import os

from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning

log = logging.getLogger(__name__)

# define the path where the simulation results in form of CSV files will be stored, for example: "examples/outputs"
# "" means no CSV files will be stored
csv_path = ""

os.makedirs("./examples/local_db", exist_ok=True)

available_examples = {
    # small examples for easier understanding of different features and configurations
    "small": {"scenario": "example_01a", "study_case": "base"},
    "small_with_exchanges": {
        "scenario": "example_01a",
        "study_case": "base_with_exchanges",
    },
    "small_dam": {"scenario": "example_01a", "study_case": "dam"},
    "small_with_opt_clearing": {
        "scenario": "example_01a",
        "study_case": "dam_with_complex_clearing",
    },
    "small_with_vre": {"scenario": "example_01b", "study_case": "base"},
    "small_with_vre_and_storage": {
        "scenario": "example_01c",
        "study_case": "eom_only",
    },
    "small_with_BB_and_LB": {
        "scenario": "example_01c",
        "study_case": "dam_with_complex_opt_clearing",
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
    "small_with_zonal_clearing": {
        "scenario": "example_01d",
        "study_case": "zonal_case",
    },
    # example_01e is used in the tutorial notebook #3: Custom unit and bidding strategy example
    "market_study_eom": {
        "scenario": "example_01f",
        "study_case": "eom_case",
    },
    "market_study_eom_and_ltm": {
        "scenario": "example_01f",
        "study_case": "ltm_case",
    },
    # example_01f is used in the tutorial notebook #5: Market configuration comparison example
    # example_01g is used in the tutorial notebook #6: Advanced order types example
    #
    # DRL references case for learning advancement testing
    "small_learning_1": {"scenario": "example_02a", "study_case": "base"},
    "small_learning_2": {"scenario": "example_02b", "study_case": "base"},
    "small_learning_3": {"scenario": "example_02c", "study_case": "base"},
    # DRL cases with lstm instead of mlp as actor neural network architecture
    "small_learning_1_lstm": {"scenario": "example_02a", "study_case": "base_lstm"},
    "small_learning_2_lstm": {"scenario": "example_02b", "study_case": "base_lstm"},
    # Further DRL example simulation showcasing learning features
    "learning_with_complex_bids": {"scenario": "example_02d", "study_case": "dam"},
    "small_learning_with_storage": {"scenario": "example_02e", "study_case": "base"},
    #
    # full year examples to show real-world scenarios
    "large_2019_eom": {"scenario": "example_03", "study_case": "base_case_2019"},
    "large_2019_eom_crm": {
        "scenario": "example_03",
        "study_case": "eom_crm_case_2019",
    },
    "large_2019_day_ahead": {
        "scenario": "example_03",
        "study_case": "dam_case_2019",
    },
    "large_2019_with_DSM": {
        "scenario": "example_03",
        "study_case": "base_case_2019_with_DSM",
    },
    "large_2019_rl": {"scenario": "example_03a", "study_case": "base_case_2019"},
    "large_2021_rl": {"scenario": "example_03b", "study_case": "base_case_2021"},
}


# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """

    # select to store the simulation results in a local database or in timescale
    # when using timescale, you need to have docker installed and can access the grafana dashboard
    data_format = "local_db"  # "local_db" or "timescale"

    # select the example to run from the available examples above
    example = "small"

    if data_format == "local_db":
        db_uri = "sqlite:///./examples/local_db/assume_db.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # load scenario
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario=available_examples[example]["scenario"],
        study_case=available_examples[example]["study_case"],
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
            scenario=available_examples[example]["scenario"],
            study_case=available_examples[example]["study_case"],
        )

    world.run()
