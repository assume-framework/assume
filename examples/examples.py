# %%
import logging
import os

from assume import World, load_scenario_folder

log = logging.getLogger(__name__)

csv_path = "./examples/outputs"
os.makedirs(csv_path, exist_ok=True)

os.makedirs("./examples/local_db", exist_ok=True)

availabe_examples = {
    "small": {"scenario": "example_01a", "study_case": "base"},
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
    "small_with_crm": {
        "scenario": "example_01c",
        "study_case": "eom_and_crm",
    },
    "small_with_heatpumps": {
        "scenario": "example_01_heatpumps",
        "study_case": "with_heat_pumps",
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
    "rl": {"scenario": "example_01_rl", "study_case": "base"},
}

# %%
if __name__ == "__main__":
    example = "rl"
    data_format = "timescale"  # "local_db" or "timescale"

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
    world.run()
