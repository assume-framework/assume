# %%
import logging
import os
from os import getenv

from assume import World, load_scenario_folder

log = logging.getLogger(__name__)

os.makedirs("./examples/outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "./examples/outputs"))

os.makedirs("./examples/local_db", exist_ok=True)

availabe_examples = {
    "small": {"scenario": "example_01a", "study_case": "example_01a"},
    "small_with_vre": {"scenario": "example_01b", "study_case": "example_01b"},
    "small_with_vre_and_storage": {
        "scenario": "example_01c",
        "study_case": "example_01c_eom",
    },
    "small_with_crm": {
        "scenario": "example_01c",
        "study_case": "example_01c_eom_crm",
    },
    "large_2019_eom": {"scenario": "example_02", "study_case": "base_case_2019"},
    "large_2019_eom_crm": {
        "scenario": "example_02",
        "study_case": "eom_and_crm_case_2019",
    },
}

# %%
if __name__ == "__main__":
    example = "small"
    data_format = "timescale"  # "local_db" or "timescale"

    if data_format == "local_db":
        DATABASE_URI = getenv(
            "DATABASE_URI", f"sqlite:///./examples/local_db/assume_db_{example}.db"
        )
    elif data_format == "timescale":
        DATABASE_URI = getenv(
            "DATABASE_URI", "postgresql://assume:assume@localhost:5432/assume"
        )

    world = World(database_uri=DATABASE_URI, export_csv_path=EXPORT_CSV_PATH)
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario=availabe_examples[example]["scenario"],
        study_case=availabe_examples[example]["study_case"],
    )
    world.run()
