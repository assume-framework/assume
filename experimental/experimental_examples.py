# %%
import logging
import os
from os import getenv

from assume import World, load_scenario_folder

log = logging.getLogger(__name__)

os.makedirs("./experimental/outputs", exist_ok=True)
csv_path = str(getenv("EXPORT_CSV_PATH", "./experimental/outputs"))

os.makedirs("./experimental/local_db", exist_ok=True)

availabe_examples = {
    "rl": {"scenario": "example_01_rl", "study_case": "rl_case"},
    "small": {"scenario": "example_01_clearing", "study_case": "example_01a"},
    "small_with_opt_clearing": {
        "scenario": "example_01_clearing",
        "study_case": "example_01a_complex_clearing",
    },
    "small_with_heatpumps": {
        "scenario": "example_01_heatpumps",
        "study_case": "with_heat_pumps",
    },
}

# %%
if __name__ == "__main__":
    example = "small_with_opt_clearing"
    data_format = "timescale"  # "local_db" or "timescale"

    if data_format == "local_db":
        db_url = getenv(
            "DATABASE_URI", f"sqlite:///./experimental/local_db/assume_db_{example}.db"
        )
    elif data_format == "timescale":
        db_url = getenv(
            "DATABASE_URI", "postgresql://assume:assume@localhost:5432/assume"
        )

    world = World(database_uri=db_url, export_csv_path=csv_path)
    load_scenario_folder(
        world,
        inputs_path="experimental/inputs",
        scenario=availabe_examples[example]["scenario"],
        study_case=availabe_examples[example]["study_case"],
    )
    world.run()
