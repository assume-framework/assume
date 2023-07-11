# %%
import logging
import os
from os import getenv

from assume import World, load_scenario_folder

log = logging.getLogger(__name__)

os.makedirs("./examples/outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "./examples/outputs"))

os.makedirs("./examples/local_db", exist_ok=True)
DATABASE_URI = getenv(
    "DATABASE_URI", "postgresql://assume:assume@localhost:5432/assume"
)

if __name__ == "__main__":
    scenario = "example_02b"
    study_case = "crm_case_2019"
    world = World(database_uri=DATABASE_URI, export_csv_path=EXPORT_CSV_PATH)
    load_scenario_folder(
        world,
        inputs_path="examples/inputs",
        scenario=scenario,
        study_case=study_case,
    )
    world.run()
