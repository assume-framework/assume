# %%
import logging
import os
from os import getenv

from assume import World, load_scenario_folder

log = logging.getLogger(__name__)

os.makedirs("./examples/outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "./examples/outputs"))

os.makedirs("./examples/local_db", exist_ok=True)
DATABASE_URI = getenv("DATABASE_URI", "sqlite:///./examples/local_db/assume_db_01a.db")
# %%
if __name__ == "__main__":
    scenario = "validation_01a_02a"
    study_case = "eom_only"

    world = World(database_uri=DATABASE_URI, export_csv_path=EXPORT_CSV_PATH)
    load_scenario_folder(
        world, inputs_path="validation_runs/inputs", scenario=scenario, study_case=study_case
    )
    world.run()    
