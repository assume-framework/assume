# %%
import logging
import os
from os import getenv

from assume import World

log = logging.getLogger(__name__)

os.makedirs("./examples/outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "./examples/outputs"))

os.makedirs("./examples/local_db", exist_ok=True)
DATABASE_URI = getenv("DATABASE_URI", "sqlite:///./examples/local_db/assume_db_01a.db")
# %%
if __name__ == "__main__":
    scenario = "example_01a"
    study_case = "base_case"

    world = World(database_uri=DATABASE_URI, export_csv=EXPORT_CSV_PATH)
    world.load_scenario(
        inputs_path="examples/inputs", scenario=scenario, study_case=study_case
    )
    world.run()
