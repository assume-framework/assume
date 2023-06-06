# %%
import logging
import os
from os import getenv

from assume import World

log = logging.getLogger(__name__)

os.makedirs("./examples/outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "./examples/outputs"))

os.makedirs("./examples/local_db", exist_ok=True)
DATABASE_URI = getenv("DATABASE_URI", "sqlite:///./examples/local_db/assume_db_02.db")

if __name__ == "__main__":
    scenario = "example_02"
    study_case = "crm_case_2019"
    world = World(database_uri=DATABASE_URI, export_csv_path=EXPORT_CSV_PATH)
    world.load_scenario(
        inputs_path="examples/inputs",
        scenario=scenario,
        study_case=study_case,
    )
    world.run()
