# %%
import logging
import os
from os import getenv

from assume import World

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

os.makedirs("./outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "validation_runs/outputs"))

DATABASE_URI = getenv(
    "DATABASE_URI", "postgresql://assume:assume@localhost:5432/assume"
)
# %%
if __name__ == "__main__":
    scenario = "validation_01a_02a"
    study_case = "eom_only"

    world = World(database_uri=DATABASE_URI, export_csv_path=EXPORT_CSV_PATH)
    world.load_scenario(
        inputs_path="validation_runs/inputs", scenario=scenario, study_case=study_case
    )
    world.run()
