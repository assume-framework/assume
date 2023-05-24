# %%
import logging
import os
from os import getenv

from assume import World

log = logging.getLogger(__name__)

os.makedirs("./examples/outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "./examples/outputs"))

DATABASE_URI = getenv(
    "DATABASE_URI", "postgresql://assume:assume@localhost:5432/assume"
)
# %%
if __name__ == "__main__":
    world = World(database_uri=DATABASE_URI, export_csv_path=EXPORT_CSV_PATH)
    world.load_scenario(
        inputs_path="examples/inputs", scenario="example_01a", study_case="example_01a"
    )
    world.run()
