# %%
import asyncio
import logging
from os import getenv
import os

from assume import World

log = logging.getLogger(__name__)
logging.getLogger('mango').setLevel(logging.WARNING)

os.makedirs("./examples/outputs", exist_ok=True)
EXPORT_CSV_PATH = str(getenv("EXPORT_CSV_PATH", "./examples/outputs"))

os.makedirs("./examples/local_db", exist_ok=True)
DATABASE_URI = getenv("DATABASE_URI", "sqlite:///./examples/local_db/assume_db.db")

# DATABASE_URI = getenv(
#     "DATABASE_URI", "postgresql://assume:assume@localhost:5432/assume"
# )

async def main():
    world = World(database_uri=DATABASE_URI, export_csv=EXPORT_CSV_PATH)
    await world.load_scenario(inputs_path="examples/inputs", scenario="example_03", study_case="base_case")
    await world.run_simulation()

if __name__ == '__main__':
    asyncio.run(main())

# %%