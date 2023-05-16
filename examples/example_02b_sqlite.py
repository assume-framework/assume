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

# DATABASE_URI = getenv(
#     "DATABASE_URI", "postgresql://assume:assume@localhost:5432/assume"
# )
#%%
if __name__ == "__main__":
    scenario = "example_02b"
    study_case = "base_case"
    world = World(database_uri=DATABASE_URI, export_csv=EXPORT_CSV_PATH)
    #%%
    world.load_scenario(
        inputs_path="inputs",
        scenario=scenario,
        study_case=study_case,
    )
    #%%
    world.run()

# %% test
import yaml

from assume.common import (
    MarketConfig,
    UnitsOperator,
    load_file,
    make_market_config,
    mango_codec_factory,
)

inputs_path="inputs"
scenario = "example_02b"

path = f"{inputs_path}/{scenario}"
with open(f"{path}/config.yml", "r") as f:
    config = yaml.safe_load(f)
    config = config[study_case]


storage_units_df = load_file(
             path=path, config=config, file_name="storage_units"
        )
# %%
for storage_name, unit_params in storage_units_df.iterrows():
    print(storage_name)
# %%
