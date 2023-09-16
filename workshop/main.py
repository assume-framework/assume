# %%
import logging
import os

from assume import World, load_scenario_folder, run_learning

log = logging.getLogger(__name__)

csv_path = "./workshop/outputs"
os.makedirs(csv_path, exist_ok=True)

os.makedirs("./workshop/local_db", exist_ok=True)

import pandas as pd

demand_paper = pd.read_csv("./inputs/learning_scenario_1/IED_DE.csv", index_col=0)
actual_demand = pd.read_csv(
    "./inputs/learning_scenario_1/demand_df.csv", index_col=0, parse_dates=True
)
actual_demand["demand_EOM"] = demand_paper["demand"].values / 10
actual_demand.to_csv("./inputs/learning_scenario_1/demand_df.csv")

# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """
    data_format = "local_db"  # "local_db" or "timescale"

    if data_format == "local_db":
        db_uri = "sqlite:///./workshop/local_db/assume_db.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    input_path = "workshop/inputs"
    scenario = "learning_scanerio_01"
    study_case = "base"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # load scenario
    load_scenario_folder(
        world,
        inputs_path=input_path,
        scenario=scenario,
        study_case=study_case,
    )

    # you can also add custom bidding strategies as follows:
    from bidding_strategy import RLStrategy

    world.bidding_strategies["pp_learning"] = RLStrategy

    if world.learning_config.get("learning_mode", False):
        # run learning if learning mode is enabled
        run_learning(
            world,
            inputs_path="examples/inputs",
            scenario=scenario,
            study_case=study_case,
        )

    world.run()
