# %%
import logging
import os

from assume import World, load_scenario_folder, run_learning

log = logging.getLogger(__name__)

csv_path = ""

os.makedirs("./workshop/local_db", exist_ok=True)

# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """
    data_format = "timescale"  # "local_db" or "timescale"

    if data_format == "local_db":
        db_uri = "sqlite:///./workshop/local_db/assume_db.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    input_path = "workshop/inputs"
    scenario = "learning_scenario_1"
    study_case = "base"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # you can also add custom bidding strategies as follows:
    from learning_strategy import RLStrategy

    world.bidding_strategies["pp_learning"] = RLStrategy

    # load scenario
    load_scenario_folder(
        world,
        inputs_path=input_path,
        scenario=scenario,
        study_case=study_case,
    )

    if world.learning_config.get("learning_mode", False):
        # run learning if learning mode is enabled
        run_learning(
            world,
            inputs_path=input_path,
            scenario=scenario,
            study_case=study_case,
        )

    world.run()
