# %%
import logging
import os

from assume import World, load_scenario_folder, run_learning

log = logging.getLogger(__name__)

csv_path = "./examples/outputs"
os.makedirs(csv_path, exist_ok=True)

os.makedirs("./examples/local_db", exist_ok=True)

availabe_examples = {
    "2020_rule_SB": {
        # merit order clearing for 24 hours
        "scenario": "2020_SB",
        "study_case": "dam",
    },
    "2020_rule_BB": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2020_BB",
        "study_case": "dam",
    },
    "2020_RL_SB": {
        # merit order clearing with RL strategies
        "scenario": "2020_RL",
        "study_case": "dam_SB",
    },
    "2020_RL_BB": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL",
        "study_case": "dam_BB",
    },
    "2020_RL_tiny": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2020_RL",
        "study_case": "tiny",
    },
    "2020_hRL_SB": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_hRL",
        "study_case": "dam_SB",
    },
    "2020_hRL_BB": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_hRL",
        "study_case": "dam_BB",
    },
    "2020_hRL_tiny": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_hRL",
        "study_case": "tiny",
    },
    "2037_RL_SB": {
        # merit order clearing with RL strategies
        "scenario": "2037_RL",
        "study_case": "dam_SB",
    },
    "2037_RL_BB": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL",
        "study_case": "dam_BB",
    },
    "2037_rule_SB": {
        # merit order clearing for 24 hours
        "scenario": "2037_SB",
        "study_case": "dam",
    },
    "2037_rule_BB": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2037_BB",
        "study_case": "dam",
    },
    "2037_hRL_tiny": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_hRL",
        "study_case": "tiny",
    },
    "2037_hRL_SB": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_hRL",
        "study_case": "dam_SB",
    },
    "2037_hRL_BB": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_hRL",
        "study_case": "dam_BB",
    },
}

# %%
if __name__ == "__main__":
    example = "2037_RL_BB"
    data_format = "timescale"  # "local_db" or "timescale"

    if data_format == "local_db":
        db_uri = f"sqlite:///./examples/local_db/assume_db_{example}.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

        world = World(database_uri=db_uri, export_csv_path=csv_path)
        load_scenario_folder(
            world,
            inputs_path="examples/inputs",
            scenario=availabe_examples[example]["scenario"],
            study_case=availabe_examples[example]["study_case"],
        )
        if world.learning_config.get("learning_mode", False):
            # run learning if learning mode is enabled
            run_learning(
                world,
                inputs_path="examples/inputs",
                scenario=availabe_examples[example]["scenario"],
                study_case=availabe_examples[example]["study_case"],
            )

        world.run()
