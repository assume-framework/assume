# %%
import logging
import os

from assume import World, load_scenario_folder, run_learning

log = logging.getLogger(__name__)

csv_path = "./examples/outputs"
os.makedirs(csv_path, exist_ok=True)

os.makedirs("./examples/local_db", exist_ok=True)

availabe_examples = {
    "base_2019": {
        # merit order clearing for 24 hours
        "scenario": "2019_SB",
        "study_case": "dam_simple",
    },
    "complex_SB_2019": {
        # merit order clearing for 24 hours
        "scenario": "2019_SB",
        "study_case": "dam",
    },
    "complex_BB_2019": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2019_BB",
        "study_case": "dam",
    },
    "complex_SB_2020": {
        # merit order clearing for 24 hours
        "scenario": "2020_SB",
        "study_case": "dam",
    },
    "complex_BB_2020": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2020_BB",
        "study_case": "dam",
    },
    "simple_RL_2020": {
        # merit order clearing with RL strategies
        "scenario": "2020_RL_SB",
        "study_case": "dam",
    },
    "complex_RL_2020": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_RL_BB",
        "study_case": "dam",
    },
    "complex_all_RL_2020": {
        # complex clearing with RL strategies and BB
        "scenario": "2020_all_RL_BB",
        "study_case": "dam",
    },
    "complex_SB_2037": {
        # merit order clearing for 24 hours
        "scenario": "2037_SB",
        "study_case": "dam",
    },
    "complex_BB_2037": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2037_BB",
        "study_case": "dam",
    },
    "complex_RL_2037": {
        # complex clearing with RL strategies and BB
        "scenario": "2037_RL_BB",
        "study_case": "dam",
    },
    "tiny_RL_2020": {
        # complex clearing with naive strategies - no blocks
        "scenario": "2020_RL_BB",
        "study_case": "tiny",
    },
    "complex_clearing_blocks": {
        # complex clearing with dam naive strategies - with profile blocks in PPs
        "scenario": "analysis_02",
        "study_case": "dam_complex_clearing",
    },
    "minimal_base": {
        "scenario": "minimal_SB",
        "study_case": "dam_case_2019",
    },
    "minimal_complex": {
        "scenario": "minimal_BB",
        "study_case": "dam_complex_clearing",
    },
}

# %%
if __name__ == "__main__":
    example = "complex_RL_2037"
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
