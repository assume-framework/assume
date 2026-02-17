# %%
import logging
import os

from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning

log = logging.getLogger(__name__)

# define the path where the simulation results in form of CSV files will be stored, for example: "examples/outputs"
# "" means no CSV files will be stored
csv_path = ""

os.makedirs("./examples/local_db", exist_ok=True)

scenario = {
    "simple": {"scenario": "simple", "study_case": "base_case"},
    "simple_complex": {"scenario":"simple", "study_case":"complex_clearing"},
    "simple_with_storage": {"scenario":"simple_with_storage", "study_case":"base_case"},
    "simple_with_storage_complex": {"scenario":"simple_with_storage", "study_case":"complex_clearing"},
    "simple_with_storage_24h": {"scenario":"simple_with_storage", "study_case":"cc_24h_foresight"},
    "simple_with_storage_48h": {"scenario":"simple_with_storage", "study_case":"cc_48h_foresight"},
    "simple_with_storage_336h": {"scenario":"simple_with_storage", "study_case":"cc_336h_foresight"},
    "simple_with_storage_8760h": {"scenario":"simple_with_storage", "study_case":"cc_8760h_foresight"},
    "simple_with_storage_price" : {"scenario":"simple_with_storage_forecast", "study_case":"price"},
    "simple_with_storage_cc_price" : {"scenario":"simple_with_storage_forecast", "study_case":"cc_price"},
    "simple_with_storage_cc_price_24h" : {"scenario":"simple_with_storage_forecast", "study_case":"cc_price_24h_foresight"},
    "simple_with_storage_cc_price_48h" : {"scenario":"simple_with_storage_forecast", "study_case":"cc_price_48h_foresight"},
    "simple_with_storage_cc_price_336h" : {"scenario":"simple_with_storage_forecast", "study_case":"cc_price_336h_foresight"},
    "simple_with_storage_cc_price_8760h" : {"scenario":"simple_with_storage_forecast", "study_case":"cc_price_8760h_foresight"},
    
    "simple_with_storage_opt_price" : {"scenario":"simple_with_storage_opt", "study_case":"price"},
    "simple_with_storage_opt_cc_price" : {"scenario":"simple_with_storage_opt", "study_case":"cc_price"},
    "simple_with_storage_opt_cc_price_24h" : {"scenario":"simple_with_storage_opt", "study_case":"cc_price_24h_foresight"},
    "simple_with_storage_opt_cc_price_48h" : {"scenario":"simple_with_storage_opt", "study_case":"cc_price_48h_foresight"},
    "simple_with_storage_opt_cc_price_336h" : {"scenario":"simple_with_storage_opt", "study_case":"cc_price_336h_foresight"},
    "simple_with_storage_opt_cc_price_8760h" : {"scenario":"simple_with_storage_opt", "study_case":"cc_price_8760h_foresight"},

    "stylized": {"scenario":"stylized", "study_case":"base_case"},
    "stylized_complex": {"scenario":"stylized", "study_case":"complex_clearing"},

    "stylized_forecast_price": {"scenario":"stylized_forecast", "study_case":"price"},
    "stylized_forecast_cc_price": {"scenario":"stylized_forecast", "study_case":"cc_price"},
    "stylized_forecast_cc_price_24h": {"scenario":"stylized_forecast", "study_case":"cc_price_24h_foresight"},
    
    "stylized_opt_price": {"scenario":"stylized_opt", "study_case":"price"},
    "stylized_opt_cc_price": {"scenario":"stylized_opt", "study_case":"cc_price"},
    "stylized_opt_cc_price_24h": {"scenario":"stylized_opt", "study_case":"cc_price_24h_foresight"},
}


# %%
if __name__ == "__main__":
    """
    Available examples:
    - local_db: without database and grafana
    - timescale: with database and grafana (note: you need docker installed)
    """

    data_format = "timescale"  # "local_db" or "timescale"

    example = "stylized_opt_cc_price_24h"  # "simple", "simple_complex", "simple_with_storage", "simple_with_storage_complex", "stylized"

    if data_format == "local_db":
        db_uri = "sqlite:///./examples/local_db/assume_db.db"
    elif data_format == "timescale":
        db_uri = "postgresql://assume:assume@localhost:5432/assume"

    # create world
    world = World(database_uri=db_uri, export_csv_path=csv_path)

    # load scenario
    current_dir = os.path.dirname(os.path.abspath(__file__))  # folder containing this file
    inputs_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "assume-examples", "inputs", "inconsistencies")  # one level up + "directory"
    print(f"Loading inputs from: {inputs_path}")

    load_scenario_folder(
        world,
        inputs_path=inputs_path,
        scenario=scenario[example]["scenario"],
        study_case=scenario[example]["study_case"],
    )

    if world.learning_mode:
        # run learning if learning mode is enabled
        run_learning(world)

    world.run()
