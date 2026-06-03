# import the main World class and the load_scenario_folder functions from assume
import os
from assume import World
from assume.scenario.loader_csv import load_scenario_folder
from assume.common.forecaster import BuildingForecaster
from assume.units.building import Building

# Define paths for input and output data
csv_path = "outputs"

# Define the data format and database URI
# Use "local_db" for SQLite database or "timescale" for TimescaleDB in Docker

# Create directories if they don't exist
os.makedirs(csv_path, exist_ok=True)
os.makedirs("local_db", exist_ok=True)

data_format = "timescale"  # "local_db" or "timescale"

if data_format == "local_db":
    db_uri = "sqlite:///local_db/assume_db.db"
elif data_format == "timescale":
    db_uri = "postgresql://assume:assume@localhost:5432/assume"

# Create the World instance
world = World(database_uri=db_uri, export_csv_path=csv_path)

# Load the scenario by providing the world instance
# The path to the inputs folder and the scenario name (subfolder in inputs)
# and the study case name (which config to use for the simulation)
load_scenario_folder(
    world,
    inputs_path="inputs",
    scenario="dist_network",
    study_case="base",
)
evs = {0: {'cap': 90, 'p_max': 22}, 1: {'cap': 80, 'p_max': 11}}
requests = {0: {'s_soc': 0.53, 'e_soc': 1.00, 's_t': 0, 'e_t': 10},
            1: {'s_soc': 0.20, 'e_soc': 0.80, 's_t': 0, 'e_t': 6}}

# - start soc @ index start time, + end soc @ index end time, 2.0 during plugged otherwise, 0.0 during unplugged
availabilites = {
    0: [-0.53, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 0.0],
    1: [-0.20, 2.0, 2.0, 2.0, 2.0, 2.0, 0.80, 0.0, 0.0, 0.0]
}
price = {"EOM": [100, 90, 80, 80, 80, 90, 100, 110, 70, 40]}
grid_tariff = [0]*10
grid_tariff[5] = 200
grid_tariff[6] = 200
grid_tariff2 = [10, 10, 10, 30, 40, 10, -20, -30, 20, 400]


pp_forecaster = world.units["Unit 1"].forecaster

forecaster = BuildingForecaster(
    index=pp_forecaster.index,
    fuel_prices=pp_forecaster.fuel_prices,
    market_prices=price,
    electricity_price=price["EOM"],
    electricity_price_flex=price["EOM"],
    # No trip_distance, but providing trip_energy_consumption as alternative
    aggregator_electric_vehicle_1_availability_profile=pp_forecaster._to_series([1, 1, 1, 1, 1, 1, 1, 1, 0, 0]),
    aggregator_electric_vehicle_2_availability_profile=pp_forecaster._to_series([1, 1, 1, 1, 1, 1, 0, 0, 0, 0]),
    aggregator_electric_vehicle_1_trip_energy_consumption=pp_forecaster._to_series([0, 0, 0, 0, 0, 0, 0, 0, 90, 0]),
    aggregator_electric_vehicle_2_trip_energy_consumption=pp_forecaster._to_series([0, 0, 0, 0, 0, 0, 64, 0, 0, 0]),
)

components = {
    "electric_vehicle_1": {
        "capacity":evs[0]["cap"],
        "min_soc": 0.0,
        "max_soc": 1.0,
        "max_power_charge": evs[0]["p_max"],
        "max_power_discharge":  evs[0]["p_max"],
        "initial_soc": requests[0]["s_soc"],
        "efficiency_charge": 1.0,
        "efficiency_discharge": 1.0,
        "ramp_up": evs[0]["p_max"],
        "ramp_down": evs[0]["p_max"],
        "mileage": 1.0,
        "power_flow_directionality": "bidirectional",
    },
    "electric_vehicle_2": {
        "capacity":evs[1]["cap"],
        "min_soc": 0.0,
        "max_soc": 1.0,
        "max_power_charge": evs[1]["p_max"],
        "max_power_discharge":  evs[1]["p_max"],
        "initial_soc": requests[1]["s_soc"],
        "efficiency_charge": 1.0,
        "efficiency_discharge": 1.0,
        "ramp_up": evs[1]["p_max"],
        "ramp_down": evs[1]["p_max"],
        "mileage": 1.0,
        "power_flow_directionality": "bidirectional",
    }
}


#world.bidding_strategies["arbitrage_ev"] = ArbitrageWithTarget
#world.unit_types["ev"] = ElectricVehicle

world.add_unit_operator("Operator South")


world.add_unit(
    id = "aggregator",
    unit_operator_id="Operator South",
    unit_type="building",
    unit_params={
        "bidding_strategies": {"EOM":"household_energy_optimization"},
        "components":components,
        "objective":"min_variable_cost",
        "flexibility_measure":"electricity_price_signal",
        "is_prosumer":"Yes",
        "node": "south",
        },
    forecaster=forecaster,
)

world.units["aggregator"].setup_model()


# Run the simulation
world.run()

import pyomo.environ as pyo

unit = world.units["aggregator"]

# Build and solve a fresh flex instance (the one solved during world.run() is discarded)
instance = unit.model.create_instance()
instance = unit.switch_to_flex(instance)
unit.solver.solve(instance)

for ev in ["electric_vehicle_1", "electric_vehicle_2"]:
    blk = instance.dsm_blocks[ev]
    print(f"\n=== {ev} ===")
    print(f"{'t':>3} {'charge':>8} {'discharge':>10} {'soc':>8}")
    for t in instance.time_steps:
        charge = pyo.value(blk.charge[t])
        discharge = pyo.value(blk.discharge[t])
        soc = pyo.value(blk.soc[t])
        print(f"{t:>3} {charge:>8.2f} {discharge:>10.2f} {soc:>8.3f}")
