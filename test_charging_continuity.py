import logging
import os

from assume import World
from assume.scenario.loader_csv import load_scenario_folder

# Set up logging
logging.basicConfig(level=logging.INFO)

# Create output directory
os.makedirs("./test_outputs", exist_ok=True)

# Create world with CSV output
world = World(database_uri=None, export_csv_path="test_outputs")

# Load the bus depot scenario
load_scenario_folder(
    world,
    inputs_path="examples/inputs",
    scenario="example_bus",
    study_case="eom",
)

# Run simulation for just a few timesteps to test the constraint
print("Testing charging continuity constraint...")
print("=" * 50)

# Run the simulation
world.run()

print("\nSimulation completed!")
print("Check the test_outputs directory for results.") 