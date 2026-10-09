"""Run matching EV and building scenarios and save a comparison plot.

Usage: python main.py <test_case_name> [--study-case <study_case>]
"""

import argparse
import time
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "scenario_data"
OUTPUT_DIR = BASE_DIR / "output_data"
PLOT_DIR = OUTPUT_DIR / "plots"


def available_study_cases():
    """Return cases that have both EV and building scenario input folders."""
    config_file = INPUT_DIR / "config.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    test_cases = list(set(["_".join(x.split("_")[:-1]) for x in config.keys()]))

    return test_cases


def run_scenario(study_case: str):
    from assume.scenario.loader_csv import load_config_and_create_forecaster, setup_world
    from assume.world import World

    world = World(export_csv_path=str(OUTPUT_DIR))
    world.scenario_data = load_config_and_create_forecaster(
        inputs_path=str(INPUT_DIR.parent),
        scenario=INPUT_DIR.name,
        study_case=study_case,
    )

    setup_world(world)
    start_time = time.perf_counter()
    world.run()
    run_time = time.perf_counter() - start_time
    print(f"{study_case} scenario run time: {run_time} seconds")
    return OUTPUT_DIR / world.scenario_data["simulation_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "study_case",
        choices=available_study_cases(),
        help="Name the scenario for which you want to compare EV and building.",
    )
    args = parser.parse_args()

    with open(INPUT_DIR / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    ev_output_dir = run_scenario(f"{args.study_case}_ev")
    print(f"EV output directory: {ev_output_dir}")
    # Some study cases (e.g. tariff comparisons) only define the _ev variant.
    if f"{args.study_case}_building" in config:
        building_output_dir = run_scenario(f"{args.study_case}_building")
        print(f"Building output directory: {building_output_dir}")


if __name__ == "__main__":
    main()
