#!/usr/bin/env python
"""Run a single ASSUME variant. Called by run_array.sh once per array task.

Adapt the body of run_simulation() to match what your examples/examples.py does.
The default below is the standard ASSUME pattern:

    World(database_uri=...)
      .load_scenario_folder(inputs_path=..., scenario=..., study_case=...)
      .run()

If your examples.py does something different (custom output dirs, extra setup,
postprocessing), copy that logic into run_simulation() here.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from assume import World
from assume.scenario.loader_csv import load_scenario_folder, run_learning


def run_simulation(scenarios_dir: Path, scenario: str, study_case: str) -> None:


    # Each variant gets its own sqlite DB inside its own folder. No cross-talk.
    output_db = scenarios_dir / scenario / "assume_db.db"
    world = World(database_uri=f"sqlite:///{output_db}", export_csv_path="")

    load_scenario_folder(
        world,
        inputs_path=str(scenarios_dir),
        scenario=scenario,
        study_case=study_case,
    )

    if world.learning_mode:
        # run learning if learning mode is enabled
        run_learning(world)

    world.run()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", required=True,
                    help="Absolute path to this variant's scenario folder")
    ap.add_argument("--study-case", required=True)
    args = ap.parse_args()

    scenario_dir = Path(args.scenario_dir).resolve()
    scenarios_dir = scenario_dir.parent
    scenario = scenario_dir.name

    print(f"=== {scenario} (study_case={args.study_case}) ===", flush=True)
    print(f"    scenarios_dir={scenarios_dir}", flush=True)
    print(f"    SLURM_ARRAY_TASK_ID={os.environ.get('SLURM_ARRAY_TASK_ID', '-')}",
          flush=True)

    run_simulation(scenarios_dir, scenario, args.study_case)


if __name__ == "__main__":
    main()