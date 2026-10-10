"""Train example_02b MATD3 base and base_lstm for seeds 11 and 22."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

SCENARIO = "example_02b"
CASES = ("base", "base_lstm")
SEEDS = (11, 22)
OUT_DIR = Path("/tmp/matd3_02b_compare") / REPO.name

_yaml_load = yaml.safe_load


def inject_eval(study_case: str, seed: int, load_path: Path) -> None:
    def seeded(stream):
        config = _yaml_load(stream)
        case = config[study_case]
        case["seed"] = seed
        learning = case["learning_config"]
        learning["learning_mode"] = False
        learning["continue_learning"] = False
        learning["trained_policies_load_path"] = str(load_path)
        return config

    yaml.safe_load = seeded


def inject(study_case: str, seed: int):
    def seeded(stream):
        config = _yaml_load(stream)
        config[study_case]["seed"] = seed
        return config

    yaml.safe_load = seeded


def run_case(study_case: str, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{study_case}_seed{seed}.json"
    if out_path.exists():
        print(f"skip existing {out_path}", flush=True)
        return

    save_dir = (
        REPO
        / "examples/inputs"
        / SCENARIO
        / "learned_strategies"
        / f"{SCENARIO}_{study_case}"
    )
    if save_dir.exists():
        shutil.rmtree(save_dir)

    inject(study_case, seed)
    from assume import World, load_scenario_folder, run_learning

    run_dir = Path(mkdtemp(prefix=f"{REPO.name}_{study_case}_seed{seed}_"))
    os.chdir(run_dir)
    database = run_dir / "run.db"
    world = World(
        database_uri=f"sqlite:///{database}",
        export_csv_path="",
        log_level="WARNING",
    )
    load_scenario_folder(
        world,
        inputs_path=str(REPO / "examples/inputs"),
        scenario=SCENARIO,
        study_case=study_case,
    )
    print(f"start {REPO.name} {study_case} seed {seed}", flush=True)
    run_learning(world)
    reward_frame = pd.read_sql(
        """
        SELECT episode, AVG(total_reward) AS avg_reward
        FROM (
            SELECT episode, unit, SUM(reward) AS total_reward
            FROM rl_params
            WHERE evaluation_mode = 1
            GROUP BY episode, unit
        )
        GROUP BY episode
        ORDER BY episode
        """,
        sqlite3.connect(database),
    )
    rewards = reward_frame["avg_reward"].astype(float).tolist()
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table'", sqlite3.connect(database)
    )
    if "market_meta" not in set(tables["name"]):
        load_path = save_dir / "last_policies"
        inject_eval(study_case, seed, load_path)
        from assume import World as EvalWorld
        from assume.scenario.loader_csv import load_scenario_folder as load_folder

        eval_dir = Path(mkdtemp(prefix=f"eval_{REPO.name}_{study_case}_seed{seed}_"))
        os.chdir(eval_dir)
        database = eval_dir / "eval.db"
        eval_world = EvalWorld(
            database_uri=f"sqlite:///{database}",
            export_csv_path="",
            log_level="WARNING",
        )
        load_folder(
            eval_world,
            inputs_path=str(REPO / "examples/inputs"),
            scenario=SCENARIO,
            study_case=study_case,
        )
        eval_world.run()
    prices = pd.read_sql(
        "SELECT time, price FROM market_meta", sqlite3.connect(database)
    )
    out_path.write_text(
        json.dumps(
            {
                "repo": REPO.name,
                "study_case": study_case,
                "seed": seed,
                "rewards": rewards,
                "time": prices["time"].astype(str).tolist(),
                "price": prices["price"].astype(float).tolist(),
            }
        )
    )
    print(
        f"saved {out_path} rewards={len(rewards)} prices={len(prices)}",
        flush=True,
    )


if __name__ == "__main__":
    for study_case in CASES:
        for seed in SEEDS:
            run_case(study_case, seed)
