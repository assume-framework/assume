# SPDX-License-Identifier: AGPL-3.0-or-later
"""Short validation run of steel_base/fokusH2_2040 (base_case_2040_smoketest,
2 weeks) with ALL fixes staged: lignite efficiency, CO2/waste, storage
cross-market + aFRR energy pricing + staggered timing, and the new
per-interconnector import prices.

Usage:
  python examples/run_validation_smoketest.py new   -> exports to outputs/_validation/new
  python examples/run_validation_smoketest.py noimp -> import prices OFF (control)
"""
import os, sys, shutil, time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)
MODE = sys.argv[1] if len(sys.argv) > 1 else "new"
INPUTS = REPO / "examples/inputs/steel_base"
SCEN = "fokusH2_2040"
STUDY = "base_case_2040_smoketest"
OUT = REPO / "examples/outputs/_validation" / MODE
PRICE = INPUTS / SCEN / "exchange_prices_df.csv"
PRICE_OFF = INPUTS / SCEN / "exchange_prices_df.csv.OFF"

# control: temporarily hide the import-price file so the loader falls back to EUR0
toggled = False
if MODE == "noimp" and PRICE.exists():
    shutil.move(str(PRICE), str(PRICE_OFF)); toggled = True
try:
    OUT.mkdir(parents=True, exist_ok=True)
    from assume import World
    from assume.scenario.loader_csv import load_scenario_folder
    t0 = time.perf_counter()
    world = World(database_uri="", export_csv_path=str(OUT))
    load_scenario_folder(world, inputs_path=str(INPUTS), scenario=SCEN, study_case=STUDY)
    world.run()
    print(f"=== DONE {MODE}: {(time.perf_counter()-t0)/60:.1f} min -> {OUT}")
finally:
    if toggled and PRICE_OFF.exists():
        shutil.move(str(PRICE_OFF), str(PRICE))
