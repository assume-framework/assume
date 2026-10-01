# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Full-year run of the 8 macro-case input folders (6 risk-neutral + WCMean/WCTail), each with
6 families x 4 years = 24 scenarios (192 jobs). The *_BID folders are byte-identical copies of
these inputs, so they are not simulated again.

Each finished scenario's 3 output tables (grid_line_loading, market_meta, market_orders) are
packed into <case>/<scenario>.tar.gz and the uncompressed copy is removed immediately.
Archives are written to a temp name, integrity-checked (gzip -t) and renamed atomically, and an
existing archive is only skipped if it passes gzip -t -- so an interrupted run can never leave
a truncated archive that a restart would trust.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import logging
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_PATH = "examples/inputs"
OUTPUTS_ROOT = REPO_ROOT / "examples/outputs"
LOG_DIR = OUTPUTS_ROOT / "logs_8cases"
SUMMARY = OUTPUTS_ROOT / "batch8_summary.csv"

# worst cases first, then the six risk-neutral cases
CASES = ["WCMean", "WCTail", "AktuellePolitiken_RN", "FokusH2_RN", "FokusStrom_RN",
         "HoheNachfrage_RN", "NachfrageNiedrig_RN", "Technologiemix_RN"]
FAMILIES = ["aktuellepolitiken", "fokusH2", "fokusstrom", "hohenachfrage",
            "niedrigenachfrage", "technologiemix"]
YEARS = [2030, 2035, 2040, 2045]


def gzip_ok(path: Path) -> bool:
    return subprocess.run(["gzip", "-t", str(path)], capture_output=True).returncode == 0


def compress_and_remove(scenario_dir: Path) -> tuple[bool, str]:
    if not scenario_dir.exists():
        return False, "output dir missing, nothing to compress"
    archive = scenario_dir.with_suffix(".tar.gz")
    tmp = archive.with_name(archive.name + ".tmp")
    try:
        subprocess.run(["tar", "-czf", str(tmp), "-C", str(scenario_dir.parent), scenario_dir.name],
                       check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        tmp.unlink(missing_ok=True)
        return False, f"tar failed: {exc.stderr[-300:]}"
    if not gzip_ok(tmp):
        tmp.unlink(missing_ok=True)
        return False, "archive failed gzip -t"
    os.replace(tmp, archive)
    shutil.rmtree(scenario_dir, ignore_errors=True)
    return True, f"{archive.stat().st_size / 1e6:.1f} MB"


def run_one(job: tuple[str, str, int]) -> dict:
    case, family, year = job
    scenario = f"{family}_{year}"
    study_case = f"base_case_{year}"
    name = f"{case}/{scenario}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{case}__{scenario}.log"

    os.chdir(REPO_ROOT)
    csv_path = str(OUTPUTS_ROOT / case)
    scenario_out_dir = OUTPUTS_ROOT / case / f"{scenario}_{study_case}"
    archive = scenario_out_dir.with_suffix(".tar.gz")

    if archive.exists():
        if gzip_ok(archive):
            return {"case": case, "scenario": scenario, "name": name, "sim_seconds": 0.0, "error": None,
                    "compressed": True, "note": "already done, skipped"}
        archive.unlink()  # stale/corrupt -> redo
    if scenario_out_dir.exists():
        shutil.rmtree(scenario_out_dir, ignore_errors=True)  # partial output of an interrupted run

    start = time.perf_counter()
    saved_out, saved_err = os.dup(1), os.dup(2)
    error = None
    try:
        with open(log_file, "w") as fh:
            os.dup2(fh.fileno(), 1)
            os.dup2(fh.fileno(), 2)
            logging.basicConfig(level=logging.WARNING, force=True)
            try:
                from assume import World
                from assume.scenario.loader_csv import load_scenario_folder

                world = World(database_uri="", export_csv_path=csv_path)
                load_scenario_folder(world, inputs_path=f"{INPUTS_PATH}/{case}", scenario=scenario,
                                     study_case=study_case)
                world.run()
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                logging.exception("run failed")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)

    sim_seconds = time.perf_counter() - start
    compressed, note = (False, "skipped (run failed)")
    if error is None:
        compressed, note = compress_and_remove(scenario_out_dir)
    return {"case": case, "scenario": scenario, "name": name, "sim_seconds": round(sim_seconds, 1),
            "error": error, "compressed": compressed, "note": note}


def build_jobs() -> list[tuple[str, str, int]]:
    return [(c, f, y) for c in CASES for f in FAMILIES for y in YEARS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    jobs = build_jobs()
    if args.list:
        for j in jobs:
            print(j)
        print(f"{len(jobs)} jobs")
        sys.exit(0)

    os.chdir(REPO_ROOT)
    print(f"running {len(jobs)} full-year simulations, {args.workers} workers, worst cases first", flush=True)
    rows = []
    t_start = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            rows.append(row)
            status = "OK" if row["error"] is None else "FAILED"
            comp = f"compressed {row['note']}" if row["compressed"] else row["note"]
            print(f"[{i}/{len(jobs)}] {status:7s} {row['name']:45s} {row['sim_seconds']/60:6.1f}min  {comp}"
                  + (f"  ERROR: {row['error']}" if row["error"] else ""), flush=True)
            pd.DataFrame(rows).to_csv(SUMMARY, index=False)

    total_h = (time.perf_counter() - t_start) / 3600
    df = pd.DataFrame(rows)
    n_ok = int(df.error.isna().sum())
    print(f"\n=== DONE: {len(df)} scenarios in {total_h:.2f} h ===")
    print(f"OK: {n_ok}  FAILED: {len(df) - n_ok}")
    if df.error.notna().any():
        print(df[df.error.notna()][["case", "scenario", "error"]].to_string(index=False))
