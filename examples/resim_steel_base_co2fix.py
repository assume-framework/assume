# SPDX-License-Identifier: AGPL-3.0-or-later
"""Full re-run of steel_base (6 families x 4 years = 24) after the 2026-11-27
CO2-price correction (-> constant Ariadne values) and waste-price NaN fix
(fokusH2 2035/2040 -> 0.0). steel_base is the canonical no-actor BASE price
signal for the investment model.

Old archives are MOVED to outputs/_stale_co2fix/steel_base/ (reversible), then
run_one (from run_batch_8cases) regenerates each scenario. Study_case base_case_<year>.
"""
import os, sys, shutil, time, argparse
import multiprocessing as mp
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_batch_8cases import run_one, OUTPUTS_ROOT  # noqa: E402
import pandas as pd  # noqa: E402

CASE = "steel_base"
FAMILIES = ["aktuellepolitiken", "fokusH2", "fokusstrom",
            "hohenachfrage", "niedrigenachfrage", "technologiemix"]
YEARS = [2030, 2035, 2040, 2045]
STALE_DIR = OUTPUTS_ROOT / "_stale_co2fix" / CASE
LOG_SUMMARY = OUTPUTS_ROOT / "resim_steel_base_co2fix_summary.csv"


def build_jobs():
    return [(CASE, f, y) for f in FAMILIES for y in YEARS]


def stash_stale(jobs):
    moved = 0
    for case, family, year in jobs:
        arch = OUTPUTS_ROOT / case / f"{family}_{year}_base_case_{year}.tar.gz"
        if arch.exists():
            STALE_DIR.mkdir(parents=True, exist_ok=True)
            dest = STALE_DIR / arch.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(arch), str(dest))
            moved += 1
    return moved


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs()
    if args.list:
        for j in jobs:
            print(j)
        print(f"{len(jobs)} jobs")
        sys.exit(0)

    os.chdir(HERE.parent)
    moved = stash_stale(jobs)
    print(f"stashed {moved} stale archives -> {STALE_DIR}", flush=True)
    print(f"running {len(jobs)} steel_base re-sims, {args.workers} workers", flush=True)

    rows = []
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            rows.append(row)
            status = "OK" if row["error"] is None else "FAILED"
            comp = f"compressed {row['note']}" if row["compressed"] else row["note"]
            print(f"[{i}/{len(jobs)}] {status:7s} {row['name']:34s} "
                  f"{row['sim_seconds']/60:6.1f}min  {comp}"
                  + (f"  ERROR: {row['error']}" if row["error"] else ""), flush=True)
            pd.DataFrame(rows).to_csv(LOG_SUMMARY, index=False)

    dt = (time.perf_counter() - t0) / 3600
    df = pd.DataFrame(rows)
    nok = int(df.error.isna().sum())
    print(f"\n=== DONE: {len(df)} runs in {dt:.2f} h | OK {nok} FAILED {len(df)-nok} ===", flush=True)
    if df.error.notna().any():
        print(df[df.error.notna()][["case", "scenario", "error"]].to_string(index=False), flush=True)
