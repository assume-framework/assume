# SPDX-License-Identifier: AGPL-3.0-or-later
"""Re-simulate the 2040 scenarios affected by the lignite-efficiency fix (2026-11-26).

Scope: 3 families (fokusH2, fokusstrom, technologiemix) x 2040 x 18 input folders
(8 price-taker cases + 8 price-setter _BID + steel_base + cement_base no-actor bases)
= 54 runs. Old archives are MOVED to outputs/_stale_2040_prelignitefix/ (reversible),
not deleted, so before/after can be compared.

Reuses run_one from run_batch_8cases (handles full-year sim + atomic gzip archive).
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

# Base-only: the no-actor runs (no steel/cement/aviation+maritime demand added).
# Their corrected DA price signal feeds the investment model; the actor runs follow later.
CASE_DIRS = [
    "steel_base", "cement_base",
]
FAMILIES = ["fokusH2", "fokusstrom", "technologiemix"]
YEAR = 2040
STALE_DIR = OUTPUTS_ROOT / "_stale_2040_prelignitefix"
LOG_SUMMARY = OUTPUTS_ROOT / "resim_2040_summary.csv"


def build_jobs():
    return [(c, f, YEAR) for c in CASE_DIRS for f in FAMILIES]


FIX_CUTOFF = time.mktime(time.strptime("2026-11-26", "%Y-%m-%d"))


def stash_stale(jobs):
    """Move only PRE-FIX archives aside. An archive dated on/after the lignite-fix
    cutoff is a fresh, already-corrected run -> leave it so run_one skips it, never
    re-stash it (that was what wiped completed runs on an earlier restart)."""
    moved = 0
    for case, family, year in jobs:
        arch = OUTPUTS_ROOT / case / f"{family}_{year}_base_case_{year}.tar.gz"
        if arch.exists() and arch.stat().st_mtime < FIX_CUTOFF:
            dest = STALE_DIR / case / arch.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                dest.unlink()  # never overwrite an already-stashed pre-fix original
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
    print(f"running {len(jobs)} re-sims, {args.workers} workers", flush=True)

    rows = []
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, jobs, chunksize=1), 1):
            rows.append(row)
            status = "OK" if row["error"] is None else "FAILED"
            comp = f"compressed {row['note']}" if row["compressed"] else row["note"]
            print(f"[{i}/{len(jobs)}] {status:7s} {row['name']:42s} "
                  f"{row['sim_seconds']/60:6.1f}min  {comp}"
                  + (f"  ERROR: {row['error']}" if row["error"] else ""), flush=True)
            pd.DataFrame(rows).to_csv(LOG_SUMMARY, index=False)

    dt = (time.perf_counter() - t0) / 3600
    df = pd.DataFrame(rows)
    nok = int(df.error.isna().sum())
    print(f"\n=== DONE: {len(df)} runs in {dt:.2f} h | OK {nok} FAILED {len(df)-nok} ===", flush=True)
    if df.error.notna().any():
        print(df[df.error.notna()][["case", "scenario", "error"]].to_string(index=False), flush=True)
