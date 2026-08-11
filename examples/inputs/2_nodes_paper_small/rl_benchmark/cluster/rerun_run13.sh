#!/bin/bash
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# ===========================================================================
# Re-run run 13 (eleven learning agents, inc_dec_learning) on SLURM, in ONE call
# ===========================================================================
#
#     bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/rerun_run13.sh
#
# That single call preflights the checkout, submits an 18-task array (6
# conditions x 3 seeds, one trial per task), and chains a collector job that
# regenerates the report and the figures and packs everything into one
# tarball. Nothing else has to be run by hand, and the script does not have to
# be edited for the cluster: --chdir is derived from where this file sits.
#
# The script is its own array body and its own collector -- SLURM re-invokes it
# with MODE=trial / MODE=collect. Run it with `bash`, not `sbatch`.
#
# Why re-run at all (HANDOFF.md, "Caveats on the current results"):
#   * run 13's archived films came from a working-tree inc_dec_learning; that
#     config is now committed (d717a2a5), so the re-run is reproducible;
#   * its `rewards` arrays used REWARD_WINDOW = 62, inherited from the
#     single-agent case, against a 69-transition episode. The recorder is now
#     69, so the reward trace WILL differ from the archive -- no bid, critic or
#     act_share number will;
#   * the matd3.py:618-628 debug prints, live for runs 09-12, are commented out;
#   * local seeds were memory-bound at ~0.85 GB and 4 concurrent workers; here
#     each trial owns a task.
#
# Knobs, all overridable from the environment:
#   ASSUME_PYTHON   interpreter (default $HOME/miniconda3/envs/assume/bin/python)
#   PARTITION       CPU partition (default cpu_il). NO GPU: two small MLPs
#                   against a CPU-bound simulator, a GPU only queues longer.
#   WALLTIME        per task (default 06:00:00 -- ~59 min locally for a
#                   50-episode trial, with headroom for a slower core)
#   MEM / CPUS      per task (default 4G / 1). CPUS stays at 1 and --threads at
#                   1 so the results are comparable with the archive: run 08
#                   found BLAS thread count alone flipping a surrogate seed.
#   MAX_PARALLEL    array concurrency cap (default 18 = no cap)
#   CONDITIONS      space-separated subset (default: all six)
#   SEEDS           space-separated subset (default: 42 1 2)
#
# Resubmitting only the failures:
#   MODE is per-task, so just re-run the whole thing -- a trial whose .npz
#   already validates is skipped in seconds by the runner's own guard.
# ===========================================================================

set -euo pipefail

MODE="${MODE:-submit}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(dirname "$HERE")"                     # .../rl_benchmark
REPO="$(cd "$BENCH/../../../.." && pwd)"       # repo root
PYTHON="${ASSUME_PYTHON:-$HOME/miniconda3/envs/assume/bin/python}"

OUT="$REPO/examples/outputs/2_nodes_paper_small/rl_benchmark/runs"
DATA="$OUT/data/13-multiagent-actshare"
LOGS="$OUT/logs/13-multiagent-actshare"
EXPORTS="$OUT/exports"
TRIALS="$LOGS/trials.txt"

RUNNER="$BENCH/real_matd3/assume_multiagent_actshare.py"

CONDITIONS="${CONDITIONS:-baseline-25 act-all-x2 act-all-x15 baseline act-all-x2-50 act-own-x15}"
SEEDS="${SEEDS:-42 1 2}"

PARTITION="${PARTITION:-cpu_il}"
WALLTIME="${WALLTIME:-06:00:00}"
MEM="${MEM:-4G}"
CPUS="${CPUS:-1}"

# ---------------------------------------------------------------- preflight

preflight() {
    [[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set ASSUME_PYTHON)" >&2; exit 1; }
    [[ -f "$RUNNER" ]] || { echo "no runner at $RUNNER" >&2; exit 1; }

    # 1. the inc-dec reward shaping is a source edit and must stay commented out
    if grep -Eq '^ {8}if reward > 0:' "$REPO/assume/strategies/learning_strategies.py"; then
        echo "PREFLIGHT: the reward shaping at learning_strategies.py:1583 is UNCOMMENTED." >&2
        echo "Run 13 is a true-reward run; comment it back out first." >&2
        exit 1
    fi

    # 2. inc_dec_learning must be the config run 13 used (RUNS.md section 3).
    #    The committed 5 h / train_freq 1h version dies with "No rewards were
    #    collected during evaluation run" -- after burning the walltime.
    "$PYTHON" - "$REPO" <<'PY' || exit 1
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(
    open(Path(sys.argv[1]) / "examples/inputs/2_nodes_paper_small/config.yaml")
)["inc_dec_learning"]
lc = cfg["learning_config"]
want = {
    "end_date": "2019-01-04 00:00", "learning_rate": 0.0001,
    "training_episodes": 50, "episodes_collecting_initial_experience": 5,
    "train_freq": "12h", "validation_episodes_interval": 5,
}
have = {"end_date": str(cfg["end_date"]),
        **{k: lc.get(k) for k in list(want)[1:]}}
bad = {k: (have[k], v) for k, v in want.items() if str(have[k]) != str(v)}
if bad:
    print("PREFLIGHT: inc_dec_learning is not run 13's config:", file=sys.stderr)
    for k, (got, exp) in bad.items():
        print(f"  {k}: {got!r}, expected {exp!r}", file=sys.stderr)
    sys.exit(1)
print("  preflight: inc_dec_learning matches run 13's table")
PY

    # 3. sbatch does not create missing --output directories; a job whose log
    #    path does not exist fails at launch with nothing to read.
    mkdir -p "$LOGS" "$DATA" "$EXPORTS"

    # 4. this file is submitted as the batch script for both the array and the
    #    collector; a checkout that lost the exec bit would fail at sbatch time
    chmod +x "${BASH_SOURCE[0]}" 2>/dev/null || true
}

# ------------------------------------------------------------------- submit

if [[ "$MODE" == "submit" ]]; then
    preflight

    : > "$TRIALS"
    for c in $CONDITIONS; do
        for s in $SEEDS; do
            echo "$c $s" >> "$TRIALS"
        done
    done
    N=$(wc -l < "$TRIALS")
    ARRAY="1-$N"
    [[ -n "${MAX_PARALLEL:-}" ]] && ARRAY="$ARRAY%$MAX_PARALLEL"

    echo
    echo "  repo    : $REPO"
    echo "  python  : $PYTHON"
    echo "  trials  : $N  ($TRIALS)"
    echo "  data    : $DATA"
    echo "  logs    : $LOGS"
    echo

    JOB=$(sbatch --parsable \
        --job-name=run13 \
        --partition="$PARTITION" \
        --time="$WALLTIME" \
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" \
        --chdir="$REPO" \
        --array="$ARRAY" \
        --output="$LOGS/run13-%A_%a.out" \
        --error="$LOGS/run13-%A_%a.err" \
        --export=ALL,MODE=trial \
        "${BASH_SOURCE[0]}")
    echo "  array job : $JOB"

    # afterany, not afterok: a partially failed batch is still worth collecting,
    # and the runner's validate_result() marks the incomplete trials in the
    # report rather than letting them pass silently.
    COLLECT=$(sbatch --parsable \
        --job-name=run13_collect \
        --partition="$PARTITION" \
        --time=00:30:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
        --chdir="$REPO" \
        --dependency=afterany:"$JOB" \
        --output="$LOGS/run13-collect-%j.out" \
        --error="$LOGS/run13-collect-%j.err" \
        --export=ALL,MODE=collect,ARRAY_JOB="$JOB" \
        "${BASH_SOURCE[0]}")
    echo "  collect   : $COLLECT (runs after the array, whatever it does)"
    echo
    echo "  watch     : squeue -j $JOB,$COLLECT"
    echo "  accounting: sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS"
    echo
    echo "  when it is done, from your laptop:"
    echo "    scp <cluster>:$EXPORTS/run13_${JOB}.tar.gz ."
    echo
    exit 0
fi

# -------------------------------------------------------------------- trial

if [[ "$MODE" == "trial" ]]; then
    TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TRIALS")
    [[ -n "$TRIAL" ]] || { echo "no trial on line $SLURM_ARRAY_TASK_ID of $TRIALS" >&2; exit 1; }
    COND=${TRIAL% *}
    SEED=${TRIAL#* }

    echo "=========================================="
    echo "condition : $COND"
    echo "seed      : $SEED"
    echo "task      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
    echo "started   : $(date -Is)"
    echo "=========================================="

    # one torch thread and one worker: SLURM does the scheduling, and a crashed
    # trial takes only its own task down. --disable-tensorboard is already set
    # by the runner for every child (run 11 lost six trials to a concurrent
    # TensorBoard writer race).
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

    "$PYTHON" "$RUNNER" \
        --conditions "$COND" \
        --seeds "$SEED" \
        --workers 1 \
        --out-dir "$DATA"

    echo "finished  : $(date -Is)"
    exit 0
fi

# ------------------------------------------------------------------ collect

if [[ "$MODE" == "collect" ]]; then
    STAMP="${ARRAY_JOB:-$(date +%Y%m%d-%H%M%S)}"
    REPORT="$LOGS/report_${STAMP}.txt"

    echo "  regenerating the report and the figures"
    # --report-only re-reads every archived .npz; it never retrains, so an
    # incomplete batch reports the trials it has and names the missing ones.
    "$PYTHON" "$RUNNER" --report-only \
        --conditions $CONDITIONS --seeds $SEEDS --out-dir "$DATA" \
        2>&1 | tee "$REPORT" || true
    "$PYTHON" "$BENCH/real_matd3/assume_multiagent_grids.py" || true
    "$PYTHON" "$BENCH/real_matd3/assume_multiagent_window.py" \
        2>&1 | tee -a "$REPORT" || true

    # accounting goes in the tarball: MaxRSS is what sets the next --mem
    if command -v sacct >/dev/null && [[ -n "${ARRAY_JOB:-}" ]]; then
        sacct -j "$ARRAY_JOB" \
            --format=JobID,JobName%18,State,ExitCode,Elapsed,MaxRSS \
            > "$LOGS/sacct_${STAMP}.txt" || true
    fi

    TARBALL="$EXPORTS/run13_${STAMP}.tar.gz"
    mkdir -p "$EXPORTS"
    # Everything needed to read the run, and nothing that cannot be regenerated:
    # the films, the per-trial logs, the SLURM streams, the figures. The scratch
    # databases and saved policies are excluded -- they are the bulk and no
    # analysis reads them.
    tar czf "$TARBALL" -C "$OUT" \
        --exclude='*.db' \
        --exclude='*.db-journal' \
        --exclude='scratch/*/learned_strategies' \
        "data/13-multiagent-actshare" \
        "logs/13-multiagent-actshare" \
        $( [[ -d "$OUT/img" ]] && echo "img" )

    echo
    echo "  packed $(du -h "$TARBALL" | cut -f1) into $TARBALL"
    echo "  scp <cluster>:$TARBALL ."
    exit 0
fi

echo "unknown MODE=$MODE (expected submit, trial or collect)" >&2
exit 1
