#!/bin/bash
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# ===========================================================================
# Run 18c -- the same hyperparameter grid on the EOM case p1, in ONE call
# ===========================================================================
#
#     bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/hpo_eom.sh
#
# Submits one task per (case, cell, seed) -- 20 cells x 3 seeds = 60 tasks by
# default -- and chains a collector that regenerates the report and packs
# everything into one tarball. Same shape as eom_critic_evolution.sh, which it
# is a sibling of rather than a fork: the trials go through the SAME runner
# (real_matd3/eom_critic_film.py), with --hp as the extra axis.
#
# Why p1
# ------
# p1 is example_02_single_bid's undistorted demand series, i.e. p1 IS sb02b and
# is the pivotal-frequency ladder's own control: five learners, single bid,
# 13 % of hours pivotal. Run 14b found the critic fits the regime it sees often
# and leaves the rare one as noise, so this sweep runs with OBS_REGIMES=1 by
# default and the films are stratified by Nash equilibrium -- the question is
# not just "does a setting help" but "does it help in the regime the critic is
# currently ignoring". A setting that lifts the aggregate by fitting the common
# regime harder is not an improvement for this benchmark's purpose.
#
# Under OBS_REGIMES=1, NOBS is PER REGIME, so the film grows by the number of
# regimes present (2 for p1). RERUN=1 is needed if only the RECORDING changed:
# validate_result() checks schema, label and seed, not how observations were
# sampled, so an existing film is otherwise skipped and you get the old one.
#
# The grid (real_matd3/hpo_grid.py, shared with cluster/critic_arch.sh)
# --------------------------------------------------------------------
#   lr      3 learning rates x {const, linear, cosine}       9 cells
#   batch   64 / 128 / 256 / 512                             4
#   delay   policy_delay 1 / 2 / 4 / 8                       4
#   wd      weight_decay 0.0 / 0.1                           2
#   centre  'default' -- the study case's own settings       1
#
# A coordinate sweep, not a full cross: 108 cells crossed against 20 here, and
# nothing in runs 09-17 suggests these four interact -- the failure they are
# aimed at is a critic that never develops a slope. Learning rate and its
# schedule ARE crossed, because a schedule is a statement about the rate.
#
# lr0.001-const reproduces 'default' exactly (both study cases run 1e-3 with no
# schedule). That redundancy is deliberate: a 3x3 grid with a hole where its
# centre belongs is harder to read than one repeated cell, and the two agreeing
# is a check that the sweep machinery is not itself changing the run.
#
# Weight decay is the axis with no config field. matd3.py builds
# AdamW(params, lr=...) and passes nothing else, so every run in this
# benchmark's archive trained at torch's default 0.01 rather than at none --
# "the current runs use no regularization" is false. Applied by the monkeypatch
# in real_matd3/optim_patches.py, and NOT applied for cells at 0.01, so those
# films stay bit-comparable with the run 14/15 batches.
#
#   CELLS=lr  bash .../hpo_eom.sh              # just the 3x3 grid, 27 tasks
#   CASES="p1 p4" bash .../hpo_eom.sh          # two rungs of the ladder
#
# Knobs, all overridable from the environment:
#   ASSUME_PYTHON   interpreter (default $HOME/miniconda3/envs/assume/bin/python)
#   PARTITION       CPU partition (default cpu_il). No GPU is wanted.
#   WALLTIME        per task (default 06:00:00). p1 runs a 744 h episode in
#                   ~10 s locally, so a 100-episode trial is ~20 min; the
#                   default is headroom for a slower core.
#   MEM / CPUS      per task (default 8G / 1)
#   MAX_PARALLEL    array concurrency cap (default: none)
#   CASES           subset (default "p1"). Any case eom_critic_film.py knows.
#   CELLS           cell names, an axis group (centre, lr, batch, delay, wd)
#                   or "all" (default "all")
#   SEEDS           subset (default: 42 1 2)
#   EPISODES        override training_episodes
#   GRID/NOBS/EVERY recorder resolution (default 201 / 4 / 4). NOBS is per
#                   regime while OBS_REGIMES=1.
#   OBS_REGIMES     0 to switch off the per-equilibrium stratification
#                   (default 1 -- it is the point of running p1)
#   CRITIC_ARCH     critic architecture for every task (default baseline).
#                   Cross this with the grid only after run 18a has said which
#                   architecture is worth tuning.
#   KEEP_DB         0 to leave the sqlite databases out (drops exploitability)
#   RUN_NAME        data/log/tarball basename (default 18c-hpo-eom)
#   RERUN           1 to re-run trials that already have a valid .npz
# ===========================================================================

set -euo pipefail

MODE="${MODE:-submit}"

# SLURM copies the batch script into its spool directory, so ${BASH_SOURCE[0]}
# is useless in MODE=trial/collect; the submit branch exports both paths.
if [[ -n "${ASSUME_BENCH:-}" && -n "${ASSUME_REPO:-}" ]]; then
    BENCH="$ASSUME_BENCH"
    REPO="$ASSUME_REPO"
else
    HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BENCH="$(dirname "$HERE")"                 # .../rl_benchmark
    REPO="$(cd "$BENCH/../../../.." && pwd)"   # repo root
fi
PYTHON="${ASSUME_PYTHON:-$HOME/miniconda3/envs/assume/bin/python}"

OUT="$REPO/examples/outputs/2_nodes_paper_small/rl_benchmark/runs"
NAME="${RUN_NAME:-18c-hpo-eom}"
DATA="$OUT/data/$NAME"
LOGS="$OUT/logs/$NAME"
EXPORTS="$OUT/exports"
TRIALS="$LOGS/trials.txt"

RUNNER="$BENCH/real_matd3/eom_critic_film.py"
FIGURES="$BENCH/analysis/eom_critic_evolution.py"

CASES="${CASES:-p1}"
CELLS="${CELLS:-all}"
SEEDS="${SEEDS:-42 1 2}"
GRID="${GRID:-201}"
NOBS="${NOBS:-4}"
EVERY="${EVERY:-4}"
OBS_REGIMES="${OBS_REGIMES:-1}"
CRITIC_ARCH="${CRITIC_ARCH:-baseline}"

PARTITION="${PARTITION:-cpu_il}"
WALLTIME="${WALLTIME:-06:00:00}"
MEM="${MEM:-8G}"
CPUS="${CPUS:-1}"
KEEP_DB="${KEEP_DB:-1}"

# ---------------------------------------------------------------- preflight

preflight() {
    [[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set ASSUME_PYTHON)" >&2; exit 1; }
    [[ -f "$RUNNER" ]] || { echo "no runner at $RUNNER" >&2; exit 1; }

    # The inc-dec reward shaping is a source edit and fires unconditionally --
    # it would silently reshape these EOM runs too.
    if grep -Eq '^ {8}if reward > 0:' "$REPO/assume/strategies/learning_strategies.py"; then
        echo "PREFLIGHT: the reward shaping at learning_strategies.py:1583 is UNCOMMENTED." >&2
        echo "Comment it back out before running anything on the true reward." >&2
        exit 1
    fi

    # resolve the case and cell names early, before 60 tasks queue up and all
    # fail identically
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$BENCH/real_matd3')
sys.path.insert(0, '$BENCH')
from eom_critic_film import CASES
from hpo_grid import resolve
bad = [c for c in '''$CASES'''.split() if c not in CASES]
if bad:
    sys.exit(f'PREFLIGHT: unknown case(s) {bad}; known: {list(CASES)}')
resolve('''$CELLS'''.split())
" || exit 1

    mkdir -p "$LOGS" "$DATA" "$EXPORTS"
    chmod +x "${BASH_SOURCE[0]}" 2>/dev/null || true
}

# ------------------------------------------------------------------- submit

if [[ "$MODE" == "submit" ]]; then
    preflight

    # the cell list comes from Python, so the shell never carries a second copy
    # of the grid that could drift out of step with hpo_grid.py
    CELL_NAMES=$("$PYTHON" -c "
import sys
sys.path.insert(0, '$BENCH/real_matd3')
sys.path.insert(0, '$BENCH')
from hpo_grid import resolve
print(' '.join(resolve('''$CELLS'''.split())))
")

    : > "$TRIALS"
    for c in $CASES; do
        for cell in $CELL_NAMES; do
            for seed in $SEEDS; do
                echo "$c $cell $seed" >> "$TRIALS"
            done
        done
    done
    N=$(wc -l < "$TRIALS")
    ARRAY="1-$N"
    [[ -n "${MAX_PARALLEL:-}" ]] && ARRAY="$ARRAY%$MAX_PARALLEL"

    echo
    echo "  repo    : $REPO"
    echo "  python  : $PYTHON"
    echo "  cases   : $CASES"
    echo "  cells   : $CELL_NAMES"
    echo "  trials  : $N  ($TRIALS)"
    echo "  run     : $NAME"
    echo "  critic  : $CRITIC_ARCH"
    echo "  recorder: grid $GRID, $NOBS observations$( [[ "$OBS_REGIMES" == "1" ]] && echo " PER REGIME" ), every $EVERY blocks"
    echo "  existing: $( [[ "${RERUN:-0}" == "1" ]] && echo "re-run (--rerun)" || echo "skipped if valid -- set RERUN=1 to re-run" )"
    echo "  data    : $DATA"
    echo "  logs    : $LOGS"
    echo

    JOB=$(sbatch --parsable \
        --job-name=hpo_eom \
        --partition="$PARTITION" \
        --time="$WALLTIME" \
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" \
        --chdir="$REPO" \
        --array="$ARRAY" \
        --output="$LOGS/hpo-%A_%a.out" \
        --error="$LOGS/hpo-%A_%a.err" \
        --export=ALL,MODE=trial,ASSUME_REPO="$REPO",ASSUME_BENCH="$BENCH" \
        "${BASH_SOURCE[0]}")
    echo "  array job : $JOB"

    COLLECT=$(sbatch --parsable \
        --job-name=hpo_collect \
        --partition="$PARTITION" \
        --time=01:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
        --chdir="$REPO" \
        --dependency=afterany:"$JOB" \
        --output="$LOGS/hpo-collect-%j.out" \
        --error="$LOGS/hpo-collect-%j.err" \
        --export=ALL,MODE=collect,ARRAY_JOB="$JOB",ASSUME_REPO="$REPO",ASSUME_BENCH="$BENCH" \
        "${BASH_SOURCE[0]}")
    echo "  collect   : $COLLECT (runs after the array, whatever it does)"
    echo
    echo "  watch     : squeue -j $JOB,$COLLECT"
    echo "  accounting: sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS"
    echo
    echo "  when it is done, from your laptop:"
    echo "    scp <cluster>:$EXPORTS/hpo_${JOB}.tar.gz ."
    echo
    exit 0
fi

# -------------------------------------------------------------------- trial

if [[ "$MODE" == "trial" ]]; then
    [[ -f "$RUNNER" ]] || { echo "no runner at $RUNNER -- ASSUME_REPO/ASSUME_BENCH did not survive the submit (got REPO=$REPO)" >&2; exit 1; }
    [[ -f "$TRIALS" ]] || { echo "no trial list at $TRIALS" >&2; exit 1; }

    TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TRIALS")
    [[ -n "$TRIAL" ]] || { echo "no trial on line $SLURM_ARRAY_TASK_ID of $TRIALS" >&2; exit 1; }
    read -r CASE CELL SEED <<< "$TRIAL"

    echo "=========================================="
    echo "case      : $CASE"
    echo "cell      : $CELL"
    echo "seed      : $SEED"
    echo "critic    : $CRITIC_ARCH"
    echo "task      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
    echo "started   : $(date -Is)"
    echo "=========================================="

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

    ARGS=(--cases "$CASE" --seeds "$SEED" --hp "$CELL" --workers 1 --threads 1
          --grid "$GRID" --n-obs "$NOBS" --every "$EVERY"
          --critic-arch "$CRITIC_ARCH" --out-dir "$DATA")
    [[ -n "${EPISODES:-}" ]] && ARGS+=(--episodes "$EPISODES")
    [[ "$OBS_REGIMES" == "1" ]] && ARGS+=(--obs-regimes)
    [[ "${RERUN:-0}" == "1" ]] && ARGS+=(--rerun)

    "$PYTHON" "$RUNNER" "${ARGS[@]}"

    echo "finished  : $(date -Is)"
    exit 0
fi

# ------------------------------------------------------------------ collect

if [[ "$MODE" == "collect" ]]; then
    STAMP="${ARRAY_JOB:-$(date +%Y%m%d-%H%M%S)}"
    REPORT="$LOGS/report_${STAMP}.txt"

    CELL_NAMES=$("$PYTHON" -c "
import sys
sys.path.insert(0, '$BENCH/real_matd3')
sys.path.insert(0, '$BENCH')
from hpo_grid import resolve
print(' '.join(resolve('''$CELLS'''.split())))
")

    echo "  regenerating the report, one cell at a time"
    # --report-only re-reads the archived .npz files and never retrains, so an
    # incomplete batch reports what it has and names what is missing. One
    # invocation per cell: --hp selects which films are read.
    : > "$REPORT"
    for cell in $CELL_NAMES; do
        "$PYTHON" "$RUNNER" --report-only \
            --cases $CASES --seeds $SEEDS --hp "$cell" --out-dir "$DATA" \
            2>&1 | tee -a "$REPORT" || true
    done

    # The figure scripts do NOT run here: analysis/ imports the house palette
    # from sweeps/run_benchmark.py, which imports stable_baselines3 at module
    # level, and SB3 is not in the cluster environment. Attempted anyway, on
    # the chance the environment has it, and every step is || true so a broken
    # figure cannot cost the tarball. Redraw locally -- see cluster/README.md.
    "$PYTHON" "$FIGURES" --cases $CASES --seeds $SEEDS \
        --data-dir "$DATA" --img-dir "$OUT/img" 2>&1 | tee -a "$REPORT" || true

    if command -v sacct >/dev/null && [[ -n "${ARRAY_JOB:-}" ]]; then
        sacct -j "$ARRAY_JOB" \
            --format=JobID,JobName%18,State,ExitCode,Elapsed,MaxRSS \
            > "$LOGS/sacct_${STAMP}.txt" || true
    fi

    TARBALL="$EXPORTS/hpo_${STAMP}.tar.gz"
    mkdir -p "$EXPORTS"
    EXCLUDES=(--exclude='scratch/*/learned_strategies')
    if [[ "$KEEP_DB" == "0" ]]; then
        EXCLUDES+=(--exclude='*.db' --exclude='*.db-journal')
    else
        # the sqlite files carry rl_exploitability, which is the second reading
        # these EOM runs exist for -- keep them, drop only the journals
        EXCLUDES+=(--exclude='*.db-journal')
    fi
    tar czf "$TARBALL" -C "$OUT" \
        "${EXCLUDES[@]}" \
        "data/$NAME" \
        "logs/$NAME" \
        $( [[ -d "$OUT/img" ]] && echo "img" )

    echo
    echo "  packed $(du -h "$TARBALL" | cut -f1) into $TARBALL"
    echo "  scp <cluster>:$TARBALL ."
    exit 0
fi

echo "unknown MODE=$MODE (expected submit, trial or collect)" >&2
exit 1
