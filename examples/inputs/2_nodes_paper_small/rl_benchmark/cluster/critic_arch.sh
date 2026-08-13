#!/bin/bash
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# ===========================================================================
# Run 18 -- critic architecture and hyperparameters, LIVE, on inc-dec, in ONE call
# ===========================================================================
#
#     bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/critic_arch.sh
#
# Submits one task per (round, cell, seed) and chains a collector that
# regenerates the report and packs everything into one tarball. Same shape as
# rerun_run13.sh and eom_critic_evolution.sh: the script is its own array body
# and its own collector, SLURM re-invokes it with MODE=trial / MODE=collect,
# and --chdir is derived from where this file sits. Run it with `bash`.
#
# TWO ROUNDS, and by default only the first
# -----------------------------------------
#   arch   19 cells x 3 seeds = 57 tasks   critic architecture
#   hpo    20 cells x 3 seeds = 60 tasks   optimizer settings (hpo_grid.py)
#
#   ROUNDS="arch hpo" bash .../critic_arch.sh          # all 117 tasks
#   ROUNDS=hpo CELLS=lr bash .../critic_arch.sh        # just the 3x3 lr grid
#
# Round `arch`: does run 17 survive the live loop?
# ------------------------------------------------
# The offline gamma = 0 screen found (a) RSNorm disqualifying -- six variants
# carrying it pinned at argmax 100.0 at every width from 143 k to 8.5 M -- and
# (b) SimBa's residual trunk worth about 15x the parameters, 548 k against
# 8.48 M for a better argmax. Offline there is no bootstrap, no growing buffer
# and no actor moving the action distribution; all three are back here. The
# RSNorm carriers are NOT re-run: 24 more tasks of the same answer.
#
# The ladder is a GRID, not a line. Run 17 moved width at fixed depth, so
# "capacity" and "width" were one variable. Here each family runs at two depths
# up the same four-rung parameter ladder (100k / 500k / 2M / 8M), so depth and
# width separate. One SimBa block is two Linear layers, so the two families'
# depth units differ and each family's curve is read against itself.
#
# `split` is the new cell: observation and action each get their own encoder of
# equal width, merged at layer 2. Late injection gives the action its own
# weight matrix but it still arrives raw and outnumbered; `split` fixes the
# COUNT. Equal scale is what run 12's act_share moved, and act_share is the
# invented quantity this workstream exists to replace -- so the two have to be
# separable, and this is where.
#
# Round `hpo`: 20 cells, coordinate sweep
# ---------------------------------------
# Shared with the EOM study (cluster/hpo_eom.sh) so the two tables can be laid
# side by side. Axes: 3 learning rates x {const, linear, cosine}, batch size,
# policy delay, weight decay. Weight decay is the one with no LearningConfig
# field -- matd3.py constructs AdamW(params, lr=...) and nothing else, so every
# archived run trained at torch's default 0.01 rather than at none. See
# real_matd3/optim_patches.py.
#
# Knobs, all overridable from the environment:
#   ASSUME_PYTHON   interpreter (default $HOME/miniconda3/envs/assume/bin/python)
#   PARTITION       CPU partition (default cpu_il). No GPU is wanted.
#   WALLTIME        per task (default 06:00:00). A 40-episode inc-dec trial is
#                   ~1 h locally at the baseline width; the 8 M cells are
#                   compute-bound and were not timed before this was written,
#                   so the default is headroom rather than an estimate.
#   MEM / CPUS      per task (default 8G / 1). The 8 M critics are what set the
#                   memory; 4G is enough for everything up to 500 k.
#   MAX_PARALLEL    array concurrency cap (default: none)
#   ROUNDS          subset of "arch hpo" (default "arch")
#   ARCHS           subset of the architectures (default: all 19)
#   CELLS           subset of the hyperparameter cells, or an axis group
#                   (centre, lr, batch, delay, wd) or "all" (default "all")
#   SEEDS           subset (default: 42 1 2, run 13's seeds)
#   EPISODES        override the 40-episode budget
#   GRID/NOBS       recorder resolution (default 401 / 6, run 12's)
#   RUN_NAME        data/log/tarball basename (default 18-live-arch)
#   RERUN           set to 1 to re-run trials that already have a valid .npz
# ===========================================================================

set -euo pipefail

MODE="${MODE:-submit}"

# Where the code is. At submit time this file sits in the checkout, so the
# paths can be derived from it -- but SLURM COPIES the batch script into its
# spool directory before running it, so in MODE=trial / MODE=collect
# ${BASH_SOURCE[0]} is /var/spool/slurmd/.../slurm_script and climbing four
# levels off it lands on "/". The submit branch therefore exports both paths
# and the array tasks take them from the environment.
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
NAME="${RUN_NAME:-18-live-arch}"
DATA="$OUT/data/$NAME"
LOGS="$OUT/logs/$NAME"
EXPORTS="$OUT/exports"
TRIALS="$LOGS/trials.txt"

RUNNER="$BENCH/real_matd3/assume_arch_sweep.py"

ROUNDS="${ROUNDS:-arch}"
SEEDS="${SEEDS:-42 1 2}"
CELLS="${CELLS:-all}"
GRID="${GRID:-401}"
NOBS="${NOBS:-6}"

PARTITION="${PARTITION:-cpu_il}"
WALLTIME="${WALLTIME:-06:00:00}"
MEM="${MEM:-8G}"
CPUS="${CPUS:-1}"

# ---------------------------------------------------------------- preflight

preflight() {
    [[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set ASSUME_PYTHON)" >&2; exit 1; }
    [[ -f "$RUNNER" ]] || { echo "no runner at $RUNNER" >&2; exit 1; }

    # The inc-dec reward shaping is a source edit and fires unconditionally
    # once uncommented. Every cell here is a true-reward run.
    if grep -Eq '^ {8}if reward > 0:' "$REPO/assume/strategies/learning_strategies.py"; then
        echo "PREFLIGHT: the reward shaping at learning_strategies.py:1583 is UNCOMMENTED." >&2
        echo "Comment it back out before running anything on the true reward." >&2
        exit 1
    fi

    # The runner's own preflight() checks the frozen starting buffer's sha256
    # and cross-checks run 11's BASELINE; this only resolves the cell names
    # early, before 117 tasks queue up and all fail identically.
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$BENCH/real_matd3')
sys.path.insert(0, '$BENCH')
from assume_arch_sweep import ARCHS
from hpo_grid import resolve
bad = [a for a in '''${ARCHS:-}'''.split() if a not in ARCHS]
if bad:
    sys.exit(f'PREFLIGHT: unknown architecture(s) {bad}')
resolve('''$CELLS'''.split())
for r in '''$ROUNDS'''.split():
    if r not in ('arch', 'hpo'):
        sys.exit(f'PREFLIGHT: unknown round {r!r}; expected arch or hpo')
" || exit 1

    # sbatch does not create missing --output directories.
    mkdir -p "$LOGS" "$DATA" "$EXPORTS"
    chmod +x "${BASH_SOURCE[0]}" 2>/dev/null || true
}

# ------------------------------------------------------------------- submit

if [[ "$MODE" == "submit" ]]; then
    preflight

    # the cell list per round comes from Python, so the shell never carries a
    # second copy of the grid that could drift out of step with hpo_grid.py
    : > "$TRIALS"
    for r in $ROUNDS; do
        NAMES=$("$PYTHON" -c "
import sys
sys.path.insert(0, '$BENCH/real_matd3')
sys.path.insert(0, '$BENCH')
if '$r' == 'arch':
    from assume_arch_sweep import ARCHS
    names = '''${ARCHS:-}'''.split() or ARCHS
else:
    from hpo_grid import resolve
    names = resolve('''$CELLS'''.split())
print(' '.join(names))
")
        for n in $NAMES; do
            for s in $SEEDS; do
                echo "$r $n $s" >> "$TRIALS"
            done
        done
    done
    N=$(wc -l < "$TRIALS")
    ARRAY="1-$N"
    [[ -n "${MAX_PARALLEL:-}" ]] && ARRAY="$ARRAY%$MAX_PARALLEL"

    echo
    echo "  repo    : $REPO"
    echo "  python  : $PYTHON"
    echo "  rounds  : $ROUNDS"
    echo "  trials  : $N  ($TRIALS)"
    echo "  run     : $NAME"
    echo "  recorder: grid $GRID, $NOBS observations, every training block"
    echo "  data    : $DATA"
    echo "  logs    : $LOGS"
    echo

    JOB=$(sbatch --parsable \
        --job-name=arch18 \
        --partition="$PARTITION" \
        --time="$WALLTIME" \
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" \
        --chdir="$REPO" \
        --array="$ARRAY" \
        --output="$LOGS/arch-%A_%a.out" \
        --error="$LOGS/arch-%A_%a.err" \
        --export=ALL,MODE=trial,ASSUME_REPO="$REPO",ASSUME_BENCH="$BENCH" \
        "${BASH_SOURCE[0]}")
    echo "  array job : $JOB"

    COLLECT=$(sbatch --parsable \
        --job-name=arch18_collect \
        --partition="$PARTITION" \
        --time=01:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
        --chdir="$REPO" \
        --dependency=afterany:"$JOB" \
        --output="$LOGS/arch-collect-%j.out" \
        --error="$LOGS/arch-collect-%j.err" \
        --export=ALL,MODE=collect,ARRAY_JOB="$JOB",ASSUME_REPO="$REPO",ASSUME_BENCH="$BENCH" \
        "${BASH_SOURCE[0]}")
    echo "  collect   : $COLLECT (runs after the array, whatever it does)"
    echo
    echo "  watch     : squeue -j $JOB,$COLLECT"
    echo "  accounting: sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS"
    echo
    echo "  when it is done, from your laptop:"
    echo "    scp <cluster>:$EXPORTS/arch18_${JOB}.tar.gz ."
    echo
    exit 0
fi

# -------------------------------------------------------------------- trial

if [[ "$MODE" == "trial" ]]; then
    [[ -f "$RUNNER" ]] || { echo "no runner at $RUNNER -- ASSUME_REPO/ASSUME_BENCH did not survive the submit (got REPO=$REPO)" >&2; exit 1; }
    [[ -f "$TRIALS" ]] || { echo "no trial list at $TRIALS" >&2; exit 1; }

    TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TRIALS")
    [[ -n "$TRIAL" ]] || { echo "no trial on line $SLURM_ARRAY_TASK_ID of $TRIALS" >&2; exit 1; }
    read -r ROUND CELL SEED <<< "$TRIAL"

    echo "=========================================="
    echo "round     : $ROUND"
    echo "cell      : $CELL"
    echo "seed      : $SEED"
    echo "task      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
    echo "started   : $(date -Is)"
    echo "=========================================="

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

    ARGS=(--round "$ROUND" --seeds "$SEED" --workers 1 --threads 1
          --grid "$GRID" --n-obs "$NOBS" --out-dir "$DATA")
    if [[ "$ROUND" == "arch" ]]; then
        ARGS+=(--archs "$CELL")
    else
        ARGS+=(--cells "$CELL")
    fi
    [[ -n "${EPISODES:-}" ]] && ARGS+=(--episodes "$EPISODES")
    [[ "${RERUN:-0}" == "1" ]] && ARGS+=(--rerun)

    "$PYTHON" "$RUNNER" "${ARGS[@]}"

    echo "finished  : $(date -Is)"
    exit 0
fi

# ------------------------------------------------------------------ collect

if [[ "$MODE" == "collect" ]]; then
    STAMP="${ARRAY_JOB:-$(date +%Y%m%d-%H%M%S)}"
    REPORT="$LOGS/report_${STAMP}.txt"

    echo "  regenerating the report"
    # --report-only re-reads the archived .npz files and never retrains, so an
    # incomplete batch reports what it has and leaves the rest blank.
    ARGS=(--report-only --round $ROUNDS --seeds $SEEDS --out-dir "$DATA"
          --cells $CELLS)
    [[ -n "${ARCHS:-}" ]] && ARGS+=(--archs $ARCHS)
    [[ -n "${EPISODES:-}" ]] && ARGS+=(--episodes "$EPISODES")
    "$PYTHON" "$RUNNER" "${ARGS[@]}" 2>&1 | tee "$REPORT" || true

    # The figures do NOT run here: analysis/ imports the house palette from
    # sweeps/run_benchmark.py, which imports stable_baselines3 at module level,
    # and SB3 is not in the cluster environment. Redraw locally after
    # unpacking -- see cluster/README.md.

    if command -v sacct >/dev/null && [[ -n "${ARRAY_JOB:-}" ]]; then
        sacct -j "$ARRAY_JOB" \
            --format=JobID,JobName%18,State,ExitCode,Elapsed,MaxRSS \
            > "$LOGS/sacct_${STAMP}.txt" || true
    fi

    TARBALL="$EXPORTS/arch18_${STAMP}.tar.gz"
    mkdir -p "$EXPORTS"
    # the films, the logs and the accounting. The scratch databases hold only
    # the first two products of each episode (RUNS.md correction 16) and the
    # saved policies are bulk no analysis reads.
    tar czf "$TARBALL" -C "$OUT" \
        --exclude='*.db' \
        --exclude='*.db-journal' \
        --exclude='scratch/*/learned_strategies' \
        "data/$NAME" \
        "logs/$NAME"

    echo
    echo "  packed $(du -h "$TARBALL" | cut -f1) into $TARBALL"
    echo "  scp <cluster>:$TARBALL ."
    exit 0
fi

echo "unknown MODE=$MODE (expected submit, trial or collect)" >&2
exit 1
