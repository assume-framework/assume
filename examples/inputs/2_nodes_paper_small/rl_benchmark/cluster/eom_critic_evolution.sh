#!/bin/bash
# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# ===========================================================================
# Critic evolution on the plain-EOM examples, on SLURM, in ONE call
# ===========================================================================
#
#     bash examples/inputs/2_nodes_paper_small/rl_benchmark/cluster/eom_critic_evolution.sh
#
# Submits a 9-task array (3 cases x 3 seeds, one trial per task) and chains
# a collector that regenerates the report and the figures and packs everything
# into one tarball. Same shape as rerun_run13.sh: the script is its own array
# body and its own collector, SLURM re-invokes it with MODE=trial / MODE=collect,
# and --chdir is derived from where this file sits. Run it with `bash`.
#
# What it films (HANDOFF.md, workstream C). Six cases, three run by default:
#
#   case    scenario / study case                 learners  bid structure
#   02a     example_02a / base                    pp_6      two-bid
#   02b     example_02b / base                    pp_6-10   two-bid
#   02c     example_02c / base                    pp_6-15   two-bid
#   sb02a   example_02_single_bid / 02a           pp_6      SINGLE bid
#   sb02b   example_02_single_bid / 02b           pp_6-10   SINGLE bid
#   sb02c   example_02_single_bid / 02c           pp_6-15   SINGLE bid
#
# The default is the sb* trio. Each sb case is the same fleet, demand, fuel
# prices and market as the 02x case above it, bidding with
# EnergyLearningSingleBidStrategy (act_dim 1, one bid for the unit's whole
# max_power) instead of EnergyLearningStrategy (act_dim 2, inflexible +
# flexible block) -- so the pair is an A/B with the bid structure as the only
# deliberate difference. (The single-bid strategy also defaults foresight to 24
# rather than 12, so its observation is 50-dimensional against 26. That is the
# strategy's own default, not a choice of this scenario, but it means the
# critic's input count moves for two reasons at once.)
#
# Only the LEARNING units are recorded -- learning_role.rl_strats holds exactly
# those, and the naive units never enter a critic's input. The two-bid cases
# take three sweeps per agent (one per action component, plus the diagonal);
# the single-bid cases take one, named "diag" so every reader works on both.
#
# These are also the scenarios exploitability is valid on -- one pay-as-clear
# EOM, no storage, no redispatch (see the SCOPE note in
# assume/reinforcement_learning/exploitability.py). The runs write
# rl_exploitability on every evaluation episode, and the collector keeps those
# databases in the tarball for exactly that reason. They are the bulk of it.
# The sb* cases are the cleanest of the six for it: one bid per unit is exactly
# what the probe handles, with no ordered-bids decomposition involved.
#
# Knobs, all overridable from the environment:
#   ASSUME_PYTHON   interpreter (default $HOME/miniconda3/envs/assume/bin/python)
#   PARTITION       CPU partition (default cpu_il). No GPU is wanted.
#   WALLTIME        per task (default 06:00:00). Measured locally, 02b runs a
#                   744 h episode in ~10 s, so its 100-episode trial is ~20 min
#                   and 02c's ten learners maybe 3x that -- the default is
#                   headroom for a slower core, not an estimate.
#   MEM / CPUS      per task (default 8G / 1)
#   MAX_PARALLEL    array concurrency cap (default: none)
#   CASES           subset (default "sb02a sb02b sb02c"). Pass "02a 02b 02c"
#                   for the two-bid originals, or all six for both ladders.
#   SEEDS           subset (default: 42 1 2, run 13's seeds)
#   EPISODES        override training_episodes (default: the study case's 100)
#   GRID/NOBS/EVERY recorder resolution. Defaults 201 / 4 / 4 keep 02c near
#                   30 MB per seed; --grid 401 --every 1 is ~1.5 GB per seed and
#                   is what makes results un-scp-able. Say which was used in the
#                   run's RUNS_Continuation.md section.
#   KEEP_DB         set to 0 to leave the sqlite databases out of the tarball
#                   (drops exploitability, keeps the films)
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
NAME="14-eom-critic-evolution"
DATA="$OUT/data/$NAME"
LOGS="$OUT/logs/$NAME"
EXPORTS="$OUT/exports"
TRIALS="$LOGS/trials.txt"

RUNNER="$BENCH/real_matd3/eom_critic_film.py"
FIGURES="$BENCH/analysis/eom_critic_evolution.py"

CASES="${CASES:-sb02a sb02b sb02c}"
SEEDS="${SEEDS:-42 1 2}"
GRID="${GRID:-201}"
NOBS="${NOBS:-4}"
EVERY="${EVERY:-4}"

PARTITION="${PARTITION:-cpu_il}"
WALLTIME="${WALLTIME:-06:00:00}"
MEM="${MEM:-8G}"
CPUS="${CPUS:-1}"
KEEP_DB="${KEEP_DB:-1}"

# ---------------------------------------------------------------- preflight

preflight() {
    [[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set ASSUME_PYTHON)" >&2; exit 1; }
    [[ -f "$RUNNER" ]] || { echo "no runner at $RUNNER" >&2; exit 1; }

    # The inc-dec reward shaping is a source edit and fires unconditionally once
    # uncommented -- it would silently reshape these EOM runs too.
    if grep -Eq '^ {8}if reward > 0:' "$REPO/assume/strategies/learning_strategies.py"; then
        echo "PREFLIGHT: the reward shaping at learning_strategies.py:1583 is UNCOMMENTED." >&2
        echo "Comment it back out before running anything on the true reward." >&2
        exit 1
    fi

    # the runner's own preflight() resolves each case to its scenario folder
    # and checks the config exists; this only catches a typo'd case name early,
    # before 9 tasks queue up and all fail identically
    "$PYTHON" -c "
import sys
sys.path.insert(0, '$BENCH/real_matd3')
sys.path.insert(0, '$BENCH')
from eom_critic_film import CASES
bad = [c for c in '''$CASES'''.split() if c not in CASES]
if bad:
    sys.exit(f'PREFLIGHT: unknown case(s) {bad}; known: {list(CASES)}')
" || exit 1

    # sbatch does not create missing --output directories.
    mkdir -p "$LOGS" "$DATA" "$EXPORTS"

    # this file is submitted as the batch script for both the array and the
    # collector; a checkout that lost the exec bit would fail at sbatch time
    chmod +x "${BASH_SOURCE[0]}" 2>/dev/null || true
}

# ------------------------------------------------------------------- submit

if [[ "$MODE" == "submit" ]]; then
    preflight

    : > "$TRIALS"
    for c in $CASES; do
        for seed in $SEEDS; do
            echo "$c $seed" >> "$TRIALS"
        done
    done
    N=$(wc -l < "$TRIALS")
    ARRAY="1-$N"
    [[ -n "${MAX_PARALLEL:-}" ]] && ARRAY="$ARRAY%$MAX_PARALLEL"

    echo
    echo "  repo    : $REPO"
    echo "  python  : $PYTHON"
    echo "  trials  : $N  ($TRIALS)"
    echo "  recorder: grid $GRID, $NOBS observations, every $EVERY blocks"
    echo "  data    : $DATA"
    echo "  logs    : $LOGS"
    echo

    JOB=$(sbatch --parsable \
        --job-name=eom_film \
        --partition="$PARTITION" \
        --time="$WALLTIME" \
        --nodes=1 --ntasks=1 --cpus-per-task="$CPUS" --mem="$MEM" \
        --chdir="$REPO" \
        --array="$ARRAY" \
        --output="$LOGS/eom-%A_%a.out" \
        --error="$LOGS/eom-%A_%a.err" \
        --export=ALL,MODE=trial,ASSUME_REPO="$REPO",ASSUME_BENCH="$BENCH" \
        "${BASH_SOURCE[0]}")
    echo "  array job : $JOB"

    COLLECT=$(sbatch --parsable \
        --job-name=eom_collect \
        --partition="$PARTITION" \
        --time=01:00:00 \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G \
        --chdir="$REPO" \
        --dependency=afterany:"$JOB" \
        --output="$LOGS/eom-collect-%j.out" \
        --error="$LOGS/eom-collect-%j.err" \
        --export=ALL,MODE=collect,ARRAY_JOB="$JOB",ASSUME_REPO="$REPO",ASSUME_BENCH="$BENCH" \
        "${BASH_SOURCE[0]}")
    echo "  collect   : $COLLECT (runs after the array, whatever it does)"
    echo
    echo "  watch     : squeue -j $JOB,$COLLECT"
    echo "  accounting: sacct -j $JOB --format=JobID,State,ExitCode,Elapsed,MaxRSS"
    echo
    echo "  when it is done, from your laptop:"
    echo "    scp <cluster>:$EXPORTS/eom_${JOB}.tar.gz ."
    echo
    exit 0
fi

# -------------------------------------------------------------------- trial

if [[ "$MODE" == "trial" ]]; then
    # fail with the cause rather than with sed's "no such file": a wrong REPO
    # makes every path here wrong at once
    [[ -f "$RUNNER" ]] || { echo "no runner at $RUNNER -- ASSUME_REPO/ASSUME_BENCH did not survive the submit (got REPO=$REPO)" >&2; exit 1; }
    [[ -f "$TRIALS" ]] || { echo "no trial list at $TRIALS" >&2; exit 1; }

    TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TRIALS")
    [[ -n "$TRIAL" ]] || { echo "no trial on line $SLURM_ARRAY_TASK_ID of $TRIALS" >&2; exit 1; }
    CASE=${TRIAL% *}
    SEED=${TRIAL#* }

    echo "=========================================="
    echo "case      : $CASE"
    echo "seed      : $SEED"
    echo "task      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
    echo "started   : $(date -Is)"
    echo "=========================================="

    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

    ARGS=(--cases "$CASE" --seeds "$SEED" --workers 1 --threads 1
          --grid "$GRID" --n-obs "$NOBS" --every "$EVERY" --out-dir "$DATA")
    [[ -n "${EPISODES:-}" ]] && ARGS+=(--episodes "$EPISODES")

    "$PYTHON" "$RUNNER" "${ARGS[@]}"

    echo "finished  : $(date -Is)"
    exit 0
fi

# ------------------------------------------------------------------ collect

if [[ "$MODE" == "collect" ]]; then
    STAMP="${ARRAY_JOB:-$(date +%Y%m%d-%H%M%S)}"
    REPORT="$LOGS/report_${STAMP}.txt"

    echo "  regenerating the report and the figures"
    # --report-only re-reads the archived .npz files and never retrains, so an
    # incomplete batch reports what it has and names what is missing.
    "$PYTHON" "$RUNNER" --report-only \
        --cases $CASES --seeds $SEEDS --out-dir "$DATA" \
        2>&1 | tee "$REPORT" || true
    "$PYTHON" "$FIGURES" \
        --cases $CASES --seeds $SEEDS \
        --data-dir "$DATA" --img-dir "$OUT/img" \
        2>&1 | tee -a "$REPORT" || true

    if command -v sacct >/dev/null && [[ -n "${ARRAY_JOB:-}" ]]; then
        sacct -j "$ARRAY_JOB" \
            --format=JobID,JobName%18,State,ExitCode,Elapsed,MaxRSS \
            > "$LOGS/sacct_${STAMP}.txt" || true
    fi

    TARBALL="$EXPORTS/eom_${STAMP}.tar.gz"
    mkdir -p "$EXPORTS"
    EXCLUDES=(--exclude='scratch/*/learned_strategies')
    if [[ "$KEEP_DB" == "0" ]]; then
        EXCLUDES+=(--exclude='*.db' --exclude='*.db-journal')
    else
        # the sqlite files carry rl_exploitability, which is the second reading
        # these runs exist for -- keep them, drop only the journals
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
