#!/bin/bash
#SBATCH --job-name=assume_sweep
#SBATCH --partition=cpu_il        # cpu_il | cpu | highmem. NO GPU IS NEEDED:
                                  # ASSUME learning is two small MLPs against a
                                  # CPU-bound simulator; a GPU only queues longer.
#SBATCH --time=00:30:00                 # walltime hh:mm:ss
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# #SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --chdir=/pfs/work9/workspace/scratch/fr_fr1096-finn_basic/code/ADAPT/assume
#SBATCH --output=examples/outputs/slurm/sweep-%A_%a.out
#SBATCH --error=examples/outputs/slurm/sweep-%A_%a.err
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=you@example.com

# ----------------------------------------------------------------------------
# Usage: sbatch --array=1-N[%K] run_array.sh <manifest.tsv>
#   N = number of variants (use submit.sh to compute automatically)
#   K = optional cap on concurrent tasks (polite on small partitions)
#
# The #SBATCH lines above are FALLBACK DEFAULTS for a bare `sbatch run_array.sh`.
# submit.sh overrides partition, time, cpus, mem and both log paths from the
# sweep file's `slurm:` and `output:` blocks, so this file rarely needs editing
# -- with one exception: --chdir must point at your repo checkout on the cluster.
#
# Logs land in two places, both beside the outputs they describe:
#   <scenarios_dir>/logs/sweep-<jobid>_<task>.{out,err}   the SLURM streams
#   <variant_dir>/logs/run.log                            this variant's own log
# ----------------------------------------------------------------------------

set -euo pipefail

MANIFEST="${1:?Usage: sbatch --array=1-N run_array.sh <manifest.tsv>}"

if [[ ! -f "$MANIFEST" ]]; then
    echo "Manifest not found: $MANIFEST" >&2
    exit 1
fi

# Pull the row for THIS array task. Match on the array_idx column for safety.
ROW=$(awk -F'\t' -v idx="$SLURM_ARRAY_TASK_ID" \
      'NR > 1 && $1 == idx { print; exit }' "$MANIFEST")

if [[ -z "$ROW" ]]; then
    echo "No manifest row for array_idx=$SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

# Columns: array_idx \t run_id \t scenario_dir \t study_case \t <params...>
RUN_ID=$(echo "$ROW"        | cut -f2)
SCENARIO_DIR=$(echo "$ROW"  | cut -f3)
STUDY_CASE=$(echo "$ROW"    | cut -f4)

# This variant's own log, next to its config.yaml and assume_db.db.
RUN_LOG_DIR="$SCENARIO_DIR/logs"
mkdir -p "$RUN_LOG_DIR"
RUN_LOG="$RUN_LOG_DIR/run.log"

echo "=========================================="
echo "Array task : $SLURM_ARRAY_TASK_ID"
echo "Run ID     : $RUN_ID"
echo "Scenario   : $SCENARIO_DIR"
echo "Study case : $STUDY_CASE"
echo "Job        : $SLURM_JOB_ID ($SLURM_ARRAY_JOB_ID)"
echo "Node       : $SLURM_JOB_NODELIST"
echo "Started    : $(date -Is)"
echo "GPU        : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Run log    : $RUN_LOG"
echo "=========================================="

# Environment ---------------------------------------------------------------
#module load devel/miniforge/25.3.1-python-3.12

# Initialize conda's shell functions BEFORE calling activate.
# `module load` doesn't set up `activate`/`deactivate` in non-interactive
# shells, and once an env is in CONDA_DEFAULT_ENV, activating a new env
# triggers an internal deactivate that fails with:
#   "CondaError: Run 'conda init' before 'conda deactivate'"
# Sourcing conda.sh from the miniconda3 install that actually owns the env
# avoids any ambiguity between miniforge (loaded above) and miniconda3.
#source "$HOME/miniconda3/etc/profile.d/conda.sh"
#conda activate "$HOME/miniconda3/envs/assume"



export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Run -----------------------------------------------------------------------
#module load devel/miniforge/25.3.1-python-3.12
#source "$HOME/miniconda3/etc/profile.d/conda.sh"
#conda activate "$HOME/miniconda3/envs/assume"
#python examples/sweep/run_simulation.py \

PYTHON="$HOME/miniconda3/envs/assume/bin/python"

# tee into the variant's own log so a run is self-contained: the config, the
# database and the log that produced them all sit in one folder. `pipefail`
# (set above) keeps python's exit status as the task's exit status.
{
    echo "=== $RUN_ID | $STUDY_CASE | job ${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} | $(date -Is) ==="
    "$PYTHON" examples/sweep/run_simulation.py \
        --scenario-dir "$SCENARIO_DIR" \
        --study-case   "$STUDY_CASE"
} 2>&1 | tee -a "$RUN_LOG"

echo "Finished   : $(date -Is)"