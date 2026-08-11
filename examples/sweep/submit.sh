#!/bin/bash
# One-command sweep launcher: expand variants, then submit the array job.
#
# Usage: bash submit.sh [sweep.yaml]
#        bash examples/sweep/submit.sh examples/sweep/sweep.yaml
#
# Everything sbatch needs -- array size, concurrency cap, log paths and the
# per-task resources -- is read from the sweep file, so run_array.sh does not
# have to be edited per sweep. Its #SBATCH lines are only the fallback for a
# bare `sbatch run_array.sh`.

set -euo pipefail

SWEEP_FILE="${1:-sweep.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "Sweep file not found: $SWEEP_FILE" >&2
    exit 1
fi

# 1) Generate variants (also creates the log directories)
python "$SCRIPT_DIR/expand.py" --sweep "$SWEEP_FILE"

# 2) Read manifest path, log dir, concurrency cap and resources from the sweep
#    file in one go. "-" means "not set, leave sbatch to its default".
IFS=$'\t' read -r MANIFEST LOGS_DIR MAX_PAR PARTITION WALLTIME CPUS MEM GRES < <(
python - "$SWEEP_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

s = yaml.safe_load(open(sys.argv[1]))
out = s.get("output") or {}
scen = Path(out["scenarios_dir"])
logs = out.get("logs_dir") or (scen / "logs")
sl = s.get("slurm") or {}
mp = s.get("max_parallel")


def f(v):
    return str(v) if v not in (None, "") else "-"


print(
    "\t".join(
        [
            str(Path(out["manifest"]).resolve()),
            str(Path(logs).resolve()),
            str(mp) if isinstance(mp, int) and mp > 0 else "-",
            f(sl.get("partition")),
            f(sl.get("time")),
            f(sl.get("cpus_per_task")),
            f(sl.get("mem")),
            f(sl.get("gres")),
        ]
    )
)
PY
)

N=$(($(wc -l < "$MANIFEST") - 1))
if [[ "$N" -lt 1 ]]; then
    echo "No variants in manifest: $MANIFEST" >&2
    exit 1
fi

ARRAY_SPEC="1-$N"
if [[ "$MAX_PAR" != "-" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PAR}"
fi

mkdir -p "$LOGS_DIR"

# 3) Build the sbatch argument list
SBATCH_ARGS=(
    "--array=$ARRAY_SPEC"
    "--output=$LOGS_DIR/sweep-%A_%a.out"
    "--error=$LOGS_DIR/sweep-%A_%a.err"
)
[[ "$PARTITION" != "-" ]] && SBATCH_ARGS+=("--partition=$PARTITION")
[[ "$WALLTIME"  != "-" ]] && SBATCH_ARGS+=("--time=$WALLTIME")
[[ "$CPUS"      != "-" ]] && SBATCH_ARGS+=("--cpus-per-task=$CPUS")
[[ "$MEM"       != "-" ]] && SBATCH_ARGS+=("--mem=$MEM")
[[ "$GRES"      != "-" ]] && SBATCH_ARGS+=("--gres=$GRES")

echo
echo "Submitting array job"
echo "  array : $ARRAY_SPEC"
echo "  logs  : $LOGS_DIR/sweep-%A_%a.{out,err}"
echo "  sbatch: ${SBATCH_ARGS[*]}"
echo

sbatch "${SBATCH_ARGS[@]}" "$SCRIPT_DIR/run_array.sh" "$MANIFEST"
