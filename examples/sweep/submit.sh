#!/bin/bash
# One-command sweep launcher: expand variants, then submit the array job.
#
# Usage: bash submit.sh [sweep.yaml]
#        bash examples/sweep/submit.sh examples/sweep/sweep.yaml

set -euo pipefail

SWEEP_FILE="${1:-sweep.yaml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$SWEEP_FILE" ]]; then
    echo "Sweep file not found: $SWEEP_FILE" >&2
    exit 1
fi

# 1) Generate variants
python "$SCRIPT_DIR/expand.py" --sweep "$SWEEP_FILE"

# 2) Extract manifest path + concurrency cap from sweep file
MANIFEST=$(python -c "
import yaml, sys
s = yaml.safe_load(open('$SWEEP_FILE'))
print(s['output']['manifest'])
")

MAX_PAR=$(python -c "
import yaml
s = yaml.safe_load(open('$SWEEP_FILE'))
mp = s.get('max_parallel')
print(mp if isinstance(mp, int) and mp > 0 else '')
")

N=$(($(wc -l < "$MANIFEST") - 1))

if [[ "$N" -lt 1 ]]; then
    echo "No variants in manifest: $MANIFEST" >&2
    exit 1
fi

ARRAY_SPEC="1-$N"
if [[ -n "$MAX_PAR" ]]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_PAR}"
fi

# 3) Submit
echo
echo "Submitting array job: --array=$ARRAY_SPEC"
sbatch --array="$ARRAY_SPEC" "$SCRIPT_DIR/run_array.sh" "$MANIFEST"