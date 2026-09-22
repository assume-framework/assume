#!/bin/bash
# Progress of the 384-scenario full-year batch (examples/run_full_batch.py).
cd "$(dirname "$0")/.." || exit 1
LOG=examples/outputs/full_batch.log
DONE=$(grep -cE '^\[[0-9]+/384\]' "$LOG" 2>/dev/null); DONE=${DONE:-0}
OK=$(grep -cE '^\[[0-9]+/384\] +OK' "$LOG" 2>/dev/null); OK=${OK:-0}
FAIL=$((DONE - OK))
WORKERS=$(ps aux | grep -c '[r]un_full_batch.py --workers\|multiprocessing-fork')

echo "=== full batch: $DONE/384 done ($OK ok, $FAIL failed) ==="
free -g | awk 'NR==2{printf "RAM %s/%s GB used\n", $3, $2}'
df -h /root | tail -1 | awk '{print "disk:", $3"/"$2, "used,", $4, "free"}'
echo

if [ "$FAIL" -gt 0 ]; then
  echo "--- failures ---"
  grep -E '^\[[0-9]+/384\] +FAILED' "$LOG" | tail -10
  echo
fi

echo "--- compressed archives so far ---"
find examples/outputs -maxdepth 2 -name "*.tar.gz" 2>/dev/null | wc -l
du -sh examples/outputs/*_RN* examples/outputs/WCMean* examples/outputs/WCTail* 2>/dev/null | tail -20

echo
echo "--- last 8 completed ---"
tail -8 "$LOG"
