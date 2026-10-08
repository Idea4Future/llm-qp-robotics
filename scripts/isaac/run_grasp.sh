#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUTPUT="$ROOT/runs/isaac_grasp_$STAMP"
LOG="$ROOT/logs/isaac_grasp_$STAMP.log"
for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        exec bash "$ROOT/scripts/isaac/run_python.sh" "$ROOT/scripts/isaac/run_grasp.py" --output "$OUTPUT" --help
    fi
    if [[ "$argument" == "--output" || "$argument" == --output=* ]]; then
        echo "run_grasp.sh chooses a fresh runs/ output directory." >&2
        exit 2
    fi
done
mkdir -p "$ROOT/logs"
echo "Run directory: $OUTPUT"
echo "Process log: $LOG"
STATUS=0
bash "$ROOT/scripts/isaac/run_python.sh" "$ROOT/scripts/isaac/run_grasp.py" --output "$OUTPUT" --scene warehouse --video "$@" > "$LOG" 2>&1 || STATUS=$?
if [[ "$STATUS" != 0 ]]; then
    tail -n 25 "$LOG"
    exit "$STATUS"
fi
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("Grasp:", "PASS" if r.get("passed") else "FAIL"); print(r.get("error", "")); sys.exit(0 if r.get("passed") else 1)' "$OUTPUT/summary.json"
