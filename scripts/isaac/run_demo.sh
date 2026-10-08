#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT="$ROOT/runs/isaac_$(date +%Y%m%d_%H%M%S)_$$"
for argument in "$@"; do
    if [[ "$argument" == "--help" || "$argument" == "-h" ]]; then
        exec bash "$ROOT/scripts/isaac/run_python.sh" "$ROOT/scripts/isaac/run_robot.py" --output "$OUTPUT" --help
    fi
    if [[ "$argument" == "--output" || "$argument" == --output=* ]]; then
        echo "run_demo.sh chooses a fresh runs/ output directory. Use run_python.sh run_robot.py for a custom --output." >&2
        exit 2
    fi
done
echo "Run directory: $OUTPUT"
bash "$ROOT/scripts/isaac/run_python.sh" "$ROOT/scripts/isaac/run_robot.py" --output "$OUTPUT" --seconds 30 --realtime "$@"
python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print("Demo:", "PASS" if r.get("passed") else "FAIL"); sys.exit(0 if r.get("passed") else 1)' "$OUTPUT/summary.json"
