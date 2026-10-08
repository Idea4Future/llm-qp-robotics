#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="$ROOT/third_party/rby1_isaac"
COMMIT=2417a2b2c83bc80b3ad605ab14f4d508d90089a9
if [[ ! -d "$DEST/.git" ]]; then
    mkdir -p "$ROOT/third_party"
    git clone --filter=blob:none --no-checkout https://github.com/RainbowRobotics/rby1-sim-isaac.git "$DEST"
    git -C "$DEST" sparse-checkout init --no-cone
    git -C "$DEST" sparse-checkout set '/README.md' '/src/' '/assets/model_v_1_2_a.usd' '/assets/gripper/rb_gripper/'
    git -C "$DEST" checkout --detach "$COMMIT"
fi
[[ "$(git -C "$DEST" rev-parse HEAD)" == "$COMMIT" ]] || { echo "Unexpected RB-Y1 revision; keep local work intact and inspect it." >&2; exit 1; }
test -s "$DEST/assets/model_v_1_2_a.usd"
test -s "$DEST/assets/gripper/rb_gripper/rb_gripper_left.usd"
test -s "$DEST/assets/gripper/rb_gripper/rb_gripper_right.usd"
python3 "$ROOT/scripts/isaac/verify_sources.py"
printf 'RB-Y1 source ready: %s\n' "$COMMIT"
