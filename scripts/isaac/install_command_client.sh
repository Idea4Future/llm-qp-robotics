#!/usr/bin/env bash
# The planning/HTTP client uses only the Python standard library, separately from Kit.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CLIENT_PYTHON="${OPTI_COMMAND_PYTHON:-/usr/bin/python3}"
CLIENT_ENV="$ROOT/.venv"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
"$CLIENT_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use system Python 3.10 for the command client."'
if [[ ! -x "$CLIENT_ENV/bin/python" ]]; then
    "$CLIENT_PYTHON" -m venv "$CLIENT_ENV"
fi
# Do not reinstall the old MuJoCo dependencies or alter an existing valid venv.
"$CLIENT_ENV/bin/python" - "$ROOT" <<'PY'
import sys
from pathlib import Path
assert sys.version_info[:2] == (3, 10), "Existing .venv must be Python 3.10."
root = Path(sys.argv[1])
sys.path.insert(0, str(root / "src"))
from opti_robot.command_executor import extract_supported_intent
from opti_robot.task_proposal import TypedThinkingLocalPlanner
from opti_robot.nav2_adapter import request_factory_path
from opti_web.server import ChatServer
assert extract_supported_intent("A 입고대의 빨간 용기를 B 조립대 첫 번째 자리에 가져다 놓아줘.")["object_id"] == "redbin01"
print("Command client ready:", sys.executable)
print("No model, GPU, ROS node or Isaac engine was launched.")
PY
