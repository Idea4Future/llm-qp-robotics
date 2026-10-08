#!/usr/bin/env bash
# The local planner uses core Python3.10. Isaac is a separate Python3.11 child.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
    echo "Project .venv is missing; run: bash scripts/isaac/install_command_client.sh" >&2
    exit 1
fi
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/scripts/isaac/command_runtime.py" "$@"
