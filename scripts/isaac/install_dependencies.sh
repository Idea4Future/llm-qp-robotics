#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${OPTI_ISAAC_ENV:-$ROOT/.envs/isaacsim}"
PY="$PREFIX/bin/python"
LAB="$ROOT/third_party/IsaacLab"
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$ROOT/.cache/pip-isaac"
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$ROOT/logs/isaac_setup" "$ROOT/.cache/pip-isaac"
CONSTRAINTS=()
if [[ -f "$ROOT/configs/isaac/pip_requirements.lock.txt" ]]; then
    CONSTRAINTS=(-c "$ROOT/configs/isaac/pip_requirements.lock.txt")
fi
"$PY" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
"$PY" -m pip install 'pip==25.2' 'setuptools==80.9.0' 'wheel==0.45.1'
"$PY" -m pip install "${CONSTRAINTS[@]}" 'torch==2.7.0' 'torchvision==0.22.0' --index-url https://download.pytorch.org/whl/cu128 --report "$ROOT/logs/isaac_setup/torch-install.json"
"$PY" -m pip install "${CONSTRAINTS[@]}" 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com --report "$ROOT/logs/isaac_setup/sim-install.json"
if [[ ! -d "$LAB/.git" ]]; then
    git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git "$LAB"
fi
git -C "$LAB" describe --tags --exact-match HEAD | grep -Fx 'v2.3.0'
[[ "$(git -C "$LAB" rev-parse HEAD)" == "3c6e67bb5c7ada942a6d1884ab69338f57596f77" ]] || { echo "Isaac Lab source revision differs from the pinned version" >&2; exit 1; }
"$PY" -m pip install "${CONSTRAINTS[@]}" --no-build-isolation -c "$ROOT/configs/isaac/compatibility.constraints.txt" -e "$LAB/source/isaaclab" 'click==8.1.7' 'idna==3.10' --report "$ROOT/logs/isaac_setup/lab-install.json"
"$PY" -m pip install "${CONSTRAINTS[@]}" --no-build-isolation -c "$ROOT/configs/isaac/compatibility.constraints.txt" -e "$LAB/source/isaaclab_assets" -e "$LAB/source/isaaclab_tasks" -e "$LAB/source/isaaclab_rl" --report "$ROOT/logs/isaac_setup/lab-extensions-install.json"
# Sim's older OSQP pin is part of this tested dependency intersection. Installing
# a current CVXPY without this pin may silently replace the Kit-compatible solver.
"$PY" -m pip install "${CONSTRAINTS[@]}" 'cvxpy==1.6.5' 'osqp==0.6.7.post3' --report "$ROOT/logs/isaac_setup/qp-install.json"
"$PY" -m pip check
"$PY" -m pip freeze --all > "$ROOT/logs/isaac_setup/pip-freeze.txt"
git -C "$LAB" rev-parse HEAD > "$ROOT/logs/isaac_setup/isaaclab-commit.txt"
