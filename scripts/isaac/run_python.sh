#!/usr/bin/env bash
# Keep Isaac's Python 3.11 independent of system ROS Humble's Python 3.10.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${OPTI_ISAAC_ENV:-$ROOT/.envs/isaacsim}"
if [[ ! -x "$PREFIX/bin/python" ]]; then
    echo "Isaac environment missing. Run scripts/isaac/bootstrap_conda.sh and install_dependencies.sh." >&2
    exit 1
fi
unset PYTHONPATH PYTHONHOME AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH ROS_PACKAGE_PATH
unset LD_LIBRARY_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export PYTHONNOUSERSITE=1
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1
export XDG_CACHE_HOME="$ROOT/.cache/isaac"
export CUDA_CACHE_PATH="$ROOT/.cache/isaac/cuda"
for argument in "$@"; do
    if [[ "$argument" == "--ros2" || "$argument" == "--check-ros-clock" ]]; then
        export OPTI_ISAAC_ROS2=1
    fi
done
if [[ "${OPTI_ISAAC_ROS2:-0}" == "1" ]]; then
    export ROS_DISTRO=humble
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
    export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-172}"
    export ROS_LOCALHOST_ONLY=1
    export LD_LIBRARY_PATH="$PREFIX/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble/lib"
fi
mkdir -p "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
cd "$ROOT"
exec "$PREFIX/bin/python" "$@"
