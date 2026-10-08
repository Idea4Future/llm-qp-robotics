#!/usr/bin/env bash
# Bootstrap only. Run install_dependencies.sh after this finishes.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFIX="${OPTI_ISAAC_ENV:-$ROOT/.envs/isaacsim}"
CONDA_BIN="${OPTI_CONDA_EXE:-${CONDA_EXE:-$HOME/anaconda3/bin/conda}}"
export PYTHONNOUSERSITE=1
if [[ ! -x "$CONDA_BIN" ]]; then
    CONDA_BIN="$(command -v conda || true)"
fi
if [[ -z "$CONDA_BIN" || ! -x "$CONDA_BIN" ]]; then
    echo "Install Miniconda/Anaconda first, or set OPTI_CONDA_EXE." >&2
    exit 1
fi
mkdir -p "$ROOT/.cache/conda-pkgs" "$ROOT/logs/isaac_setup"
export CONDA_PKGS_DIRS="$ROOT/.cache/conda-pkgs"
if [[ -x "$PREFIX/bin/python" ]]; then
    "$PREFIX/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)'
else
    if [[ -f "$ROOT/configs/isaac/conda-linux-64.explicit.txt" ]]; then
        "$CONDA_BIN" create --yes --prefix "$PREFIX" --file "$ROOT/configs/isaac/conda-linux-64.explicit.txt"
    else
        "$CONDA_BIN" create --yes --prefix "$PREFIX" --override-channels --channel conda-forge python=3.11 pip
    fi
fi
"$PREFIX/bin/python" -m pip --version
