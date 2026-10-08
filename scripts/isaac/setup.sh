#!/usr/bin/env bash
# One-time setup. Run this from the extracted runtime source package.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
bash "$ROOT/scripts/isaac/bootstrap_conda.sh"
bash "$ROOT/scripts/isaac/install_dependencies.sh"
bash "$ROOT/scripts/isaac/fetch_robot.sh"
bash "$ROOT/scripts/isaac/install_command_client.sh"
printf '\nSetup finished. Start with: bash scripts/isaac/run_demo.sh\n'
