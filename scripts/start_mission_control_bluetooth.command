#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/christopherlin/dev/cwsf2026/sim"
cd "$REPO_ROOT"

source .venv/bin/activate

python -m mission_control.main \
  --ble-enable \
  --ble-device-name robot_0 \
  --ble-timeout 5.0 \
  --log-level INFO
