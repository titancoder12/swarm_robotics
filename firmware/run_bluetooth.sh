#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT_DIR="$REPO_ROOT/checkpoints"
ROBOT_PORT="/dev/ttyUSB0"
ROBOT_ID="robot_0"
BLE_DEVICE_NAME=""
BLE_SERVICE_UUID="6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_WRITE_CHAR_UUID="6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_NOTIFY_CHAR_UUID="6E400003-B5A3-F393-E0A9-E50E24DCCA9E"
BLE_TIMEOUT="1.0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-id)
      ROBOT_ID="$2"
      shift 2
      ;;
    --port)
      ROBOT_PORT="$2"
      shift 2
      ;;
    --ble-device-name)
      BLE_DEVICE_NAME="$2"
      shift 2
      ;;
    --ble-service-uuid)
      BLE_SERVICE_UUID="$2"
      shift 2
      ;;
    --ble-write-char-uuid)
      BLE_WRITE_CHAR_UUID="$2"
      shift 2
      ;;
    --ble-notify-char-uuid)
      BLE_NOTIFY_CHAR_UUID="$2"
      shift 2
      ;;
    --ble-timeout)
      BLE_TIMEOUT="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --help|-h)
      cat <<'EOF'
Usage: bash firmware/run_bluetooth.sh [launcher flags] [-- run.py args]

Launcher flags:
  --robot-id ID
  --port DEVICE
  --ble-device-name NAME
  --ble-service-uuid UUID
  --ble-write-char-uuid UUID
  --ble-notify-char-uuid UUID
  --ble-timeout SECONDS
  --checkpoint-dir DIR

Examples:
  bash firmware/run_bluetooth.sh
  bash firmware/run_bluetooth.sh --robot-id robot_1
  bash firmware/run_bluetooth.sh --robot-id robot_1 --port /dev/ttyUSB1
  bash firmware/run_bluetooth.sh --robot-id robot_1 --ble-device-name robot_1_ble
  bash firmware/run_bluetooth.sh --robot-id robot_1 -- --max-steps 20
EOF
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$BLE_DEVICE_NAME" ]]; then
  BLE_DEVICE_NAME="$ROBOT_ID"
fi

cd "$REPO_ROOT"

python firmware/run.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --port "$ROBOT_PORT" \
  --robot-id "$ROBOT_ID" \
  --cc-ble-enable \
  --cc-ble-device-name "$BLE_DEVICE_NAME" \
  --cc-ble-service-uuid "$BLE_SERVICE_UUID" \
  --cc-ble-write-char-uuid "$BLE_WRITE_CHAR_UUID" \
  --cc-ble-notify-char-uuid "$BLE_NOTIFY_CHAR_UUID" \
  --cc-ble-timeout "$BLE_TIMEOUT" \
  --debug \
  "${EXTRA_ARGS[@]}"
