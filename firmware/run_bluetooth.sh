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
CONTROL_MODE="policy"
CAMERA_ENABLE=1
CAMERA_INDEX="0"
CAMERA_WIDTH="640"
CAMERA_HEIGHT="480"
CAMERA_HORIZONTAL_FOV_DEG="62.0"
CAMERA_TARGET_WIDTH_CM="6.0"
CAMERA_MIN_AREA_PX="100"
CAMERA_HSV_LOWER="35,70,70"
CAMERA_HSV_UPPER="100,255,255"
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
    --heuristic-control)
      CONTROL_MODE="heuristic"
      shift
      ;;
    --hybrid-control)
      CONTROL_MODE="hybrid"
      shift
      ;;
    --policy-control)
      CONTROL_MODE="policy"
      shift
      ;;
    --camera-enable)
      CAMERA_ENABLE=1
      shift
      ;;
    --no-camera)
      CAMERA_ENABLE=0
      shift
      ;;
    --camera-index)
      CAMERA_INDEX="$2"
      shift 2
      ;;
    --camera-width)
      CAMERA_WIDTH="$2"
      shift 2
      ;;
    --camera-height)
      CAMERA_HEIGHT="$2"
      shift 2
      ;;
    --camera-horizontal-fov-deg)
      CAMERA_HORIZONTAL_FOV_DEG="$2"
      shift 2
      ;;
    --camera-target-width-cm)
      CAMERA_TARGET_WIDTH_CM="$2"
      shift 2
      ;;
    --camera-min-area-px)
      CAMERA_MIN_AREA_PX="$2"
      shift 2
      ;;
    --camera-hsv-lower)
      CAMERA_HSV_LOWER="$2"
      shift 2
      ;;
    --camera-hsv-upper)
      CAMERA_HSV_UPPER="$2"
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
  --heuristic-control
  --hybrid-control
  --policy-control
  --no-camera
  --camera-enable
  --camera-index INDEX
  --camera-width PIXELS
  --camera-height PIXELS
  --camera-horizontal-fov-deg DEGREES
  --camera-target-width-cm CM
  --camera-min-area-px PIXELS
  --camera-hsv-lower H,S,V
  --camera-hsv-upper H,S,V

Examples:
  bash firmware/run_bluetooth.sh
  bash firmware/run_bluetooth.sh --heuristic-control
  bash firmware/run_bluetooth.sh --hybrid-control
  bash firmware/run_bluetooth.sh --no-camera
  bash firmware/run_bluetooth.sh --robot-id robot_1
  bash firmware/run_bluetooth.sh --robot-id robot_1 --port /dev/ttyUSB1
  bash firmware/run_bluetooth.sh --robot-id robot_1 --ble-device-name robot_1_ble
  bash firmware/run_bluetooth.sh --camera-enable
  bash firmware/run_bluetooth.sh --camera-enable --camera-hsv-lower 35,70,70 --camera-hsv-upper 100,255,255
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

CAMERA_ARGS=()
if [[ "$CAMERA_ENABLE" == "1" ]]; then
  CAMERA_ARGS+=(
    --camera-enable
    --camera-index "$CAMERA_INDEX"
    --camera-width "$CAMERA_WIDTH"
    --camera-height "$CAMERA_HEIGHT"
    --camera-horizontal-fov-deg "$CAMERA_HORIZONTAL_FOV_DEG"
    --camera-target-width-cm "$CAMERA_TARGET_WIDTH_CM"
    --camera-min-area-px "$CAMERA_MIN_AREA_PX"
    --camera-hsv-lower "$CAMERA_HSV_LOWER"
    --camera-hsv-upper "$CAMERA_HSV_UPPER"
  )
fi

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
  --control-mode "$CONTROL_MODE" \
  "${CAMERA_ARGS[@]}" \
  --debug \
  "${EXTRA_ARGS[@]}"
