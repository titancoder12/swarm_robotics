#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT_DIR="$REPO_ROOT/checkpoints"
ROBOT_PORT="/dev/ttyUSB0"
ROBOT_ID="robot_0"
RELAY_URL="https://relay.christopherlin.ca"
RELAY_SESSION=""
RELAY_TIMEOUT="1.0"
CAMERA_ENABLE=0
CAMERA_INDEX="0"
CAMERA_WIDTH="640"
CAMERA_HEIGHT="480"
CAMERA_HORIZONTAL_FOV_DEG="62.0"
CAMERA_TARGET_WIDTH_CM="6.0"
CAMERA_MIN_AREA_PX="400"
CAMERA_HSV_LOWER="20,120,120"
CAMERA_HSV_UPPER="40,255,255"
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
    --relay-url)
      RELAY_URL="$2"
      shift 2
      ;;
    --relay-session)
      RELAY_SESSION="$2"
      shift 2
      ;;
    --relay-timeout)
      RELAY_TIMEOUT="$2"
      shift 2
      ;;
    --checkpoint-dir)
      CHECKPOINT_DIR="$2"
      shift 2
      ;;
    --camera-enable)
      CAMERA_ENABLE=1
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
Usage: bash firmware/run_relay.sh [launcher flags] [-- run.py args]

Launcher flags:
  --robot-id ID
  --port DEVICE
  --relay-url URL
  --relay-session SESSION
  --relay-timeout SECONDS
  --checkpoint-dir DIR
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
  bash firmware/run_relay.sh
  bash firmware/run_relay.sh --robot-id robot_1
  bash firmware/run_relay.sh --robot-id robot_1 --port /dev/ttyUSB1
  bash firmware/run_relay.sh --camera-enable
  bash firmware/run_relay.sh --camera-enable --camera-hsv-lower 20,120,120 --camera-hsv-upper 40,255,255
  bash firmware/run_relay.sh --robot-id robot_1 -- --max-steps 20
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

if [[ -z "$RELAY_SESSION" ]]; then
  RELAY_SESSION="$ROBOT_ID"
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
  --cc-relay-url "$RELAY_URL" \
  --cc-relay-session "$RELAY_SESSION" \
  --cc-relay-timeout "$RELAY_TIMEOUT" \
  "${CAMERA_ARGS[@]}" \
  --debug \
  "${EXTRA_ARGS[@]}"
