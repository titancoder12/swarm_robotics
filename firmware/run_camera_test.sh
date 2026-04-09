#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CAMERA_INDEX="0"
CAMERA_WIDTH="640"
CAMERA_HEIGHT="480"
CAMERA_HORIZONTAL_FOV_DEG="62.0"
CAMERA_TARGET_WIDTH_CM="6.0"
CAMERA_MIN_AREA_PX="100"
CAMERA_HSV_LOWER="40,90,90"
CAMERA_HSV_UPPER="90,255,255"
CAMERA_FPS="2.0"
CAMERA_SHOW="0"
CAMERA_SAVE="0"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --fps)
      CAMERA_FPS="$2"
      shift 2
      ;;
    --show)
      CAMERA_SHOW="1"
      shift
      ;;
    --save)
      CAMERA_SAVE="1"
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage: bash firmware/run_camera_test.sh [options] [-- test_camera.py args]

Defaults:
  continuous terminal output enabled
  green target HSV defaults enabled
  image saving disabled
  preview window disabled

Options:
  --camera-index INDEX
  --camera-width PIXELS
  --camera-height PIXELS
  --camera-horizontal-fov-deg DEGREES
  --camera-target-width-cm CM
  --camera-min-area-px PIXELS
  --camera-hsv-lower H,S,V
  --camera-hsv-upper H,S,V
  --fps FPS
  --show
  --save

Examples:
  bash firmware/run_camera_test.sh
  bash firmware/run_camera_test.sh --show
  bash firmware/run_camera_test.sh --camera-hsv-lower 40,120,120 --camera-hsv-upper 85,255,255
  bash firmware/run_camera_test.sh --fps 4
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

cd "$REPO_ROOT"

SAVE_ARGS=(--no-save)
if [[ "$CAMERA_SAVE" == "1" ]]; then
  SAVE_ARGS=()
fi

SHOW_ARGS=()
if [[ "$CAMERA_SHOW" == "1" ]]; then
  SHOW_ARGS=(--show)
fi

python firmware/test_camera.py \
  --continuous \
  --camera-index "$CAMERA_INDEX" \
  --camera-width "$CAMERA_WIDTH" \
  --camera-height "$CAMERA_HEIGHT" \
  --camera-horizontal-fov-deg "$CAMERA_HORIZONTAL_FOV_DEG" \
  --camera-target-width-cm "$CAMERA_TARGET_WIDTH_CM" \
  --camera-min-area-px "$CAMERA_MIN_AREA_PX" \
  --camera-hsv-lower "$CAMERA_HSV_LOWER" \
  --camera-hsv-upper "$CAMERA_HSV_UPPER" \
  --fps "$CAMERA_FPS" \
  "${SAVE_ARGS[@]}" \
  "${SHOW_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
