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

Examples:
  bash firmware/run_relay.sh
  bash firmware/run_relay.sh --robot-id robot_1
  bash firmware/run_relay.sh --robot-id robot_1 --port /dev/ttyUSB1
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

python firmware/run.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --port "$ROBOT_PORT" \
  --robot-id "$ROBOT_ID" \
  --cc-relay-url "$RELAY_URL" \
  --cc-relay-session "$RELAY_SESSION" \
  --cc-relay-timeout "$RELAY_TIMEOUT" \
  --debug \
  "${EXTRA_ARGS[@]}"
