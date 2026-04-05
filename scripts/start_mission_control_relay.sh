#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Missing .venv in $REPO_ROOT" >&2
  exit 1
fi

source .venv/bin/activate

RELAY_URL="${MISSION_CONTROL_RELAY_URL:-https://relay.christopherlin.ca}"
SESSION_ID="${MISSION_CONTROL_RELAY_SESSION:-robot_0}"
TCP_HOST="${MISSION_CONTROL_TCP_HOST:-127.0.0.1}"
TCP_PORT="${MISSION_CONTROL_TCP_PORT:-8765}"
MC_LOG_LEVEL="${MISSION_CONTROL_LOG_LEVEL:-INFO}"
BRIDGE_DEBUG="${MISSION_CONTROL_BRIDGE_DEBUG:-1}"

MC_PID=""
BRIDGE_PID=""

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ -n "${BRIDGE_PID}" ]] && kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    kill "${BRIDGE_PID}" 2>/dev/null || true
    wait "${BRIDGE_PID}" 2>/dev/null || true
  fi

  if [[ -n "${MC_PID}" ]] && kill -0 "${MC_PID}" 2>/dev/null; then
    kill "${MC_PID}" 2>/dev/null || true
    wait "${MC_PID}" 2>/dev/null || true
  fi

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

echo "Starting Mission Control on ${TCP_HOST}:${TCP_PORT}"
python -m mission_control.main \
  --tcp-host "${TCP_HOST}" \
  --tcp-port "${TCP_PORT}" \
  --log-level "${MC_LOG_LEVEL}" &
MC_PID=$!

sleep 1

BRIDGE_ARGS=(
  -m mission_control.relay_bridge
  --relay-url "${RELAY_URL}"
  --session "${SESSION_ID}"
  --tcp-host "${TCP_HOST}"
  --tcp-port "${TCP_PORT}"
)

if [[ "${BRIDGE_DEBUG}" != "0" ]]; then
  BRIDGE_ARGS+=(--debug)
fi

echo "Starting relay bridge for session ${SESSION_ID} via ${RELAY_URL}"
python "${BRIDGE_ARGS[@]}" &
BRIDGE_PID=$!

while true; do
  if ! kill -0 "${MC_PID}" 2>/dev/null; then
    echo "Mission Control exited; stopping relay bridge"
    break
  fi
  if ! kill -0 "${BRIDGE_PID}" 2>/dev/null; then
    echo "Relay bridge exited; stopping Mission Control"
    break
  fi
  sleep 1
done

cleanup
