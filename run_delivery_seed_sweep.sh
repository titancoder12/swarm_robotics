#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 6 ]]; then
  echo "Usage: bash run_delivery_seed_sweep.sh <checkpoint_dir> <seed_start> [seed_end] [episodes_per_seed] [output_dir] [python_bin]"
  echo "Example: bash run_delivery_seed_sweep.sh checkpoints/mappo_f/latest 0 49 1 runs/seed_sweeps .venv/bin/python"
  exit 1
fi

CHECKPOINT_DIR="$1"
SEED_START="$2"
SEED_END="${3:-$SEED_START}"
EPISODES_PER_SEED="${4:-1}"
OUTPUT_DIR="${5:-runs/seed_sweeps}"
PYTHON_BIN="${6:-.venv/bin/python}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

METADATA_PATH="$CHECKPOINT_DIR/metadata.json"
if [[ ! -f "$METADATA_PATH" ]]; then
  echo "Checkpoint metadata not found: $METADATA_PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$OUTPUT_DIR/delivery_seed_sweep_${TIMESTAMP}"
RAW_DIR="$RUN_DIR/raw"
mkdir -p "$RAW_DIR"

read_metadata_field() {
  local field="$1"
  "$PYTHON_BIN" - <<'PY' "$METADATA_PATH" "$field"
import json
import sys

metadata_path = sys.argv[1]
field = sys.argv[2]
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)
value = metadata.get(field, "")
print(value)
PY
}

N_AGENTS="$(read_metadata_field "n_agents")"
ACTIVE_TARGETS="$(read_metadata_field "active_targets")"
MAX_STEPS="$(read_metadata_field "max_steps")"

SUMMARY_CSV="$RUN_DIR/deliveries_by_seed.csv"
{
  echo "seed,episodes,food_delivered,food_picked_up,food_discovered,pheromone_usage,food_source_respawns,food_units_remaining,episode_length,checkpoint_dir"
} > "$SUMMARY_CSV"

echo "[seed_sweep] checkpoint=$CHECKPOINT_DIR"
echo "[seed_sweep] n_agents=$N_AGENTS active_targets=$ACTIVE_TARGETS max_steps=$MAX_STEPS"
echo "[seed_sweep] seeds=${SEED_START}..${SEED_END} episodes_per_seed=$EPISODES_PER_SEED"
echo "[seed_sweep] run_dir=$RUN_DIR"

for (( seed=SEED_START; seed<=SEED_END; seed++ )); do
  FILENAME="seed_${seed}"
  "$PYTHON_BIN" analysis/evaluate.py \
    --policy-kind mappo_gru \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --n-agents "$N_AGENTS" \
    --episodes "$EPISODES_PER_SEED" \
    --seed "$seed" \
    --headless \
    --output-dir "$RAW_DIR" \
    --filename "$FILENAME" \
    --eval-steps "$MAX_STEPS" \
    --active-targets "$ACTIVE_TARGETS"

  METRICS_CSV="$RAW_DIR/${FILENAME}_eval_metrics.csv"
  "$PYTHON_BIN" - <<'PY' "$METRICS_CSV" "$SUMMARY_CSV" "$seed" "$EPISODES_PER_SEED" "$CHECKPOINT_DIR"
import csv
import sys

metrics_csv = sys.argv[1]
summary_csv = sys.argv[2]
seed = int(sys.argv[3])
episodes = int(sys.argv[4])
checkpoint_dir = sys.argv[5]

rows = []
with open(metrics_csv, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def mean_float(key: str) -> float:
    if not rows:
        return 0.0
    values = [float(row.get(key, 0.0) or 0.0) for row in rows]
    return sum(values) / len(values)

food_delivered = mean_float("food_delivered")
food_picked_up = mean_float("food_picked_up")
food_discovered = mean_float("food_discovered")
pheromone_usage = mean_float("pheromone_usage")
food_source_respawns = mean_float("food_source_respawns")
food_units_remaining = mean_float("food_units_remaining")
episode_length = mean_float("episode_length")

with open(summary_csv, "a", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        seed,
        episodes,
        food_delivered,
        food_picked_up,
        food_discovered,
        pheromone_usage,
        food_source_respawns,
        food_units_remaining,
        episode_length,
        checkpoint_dir,
    ])

print(
    f"[seed_sweep] seed={seed} delivered={food_delivered:.3f} "
    f"picked_up={food_picked_up:.3f} respawns={food_source_respawns:.3f}"
)
PY
done

echo "[seed_sweep] complete summary_csv=$SUMMARY_CSV"
