You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Run the evaluation and comparison graph workflows against the new checkpoint folder layout to verify that:

1. the new `checkpoints/<folder_name>/...` structure works,
2. `full_policy/` checkpoints can be loaded correctly,
3. graphs and CSV outputs are generated successfully.

Goal:
I want a clean validation pass that proves the new checkpoint-folder organization works end to end for evaluation and graph generation.

--------------------------------------------------
PART 1 — TEST THE NEW CHECKPOINT LAYOUT
--------------------------------------------------

Assume training now saves checkpoints like:

```text
checkpoints/
  with_pheremone/
    metadata.json
    1_4_trained/
    1_2_trained/
    3_4_trained/
    full_policy/
  without_pheremone/
    metadata.json
    1_4_trained/
    1_2_trained/
    3_4_trained/
    full_policy/
```

Use the `full_policy/` folders as the evaluation checkpoint inputs.

--------------------------------------------------
PART 2 — SINGLE-MODEL EVALUATION SMOKE TEST
--------------------------------------------------

Run a quick single-model evaluation for each checkpoint to confirm loading works.

Example commands:

```bash
./.venv/bin/python analysis/evaluate.py \
  --checkpoint-dir checkpoints/with_pheremone/full_policy \
  --shared-policy \
  --filename eval_with_pheremone \
  --output-dir runs/eval \
  --headless \
  --episodes 1 \
  --n-agents 6 \
  --eval-steps 50 \
  --active-targets 4 \
  --use-pheromone
```

```bash
./.venv/bin/python analysis/evaluate.py \
  --checkpoint-dir checkpoints/without_pheremone/full_policy \
  --shared-policy \
  --filename eval_without_pheremone \
  --output-dir runs/eval \
  --headless \
  --episodes 1 \
  --n-agents 6 \
  --eval-steps 50 \
  --active-targets 4 \
  --no-use-pheromone
```

Confirm that each run writes:
- `*_eval_metrics.csv`
- `*_eval_summary.json`

--------------------------------------------------
PART 3 — COMPARISON GRAPH TEST
--------------------------------------------------

Run the comparison workflow using the new checkpoint folder layout:

```bash
./.venv/bin/python analysis/evaluate_comparison.py \
  --checkpoint-with-pheromone checkpoints/with_pheremone/full_policy \
  --checkpoint-without-pheromone checkpoints/without_pheremone/full_policy \
  --filename checkpoint_layout_smoke \
  --agent-min 1 \
  --agent-max 3 \
  --agent-step 1 \
  --episodes-per-agent 1 \
  --output-dir experiments/experiment_data \
  --headless \
  --max-steps 50 \
  --active-targets 4 \
  --shared-policy
```

This should test:
- checkpoint loading from the new folder layout
- per-condition raw CSV generation
- summary CSV generation
- graph generation
- exploration graph generation

--------------------------------------------------
PART 4 — WHAT TO VERIFY
--------------------------------------------------

Check that these exist after the run:

Top-level combined outputs:
- `experiments/experiment_data/raw/checkpoint_layout_smoke_all_conditions_raw.csv`
- `experiments/experiment_data/raw/checkpoint_layout_smoke_summary.csv`
- `experiments/experiment_data/graphs/PNG/checkpoint_layout_smoke_agents_vs_targets_collected.png`
- `experiments/experiment_data/graphs/PDF/checkpoint_layout_smoke_agents_vs_targets_collected.pdf`

Per-condition outputs:
- `experiments/experiment_data/trained_pheremone_use_pheremone/raw/checkpoint_layout_smoke_raw.csv`
- `experiments/experiment_data/trained_pheremone_no_use_pheremone/raw/checkpoint_layout_smoke_raw.csv`
- `experiments/experiment_data/trained_without_pheremone/raw/checkpoint_layout_smoke_raw.csv`
- `experiments/experiment_data/random_walk/raw/checkpoint_layout_smoke_raw.csv`

Also verify:
- `graphs/PNG/`
- `graphs/PDF/`
- `exploration_graphs/PNG/`
- `exploration_graphs/PDF/`

inside each per-condition folder.

--------------------------------------------------
PART 5 — FAILURE CHECKS
--------------------------------------------------

If something fails, identify whether the failure is due to:
- wrong checkpoint path
- missing `shared.pt`
- stale references to the old `checkpoints/shared.pt` layout
- output path issues
- CSV schema issues
- plot generation issues

--------------------------------------------------
PART 6 — DELIVERABLES
--------------------------------------------------

After running the test, provide:

1. The exact commands run
2. Whether checkpoint loading worked from the new folder layout
3. Whether single-model evaluation succeeded
4. Whether comparison graph generation succeeded
5. Which output files were created
6. Any remaining bugs or path mismatches found

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Use the new checkpoint folder layout as the source of truth.
Do not test with the old flat `checkpoints/shared.pt` layout.
Keep the repo runnable.
