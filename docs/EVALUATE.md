# Evaluation Guide

This document explains the practical evaluation paths in this repo:

- evaluate one saved model or baseline
- compare multiple saved models across swarm sizes
- understand where evaluation outputs are written

The main entry points are:

- [analysis/evaluate.py](../analysis/evaluate.py)
- [analysis/evaluate_comparison.py](../analysis/evaluate_comparison.py)
- [experiments/benchmark_configs.py](../experiments/benchmark_configs.py)

## 1. Single-Model Evaluation

Use [analysis/evaluate.py](../analysis/evaluate.py) when you want to evaluate one checkpoint or the rule-based baseline.

### Shared-policy DQN checkpoint

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/dqn_foraging/full_policy --shared-policy --filename eval_shared --output-dir runs/eval --headless --episodes 10 --n-agents 6 --eval-steps 2000 --active-targets 4
```

### Pheromone-enabled vs pheromone-disabled

Pheromone on:

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/dqn_foraging/full_policy --shared-policy --filename eval_pheromone_on --output-dir runs/eval --headless --episodes 10 --n-agents 6 --eval-steps 2000 --active-targets 4 --use-pheromone
```

Pheromone off:

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/dqn_foraging/full_policy --shared-policy --filename eval_pheromone_off --output-dir runs/eval --headless --episodes 10 --n-agents 6 --eval-steps 2000 --active-targets 4 --no-use-pheromone
```

### Rule-based baseline

```bash
python analysis/evaluate.py --policy-kind rule_based --filename eval_rule_based --output-dir runs/rule_eval --headless --episodes 10 --n-agents 6 --eval-steps 2000 --active-targets 4 --no-use-pheromone
```

### MAPPO GRU checkpoint

```bash
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --filename eval_mappo_current --output-dir runs/eval --headless --episodes 10 --n-agents 6 --eval-steps 2000 --active-targets 3
```

### Presentation-seed scan for one checkpoint

```bash
for s in $(seq 1 100); do
  python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_g/latest --episodes 1 --seed "$s" --headless --eval-steps 800 --output-dir runs/presentation_seed_scan --filename "seed_$s"
done
```

Use the resulting `runs/presentation_seed_scan/*_eval_metrics.csv` files to rank seeds by:

- `food_delivered` first
- then `pheromone_usage`
- then `exploration_coverage`

At the time of the latest scan, the strongest presentation seeds were:

- `45`
- `40`
- `58`
- `31`
- `51`

## 2. What `analysis/evaluate.py` Writes

For a run such as:

```bash
python analysis/evaluate.py --filename eval_shared --output-dir runs/eval ...
```

the script writes:

- `runs/eval/eval_shared_eval_metrics.csv`
- `runs/eval/eval_shared_eval_summary.json`

Important metric meanings:

- `targets_collected` = pickup events, not delivery
- `targets_picked_up` = same pickup count, kept explicitly for clarity
- `food_delivered` = delivery / return-to-nest count
- `time_to_first_discovery` = first step where any pickup happens
- `coverage_efficiency` = `exploration_coverage / total_steps_taken`

Evaluation uses a fixed horizon through `--eval-steps`. In the current evaluation path, targets respawn to maintain `--active-targets` throughout the episode.

## 3. Multi-Model Swarm-Scaling Comparison

Use [analysis/evaluate_comparison.py](../analysis/evaluate_comparison.py) when you want one combined comparison across swarm sizes.

It currently evaluates four conditions:

1. `trained_with_pheromone__eval_with_pheromone`
2. `trained_without_pheromone__eval_without_pheromone`
3. `trained_with_pheromone__eval_without_pheromone`
4. `random_walk`

Condition 3 reuses the `--checkpoint-with-pheromone` checkpoint.

Condition 4 uses no checkpoint and samples random actions each step.

### Main comparison command

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/dqn_pheromone/full_policy --checkpoint-without-pheromone checkpoints/dqn_no_pheromone/full_policy --filename pheromone_vs_random --agent-min 1 --agent-max 30 --agent-step 1 --episodes-per-agent 10 --output-dir experiments/experiment_data --headless --max-steps 2000 --active-targets 4 --shared-policy
```

### Smaller smoke test

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/dqn_pheromone/full_policy --checkpoint-without-pheromone checkpoints/dqn_no_pheromone/full_policy --filename smoke_compare --agent-min 1 --agent-max 3 --agent-step 1 --episodes-per-agent 1 --output-dir experiments/experiment_data --headless --max-steps 50 --active-targets 4 --shared-policy
```

## 4. What `analysis/evaluate_comparison.py` Writes

By default, outputs go under [experiments/experiment_data/](../experiments/experiment_data/).

Directory structure:

- `experiments/experiment_data/raw/`
- `experiments/experiment_data/graphs/PNG/`
- `experiments/experiment_data/graphs/PDF/`
- `experiments/experiment_data/exploration_graphs/PNG/`
- `experiments/experiment_data/exploration_graphs/PDF/`

Main raw outputs:

- `<filename>_all_conditions_raw.csv`
- `<filename>_summary.csv`
- one per-condition raw CSV for each comparison label

Main plots:

- `<filename>_agents_vs_targets_collected`
- `<filename>_coverage_efficiency_vs_agents`
- `<filename>_efficiency_vs_agents`
- `<filename>_time_to_first_discovery_vs_agents`

Exploration outputs:

- one heatmap image per representative swarm size per condition

The comparison metadata file is:

- `<output-dir>/<filename>_metadata.json`

## 5. Checkpoint Conventions

Training saves checkpoints in directory form. For the current MAPPO path, the most important folders are usually:

- `checkpoints/<run_name>/latest/`
- `checkpoints/<run_name>/best_greedy_eval/`
- `checkpoints/<run_name>/stage3b_full_swarm_final/`

For DQN shared-policy runs, evaluation expects `shared.pt` inside the checkpoint directory. For MAPPO runs, evaluation expects `actor.pt` and usually `metadata.json`.

Example:

- `checkpoints/<run_name>/best_greedy_eval/actor.pt`
- `checkpoints/<run_name>/best_greedy_eval/metadata.json`

## 6. Pheromone Controls During Evaluation

The shared evaluation/config path supports:

- `--use-pheromone`
- `--no-use-pheromone`
- `--pheromone-requires-food`
- `--no-pheromone-requires-food`

When pheromone is disabled:

- deposition is disabled
- pheromone observations are zeroed
- observation shape stays compatible with trained models

If you want visible pheromone trails during free exploration in demo/eval, do not forget:

```bash
--no-pheromone-requires-food
```

Otherwise deposition can be delayed until pickup.

## 7. Existing Experiment Registry

[experiments/benchmark_configs.py](../experiments/benchmark_configs.py) is different from `experiments/experiment_data/`.

- `experiments/benchmark_configs.py` defines named experiment sweeps for [train/run_experiments.py](../train/run_experiments.py)
- `experiments/experiment_data/` stores generated comparison outputs

Use [train/run_experiments.py](../train/run_experiments.py) when you want the older benchmark-registry workflow. Use [analysis/evaluate_comparison.py](../analysis/evaluate_comparison.py) when you want the newer explicit multi-checkpoint comparison workflow.

## 8. Practical Workflow

Typical flow:

1. Train a pheromone-enabled model.
2. Train a pheromone-disabled model.
3. Confirm the checkpoints exist in directories such as:
   - `checkpoints/<folder_name>/full_policy/`
4. Run a quick single-checkpoint evaluation with [analysis/evaluate.py](../analysis/evaluate.py).
5. Run the multi-condition comparison with [analysis/evaluate_comparison.py](../analysis/evaluate_comparison.py).
6. Inspect:
   - raw CSVs in `experiments/experiment_data/raw/`
   - line plots in `experiments/experiment_data/graphs/`
   - exploration heatmaps in `experiments/experiment_data/exploration_graphs/PNG/` and `experiments/experiment_data/exploration_graphs/PDF/`

## 9. Related Docs

- [docs/EVALUATION_CUSTOM.md](EVALUATION_CUSTOM.md)
- [docs/ONBOARDING.md](ONBOARDING.md)
- [docs/API_REFERENCE.md](API_REFERENCE.md)
