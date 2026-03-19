# Experiment Guide

This project currently has two primary science-fair experiments implemented in the shared framework.

The common entry point is:

```bash
python train/run_experiments.py --experiment <name>
```

All experiments reuse the same:

- environment
- custom DQN training loop
- shared evaluator
- logging outputs
- aggregation path

## Shared Output Format

Every experiment writes:

- raw run artifacts in `runs/<experiment>/...`
- trial-level metrics in `results/<experiment>/trial_metrics.csv`
- aggregated metrics in `results/<experiment>/aggregate_metrics.csv`
- plots in `analysis/<experiment>/`

## 1. Collective Intelligence Scaling

Experiment name:

```bash
collective_intelligence_scaling
```

### Goal

Test whether larger swarms improve performance, and whether that improvement depends on stigmergic communication through pheromones.

### Variables

- swarm sizes: `1, 2, 3, 5, 10`
- condition A: pheromone enabled
- condition B: pheromone disabled
- default trials: `20`

### Run Command

```bash
python train/run_experiments.py --experiment collective_intelligence_scaling
```

### Main Metrics

- `food_retrieved`
- `exploration_coverage`
- `pheromone_usage`
- `episode_length`
- `swarm_efficiency`
- `mean_episode_reward`
- derived: `efficiency_per_robot`

### Main Plots

- `completion_time_vs_agents.png`
- `efficiency_vs_agents.png`
- `pheromone_usage_vs_agents.png`
- `efficiency_per_robot_vs_agents.png`

### What It Tests

This experiment is designed to support the claim that performance improves non-linearly with swarm size when stigmergic communication is available.

## 2. RL Algorithm Comparison

Experiment name:

```bash
rl_algorithm_comparison
```

### Goal

Compare learned and non-learned control strategies under the same environment and metric pipeline.

### Algorithms

- `dqn`
- `shared_dqn`
- `rule_based`

### Variables

- swarm sizes: `1, 3, 5, 10`
- pheromone enabled
- pheromone disabled
- default trials: `20`

### Run Command

```bash
python train/run_experiments.py --experiment rl_algorithm_comparison
```

### Main Metrics

- `food_retrieved`
- `swarm_efficiency`
- `exploration_coverage`
- `pheromone_usage`
- `episode_length`
- `mean_episode_reward`
- derived: `convergence_speed`
- derived: `time_to_first_food`

### Main Plots

- `algorithm_food_retrieval.png`
- `algorithm_efficiency.png`
- `algorithm_convergence_speed.png`
- `algorithm_comparison_bar.png`

### What It Tests

This experiment is designed to support the claim that learned controllers scale more effectively than the rule-based baseline, especially when stigmergic communication is available.

## Useful Optional Flags

The experiment runner supports:

- `--trials`
- `--seed`
- `--total-steps`
- `--eval-every`
- `--eval-episodes`
- `--runs-dir`
- `--results-dir`
- `--analysis-dir`
- `--save-every`
- `--cuda`
- `--no-plots`

Example shorter run:

```bash
python train/run_experiments.py \
  --experiment rl_algorithm_comparison \
  --trials 1 \
  --total-steps 100 \
  --eval-every 0 \
  --eval-episodes 1
```

## How the Experiments Stay Comparable

The experiments are comparable because they share:

- the same environment implementation
- the same observation contract
- the same action space
- the same evaluator
- the same result CSV schema

This means differences in results are attributable to the experiment variables rather than to different logging or measurement pipelines.
