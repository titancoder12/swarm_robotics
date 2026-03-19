# Results Interpretation Guide

This guide explains how to read the CSV outputs and plots produced by training and experiments.

## Output Files

### Training outputs in `runs/`

Common files:

- `episode_metrics.csv`
- `eval_metrics.csv`
- `summary.json`
- `run_config.json`
- checkpoint files

### Experiment outputs in `results/`

Common files:

- `trial_metrics.csv`
- `aggregate_metrics.csv`

### Plots in `analysis/`

These include training curves and experiment comparison figures.

## Key Metrics

### `food_retrieved`

How much food was successfully delivered or collected during evaluation.

Higher is better.

### `swarm_efficiency`

Food retrieval normalized by episode length.

Higher is better. This is a compact measure of how productive the swarm is over time.

### `exploration_coverage`

Fraction of the coverage grid visited during an episode.

Higher means the swarm explored more of the environment.

### `pheromone_usage`

Mean local pheromone intensity experienced by agents.

Higher values indicate stronger interaction with the stigmergic field.

### `episode_length`

Length of the episode in steps.

Lower can be better if the swarm finishes the task quickly. Interpret it together with `food_retrieved`.

### `mean_episode_reward`

Average reward across agents in an episode.

Useful as a broad training signal, but less interpretable than task-specific metrics like `food_retrieved`.

### `efficiency_per_robot`

Derived in experiment aggregation:

`food_retrieved / n_agents`

Useful for checking whether adding robots is producing true collective gains or just adding more total effort.

### `convergence_speed`

Derived from training logs in the experiment layer.

Lower values mean the reward stabilized earlier in training.

### `time_to_first_food`

Derived from evaluation logs in the experiment layer.

Lower values mean the policy reached food sooner in evaluation.

## How to Read the CSVs

### `trial_metrics.csv`

One row per trial.

Use this file when you want:

- raw values for each seed
- custom statistical analysis
- to verify variance across repeated runs

### `aggregate_metrics.csv`

One row per experiment case.

This file contains means and standard deviations, which are the main values used for the published plots.

## What Patterns Matter

### Signs of emergent collective intelligence

Look for:

- higher `food_retrieved` as swarm size grows
- improved `swarm_efficiency` with more agents
- stable or improving `efficiency_per_robot`

If performance rises faster with pheromones than without pheromones, that is evidence for stigmergic coordination rather than simple scaling by agent count.

### Signs that pheromones help

Look for:

- higher `food_retrieved` in pheromone-on runs
- higher `swarm_efficiency` in pheromone-on runs
- meaningful `pheromone_usage` only in pheromone-enabled conditions
- better scaling curves in the collective intelligence experiment

### Signs RL beats the rule-based baseline

In the algorithm comparison experiment, look for:

- better `food_retrieved`
- better `swarm_efficiency`
- faster `time_to_first_food`
- better scaling across larger swarm sizes

If DQN or shared-policy DQN improves while the rule-based baseline plateaus, that supports the claim that learning captures coordination strategies the heuristic baseline does not.

## Important Caveats

- `mean_episode_reward` is useful, but it is not the main scientific metric.
- `episode_length` should not be read alone; pair it with `food_retrieved` and `swarm_efficiency`.
- `convergence_speed` and `time_to_first_food` are derived in the experiment layer, not in the trainer.
- very short smoke-test runs are useful for validation but not for scientific conclusions.
