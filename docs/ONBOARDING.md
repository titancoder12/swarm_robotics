# Onboarding and Run Guide

This guide is the main entry point for running the current project. It covers setup, simulation, training, evaluation, and experiments using the code as it exists now.

## What This Project Does

This repository studies stigmergic swarm intelligence in a 2D robotics simulation. Agents search for food, avoid obstacles, interact through a pheromone field, and optionally return food to a nest. The main research focus is whether indirect communication through pheromones improves collective performance.

## Quick Start

If you only want one command to see the current system working:

```bash
python train/run_experiments.py --experiment collective_intelligence_scaling
```

That command trains and evaluates the flagship experiment, then writes:

- raw logs under `runs/`
- aggregated CSVs under `results/`
- plots under `analysis/`

For a short version, see [docs/manual/QUICK_START.md](/Users/christopherlin/dev/cwsf2026/sim/docs/manual/QUICK_START.md).

## 1. Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Run a Simulation

Random rendered rollout:

```bash
python train/random_rollout.py
```

Capture example screenshots:

```bash
python train/capture_screenshots.py
```

## 3. Train a Custom DQN Policy

Headless training:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000
```

Training with checkpoints and run outputs:

```bash
python train/independent_dqn_pytorch.py \
  --headless \
  --total-steps 20000 \
  --save-dir checkpoints \
  --save-every 5000 \
  --output-dir runs \
  --experiment-name dqn_foraging \
  --eval-every 5000 \
  --eval-episodes 5
```

Important notes:

- the current observation dimension is `23`
- old 19-dimensional checkpoints are not compatible
- the current action space is still `Discrete(9)`

## 4. Render a Trained Policy

Custom DQN demo:

```bash
python train/demo.py --checkpoint-dir checkpoints
```

The demo renders the current environment, including pheromone heatmap, food, nest, obstacles, and agents.

## 5. Evaluate a Policy

Evaluate saved DQN checkpoints:

```bash
python train/evaluate.py --checkpoint-dir checkpoints --episodes 10 --output-dir runs/eval
```

Evaluate the rule-based baseline directly:

```bash
python train/evaluate.py --policy-kind rule_based --n-agents 5 --episodes 10 --output-dir runs/rule_eval
```

Outputs include:

- `eval_metrics.csv`
- `eval_summary.json`

## 6. Run Experiments

The central experiment entry point is [train/run_experiments.py](/Users/christopherlin/dev/cwsf2026/sim/train/run_experiments.py).

### Collective Intelligence Scaling

```bash
python train/run_experiments.py --experiment collective_intelligence_scaling
```

This experiment compares swarm sizes with pheromone on vs pheromone off.

### RL Algorithm Comparison

```bash
python train/run_experiments.py --experiment rl_algorithm_comparison
```

This experiment compares:

- DQN
- shared-policy DQN
- rule-based baseline

across multiple swarm sizes and pheromone conditions.

### Run All Registered Experiments

```bash
python train/run_experiments.py --experiment all
```

## 7. Output Locations

The project uses three main output directories:

- `runs/`
  - per-run logs
  - checkpoints
  - training CSVs
  - run configs
- `results/`
  - trial-level experiment CSVs
  - aggregate experiment CSVs
- `analysis/`
  - plots generated from training or experiments

## 8. Key Files to Know

- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
  - environment logic
- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
  - simulation configuration
- [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py)
  - custom DQN trainer
- [train/evaluate.py](/Users/christopherlin/dev/cwsf2026/sim/train/evaluate.py)
  - shared evaluation path
- [train/run_experiments.py](/Users/christopherlin/dev/cwsf2026/sim/train/run_experiments.py)
  - experiment runner
- [experiments/benchmark_configs.py](/Users/christopherlin/dev/cwsf2026/sim/experiments/benchmark_configs.py)
  - experiment definitions
- [models/rule_based_policy.py](/Users/christopherlin/dev/cwsf2026/sim/models/rule_based_policy.py)
  - rule-based baseline

## 9. Suggested Reading Order

1. [docs/manual/QUICK_START.md](/Users/christopherlin/dev/cwsf2026/sim/docs/manual/QUICK_START.md)
2. [docs/ARCHITECTURE.md](/Users/christopherlin/dev/cwsf2026/sim/docs/ARCHITECTURE.md)
3. [docs/manual/PROJECT_STRUCTURE.md](/Users/christopherlin/dev/cwsf2026/sim/docs/manual/PROJECT_STRUCTURE.md)
4. [docs/manual/EXPERIMENT_GUIDE.md](/Users/christopherlin/dev/cwsf2026/sim/docs/manual/EXPERIMENT_GUIDE.md)
5. [docs/manual/RESULTS_INTERPRETATION.md](/Users/christopherlin/dev/cwsf2026/sim/docs/manual/RESULTS_INTERPRETATION.md)
