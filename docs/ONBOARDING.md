# Onboarding and Run Guide

This guide is the main entry point for running the current project. It covers setup, simulation, training, evaluation, and experiments using the code as it exists now.

## What This Project Does

This repository studies stigmergic swarm intelligence in a 2D robotics simulation. Agents search for food, avoid obstacles, interact through a pheromone field, and optionally return food to a nest. The main research focus is whether indirect communication through pheromones improves collective performance.

## Quick Start

If you want the current main training path:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --stage-repeat-limit 1 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_full_current
```

If you want a much shorter smoke test:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --folder-name mappo_trail_smoke
```

Recommended demo checkpoint after a full run:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_current/best_greedy_eval --max-steps 300 --render-scale 0.75
```

If you want demo resets to cycle through only a few presentation seeds:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_current/best_greedy_eval --max-steps 0 --render-scale 0.75 --seed-list 45,40,58
```

For a short version, see [docs/manual/QUICK_START.md](manual/QUICK_START.md).

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
python train/independent_dqn_pytorch.py --headless --total-steps 20000 --save-dir checkpoints --folder-name dqn_foraging --save-every 5000 --output-dir runs --experiment-name dqn_foraging --eval-every 5000 --eval-episodes 5
```

Training with an explicit epsilon schedule:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 30000 --save-dir checkpoints --folder-name dqn_foraging --epsilon-start 1.0 --epsilon-final 0.05 --epsilon-decay-steps 20000 --warmup-steps 2000
```

Keep those commands on one line unless you are deliberately using shell line continuations. If a shell sees standalone `>` lines or detached flag lines, it can create empty files named after the flags instead of passing them to Python.

Important notes:

- the default per-agent observation is `69` dims from `3 x 23` stacked frames
- old checkpoints trained on smaller observation layouts are not compatible
- the current action space is `Discrete(18)`

## 4. Render a Trained Policy

Custom DQN demo:

```bash
python train/demo.py --checkpoint-dir checkpoints/dqn_foraging/full_policy
```

The demo renders the current environment, including pheromone heatmap, food, nest, obstacles, and agents. The window title shows the current reset seed, the HUD shows live counts for pickups, deliveries, pheromone drops, and carrying agents, `Space` pauses/resumes, and clicking an agent opens an inspector panel with that agent's current inputs and model outputs.

## 5. Evaluate a Policy

Evaluate saved DQN checkpoints:

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/dqn_foraging/full_policy --episodes 10 --output-dir runs/eval
```

Evaluate the rule-based baseline directly:

```bash
python analysis/evaluate.py --policy-kind rule_based --n-agents 5 --episodes 10 --output-dir runs/rule_eval
```

Outputs include:

- `eval_metrics.csv`
- `eval_summary.json`

Probe a checkpoint with hand-written observation vectors:

```bash
python train/policy_probe.py --list-cases
python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead
python train/policy_probe.py --checkpoint-dir checkpoints --case wall_ahead --agent-index 0
```

This is a small teaching script for manually feeding the current stacked observation into the custom DQN model and inspecting Q-values plus the chosen discrete action.

## 6. Run Experiments

The central experiment entry point is [train/run_experiments.py](../train/run_experiments.py).

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

- [env/swarm_env.py](../env/swarm_env.py)
  - environment logic
- [env/config.py](../env/config.py)
  - simulation configuration
- [train/independent_dqn_pytorch.py](../train/independent_dqn_pytorch.py)
  - custom DQN trainer
- [analysis/evaluate.py](../analysis/evaluate.py)
  - shared evaluation path
- [train/run_experiments.py](../train/run_experiments.py)
  - experiment runner
- [experiments/benchmark_configs.py](../experiments/benchmark_configs.py)
  - experiment definitions
- [models/rule_based_policy.py](../models/rule_based_policy.py)
  - rule-based baseline

## 9. Suggested Reading Order

1. [docs/manual/QUICK_START.md](manual/QUICK_START.md)
2. [docs/ARCHITECTURE.md](ARCHITECTURE.md)
3. [docs/manual/PROJECT_STRUCTURE.md](manual/PROJECT_STRUCTURE.md)
4. [docs/manual/EXPERIMENT_GUIDE.md](manual/EXPERIMENT_GUIDE.md)
5. [docs/manual/RESULTS_INTERPRETATION.md](manual/RESULTS_INTERPRETATION.md)
