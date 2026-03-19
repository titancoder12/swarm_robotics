# Quick Start

This project simulates a swarm of agents that search for food, interact through pheromones, and learn coordination behaviors.

## Fastest Way to See the Project Working

Run the flagship experiment:

```bash
python train/run_experiments.py --experiment collective_intelligence_scaling
```

That command will:

- train policies
- evaluate them
- aggregate metrics
- generate plots

## Where to Look After It Finishes

- raw run logs: `runs/collective_intelligence_scaling/`
- aggregated CSVs: `results/collective_intelligence_scaling/`
- plots: `analysis/collective_intelligence_scaling/`

## If You Want the Algorithm Comparison

```bash
python train/run_experiments.py --experiment rl_algorithm_comparison
```

That experiment compares:

- DQN
- shared-policy DQN
- rule-based baseline

## If You Want a Visual Simulation First

```bash
python train/random_rollout.py
```

## If You Want to Train a Single Policy

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000
```

## Important Current Facts

- observation space: `23`
- action space: `Discrete(9)`
- main environment features: nest, food, obstacles, pheromones
- main experiment runner: [train/run_experiments.py](/Users/christopherlin/dev/cwsf2026/sim/train/run_experiments.py)
