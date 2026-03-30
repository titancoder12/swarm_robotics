# Quick Start

This project simulates a swarm of agents that search for food, interact through pheromones, and learn coordination behaviors.

## Fastest Way To See The Current Main Path Working

Run the recurrent MAPPO trail-learning smoke test:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --folder-name mappo_trail_smoke
```

That command will:

- train the recurrent MAPPO path through the easy single-agent curriculum stages
- write run logs under `runs/`
- write checkpoints under `checkpoints/`

## Where to Look After It Finishes

- raw run logs: `runs/`
- checkpoints: `checkpoints/mappo_trail_smoke/`

## If You Want A Full Trail-Learning Run

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 180000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 5000 --eval-episodes 5 --folder-name mappo_trail_full
```

## If You Want The Older Experiment Runner

```bash
python train/run_experiments.py --experiment collective_intelligence_scaling
```

## If You Want a Visual Simulation First

```bash
python train/random_rollout.py
```

## If You Want to Train a Single Policy

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000
```

## If You Want to Manually Probe a Trained Policy

```bash
python train/policy_probe.py --list-cases
python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead
```

This is useful for understanding how the trained custom DQN responds to specific hand-written observation vectors.

## If You Want to Run the Physical Robot

```bash
python firmware/run.py --checkpoint-dir checkpoints --shared-policy
```

That path is now the canonical robot runtime. It talks directly to [ant.py](../../firmware/ant.py) and does not use the older `pi/` or `robot/` packages.

## Important Current Facts

- observation space: `69` by default from `3 x 23` stacked frames
- action space: `Discrete(18)`
- main environment features: nest, food, obstacles, pheromones
- main MAPPO trainer: [train/mappo_gru.py](../../train/mappo_gru.py)
