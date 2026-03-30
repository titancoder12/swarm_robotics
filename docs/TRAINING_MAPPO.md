# Recurrent MAPPO Training

This document describes the first runnable recurrent MAPPO path added to the
repo.

Source files:

- [train/mappo_gru.py](../train/mappo_gru.py)
- [algorithms/mappo/networks.py](../algorithms/mappo/networks.py)
- [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py)
- [algorithms/mappo/inference.py](../algorithms/mappo/inference.py)
- [docs/CTDE_STATE.md](CTDE_STATE.md)

## High-Level Design

The implementation uses:

- parameter-shared recurrent actor
- GRU actor over local observations
- recurrent centralized critic over the env training-time state
- PPO-style clipped policy updates
- GAE
- entropy regularization
- gradient clipping

Inference remains decentralized:

- the actor consumes only local observations plus recurrent hidden state
- the critic is training-only

## Curriculum

The first implementation supports three curriculum modes:

- `stage1`
- `stage1_to_2`
- `full`

The default `full` schedule is:

1. `stage1_single_agent`
2. `stage2_small_swarm`
3. `stage3_full_marl`

The essential curriculum variable is swarm size:

- stage 1 uses `n_agents = 1`
- stage 2 uses a small swarm between `2` and `5`
- stage 3 uses the requested full swarm size

Actor weights are carried across stages.

Important current limitation:

- the centralized critic input size changes with `n_agents`
- so critic weights are not reused across stage boundaries when the state shape changes
- actor transfer is preserved; critic reset is explicit and recorded in checkpoint metadata

## Example Commands

Single-agent smoke test:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 600 --rollout-steps 32 --update-epochs 2 --minibatch-size 32 --no-plots
```

Full curriculum run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 30000
```

Resume from a checkpoint:

```bash
python train/train.py --backend mappo --headless --curriculum full --resume-checkpoint checkpoints/my_run/latest
```

## Checkpoints

Each stage writes:

- `actor.pt`
- `critic.pt`
- `trainer.pt`
- `metadata.json`

There is also a rolling:

- `latest/`

The deployment-facing actor loader is in
[algorithms/mappo/inference.py](../algorithms/mappo/inference.py).
