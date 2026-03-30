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

The current implementation supports three curriculum modes:

- `stage1`
- `stage1_to_2`
- `full`

These modes now expand into explicit environment-difficulty stages instead of
only changing swarm size.

The default `full` schedule is:

1. `stage1a_single_agent_tiny`
2. `stage1b_single_agent_obstacles`
3. `stage2a_small_swarm_medium`
4. `stage2b_small_swarm_large`
5. `stage3a_full_swarm_large`
6. `stage3b_full_swarm_final`

The current curriculum stages the following environment variables:

- `n_agents`
- `width`
- `height`
- `n_targets`
- `n_obstacles`
- `max_steps`
- `active_targets`
- `target_respawn`

Intended teaching progression:

- Stage 1A: one agent, tiny easy world, single target, no obstacles
- Stage 1B: one agent, larger world, some obstacles
- Stage 2A: small swarm, medium environment
- Stage 2B: small swarm, large but not final environment
- Stage 3A: full swarm, same large but not final environment
- Stage 3B: full swarm, final large obstacle-heavy environment

Mode semantics:

- `stage1` runs the two single-agent stages
- `stage1_to_2` runs the two single-agent stages plus the two small-swarm stages
- `full` runs all six stages

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
