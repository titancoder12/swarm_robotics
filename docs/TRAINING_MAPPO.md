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

## Trail Objective

The current intended training behavior is not only “reach the target.”

The training path is now aligned to teach the full loop:

1. discover a target
2. pick it up
3. return to the nest
4. deposit pheromone on the successful return route
5. let later agents exploit that trail

This is reflected in the default reward ordering and pheromone behavior:

- pickup reward is meaningful but smaller than delivery reward
- delivery is the strongest task reward
- carrying-food progress back toward the nest gets a small signed shaping term
- pheromone following remains a small supportive signal
- pheromone deposition is gated so it is tied to carrying-food return behavior by default

The current default delivery mechanic is:

- agents may carry at most one food item at a time
- pickup sets an explicit carrying-food state
- reaching the nest while carrying counts as one completed delivery
- delivery clears the carrying state and returns the agent to its normal render color
- carrying agents render with a distinct green-highlighted body in demo mode

Food sources are now repeated-use sources instead of immediate single-use pickups:

- default total sources: `3`
- default source capacity: `4` uses each
- current semantics decrement capacity on pickup
- exhausted sources respawn elsewhere when target respawn is enabled

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
- `food_source_capacity`

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
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --n-targets 3 --active-targets 3 --food-source-capacity 4 --target-respawn --folder-name mappo_trail_smoke
```

Main trail-learning run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 180000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 5000 --eval-episodes 5 --n-targets 3 --active-targets 3 --food-source-capacity 4 --target-respawn --folder-name mappo_trail_full
```

Resume from a checkpoint:

```bash
python train/train.py --backend mappo --headless --curriculum full --resume-checkpoint checkpoints/my_run/latest
```

Rendered MAPPO demo:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_trail_full/latest --n-agents 6 --max-steps 300
```

Headless MAPPO evaluation:

```bash
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_trail_full/latest --n-agents 6 --episodes 10 --headless --output-dir runs/eval --filename mappo_trail_full_eval --active-targets 3 --food-source-capacity 4
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
