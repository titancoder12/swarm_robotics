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

1. `stage1a_single_agent_miniscule`
2. `stage1b_single_agent_tiny`
3. `stage1c_single_agent_small`
4. `stage1d_single_agent_delivery_obstacles`
5. `stage2a_small_swarm_medium`
6. `stage2b_small_swarm_large`
7. `stage3a_full_swarm_large`
8. `stage3b_full_swarm_final`

The current curriculum stages the following environment variables:

- `n_agents`
- `width`
- `height`
- `n_targets`
- `n_obstacles`
- `max_steps`
- `active_targets`
- `target_respawn`
- `action_repeat_steps`
- `reward_new_cell`

Intended teaching progression:

- Stage 1A-1C: one agent, increasingly larger empty worlds with one target
- Stage 1D: one agent, one target, obstacles, no respawn; this is the first full obstacle delivery stage
- Stage 2A: small swarm, medium environment, two fixed sources, no respawn yet
- Stage 2B: small swarm, large but not final environment, now with respawn enabled
- Stage 3A: full swarm, same large but not final environment
- Stage 3B: full swarm, final large obstacle-heavy environment

The stage-budget bug fixed in the current version was that the old curriculum
reused early budget buckets and left later stages more starved than intended.
The schedule now assigns one explicit weight per actual stage:

- `1, 1, 1, 2, 2, 3, 4, 6`

This keeps the hardest full-swarm stages from receiving only accidental
fine-tuning time.

`--total-steps` now applies to the curriculum slice you actually selected. For
example, `--curriculum stage1 --total-steps 32000` distributes that full
`32000` budget across the four single-agent stages instead of first splitting
it across all eight full-schedule stages and then discarding the unused ones.

Control/reward staging now also changes with difficulty:

- early single-agent stages use `action_repeat_steps = 1` for more responsive control
- later swarm stages use `action_repeat_steps = 2` for smoother execution
- early stages keep a slightly stronger `reward_new_cell`
- later stages reduce `reward_new_cell` so delivery and trail reuse compete less with generic wandering

Mode semantics:

- `stage1` runs all four single-agent stages
- `stage1_to_2` runs the four single-agent stages plus the two small-swarm stages
- `full` runs all eight stages

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
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_trail_full
```

Why this command is more realistic than the older shorter examples:

- `600k` total steps gives the eight-stage curriculum meaningful late-stage time
- delivery now dominates pickup more clearly
- the stronger undelivered-food penalty makes `picked up but never returned` less acceptable
- the later full-swarm stages are still hard enough that `180k` or `200k` often remains undertrained

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
