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

This is reflected in the current reward ordering and pheromone behavior:

- pickup reward is meaningful but smaller than delivery reward
- delivery is the strongest task reward
- carrying-food progress back toward the nest gets a small signed shaping term
- pheromone following remains a small supportive signal
- pheromone deposition can be gated to carrying-food nest-return behavior when the CLI flags are enabled

## Trap Recovery Objective

The current MAPPO path also includes explicit trap-recovery shaping aimed at
the common failure mode where agents keep pushing, circling, or stacking in the
same local pocket near a target or obstacle.

The training intent is:

1. detect non-progress
2. penalize prolonged local stagnation
3. reward recovery from a recent stuck state
4. discourage crowded jams
5. let local recovery override weak pheromone attraction when needed

The current trap-recovery config surface is implemented in:

- [env/config.py](../env/config.py)
- [train/experiment_utils.py](../train/experiment_utils.py)
- [env/swarm_env.py](../env/swarm_env.py)

Main trap-recovery controls:

- `reward_stuck`
- `reward_escape`
- `reward_crowding`
- `trap_min_displacement`
- `trap_escape_displacement`
- `trap_stuck_steps`
- `crowding_radius`
- `crowding_min_neighbors`

The recurrent GRU is useful here because the actor can learn patterns like:

- "I have seen similar geometry for several steps"
- "my recent actions are not changing my position enough"
- "I should switch away from this repeated local behavior"

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
- `target_prefer_edges`
- `target_prefer_obstacles`
- `agent_spawn_cluster_radius`

Intended teaching progression:

- Stage 1A: one agent, tiny `100x100` world, `2` targets, no obstacles
- Stage 1B: one agent, `420x320`, `2` targets, `2` obstacles
- Stage 2A: small swarm, `700x500`, `2` targets, `4` obstacles, modest clustered starts
- Stage 2B: small swarm, `950x700`, `3` targets, `8` obstacles, obstacle-biased targets
- Stage 3A: full swarm, `950x700`, `3` targets, `8` obstacles, stronger clustered starts
- Stage 3B: full swarm, `1400x950`, `4` targets, `35` obstacles, edge/obstacle-biased targets and the strongest clustered starts

Later stages are now deliberately harsher for trap recovery instead of only
being larger. They add:

- more obstacle pockets
- biased target placement near harder geometry
- increasingly clustered initial agent spawns
- higher congestion pressure near goals

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
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --folder-name mappo_trail_smoke
```

Main trail-learning run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 180000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 5000 --eval-episodes 5 --folder-name mappo_trail_full
```

Main trap-recovery run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 180000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 5000 --eval-episodes 5 --trap-stuck-steps 6 --trap-min-displacement 4 --trap-escape-displacement 12 --reward-stuck -0.02 --reward-escape 0.03 --reward-crowding -0.005 --folder-name mappo_trap_recovery_full
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
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_trail_full/latest --n-agents 6 --episodes 10 --headless --output-dir runs/eval --filename mappo_trail_full_eval
```

Headless trap-recovery evaluation:

```bash
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_trap_recovery_full/latest --n-agents 6 --episodes 10 --headless --output-dir runs/eval --filename mappo_trap_recovery_eval
```

Trap-recovery comparison:

```bash
python analysis/evaluate_comparison.py --policy-kind mappo_gru --checkpoint-with-pheromone checkpoints/mappo_trap_recovery_full/latest --checkpoint-without-pheromone checkpoints/mappo_trail_full/latest --agent-min 1 --agent-max 6 --episodes-per-agent 3 --headless --output-dir experiments/experiment_data/trap_recovery_compare
```

## Evaluation Signals

The MAPPO trainer and evaluators now expose direct trap-recovery metrics in
addition to reward and delivery:

- `low_displacement_fraction`
- `crowding_fraction`
- `stuck_event_count`
- `successful_escape_count`
- `mean_stuck_duration`
- `max_collision_streak`

These are logged in:

- `episode_metrics.csv`
- `eval_metrics.csv`
- `analysis/evaluate.py` summary outputs
- `analysis/evaluate_comparison.py` raw and aggregated comparison outputs

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
