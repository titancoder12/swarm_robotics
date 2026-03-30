# Centralized Training State

This document defines the current training-time centralized state interface for
CTDE algorithms such as MAPPO and QMIX.

Source of truth:

- [env/swarm_env.py](../env/swarm_env.py) `state_space()`
- [env/swarm_env.py](../env/swarm_env.py) `state()`
- [env/swarm_env.py](../env/swarm_env.py) `_get_global_state()`

This interface is for training only. It is not part of the robot-side
deployment observation contract.

## Purpose

The repo already has a local per-agent observation used by the current DQN
policies and by deployment-facing inference code. CTDE algorithms need a second
view of the environment:

- actor input at inference: local observation only
- critic or mixer input at training: centralized state

The centralized state is fixed-layout and deterministic for a given config so
that recurrent MAPPO and QMIX can consume it without depending on ad hoc
inspection of environment internals.

## Current Shape

The centralized state dimension depends on the config:

```python
state_dim = (
    n_agents * 7
    + max(n_targets, active_targets) * 3
    + n_obstacles * 4
    + 2
    + 5
)
```

With the current defaults:

- `n_agents = 6`
- `n_targets = 4`
- `active_targets = 4`
- `n_obstacles = 6`

the default centralized state size is:

```text
6 * 7 + 4 * 3 + 6 * 4 + 2 + 5 = 85
```

So the default centralized state space is:

- `Box(low=-1.0, high=1.0, shape=(85,), dtype=np.float32)`

## Layout

The state vector is concatenated in this order:

1. Per-agent block repeated `n_agents` times:
   - normalized `x`
   - normalized `y`
   - `sin(theta)`
   - `cos(theta)`
   - normalized forward speed
   - carrying-food flag
   - failed-agent flag

2. Target block repeated `max(n_targets, active_targets)` times:
   - normalized `x`
   - normalized `y`
   - present flag

3. Obstacle block repeated `n_obstacles` times:
   - normalized `x`
   - normalized `y`
   - normalized width
   - normalized height

4. Nest block:
   - normalized nest `x`
   - normalized nest `y`

5. Global scalars:
   - normalized episode progress
   - normalized food delivered
   - exploration coverage ratio
   - mean pheromone usage
   - active target ratio

## Normalization Rules

- world positions are mapped from `[0, width]` or `[0, height]` into `[-1, 1]`
- speed is normalized by `cfg.max_speed`
- booleans use `1.0` for true and `-1.0` for false in the agent/target blocks
- obstacle width/height and the global scalar block use `[0, 1]` style scaling

Unused target or obstacle slots are padded so the shape remains fixed.

## Contract Boundary

This centralized state is:

- intended for critics and mixers during training
- not required by deployment
- independent from the per-agent observation consumed by the current DQN policy

That separation preserves decentralized execution while enabling CTDE training.
