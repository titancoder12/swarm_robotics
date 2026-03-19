# Action Specification

This document is a focused reference for the current action contract implemented by [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py).

## Source of Truth

Action-space size and mapping are defined in:

- `action_space()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- `_build_action_table()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- `TankKinematicsDriver.apply()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- `HovercraftDriver.apply()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)

## Space Definition

Per-agent action space:

- `gymnasium.spaces.Discrete(cfg.num_actions)`

Default:

- `cfg.num_actions = 9`

Action input type:

- scalar integer per agent

Action batch type expected by `step()`:

```python
dict[str, int]
```

Example:

```python
actions = {
    "agent_0": 7,
    "agent_1": 4,
    "agent_2": 6,
}
```

## Mapping Table

The action table is built in this exact order:

```python
throttle_vals = [-1.0, 0.0, 1.0]
turn_vals = [-1.0, 0.0, 1.0]
for throttle in throttle_vals:
    for turn in turn_vals:
        table.append((throttle, turn))
```

Current mapping:

| Action / Index | Name | Meaning | Range | Unit | Internal interpretation | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `reverse_left` | reverse and turn left | throttle `-1`, turn `-1` | unitless command | target forward speed `-cfg.max_speed`, target yaw rate `-cfg.max_yaw_rate` | action id maps to tuple `(-1.0, -1.0)` |
| 1 | `reverse_straight` | reverse with no turn | `-1`, `0` | unitless command | target forward speed `-cfg.max_speed`, target yaw rate `0` | tuple `(-1.0, 0.0)` |
| 2 | `reverse_right` | reverse and turn right | `-1`, `1` | unitless command | target forward speed `-cfg.max_speed`, target yaw rate `+cfg.max_yaw_rate` | tuple `(-1.0, 1.0)` |
| 3 | `idle_left` | zero throttle and turn left | `0`, `-1` | unitless command | target speed `0`, target yaw rate `-cfg.max_yaw_rate` | tuple `(0.0, -1.0)` |
| 4 | `idle` | zero throttle and zero turn | `0`, `0` | unitless command | target speed `0`, target yaw rate `0` | tuple `(0.0, 0.0)` |
| 5 | `idle_right` | zero throttle and turn right | `0`, `1` | unitless command | target speed `0`, target yaw rate `+cfg.max_yaw_rate` | tuple `(0.0, 1.0)` |
| 6 | `forward_left` | forward and turn left | `1`, `-1` | unitless command | target speed `+cfg.max_speed`, target yaw rate `-cfg.max_yaw_rate` | tuple `(1.0, -1.0)` |
| 7 | `forward_straight` | forward with no turn | `1`, `0` | unitless command | target speed `+cfg.max_speed`, target yaw rate `0` | tuple `(1.0, 0.0)` |
| 8 | `forward_right` | forward and turn right | `1`, `1` | unitless command | target speed `+cfg.max_speed`, target yaw rate `+cfg.max_yaw_rate` | tuple `(1.0, 1.0)` |

## Semantics

These actions are high-level movement commands, not direct wheel PWM or low-level motor outputs.

Each action is decoded into:

- `throttle`
- `turn`

Then passed into the active dynamics driver as:

```python
driver.apply(agent_state, (throttle, turn), cfg.dt, cfg, rng)
```

## Dynamics Interpretation

### Tank mode

In `TankKinematicsDriver.apply()`:

```python
target_v = throttle * cfg.max_speed
target_omega = turn * cfg.max_yaw_rate
dv = clip(target_v - state.v, -cfg.accel * dt, cfg.accel * dt)
domega = clip(target_omega - state.omega, -cfg.ang_accel * dt, cfg.ang_accel * dt)
```

So actions are filtered by:

- forward acceleration limit `cfg.accel`
- angular acceleration limit `cfg.ang_accel`

There is no instantaneous jump to target speed or target yaw rate.

### Hover mode

In `HovercraftDriver.apply()`, the same `throttle` and `turn` commands define:

- target forward speed
- target yaw rate

but actual motion also includes:

- lateral drift
- Gaussian lateral noise
- stochastic slip

So the same action index can produce more variable motion under hover mode.

### Mixed mode

If `cfg.dynamics_mode == "mixed"`:

- `reset()` randomly picks either tank or hover for the full episode

So action semantics at the interface stay the same, but the active low-level transition model changes per episode.

## Action Validation

`step()` currently validates:

- that `actions` is a `dict`
- that every active agent key is present

It does not explicitly validate numeric bounds before indexing `self.action_table`.

Current failure behavior:

- invalid key set:
  - `ValueError`
- non-dict action container:
  - `ValueError`
- out-of-range action integer:
  - Python list index failure when accessing `self.action_table[action_id]`

## Collision and Motion Side Effects

After the driver produces a proposed next state:

- `_handle_collisions()` checks arena bounds and obstacle overlap
- if collision occurs:
  - proposed move is rejected
  - the agent keeps its previous state
  - `cfg.reward_collision` is applied

There is no agent-agent collision handling in the current implementation.

## Example Usage

### One action batch

```python
from env.config import SwarmConfig
from env.swarm_env import SwarmEnv

cfg = SwarmConfig(n_agents=2)
env = SwarmEnv(cfg, headless=True)
obs_dict, _ = env.reset(seed=0)

actions = {
    "agent_0": 7,  # forward_straight
    "agent_1": 3,  # idle_left
}
obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions)
env.close()
```

### Decode one action manually

```python
cfg = SwarmConfig()
env = SwarmEnv(cfg, headless=True)
print(env.action_table[7])  # (1.0, 0.0)
env.close()
```

## Important Notes

- The action interface has remained `Discrete(9)` through the recent environment upgrades.
- The meaning of action ids is defined only by the construction order in `_build_action_table()`.
- Any change to `num_actions` or the table order will break existing learned policies and any code that assumes the current 9-action layout.
