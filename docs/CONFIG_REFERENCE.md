# Configuration Reference

This document is a focused reference for the current configuration surface implemented by [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py) and the shared CLI/config helpers in [train/experiment_utils.py](/Users/christopherlin/dev/cwsf2026/sim/train/experiment_utils.py).

## Source of Truth

Primary config definition:

- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
  - `SwarmConfig`

CLI-to-config mapping helpers:

- [train/experiment_utils.py](/Users/christopherlin/dev/cwsf2026/sim/train/experiment_utils.py)
  - `add_env_config_args(parser)`
  - `make_swarm_config(args)`

## `SwarmConfig`

`SwarmConfig` is a dataclass. It is passed into `SwarmEnv(cfg, headless=...)` and drives world generation, observation shape, rewards, task behavior, rendering, and experiment conditions.

## Full Field Reference

| Field | Type | Default | Valid / Expected values | Affects |
| --- | --- | --- | --- | --- |
| `width` | `int` | `900` | positive integer | world width, obstacle and target placement, pheromone grid width, render window size |
| `height` | `int` | `600` | positive integer | world height, placement, pheromone grid height, render window size |
| `n_agents` | `int` | `6` | non-negative integer | number of agents, `possible_agents`, observation batch shape, number of Q-networks in independent mode |
| `n_targets` | `int` | `4` | non-negative integer | number of food targets spawned in `reset()` |
| `n_obstacles` | `int` | `6` | non-negative integer | number of random rectangular obstacles |
| `agent_radius` | `float` | `10.0` | positive | collision bounds, render radius, pheromone sampling spacing, target pickup reach, nest delivery reach |
| `target_radius` | `float` | `10.0` | positive | target render radius and pickup reach |
| `nest_radius` | `float` | `22.0` | positive | nest render radius and delivery reach |
| `max_steps` | `int` | `600` | positive integer | episode truncation threshold |
| `action_dim` | `int` | `1` | currently unused by env logic | retained field; not used to build the action space |
| `num_actions` | `int` | `9` | should match action-table size | action-space size returned by `action_space()` |
| `dt` | `float` | `0.1` | positive | timestep used by dynamics drivers |
| `max_speed` | `float` | `120.0` | positive | speed scale for actions and speed observation normalization |
| `max_yaw_rate` | `float` | `2.5` | positive | yaw-rate scale for actions |
| `accel` | `float` | `300.0` | positive | forward acceleration clamp in both drivers |
| `ang_accel` | `float` | `8.0` | positive | angular acceleration clamp in both drivers |
| `dynamics_mode` | `str` | `"tank"` | `"tank"`, `"hover"`, `"mixed"` | active driver selection |
| `hover_lat_damping` | `float` | `0.85` | usually `[0, 1]` | hovercraft lateral damping |
| `hover_lat_noise` | `float` | `5.0` | non-negative | hovercraft lateral Gaussian noise std |
| `hover_slip_chance` | `float` | `0.08` | `[0, 1]` | probability of slip event in hover mode |
| `hover_slip_scale` | `float` | `0.6` | non-negative | forward-speed multiplier during hover slip |
| `lidar_rays` | `int` | `9` | integer > 1 for current ray-angle formula | number of lidar observation features |
| `lidar_max_range` | `float` | `160.0` | positive | ray max distance and vector normalization denominator |
| `lidar_step` | `float` | `6.0` | positive | ray-marching step size |
| `obs_include_pheromone` | `bool` | `True` | boolean | if false, pheromone slots remain in obs but are filled with zeros |
| `pheromone_samples` | `int` | `3` | non-negative integer | number of pheromone observation values |
| `obs_include_food_presence` | `bool` | `True` | boolean | adds/removes the food-presence feature from obs layout |
| `obs_include_nest_direction` | `bool` | `True` | boolean | adds/removes the nest vector from obs layout |
| `obs_include_carrying` | `bool` | `True` | boolean | adds/removes the carrying-food feature from obs layout |
| `food_presence_radius` | `float` | `120.0` | positive | threshold for binary `food_presence` observation |
| `pheromone_enabled` | `bool` | `True` | boolean | enables pheromone-grid allocation and updates |
| `pheromone_cell_size` | `int` | `6` | positive integer | grid resolution and coordinate conversion |
| `pheromone_deposit` | `float` | `1.0` | non-negative | base pheromone deposit per agent per step |
| `pheromone_decay` | `float` | `0.985` | usually `[0, 1]` | multiplicative retention factor applied every step |
| `pheromone_diffuse_rate` | `float` | `0.25` | usually `[0, 1]` | interpolation weight for diffusion |
| `pheromone_deposit_carrying_scale` | `float` | `1.5` | non-negative | carry-state pheromone multiplier |
| `reward_target` | `float` | `8.0` | any float | immediate reward when targets do not require nest delivery |
| `reward_step` | `float` | `-0.01` | any float | step cost added to every agent every step |
| `reward_collision` | `float` | `-0.2` | any float | per-agent collision penalty |
| `reward_pickup` | `float` | `1.5` | any float | reward on pickup when nest delivery is required |
| `reward_nest_delivery` | `float` | `10.0` | any float | reward on successful nest return |
| `reward_exploration` | `float` | `0.02` | any float | reward per newly visited coverage cell |
| `reward_pheromone_following` | `float` | `0.0` | any float | multiplier for pheromone-usage shaping reward |
| `nest_enabled` | `bool` | `True` | boolean | enables nest spawn, render, and nest-direction observation |
| `require_nest_delivery` | `bool` | `True` | boolean | if true, food must be carried to the nest for full task completion |
| `coverage_cell_size` | `int` | `24` | positive integer | exploration-grid resolution |
| `failed_agent_count` | `int` | `0` | non-negative integer | number of agents randomly disabled per episode |
| `observation_noise_std` | `float` | `0.0` | non-negative | std of additive Gaussian observation noise |
| `render_pheromone` | `bool` | `True` | boolean | controls pheromone heatmap drawing |
| `render_scale` | `float` | `1.0` | currently unused | retained field, not used by the current renderer |
| `seed` | `int \| None` | `None` | integer or `None` | constructor RNG seed for environment instance |

## High-Impact Couplings

### Fields that change observation size

- `obs_include_nest_direction`
- `obs_include_food_presence`
- `obs_include_carrying`

These directly change `obs_dim` in `_compute_obs_dim()`.

### Fields that change pheromone behavior

- `pheromone_enabled`
  - disables the grid entirely
- `obs_include_pheromone`
  - leaves the slots in the observation but zero-fills them
- `render_pheromone`
  - only affects visualization
- `pheromone_deposit`
- `pheromone_decay`
- `pheromone_diffuse_rate`
- `pheromone_deposit_carrying_scale`

### Fields that change dynamics behavior

- `dynamics_mode`
- `dt`
- `max_speed`
- `max_yaw_rate`
- `accel`
- `ang_accel`
- all `hover_*` fields

### Fields used primarily for experiments

- `failed_agent_count`
- `observation_noise_std`
- `n_agents`
- `n_targets`
- `n_obstacles`
- `max_steps`

## CLI Mapping

The shared CLI config layer is implemented in [train/experiment_utils.py](/Users/christopherlin/dev/cwsf2026/sim/train/experiment_utils.py).

### Added CLI flags

`add_env_config_args(parser)` adds:

- `--n-targets`
- `--n-obstacles`
- `--max-steps-per-episode`
- `--dynamics-mode`
- `--pheromone-disabled`
- `--failed-agent-count`
- `--observation-noise-std`

### `make_swarm_config(args)`

`make_swarm_config(args)` maps CLI args into a `SwarmConfig` and also applies a coupling:

```python
pheromone_enabled = not getattr(args, "pheromone_disabled", False)
render_pheromone = pheromone_enabled
obs_include_pheromone = pheromone_enabled
```

So if `--pheromone-disabled` is passed:

- the pheromone grid is disabled
- pheromone rendering is disabled
- pheromone observation values become zeros

## Minimal Usage Examples

### Default config

```python
from env.config import SwarmConfig

cfg = SwarmConfig()
```

### Fewer agents and no pheromones

```python
cfg = SwarmConfig(
    n_agents=3,
    pheromone_enabled=False,
    render_pheromone=False,
    obs_include_pheromone=False,
)
```

### Mixed dynamics and noisy observations

```python
cfg = SwarmConfig(
    dynamics_mode="mixed",
    observation_noise_std=0.05,
)
```

### Build config from trainer/evaluator args

```python
from train.experiment_utils import make_swarm_config
from train.evaluate import parse_args

args = parse_args(["--n-agents", "5", "--pheromone-disabled"])
cfg = make_swarm_config(args)
```

## Implementation Notes

- `action_dim` is currently not used by the environment to construct the action space.
- `render_scale` is currently not used by the renderer.
- `pheromone_decay` acts as a multiplicative retention factor, not an explicit subtractive evaporation-rate parameter.
