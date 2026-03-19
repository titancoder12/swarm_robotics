# API Reference

This document is an implementation-grounded reference for the current swarm robotics simulation codebase. It is written for developers who want to inspect the environment contract, build training or evaluation scripts, or extend the simulator.

All behavior described here is based on the current implementation in:

- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py)
- [train/evaluate.py](/Users/christopherlin/dev/cwsf2026/sim/train/evaluate.py)
- [train/run_experiments.py](/Users/christopherlin/dev/cwsf2026/sim/train/run_experiments.py)
- [train/experiment_utils.py](/Users/christopherlin/dev/cwsf2026/sim/train/experiment_utils.py)
- [train/random_rollout.py](/Users/christopherlin/dev/cwsf2026/sim/train/random_rollout.py)
- [train/demo.py](/Users/christopherlin/dev/cwsf2026/sim/train/demo.py)
- [experiments/benchmark_configs.py](/Users/christopherlin/dev/cwsf2026/sim/experiments/benchmark_configs.py)
- [analysis/plot_metrics.py](/Users/christopherlin/dev/cwsf2026/sim/analysis/plot_metrics.py)
- [models/q_network.py](/Users/christopherlin/dev/cwsf2026/sim/models/q_network.py)
- [models/rule_based_policy.py](/Users/christopherlin/dev/cwsf2026/sim/models/rule_based_policy.py)

## 1. Environment Overview

### Primary class

- `SwarmEnv`
  - file: [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
  - base class: `pettingzoo.ParallelEnv`

### Related classes

- `AgentState`
  - file: [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
  - per-agent continuous state container
- `DynamicsDriver`
  - abstract driver interface
- `TankKinematicsDriver`
  - forward-speed plus yaw-rate kinematics with acceleration limits
- `HovercraftDriver`
  - forward-speed plus yaw-rate kinematics with lateral drift and stochastic slip
- `SwarmConfig`
  - file: [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
  - dataclass holding all environment parameters

### Instantiation

```python
from env.config import SwarmConfig
from env.swarm_env import SwarmEnv

cfg = SwarmConfig()
env = SwarmEnv(cfg, headless=True)
```

### Constructor

`SwarmEnv(cfg: SwarmConfig, headless: bool = False)`

Arguments:

- `cfg`
  - required
  - must be a `SwarmConfig` instance
- `headless`
  - if `True`, `render()` becomes a no-op and screenshots are disabled
  - if `False`, PyGame is initialized lazily on first render

### Supported modes

- headless training/evaluation mode: `headless=True`
- interactive render mode: `headless=False`

`SwarmEnv.metadata` is:

```python
{"name": "swarm_env_v0", "render_modes": ["human"], "is_parallel": True}
```

The `render()` method accepts a `mode` parameter but the current implementation ignores it and always renders to the PyGame window when not headless.

### Parallel API contract

#### `reset(seed=None, options=None) -> tuple[obs_dict, info_dict]`

Returns:

- `obs_dict: dict[str, np.ndarray]`
  - one observation vector per agent
- `info_dict: dict[str, dict]`
  - same initial info dict copied to each agent key

#### `step(action_dict) -> tuple[obs_dict, rewards_dict, terminations, truncations, infos]`

Arguments:

- `action_dict: dict[str, int]`
  - must include one discrete action for every active agent key in `self.agents`

Returns:

- `obs_dict: dict[str, np.ndarray]`
- `rewards_dict: dict[str, float]`
- `terminations: dict[str, bool]`
- `truncations: dict[str, bool]`
- `infos: dict[str, dict]`

#### `render(mode="human", fps=60) -> None`

- does nothing in headless mode
- otherwise draws the current world and flips the PyGame display
- limits the frame rate using `pygame.time.Clock.tick(fps)`

#### `close() -> None`

- shuts down PyGame only if it was initialized

### Episode lifecycle

`reset()` performs:

1. RNG reseeding if `seed` is provided
2. `step_count = 0`
3. clears termination flags
4. restores `self.agents = self.possible_agents[:]`
5. selects the active dynamics driver
6. spawns obstacles
7. spawns nest
8. spawns food targets
9. spawns agents
10. resets delivery and exploration state
11. assigns failed agents
12. creates or clears the pheromone grid
13. builds initial observations

### Termination and truncation rules

Implemented in `step()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py):

- `terminated = True` when:
  - `len(self.targets) == 0`
  - and no agent is still carrying food
- `truncated = True` when:
  - `self.step_count >= cfg.max_steps`

When either happens:

- `self.agents` is set to `[]`
- subsequent `step()` calls raise a `RuntimeError` until `reset()` is called

## 2. Quick Start

### Create and inspect the environment

```python
from env.config import SwarmConfig
from env.swarm_env import SwarmEnv

cfg = SwarmConfig(n_agents=3, pheromone_enabled=True)
env = SwarmEnv(cfg, headless=True)
obs_dict, info_dict = env.reset(seed=0)

print(env.possible_agents)
print(obs_dict["agent_0"].shape)
print(info_dict["agent_0"])
env.close()
```

### Step one action batch

```python
actions = {agent: 4 for agent in env.possible_agents}  # throttle=0, turn=0
obs_dict, rewards_dict, terminations, truncations, infos = env.step(actions)
```

### Short random rollout

```bash
python train/random_rollout.py
```

### Headless training

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000
```

### Headless evaluation

```bash
python train/evaluate.py --checkpoint-dir checkpoints --episodes 10 --output-dir runs/eval
```

### Probe a trained policy with hand-written observations

```bash
python train/policy_probe.py --list-cases
python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead
python train/policy_probe.py --checkpoint-dir checkpoints --case wall_ahead --agent-index 0
```

This loads a custom DQN checkpoint, prints the observation values by name, shows all 9 Q-values, and explains the chosen action in plain language.

## 3. Core Environment API

### Public attributes commonly used by scripts

- `env.cfg`
  - the active `SwarmConfig`
- `env.possible_agents`
  - fixed ordered list like `["agent_0", ..., "agent_n"]`
- `env.agents`
  - active agent list
  - emptied when an episode ends
- `env.action_table`
  - list of `(throttle, turn)` tuples
- `env.agent_states`
  - list of `AgentState`
- `env.targets`
  - list of `(x, y)` target coordinates
- `env.obstacles`
  - list of `pygame.Rect`
- `env.nest_position`
  - `(x, y)` tuple
- `env.pheromone_grid`
  - `np.ndarray` or `None`

### Space accessors

#### `observation_space(agent: str) -> gymnasium.spaces.Box`

- low: `-1.0`
- high: `1.0`
- shape: `(obs_dim,)`
- dtype: `np.float32`

#### `action_space(agent: str) -> gymnasium.spaces.Discrete`

- `Discrete(cfg.num_actions)`
- default `num_actions = 9`

### Agent state variables

`AgentState` fields:

- `x: float`
  - world x coordinate
  - units: world units, treated as pixels by rendering and geometry
- `y: float`
  - world y coordinate
- `theta: float`
  - heading in radians
- `v: float`
  - forward velocity
  - units: world units per second
- `omega: float`
  - angular velocity
  - units: radians per second
- `v_lat: float`
  - lateral velocity
  - only meaningful in hovercraft mode
- `carrying_food: bool`
  - whether the agent currently carries one food item

## 4. Observation Vector Specification

### Observation shape

Current default observation shape is `(23,)` per agent.

Exact formula from `_compute_obs_dim()`:

```python
cfg.lidar_rays
+ 2
+ (2 if cfg.obs_include_nest_direction else 0)
+ 2
+ 2
+ 1
+ (1 if cfg.obs_include_food_presence else 0)
+ (1 if cfg.obs_include_carrying else 0)
+ cfg.pheromone_samples
```

With the default config:

- `lidar_rays = 9`
- `obs_include_nest_direction = True`
- `obs_include_food_presence = True`
- `obs_include_carrying = True`
- `pheromone_samples = 3`

So:

- `9 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 3 = 23`

### Observation assembly order

Implemented in `_get_obs()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py):

1. lidar
2. nearest target vector
3. nest direction vector, if enabled
4. nearest neighbor vector
5. heading as `sin(theta), cos(theta)`
6. normalized forward speed
7. food-presence flag, if enabled
8. carrying-food flag, if enabled
9. pheromone samples

### Observation semantics

All observation values are local per-agent observations. No global shared state is exposed directly.

All observation arrays are `np.float32`.

Most continuous components are clipped to `[-1, 1]` either by construction or by the final observation noise step.

### Coordinate frame

Relative vectors are converted from world frame to the agent body frame by `_to_agent_frame(vec, theta)`.

Implementation:

```python
c = cos(-theta)
s = sin(-theta)
[x', y'] = [c * x - s * y, s * x + c * y]
```

This means:

- the input vector is first formed in world coordinates
- then rotated into the agent’s local frame
- positive and negative `y` values are side-relative in body coordinates

### Observation noise and failure masking

Applied at the end of `_get_obs()`:

- if the agent index is in `failed_agent_indices`
  - the entire observation vector is set to `0.0`
- else if `cfg.observation_noise_std > 0`
  - Gaussian noise is sampled with:
    - mean `0.0`
    - std `cfg.observation_noise_std`
  - added elementwise
  - result is clipped to `[-1.0, 1.0]`

### Detailed observation table

Default index layout with all current optional features enabled:

| Index | Name | Meaning | Source in code | Range | Unit | Normalization | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `lidar_0` | Ray distance sample 0 | `_lidar_scan()` | `[0, 1]` | fraction of `lidar_max_range` | `dist / cfg.lidar_max_range` | combines walls and obstacles; ray cast in body-relative angle pattern |
| 1 | `lidar_1` | Ray distance sample 1 | `_lidar_scan()` | `[0, 1]` | fraction of `lidar_max_range` | same as above | same dtype and behavior |
| 2 | `lidar_2` | Ray distance sample 2 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 3 | `lidar_3` | Ray distance sample 3 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 4 | `lidar_4` | Ray distance sample 4 | `_lidar_scan()` | `[0, 1]` | fraction | same | central ray when `lidar_rays = 9` |
| 5 | `lidar_5` | Ray distance sample 5 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 6 | `lidar_6` | Ray distance sample 6 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 7 | `lidar_7` | Ray distance sample 7 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 8 | `lidar_8` | Ray distance sample 8 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 9 | `target_dx_body_norm` | x component of nearest target vector in agent frame | `_nearest_target_vector()` | `[-1, 1]` | fraction of `lidar_max_range` | world delta rotated to body frame, then divided by `cfg.lidar_max_range`, then clipped | if no targets remain, becomes `0.0` |
| 10 | `target_dy_body_norm` | y component of nearest target vector in agent frame | `_nearest_target_vector()` | `[-1, 1]` | fraction of `lidar_max_range` | same as above | nearest target overall, not line-of-sight filtered |
| 11 | `nest_dx_body_norm` | x component of nest vector in agent frame | `_nest_direction()` | `[-1, 1]` | fraction of `lidar_max_range` | world nest delta rotated to body frame, divided by `cfg.lidar_max_range`, clipped | if `nest_enabled` is `False`, returns `0.0`; omitted entirely if `obs_include_nest_direction` is `False` |
| 12 | `nest_dy_body_norm` | y component of nest vector in agent frame | `_nest_direction()` | `[-1, 1]` | fraction of `lidar_max_range` | same | same |
| 13 | `neighbor_dx_body_norm` | x component of nearest-agent vector in agent frame | `_nearest_agent_vector()` | `[-1, 1]` | fraction of `lidar_max_range` | nearest-agent world delta rotated to body frame, divided by `cfg.lidar_max_range`, clipped | if `n_agents <= 1`, returns `0.0` |
| 14 | `neighbor_dy_body_norm` | y component of nearest-agent vector in agent frame | `_nearest_agent_vector()` | `[-1, 1]` | fraction of `lidar_max_range` | same | nearest neighbor only, not aggregated over all agents |
| 15 | `heading_sin` | `sin(theta)` | `_get_obs()` | `[-1, 1]` | unitless | direct trigonometric transform | `theta` stored in radians |
| 16 | `heading_cos` | `cos(theta)` | `_get_obs()` | `[-1, 1]` | unitless | direct trigonometric transform | same |
| 17 | `speed_norm` | normalized forward speed | `_get_obs()` | `[-1, 1]` | fraction of `max_speed` | `clip(agent.v / cfg.max_speed, -1, 1)` | uses forward velocity only, not lateral speed |
| 18 | `food_presence` | whether any target is within local food radius | `_food_presence()` | `{0, 1}` | binary flag | `1.0 if nearest_target_distance <= cfg.food_presence_radius else 0.0` | if no targets exist, returns `0.0`; omitted entirely if `obs_include_food_presence` is `False` |
| 19 | `carrying_food` | whether this agent is carrying food | `_carrying_food()` | `{0, 1}` | binary flag | `1.0 if agent.carrying_food else 0.0` | omitted entirely if `obs_include_carrying` is `False` |
| 20 | `pheromone_sample_0` | forward pheromone sample 1 | `_pheromone_samples()` | `[0, 1]` when enabled, else `0.0` | local relative grid intensity | sample raw grid at forward offset, then normalize the sample vector by its own max if that max is positive | not normalized by global grid max |
| 21 | `pheromone_sample_1` | forward pheromone sample 2 | `_pheromone_samples()` | `[0, 1]` when enabled, else `0.0` | same | same | same |
| 22 | `pheromone_sample_2` | forward pheromone sample 3 | `_pheromone_samples()` | `[0, 1]` when enabled, else `0.0` | same | same | same |

### Lidar specification

Implemented by `_lidar_scan()` and `_ray_distance()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py).

Current defaults:

- `cfg.lidar_rays = 9`
- `cfg.lidar_max_range = 160.0`
- `cfg.lidar_step = 6.0`

Ray angles:

```python
angle = agent.theta + (i - half) * (pi / (lidar_rays - 1))
half = lidar_rays // 2
```

With 9 rays, offsets are:

- `-pi/2`
- `-3pi/8`
- `-pi/4`
- `-pi/8`
- `0`
- `pi/8`
- `pi/4`
- `3pi/8`
- `pi/2`

Behavior:

- rays march forward from `(x, y)` in steps of `cfg.lidar_step`
- the first hit against either:
  - an arena boundary
  - any obstacle rectangle
  ends the ray
- no distinction is made between wall and obstacle hits
- if no hit occurs before max range, the returned distance is `cfg.lidar_max_range`

Output:

- raw returned distance is divided by `cfg.lidar_max_range`
- so each lidar element is in `[0, 1]`

### Target-relative features

Implemented by `_nearest_target_vector()`.

Behavior:

- uses the nearest target by Euclidean distance over all remaining targets
- does not check visibility or line of sight
- computes world delta:
  - `[target_x - agent.x, target_y - agent.y]`
- rotates into body frame using `_to_agent_frame`
- divides both components by `cfg.lidar_max_range`
- clips each component to `[-1, 1]`
- if there are no targets, returns `[0.0, 0.0]`

### Nest-direction features

Implemented by `_nest_direction()`.

Behavior:

- computes vector from agent to nest in world frame
- rotates into body frame
- divides by `cfg.lidar_max_range`
- clips to `[-1, 1]`
- if `cfg.nest_enabled` is `False`, returns `[0.0, 0.0]`

### Neighbor features

Implemented by `_nearest_agent_vector()`.

Behavior:

- considers only the nearest other agent
- not an average or pooled representation
- computes world delta to nearest agent
- rotates into body frame
- normalizes by `cfg.lidar_max_range`
- clips to `[-1, 1]`
- if `cfg.n_agents <= 1`, returns `[0.0, 0.0]`

### Heading features

Implemented directly in `_get_obs()`.

Behavior:

- heading is encoded as:
  - `sin(theta)`
  - `cos(theta)`
- `theta` is always stored in radians
- raw `theta` itself is not included in the observation

### Speed feature

Implemented directly in `_get_obs()`.

Behavior:

- scalar forward speed only
- `agent.v` is divided by `cfg.max_speed`
- clipped to `[-1, 1]`
- lateral speed `v_lat` is not observed

### Food-presence feature

Implemented by `_food_presence()`.

Behavior:

- computes Euclidean distance to the nearest remaining target
- returns `[1.0]` if that distance is `<= cfg.food_presence_radius`
- otherwise returns `[0.0]`
- if there are no targets, returns `[0.0]`

### Carrying-food feature

Implemented by `_carrying_food()`.

Behavior:

- binary flag from `agent.carrying_food`
- `1.0` if carrying
- `0.0` otherwise

### Pheromone features

Implemented by `_pheromone_samples()`.

Current defaults:

- `cfg.pheromone_samples = 3`
- `cfg.pheromone_cell_size = 6`
- sample distance for index `i` is:
  - `(i + 1) * cfg.agent_radius * 1.5`

Sampling behavior:

1. sample points are placed along the agent’s current forward direction
2. each sample point is converted to pheromone-grid cell coordinates using floor division by `pheromone_cell_size`
3. raw grid values are collected
4. if the maximum sampled value is positive:
   - the whole sample vector is divided by `(samples.max() + 1e-6)`

Implications:

- pheromone samples are local relative intensities, not absolute global concentrations
- if all sampled values are zero, the returned vector is all zeros
- if pheromone sensing is disabled or pheromone is globally disabled, the returned vector is zeros

## 5. Action Space Specification

### Action shape

- scalar discrete integer per agent
- action space: `Discrete(9)`

### Action mapping

Implemented by `_build_action_table()` in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py).

Construction order:

```python
throttle_vals = [-1.0, 0.0, 1.0]
turn_vals = [-1.0, 0.0, 1.0]
for throttle in throttle_vals:
    for turn in turn_vals:
        table.append((throttle, turn))
```

So the actual index mapping is:

| Action / Index | Name | Meaning | Range | Unit | Internal interpretation | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `reverse_left` | reverse throttle, turn left | throttle `-1`, turn `-1` | unitless command | target forward speed `-max_speed`, target yaw rate `-max_yaw_rate` | exact left/right sign meaning is defined by dynamics and body-frame convention |
| 1 | `reverse_straight` | reverse throttle, no turn | `-1`, `0` | unitless command | target speed `-max_speed`, target yaw rate `0` | same |
| 2 | `reverse_right` | reverse throttle, turn right | `-1`, `1` | unitless command | target speed `-max_speed`, target yaw rate `+max_yaw_rate` | same |
| 3 | `idle_left` | zero throttle, turn left | `0`, `-1` | unitless command | target speed `0`, target yaw rate `-max_yaw_rate` | pure turning command under both drivers |
| 4 | `idle` | zero throttle, zero turn | `0`, `0` | unitless command | target speed `0`, target yaw rate `0` | stop-like command, though velocity may decay rather than snap to zero |
| 5 | `idle_right` | zero throttle, turn right | `0`, `1` | unitless command | target speed `0`, target yaw rate `+max_yaw_rate` | same |
| 6 | `forward_left` | forward throttle, turn left | `1`, `-1` | unitless command | target speed `+max_speed`, target yaw rate `-max_yaw_rate` | common exploratory action |
| 7 | `forward_straight` | forward throttle, no turn | `1`, `0` | unitless command | target speed `+max_speed`, target yaw rate `0` | straight drive |
| 8 | `forward_right` | forward throttle, turn right | `1`, `1` | unitless command | target speed `+max_speed`, target yaw rate `+max_yaw_rate` | same |

### Action semantics

Actions are high-level motion commands, not direct motor commands.

They are interpreted by the active `DynamicsDriver` as target forward speed and target yaw rate, subject to acceleration limits.

### Invalid actions

The `step()` method does not explicitly validate numeric action bounds before indexing `self.action_table`.

Current behavior:

- missing agent key in `action_dict`
  - raises `ValueError`
- non-dict `actions`
  - raises `ValueError`
- action index outside `[0, 8]`
  - will fail when indexing `self.action_table[action_id]`
  - current failure mode is an index error from Python list access

### Do different dynamics modes interpret actions differently?

Yes, partially.

The same `(throttle, turn)` pair is interpreted through the same target-speed and target-yaw-rate idea, but:

- `TankKinematicsDriver`
  - updates forward velocity and yaw only
- `HovercraftDriver`
  - adds lateral drift, Gaussian lateral noise, and occasional slip

So the action interface is shared, but motion outcomes differ by driver.

## 6. Rewards, Termination, and Info

### Reward terms

Rewards are assembled in `step()` and helper methods in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py).

Per-step initialization:

- every agent starts with `cfg.reward_step`

Additional terms:

- collision penalty
  - added to the colliding agent only
  - value: `cfg.reward_collision`
- pickup reward
  - if `require_nest_delivery and nest_enabled`
    - pickup gives `cfg.reward_pickup`
  - else
    - immediate collection gives `cfg.reward_target`
- nest-delivery reward
  - carrying agent reaching the nest gets `cfg.reward_nest_delivery`
- exploration reward
  - computed as:
    - `new_cells * cfg.reward_exploration`
  - then divided equally among all agents
- pheromone-following shaping
  - computed as:
    - `cfg.reward_pheromone_following * usage * cfg.n_agents`
  - then divided equally among all agents

### Reward bookkeeping in info

`reward_breakdown` is included in each info dict with keys:

- `step`
- `pickup`
- `delivery`
- `collision`
- `exploration`
- `pheromone`

Important implementation detail:

- `reward_breakdown["step"]` is set to `cfg.reward_step * cfg.n_agents`
- pickup and delivery totals are episode-step totals over all agents

### Info dict contents

Each agent gets the same info dict each step:

| Key | Type | Meaning |
| --- | --- | --- |
| `targets_collected` | `int` | number of targets picked up during this step |
| `food_delivered` | `int` | number of carried food items delivered to nest during this step |
| `collisions` | `int` | number of agents that collided during this step |
| `new_cells_visited` | `int` | number of previously unseen coverage cells visited this step |
| `exploration_coverage` | `float` | current visited-cell fraction |
| `pheromone_usage` | `float` | mean normalized pheromone under current agent positions |
| `episode_length` | `int` | current `step_count` |
| `failed_agents` | `int` | number of failed agents for this episode |
| `reward_breakdown` | `dict` | step-level reward component totals |

### Termination recap

- `terminated`
  - all food targets are gone
  - and no agent is still carrying food
- `truncated`
  - `step_count >= cfg.max_steps`

## 7. Dynamics and State Transitions

### Driver interface

`DynamicsDriver.apply(state, action, dt, cfg, rng) -> AgentState`

Inputs:

- `state: AgentState`
- `action: tuple[float, float]`
  - `(throttle, turn)`
- `dt: float`
- `cfg: SwarmConfig`
- `rng: np.random.Generator`

Output:

- proposed next `AgentState`

### Tank kinematics

Implemented by `TankKinematicsDriver.apply()`.

Update order:

1. map action to:
   - `target_v = throttle * cfg.max_speed`
   - `target_omega = turn * cfg.max_yaw_rate`
2. acceleration limiting:
   - `dv = clip(target_v - state.v, -cfg.accel * dt, cfg.accel * dt)`
   - `domega = clip(target_omega - state.omega, -cfg.ang_accel * dt, cfg.ang_accel * dt)`
3. integrate:
   - `v = state.v + dv`
   - `omega = state.omega + domega`
   - `theta = state.theta + omega * dt`
4. forward motion:
   - `nx = state.x + cos(theta) * v * dt`
   - `ny = state.y + sin(theta) * v * dt`

### Hovercraft dynamics

Implemented by `HovercraftDriver.apply()`.

Same target-speed and target-yaw-rate logic, plus:

- lateral velocity damping:
  - `v_lat = state.v_lat * cfg.hover_lat_damping`
- additive lateral noise:
  - `v_lat += rng.normal(0.0, cfg.hover_lat_noise)`
- random slip:
  - if `rng.random() < cfg.hover_slip_chance`
    - `v *= cfg.hover_slip_scale`
- motion vector:
  - forward vector from current `theta`
  - right vector from `theta + pi / 2`
  - `vel = forward * v + right * v_lat`
- integrate:
  - `nx = state.x + vel[0] * dt`
  - `ny = state.y + vel[1] * dt`

### Driver selection

Implemented by `reset()` and `_select_driver()`.

Behavior:

- if `cfg.dynamics_mode == "mixed"`
  - `reset()` randomly selects `"tank"` or `"hover"` for the whole episode
- else
  - `reset()` selects the configured driver directly

Important implementation note:

- `_select_driver("mixed")` itself falls back to `TankKinematicsDriver`
- mixed-mode randomness is handled in `reset()`, not in `_select_driver()`

### Collision handling

Implemented by `_handle_collisions()`.

Current checks:

- wall bounds against `cfg.agent_radius`
- obstacle rectangle overlap using a bounding `pygame.Rect`

Current non-checks:

- no agent-agent collision handling
- no collision response beyond rejecting the move

If collision occurs:

- agent state is not updated
- collision penalty is applied

### Target collection

Implemented by `_handle_targets()`.

A target is collected if:

```python
(agent.x - tx)^2 + (agent.y - ty)^2 <= (target_radius + agent_radius)^2
```

Behavior:

- the first agent found in the loop claims the target
- collected targets are removed from `self.targets`
- if nest delivery is required:
  - the agent gets `carrying_food = True`
  - pickup reward is applied
- otherwise:
  - target reward is applied immediately

### Nest delivery

Implemented by `_handle_nest_delivery()`.

A carrying agent delivers when:

```python
(agent.x - nx)^2 + (agent.y - ny)^2 <= (nest_radius + agent_radius)^2
```

Behavior:

- `carrying_food` becomes `False`
- `food_delivered` counter increments
- `reward_nest_delivery` is added

### Stochasticity

Current stochastic elements:

- agent initial headings
- random obstacle placement
- random target placement
- random agent placement
- random failed-agent selection
- hovercraft lateral noise
- hovercraft slip events
- optional observation noise
- training-time epsilon-greedy action randomness

## 8. Pheromone / Stigmergy System

### Storage

- `self.pheromone_grid`
- shape:
  - `(height // pheromone_cell_size + 1, width // pheromone_cell_size + 1)`
- dtype: `np.float32`

### Deposit

Implemented by `_update_pheromone()`.

Per agent:

1. world position is converted to grid coordinates:
   - `gx = int(agent.x // cell)`
   - `gy = int(agent.y // cell)`
2. deposit amount starts as `cfg.pheromone_deposit`
3. if `agent.carrying_food`:
   - deposit is multiplied by `cfg.pheromone_deposit_carrying_scale`
4. deposit is added to the current grid cell

### Decay

Implemented as:

```python
grid *= (1.0 - (1.0 - cfg.pheromone_decay))
```

This simplifies exactly to:

```python
grid *= cfg.pheromone_decay
```

So current semantics are:

- `pheromone_decay` is multiplicative retention, not evaporation rate
- default `0.985` means values keep 98.5 percent of their current magnitude each step before diffusion

### Diffusion

If `cfg.pheromone_diffuse_rate > 0`:

1. neighbor grids are computed using `np.roll`
2. four-neighbor average is formed
3. grid is updated as:

```python
grid[:] = grid * (1.0 - diff) + neighbor_avg * diff
```

Important implementation note:

- `np.roll` wraps around edges
- so diffusion currently has wraparound boundary behavior, not clamped or zero-flux boundaries

### Usage metric

Implemented by `_mean_pheromone_usage()`.

Behavior:

- reads pheromone at each agent’s current cell
- divides by the global current grid maximum
- averages over agents
- returns `0.0` if grid is empty or the maximum is near zero

This metric is different from the observation pheromone samples:

- usage metric normalizes by global grid maximum
- observation pheromone samples normalize by local sample maximum

## 9. Configuration Reference

All configuration fields are in `SwarmConfig` in [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py).

| Field | Type | Default | Valid / Expected values | Effect |
| --- | --- | --- | --- | --- |
| `width` | `int` | `900` | positive integer | world width; affects placement, rendering, pheromone grid shape |
| `height` | `int` | `600` | positive integer | world height |
| `n_agents` | `int` | `6` | non-negative integer | number of agents; changes number of agent ids, reset state, and observation batch shape |
| `n_targets` | `int` | `4` | non-negative integer | number of spawned food targets |
| `n_obstacles` | `int` | `6` | non-negative integer | number of obstacle rectangles |
| `agent_radius` | `float` | `10.0` | positive | affects collision checks, sampling distances, delivery reach, render size |
| `target_radius` | `float` | `10.0` | positive | affects pickup reach and render size |
| `nest_radius` | `float` | `22.0` | positive | affects nest render and delivery reach |
| `max_steps` | `int` | `600` | positive integer | truncation horizon |
| `action_dim` | `int` | `1` | currently unused by env logic | retained field; action space is actually controlled by `num_actions` |
| `num_actions` | `int` | `9` | expected to match action table length | action space size |
| `dt` | `float` | `0.1` | positive | physics timestep in seconds |
| `max_speed` | `float` | `120.0` | positive | forward speed scale and speed observation normalization denominator |
| `max_yaw_rate` | `float` | `2.5` | positive | target yaw-rate scale in radians per second |
| `accel` | `float` | `300.0` | positive | forward acceleration limit |
| `ang_accel` | `float` | `8.0` | positive | yaw acceleration limit |
| `dynamics_mode` | `str` | `"tank"` | `"tank"`, `"hover"`, `"mixed"` | driver selection |
| `hover_lat_damping` | `float` | `0.85` | typically `[0, 1]` | hover lateral damping |
| `hover_lat_noise` | `float` | `5.0` | non-negative | hover lateral Gaussian noise std |
| `hover_slip_chance` | `float` | `0.08` | `[0, 1]` | hover slip probability per step |
| `hover_slip_scale` | `float` | `0.6` | non-negative | forward speed multiplier on slip |
| `lidar_rays` | `int` | `9` | integer greater than 1 in current implementation | number of lidar features; also used in angle denominator `(lidar_rays - 1)` |
| `lidar_max_range` | `float` | `160.0` | positive | max ray length and vector normalization denominator |
| `lidar_step` | `float` | `6.0` | positive | ray marching increment |
| `obs_include_pheromone` | `bool` | `True` | boolean | if false, pheromone samples become zeros though slots remain in obs |
| `pheromone_samples` | `int` | `3` | non-negative integer | number of pheromone sample features |
| `obs_include_food_presence` | `bool` | `True` | boolean | adds or removes the food-presence feature from observation size |
| `obs_include_nest_direction` | `bool` | `True` | boolean | adds or removes nest-direction vector from observation size |
| `obs_include_carrying` | `bool` | `True` | boolean | adds or removes carrying-food feature from observation size |
| `food_presence_radius` | `float` | `120.0` | positive | threshold distance for binary food-presence flag |
| `pheromone_enabled` | `bool` | `True` | boolean | enables pheromone grid creation and updates |
| `pheromone_cell_size` | `int` | `6` | positive integer | grid resolution and coordinate conversion |
| `pheromone_deposit` | `float` | `1.0` | non-negative | base pheromone deposit per step per agent |
| `pheromone_decay` | `float` | `0.985` | typically `[0, 1]` | multiplicative retention factor each step |
| `pheromone_diffuse_rate` | `float` | `0.25` | typically `[0, 1]` | interpolation weight for diffusion |
| `pheromone_deposit_carrying_scale` | `float` | `1.5` | non-negative | multiplier applied when carrying food |
| `reward_target` | `float` | `8.0` | any float | immediate reward if nest delivery is not required |
| `reward_step` | `float` | `-0.01` | any float | base reward applied to every agent every step |
| `reward_collision` | `float` | `-0.2` | any float | per-agent collision penalty |
| `reward_pickup` | `float` | `1.5` | any float | reward when a target is picked up and carried |
| `reward_nest_delivery` | `float` | `10.0` | any float | reward on nest delivery |
| `reward_exploration` | `float` | `0.02` | any float | reward per newly visited coverage cell |
| `reward_pheromone_following` | `float` | `0.0` | any float | multiplier for pheromone-usage shaping |
| `nest_enabled` | `bool` | `True` | boolean | enables nest spawn, nest render, and nest-direction obs |
| `require_nest_delivery` | `bool` | `True` | boolean | if true, pickup and delivery are separate events |
| `coverage_cell_size` | `int` | `24` | positive integer | exploration grid resolution |
| `failed_agent_count` | `int` | `0` | non-negative integer | number of agents to zero-mask and freeze each episode |
| `observation_noise_std` | `float` | `0.0` | non-negative | std of additive Gaussian obs noise |
| `render_pheromone` | `bool` | `True` | boolean | controls pheromone heatmap rendering |
| `render_scale` | `float` | `1.0` | currently unused by render implementation | retained field |
| `seed` | `int \| None` | `None` | integer or `None` | seed for constructor RNG; `reset(seed=...)` overrides active RNG |

### Config couplings that matter

- `obs_include_nest_direction`, `obs_include_food_presence`, and `obs_include_carrying`
  - change `obs_dim`
- `pheromone_enabled = False`
  - disables grid creation and updates
  - observation pheromone samples become zeros because `_pheromone_samples()` checks `pheromone_enabled`
- `obs_include_pheromone = False`
  - pheromone sample slots remain in the observation but values become zeros
- `dynamics_mode = "mixed"`
  - random driver selection is per episode
- `failed_agent_count > 0`
  - failed agents still exist in `agent_states`, but do not move and observe all zeros

## 10. Developer APIs for Training / Evaluation / Experiments

### Training entry points

#### `train.independent_dqn_pytorch.parse_args(argv=None)`

- file: [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py)
- purpose: CLI parser for custom DQN training

Important arguments:

- `--total-steps`
- `--n-agents`
- `--shared-policy`
- `--headless`
- `--cuda`
- `--seed`
- `--save-dir`
- `--save-every`
- `--experiment-name`
- `--output-dir`
- `--eval-every`
- `--eval-episodes`
- plus shared env args from `add_env_config_args()`

#### `train.independent_dqn_pytorch.train(args) -> str`

- purpose: run custom DQN training
- returns: `run_dir` path
- side effects:
  - creates run directory
  - writes CSVs and JSON summaries
  - saves checkpoints

Minimal usage:

```python
from train.independent_dqn_pytorch import parse_args, train

args = parse_args(["--headless", "--total-steps", "1000"])
run_dir = train(args)
```

#### `ReplayBuffer`

- file: [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py)
- purpose: fixed-capacity FIFO experience storage

Methods:

- `add(obs, action, reward, next_obs, done)`
- `sample(batch_size)`

#### `DQNConfig`

- dataclass of hyperparameters used internally by the custom trainer

### Evaluation entry points

#### `train.evaluate.parse_args(argv=None)`

- file: [train/evaluate.py](/Users/christopherlin/dev/cwsf2026/sim/train/evaluate.py)

Important arguments:

- `--checkpoint-dir`
- `--shared-policy`
- `--policy-kind {dqn, rule_based}`
- `--episodes`
- `--n-agents`
- `--seed`
- `--output-dir`
- shared env args via `add_env_config_args()`

#### `train.evaluate.run(args) -> str`

- purpose: evaluate a policy in headless mode
- returns: output directory
- supported policy kinds:
  - `dqn`
  - `rule_based`

Minimal usage:

```python
from train.evaluate import parse_args, run

args = parse_args(["--policy-kind", "rule_based", "--episodes", "5"])
out_dir = run(args)
```

### Policy probing utility

#### `train/policy_probe.py`

- file: [train/policy_probe.py](/Users/christopherlin/dev/cwsf2026/sim/train/policy_probe.py)
- purpose: manually feed named 23-dimensional observation vectors into a trained custom DQN checkpoint and inspect outputs

Typical commands:

```bash
python train/policy_probe.py --list-cases
python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead
python train/policy_probe.py --checkpoint-dir checkpoints --case carrying_to_nest --agent-index 0
```

Behavior:

- loads `shared.pt` when `--shared-policy` is set
- otherwise loads `agent_<index>.pt`
- uses the current 23-element observation layout
- prints observation names and values
- prints all Q-values
- prints chosen action index, name, and explanation

### Experiment utilities

#### `make_run_dir(output_dir, experiment_name) -> str`

- creates a timestamped run folder

#### `write_json(path, payload) -> None`

- writes stable sorted JSON

#### `CSVLogger(path, fieldnames)`

Methods:

- `log(row)`
- `close()`

#### `plot_training_metrics(csv_path, out_dir) -> None`

- reads `episode_metrics.csv`
- writes:
  - `reward_vs_episode.png`
  - `food_retrieval_vs_episode.png`
  - `swarm_efficiency_vs_episode.png`

#### `add_env_config_args(parser) -> None`

Adds shared CLI flags:

- `--n-targets`
- `--n-obstacles`
- `--max-steps-per-episode`
- `--dynamics-mode`
- `--pheromone-disabled`
- `--failed-agent-count`
- `--observation-noise-std`

#### `make_swarm_config(args) -> SwarmConfig`

- builds `SwarmConfig` from parsed CLI args
- also sets:
  - `render_pheromone = pheromone_enabled`
  - `obs_include_pheromone = pheromone_enabled`

#### `load_csv_rows(path) -> list[dict[str, str]]`

- convenience loader for result CSVs

#### `aggregate_rows(rows, metric_keys) -> dict[str, float]`

- computes `<metric>_mean` and `<metric>_std`

### Experiment runner

#### `experiments.benchmark_configs.get_experiment_cases(name)`

- file: [experiments/benchmark_configs.py](/Users/christopherlin/dev/cwsf2026/sim/experiments/benchmark_configs.py)
- returns:
  - a registry dict for `"all"`
  - or a single `{name: spec}` dict

Experiment spec shape:

```python
{
    "default_trials": int,
    "cases": [
        {
            "case_name": str,
            "train_args": list[str],
            # optional:
            "eval_args": list[str],
            "skip_training": bool,
            "n_agents": int,
            "condition": str,
            "algorithm": str,
            "failed_agents": int,
            "noise_std": float,
        }
    ],
}
```

#### `train.run_experiments.parse_args(argv=None)`

Important arguments:

- `--experiment`
- `--trials`
- `--seed`
- `--total-steps`
- `--eval-every`
- `--eval-episodes`
- `--runs-dir`
- `--results-dir`
- `--analysis-dir`
- `--save-every`
- `--headless`
- `--cuda`
- `--no-plots`

#### `train.run_experiments.run_experiments(args) -> None`

- loops over benchmark cases
- runs training unless `skip_training` is set
- always runs shared evaluation
- aggregates trial metrics
- writes experiment CSVs
- generates comparison plots

### Demo and rollout utilities

#### `train.random_rollout.main()`

- creates default env
- runs random actions for up to 300 steps
- renders continuously
- validates reward and obs shapes

#### `train.demo.parse_args(argv=None)`

- CLI for interactive demonstrations
- supports backends:
  - `custom`
  - `sb3`
  - `rllib`

#### `train.demo.load_models(...)`

- loads custom DQN checkpoints into `QNetwork`

#### `train.demo.main()`

- constructs env
- dispatches to backend-specific demo loop

#### `train.train.main()`

- simple backend dispatcher between:
  - custom
  - sb3
  - rllib

### Plotting APIs

#### `analysis.plot_metrics.main(argv=None)`

- CLI wrapper for single-run training plots

#### `plot_mean_std_curve(...)`

- plots mean curves with std shading across multiple runs

#### `plot_bar_with_error(...)`

- plots bar charts with error bars

#### `plot_grouped_errorbar(...)`

- plots grouped mean/std lines versus a numeric x-axis

## 11. Extension Guide

### Add a new observation component

Files to modify:

- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- optionally [models/rule_based_policy.py](/Users/christopherlin/dev/cwsf2026/sim/models/rule_based_policy.py)
- optionally [robot/observation_builder.py](/Users/christopherlin/dev/cwsf2026/sim/robot/observation_builder.py)

Steps:

1. add any config flags or thresholds to `SwarmConfig`
2. update `_compute_obs_dim()`
3. implement a helper like `_my_feature(...)`
4. insert the new component in `_get_obs()` at the exact intended location
5. update any policy code that assumes the old layout
6. retrain models, because checkpoint compatibility will usually break

### Add a new reward term

Files to modify:

- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)

Typical insertion points:

- directly in `step()`
- as a helper like `_apply_exploration_reward()`
- update `reward_breakdown` if you want it logged

If you want the trainer and evaluator to expose the new signal, also update:

- [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py)
- [train/evaluate.py](/Users/christopherlin/dev/cwsf2026/sim/train/evaluate.py)

### Add a new action

Files to modify:

- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- [models/q_network.py](/Users/christopherlin/dev/cwsf2026/sim/models/q_network.py) only if action count changes in downstream code assumptions
- any policy code that decodes the current 9-action layout

Steps:

1. update `cfg.num_actions`
2. update `_build_action_table()`
3. ensure `action_space()` matches the new size
4. update any hard-coded action lookup tables, such as:
   - [models/rule_based_policy.py](/Users/christopherlin/dev/cwsf2026/sim/models/rule_based_policy.py)
5. retrain checkpoints

### Add a new dynamics mode

Files to modify:

- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- optionally [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)

Steps:

1. implement a new `DynamicsDriver` subclass with `apply(...)`
2. extend `_select_driver()` to return it
3. extend `SwarmConfig.dynamics_mode` documentation and any CLI parser choices if needed

### Add a new task or scenario

Likely files:

- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
- [experiments/benchmark_configs.py](/Users/christopherlin/dev/cwsf2026/sim/experiments/benchmark_configs.py)

Typical task hooks:

- spawn functions:
  - `_spawn_obstacles()`
  - `_spawn_targets()`
  - `_spawn_nest()`
- transition helpers:
  - `_handle_targets()`
  - `_handle_nest_delivery()`
- reward helpers
- observation helpers

### Add a new experiment

Files to modify:

- [experiments/benchmark_configs.py](/Users/christopherlin/dev/cwsf2026/sim/experiments/benchmark_configs.py)
- optionally [train/run_experiments.py](/Users/christopherlin/dev/cwsf2026/sim/train/run_experiments.py)
- optionally [analysis/plot_metrics.py](/Users/christopherlin/dev/cwsf2026/sim/analysis/plot_metrics.py)

If the new experiment fits the existing runner pattern:

1. add a new registry entry in `get_experiment_cases()`
2. define case metadata and CLI args
3. reuse `run_experiments()`

If the new experiment needs special plots:

4. add a new plotting branch in `run_experiments()`
5. reuse `plot_grouped_errorbar()` or `plot_bar_with_error()` where possible

## 12. Known Gaps / Implementation Notes

### Implementation-grounded notes

- agent-agent collisions are not modeled
- observation pheromone samples use local max normalization, while `pheromone_usage` uses global max normalization
- failed agents remain in the world but:
  - do not move
  - observe zeros
  - can still be seen by other agents through the nearest-neighbor calculation because `_nearest_agent_vector()` does not exclude them
- `_sample_free_position()` checks clearance against already spawned agents and targets, but not future entities
- target pickup uses the first matching agent in loop order, not a tie-break by distance

### Mismatches between current implementation and earlier docs/comments

- older docs in the repo described a 19-dimensional observation vector
  - actual current default is 23 dimensions
- older docs described only target collection
  - current code includes nest-aware foraging with carrying-food state
- `render_scale` exists in `SwarmConfig` but is not used by the current renderer
- `action_dim` exists in `SwarmConfig` but the environment uses `num_actions` for its actual action space
- `render(mode="human", fps=60)` accepts `mode`, but current implementation ignores the argument
- diffusion currently uses `np.roll`, which gives wraparound edges; this may not match earlier intuitive “bounded diffusion” expectations

### Areas that are clear in code but easy to misread

- `pheromone_decay` is already a multiplicative retention factor, not an evaporation percentage
- `mixed` dynamics selection happens in `reset()`, not inside `_select_driver()`
- target vectors are not visibility-filtered; they use nearest remaining target globally
