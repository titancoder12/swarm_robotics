# Observation Specification

This document is a focused reference for the current observation contract implemented by [env/swarm_env.py](../env/swarm_env.py).

## Source of Truth

Observation shape and ordering are defined in:

- `_compute_obs_dim()` in [env/swarm_env.py](../env/swarm_env.py)
- `_get_obs()` in [env/swarm_env.py](../env/swarm_env.py)

Per-component helper methods:

- `_lidar_scan()`
- `_nearest_target_features()`
- `_nest_direction()`
- `_nearest_agent_vector()`
- `_food_presence()`
- `_carrying_food()`
- `_pheromone_samples()`
- `_to_agent_frame()`

## Default Shape

Default per-agent observation shape is `(69,)`.

The environment now returns a short sliding window of observation frames. With
the default `observation_history_steps = 3`, the agent receives the last 3
frames concatenated oldest-to-newest.

One frame now has shape `(23,)`.

Formula:

```python
(cfg.lidar_rays
 + 2
 + (2 if cfg.obs_include_nest_direction else 0)
 + 2
 + 2
 + 1
 + (1 if cfg.obs_include_food_presence else 0)
 + (1 if cfg.obs_include_carrying else 0)
 + cfg.pheromone_samples) * cfg.observation_history_steps
```

With defaults:

- `lidar_rays = 9`
- `obs_include_nest_direction = True`
- `obs_include_food_presence = True`
- `obs_include_carrying = True`
- `pheromone_samples = 3`
- `observation_history_steps = 3`

Total:

- one frame: `9 + 2 + 2 + 2 + 2 + 1 + 1 + 1 + 3 = 23`
- stacked history: `23 * 3 = 69`

## Assembly Order

Observation assembly order for one frame in `_get_obs_frame()` is:

1. lidar rays
2. nearest detectable target distance and angle
3. nest direction vector, if enabled
4. nearest neighbor vector
5. heading as `sin(theta), cos(theta)`
6. normalized forward speed
7. food-presence flag, if enabled
8. carrying-food flag, if enabled
9. pheromone samples

`_get_obs()` then concatenates the most recent `cfg.observation_history_steps`
frames in chronological order.

## Coordinate Frame

Relative vectors are rotated into the agent/body frame using `_to_agent_frame(vec, theta)`:

```python
c = math.cos(-theta)
s = math.sin(-theta)
return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)
```

This means:

- target, nest, and nearest-neighbor vectors are computed in world coordinates first
- then rotated into the agent-local frame
- then normalized by `cfg.lidar_max_range`

`theta` is in radians.

## Noise and Failure Handling

Applied at the end of `_get_obs()`:

- if the agent index is in `failed_agent_indices`
  - the entire observation becomes all zeros
- else if `cfg.observation_noise_std > 0`
  - additive Gaussian noise is sampled with mean `0.0` and std `cfg.observation_noise_std`
  - the full vector is then clipped to `[-1.0, 1.0]`

## Default Index Table

| Index | Name | Meaning | Source in code | Range | Unit | Normalization | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `lidar_0` | Ray distance sample 0 | `_lidar_scan()` | `[0, 1]` | fraction of `lidar_max_range` | `dist / cfg.lidar_max_range` | walls and obstacles both terminate the ray |
| 1 | `lidar_1` | Ray distance sample 1 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 2 | `lidar_2` | Ray distance sample 2 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 3 | `lidar_3` | Ray distance sample 3 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 4 | `lidar_4` | Ray distance sample 4 | `_lidar_scan()` | `[0, 1]` | fraction | same | center ray with default 9-ray setup |
| 5 | `lidar_5` | Ray distance sample 5 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 6 | `lidar_6` | Ray distance sample 6 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 7 | `lidar_7` | Ray distance sample 7 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 8 | `lidar_8` | Ray distance sample 8 | `_lidar_scan()` | `[0, 1]` | fraction | same | same |
| 9 | `target_distance_norm` | nearest detectable target distance | `_nearest_target_features()` | `[0, 1]` | fraction of `lidar_max_range` | `distance / cfg.lidar_max_range` | zero when no target is within lidar range, inside the front-facing 180 degrees, and in line of sight |
| 10 | `target_angle_norm` | nearest detectable target relative bearing | `_nearest_target_features()` | `[-1, 1]` | fraction of `pi` radians | `angle / pi` | angle is in the agent/body frame; zero when no target is within lidar range, inside the front-facing 180 degrees, and in line of sight |
| 11 | `nest_dx_body_norm` | nest x component in body frame | `_nest_direction()` | `[-1, 1]` | fraction of `lidar_max_range` | rotated world delta divided by `cfg.lidar_max_range`, clipped | zero if nest disabled; omitted if `obs_include_nest_direction=False` |
| 12 | `nest_dy_body_norm` | nest y component in body frame | `_nest_direction()` | `[-1, 1]` | fraction | same | same |
| 13 | `neighbor_dx_body_norm` | nearest-agent x component in body frame | `_nearest_agent_vector()` | `[-1, 1]` | fraction of `lidar_max_range` | rotated world delta divided by `cfg.lidar_max_range`, clipped | nearest other agent only; zero when `n_agents <= 1` |
| 14 | `neighbor_dy_body_norm` | nearest-agent y component in body frame | `_nearest_agent_vector()` | `[-1, 1]` | fraction | same | same |
| 15 | `heading_sin` | `sin(theta)` | `_get_obs()` | `[-1, 1]` | unitless | direct transform | `theta` is in radians |
| 16 | `heading_cos` | `cos(theta)` | `_get_obs()` | `[-1, 1]` | unitless | direct transform | raw `theta` is not included |
| 17 | `speed_norm` | normalized forward speed | `_get_obs()` | `[-1, 1]` | fraction of `max_speed` | `clip(agent.v / cfg.max_speed, -1, 1)` | forward speed only; ignores `v_lat` |
| 18 | `food_presence` | binary local target-visibility cue | `_food_presence()` | `{0, 1}` | binary | `1.0` if a target is within lidar range, inside the front-facing 180 degrees, and in line of sight | zero if no detectable target; omitted if `obs_include_food_presence=False` |
| 19 | `carrying_food` | whether agent carries food | `_carrying_food()` | `{0, 1}` | binary | `1.0 if agent.carrying_food else 0.0` | omitted if `obs_include_carrying=False` |
| 20 | `pheromone_sample_0` | first forward pheromone sample | `_pheromone_samples()` | `[0, 1]` when enabled, else `0.0` | local relative grid intensity | raw sample vector divided by its own max if positive | not global-max normalized |
| 21 | `pheromone_sample_1` | second forward pheromone sample | `_pheromone_samples()` | `[0, 1]` when enabled, else `0.0` | same | same | same |
| 22 | `pheromone_sample_2` | third forward pheromone sample | `_pheromone_samples()` | `[0, 1]` when enabled, else `0.0` | same | same | same |

## Component Details

### Lidar

Defaults:

- `cfg.lidar_rays = 9`
- `cfg.lidar_max_range = 160.0`
- `cfg.lidar_step = 6.0`

Ray angle formula:

```python
angle = agent.theta + (i - half) * (math.pi / (cfg.lidar_rays - 1))
half = cfg.lidar_rays // 2
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

Hit conditions:

- world boundary
- any obstacle rectangle

Unavailable case:

- none; if no hit occurs before max range, the value is exactly `1.0`

### Target Vector

Source: `_nearest_target_features()`

Representation:

- normalized distance
- normalized relative angle
- nearest target by Euclidean distance among targets that are within `cfg.lidar_max_range`, inside the agent's front-facing 180 degrees, and not blocked by obstacles

Unavailable case:

- both target features are `0.0` when no target is detectable

#### Hardware Camera Mapping

On the physical robot runtime, these same two slots can be filled by the
optional camera detector in [firmware/camera.py](../firmware/camera.py) and
[firmware/run.py](../firmware/run.py) instead of simulator target geometry.

What counts as a target on hardware:

- the largest visible image blob inside the configured HSV color range
- after morphological cleanup and contour extraction
- if its contour area is at least `--camera-min-area-px`

Current detector pipeline:

1. capture a frame
2. convert to HSV
3. threshold with `cv2.inRange(hsv_lower, hsv_upper)`
4. run morphological open / close
5. find contours
6. choose the largest contour
7. reject it if area is below `min_area_px`
8. estimate angle from horizontal image offset
9. estimate distance from apparent blob width and `--camera-target-width-cm`

Hardware slot mapping:

- index `9`: `target_distance_norm`
  - `clip(distance_m / cfg.lidar_max_range_m, 0, 1)`
- index `10`: `target_angle_norm`
  - `clip(angle_rad / pi, -1, 1)`

This is a feature-level approximation of the simulator target signal, not a
class-based learned detector.

### Nest Vector

Source: `_nest_direction()`

Representation:

- `dx, dy` in body frame from agent to nest

Unavailable case:

- `[0.0, 0.0]` when `cfg.nest_enabled` is `False`
- feature removed from the vector entirely when `obs_include_nest_direction` is `False`

### Neighbor Vector

Source: `_nearest_agent_vector()`

Representation:

- `dx, dy` to the nearest other agent
- body-frame coordinates
- only one neighbor is represented

Unavailable case:

- `[0.0, 0.0]` when `cfg.n_agents <= 1`

### Heading

Source: `_get_obs()`

Representation:

- `sin(theta)`
- `cos(theta)`

### Speed

Source: `_get_obs()`

Representation:

- scalar forward speed only
- normalized by `cfg.max_speed`

### Food Presence

Source: `_food_presence()`

Representation:

- binary
- `1.0` if any remaining target is within `cfg.lidar_max_range`, inside the front-facing 180 degrees, and not blocked by obstacles

#### Hardware Camera Mapping

On the physical robot runtime, index `18` is set from the camera detector:

- `1.0` when the camera target detector reports `found=True`
- `0.0` when no valid target blob is detected

So on hardware the camera currently influences exactly these three per-frame
features:

- index `9`: target distance
- index `10`: target angle
- index `18`: food presence

When no camera detection is available, all three fall back to zero.

### Carrying Food

Source: `_carrying_food()`

Representation:

- binary
- direct encoding of `agent.carrying_food`

### Pheromone Samples

Source: `_pheromone_samples()`

Sampling distance for index `i`:

```python
(i + 1) * cfg.agent_radius * cfg.pheromone_sample_spacing_scale
```

Behavior:

1. sample along the current forward direction
2. convert to pheromone-grid indices with floor division by `cfg.pheromone_cell_size`
3. read raw grid values
4. divide the sample vector by its own max if that max is positive

Awareness radius:

- pheromone can only affect the observation out to the farthest forward sample distance
- with the defaults, `3 * 7.0 * 1.5 = 31.5`
- pheromone outside that sensing radius does not contribute to the observation

Unavailable case:

- zeros when pheromone is disabled globally
- zeros when `obs_include_pheromone` is `False`
- zeros when all sampled values are zero

## Important Notes

- The current default observation is 23-D per frame and 69-D after 3-frame stacking.
- Pheromone samples are normalized locally by the sample vector maximum, not globally by the grid maximum.
- Pheromone sensing is bounded by the derived awareness radius rather than being treated as globally visible.
- Failed agents observe all zeros.
- Observation noise, when enabled, is applied after the full vector is assembled.
