# Complex-Environment Hypothesis Study: Stage max_10

Hypothesis:

- pheromone should become more useful in a more complex repeated-source task,
  because rediscovery is harder and route reuse is more valuable

Design:

- same-curriculum matched checkpoints
- one repeated source
- farther target placement
- more obstacles than the normal final-stage environment
- longer horizon
- swarm sizes: `1`, `3`, `6`, `10`
- `20` paired seeds per size and condition

Conditions:

- trained with pheromone, eval with pheromone
- trained with pheromone, eval without pheromone
- trained without pheromone, eval without pheromone

Complexity overrides:

- `n_targets = 1`
- `active_targets = 1`
- `target_respawn = false`
- `food_source_capacity = 15`
- `n_obstacles = 24`
- `max_steps = 1400`
- `target_nest_distance_min = 320`
- `target_nest_distance_max = 520`
- `target_nest_corridor_clearance = 80`

Predeclared extension rule:

- extend to `max_20` only if the `10`-agent result is promising:
  - `food_delivered_mean` is higher for pheromone-on than for both comparison conditions
  - and `post_discovery_deliveries_mean` is higher than the true no-pheromone condition
  - and at least one `10`-agent matched test against the true no-pheromone condition has
    `p < 0.10` or `cohens_d >= 0.3`
