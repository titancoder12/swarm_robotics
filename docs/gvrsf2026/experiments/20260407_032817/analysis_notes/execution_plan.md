# Single-Gap Chokepoint Hypothesis Study: Stage max_10

Hypothesis:

- pheromone should help more when a repeated source sits behind a single narrow passage, but the task still remains solvable enough to avoid floor effects

Design:

- same-curriculum matched checkpoints
- one repeated source with high capacity
- fixed single-wall chokepoint world with one gap
- nest and initial swarm cluster on the left side
- target on the right side, roughly aligned to the gap
- longer horizon
- swarm sizes: `1`, `3`, `6`, `10`
- `20` paired seeds per size and condition

Conditions:

- trained with pheromone, eval with pheromone
- trained with pheromone, eval without pheromone
- trained without pheromone, eval without pheromone

Predeclared extension rule:

- extend to `max_20` only if the `10`-agent result is promising:
  - `food_delivered_mean` is higher for pheromone-on than for both comparison conditions
  - and `post_discovery_deliveries_mean` is higher than the true no-pheromone condition
  - and at least one `10`-agent matched test against the true no-pheromone condition has
    `p < 0.10` or `cohens_d >= 0.3`
