# Chokepoint Route-Reuse Hypothesis Study: Stage max_10

Hypothesis:

- pheromone should help more when the swarm must repeatedly traverse a narrow, rediscovery-expensive route

Design:

- same-curriculum matched checkpoints
- one repeated source with high capacity
- fixed two-wall chokepoint world with offset gaps
- nest and initial swarm cluster on the left side
- target on the right side
- longer horizon
- swarm sizes: `1`, `3`, `6`, `10`
- `20` paired seeds per size and condition

World geometry:

- arena from the final-stage checkpoint metadata
- two vertical barrier walls
- first wall gap near the upper corridor
- second wall gap near the lower corridor
- small per-seed gap jitter to avoid a single memorized layout

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
