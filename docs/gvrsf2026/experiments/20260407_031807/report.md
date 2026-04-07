# Chokepoint Route-Reuse Report: Stage max_10

## Bundle

**Bundle folder:** `docs/gvrsf2026/experiments/20260407_031807/`

## Abstract

This bundle tests the hypothesis that pheromone-based stigmergy becomes more valuable when the swarm must repeatedly traverse a narrow rediscovery-expensive route. The study uses same-curriculum matched checkpoints, one repeated source, a fixed two-wall chokepoint layout with offset gaps, a longer horizon, and swarm sizes `1`, `3`, `6`, and `10` with `20` paired seeds per condition.

At `10` agents, the pheromone-enabled condition reached mean `food_delivered = 0.00` versus `0.00` for trained-with-pheromone but eval-without-pheromone and `0.00` for the true no-pheromone matched control. The paired test against the true no-pheromone control gave `p = nan` for `food_delivered` and `p = nan` for `post_discovery_deliveries`. By the predeclared extension rule, this stage is **not promising**, so the recommendation is: **stop at max_10 and do not extend**.

## Design

- same-curriculum matched checkpoints
- one repeated source
- fixed two-wall chokepoint world with offset gaps
- `food_source_capacity = 18`
- `max_steps = 1600`
- swarm sizes: `1`, `3`, `6`, `10`
- conditions:
  - trained with pheromone, eval with pheromone
  - trained with pheromone, eval without pheromone
  - trained without pheromone, eval without pheromone

## Figures

![Food Delivered by Swarm Size](./figures/food_delivered_by_swarm_size.png)

![Late Deliveries by Swarm Size](./figures/late_deliveries_by_swarm_size.png)

![Post-Discovery Deliveries by Swarm Size](./figures/post_discovery_deliveries_by_swarm_size.png)

![Pickup-to-Delivery Latency by Swarm Size](./figures/pickup_to_delivery_latency_by_swarm_size.png)

## `10`-Agent Checkpoint For Extension Decision

- with pheromone: `food_delivered = 0.00`
- trained with pheromone, eval without pheromone: `food_delivered = 0.00`
- true no-pheromone control: `food_delivered = 0.00`
- paired `food_delivered` vs true no-pheromone:
  - `p = nan`
  - `cohens_d = 0.000`
- paired `post_discovery_deliveries` vs true no-pheromone:
  - `p = nan`
  - `cohens_d = 0.000`

## Conclusion

This stage is **not promising** under the predeclared extension rule. The next action is to **stop at max_10 and do not extend**.
