# Stronger Focused Pheromone Experiment Plan

This campaign is a second focused rerun of the strengthened pheromone experiment.

Why another rerun was needed:

- The first focused repeated-source design produced the right qualitative trend.
- However, at `n=20` the strongest late-delivery effects were still statistically borderline.
- This rerun increases the paired sample size for the primary `6`-agent condition to `50` seeds and uses paired statistics directly.

Primary design:

- conditions:
  - trained with pheromone, evaluated with pheromone
  - trained with pheromone, evaluated without pheromone
  - trained without pheromone, evaluated without pheromone
- task:
  - one active food source
  - source capacity `12`
  - horizon `1200`
- primary swarm size:
  - `6` agents, `50` paired seeds

Supporting context:

- `1` and `3` agent runs remain included as smaller supporting samples with `20` paired seeds each

Primary hypothesis test:

> At `6` agents, the pheromone-trained policy should produce significantly more total deliveries and more late/post-discovery deliveries than the no-pheromone-trained policy in the repeated-source task.
