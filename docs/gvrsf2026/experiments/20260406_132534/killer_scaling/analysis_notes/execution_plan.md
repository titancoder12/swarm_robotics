# 30-Agent Stigmergy Scaling Plan

This campaign extends the focused repeated-source pheromone study to larger swarm sizes.

Primary goal:

- test whether the previously observed stigmergy result remains supported when swarm size increases up to `30`

Design:

- conditions:
  - trained with pheromone, evaluated with pheromone
  - trained with pheromone, evaluated without pheromone
  - trained without pheromone, evaluated without pheromone
- task:
  - one active food source
  - source capacity `12`
  - horizon `1200`
- swarm sizes:
  - `1`, `3`, `6`, `10`, `15`, `20`, `30`

Primary paired tests:

- `6` agents with `50` paired seeds for continuity with prior evidence
- `30` agents with `50` paired seeds as the new upper-bound confirmation test

Supporting sizes:

- `1`, `3`, `10`, `15`, and `20` agents with `20` paired seeds each
