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
  - full final-stage arena from `checkpoints/mappo_g/latest`
  - `3` active targets
  - `18` obstacles
  - horizon `800`
- swarm sizes:
  - `1`, `3`, `6`, `10`, `15`, `20`, `30`

Repetitions:

- `20` paired seeds for every swarm size and condition
