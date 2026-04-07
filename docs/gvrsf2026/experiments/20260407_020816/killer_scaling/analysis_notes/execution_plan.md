# 30-Agent Stigmergy Scaling Plan

This campaign redesigns the decisive pheromone study to be more sensitive to
trail reuse while keeping the condition comparison fair.

Primary goal:

- test whether pheromone-enabled training and evaluation produce stronger
  repeated-source exploitation than matched no-pheromone conditions as swarm
  size increases up to `30`

Design:

- conditions:
  - trained with pheromone, evaluated with pheromone
  - trained with pheromone, evaluated without pheromone
  - trained without pheromone, evaluated without pheromone
- task:
  - full final-stage arena geometry from `checkpoints/mappo_g/latest`
  - `1` active target
  - `1` total target source
  - `food_source_capacity = 12`
  - `target_respawn = false`
  - `18` obstacles
  - target constrained away from the nest to reduce trivial finds
  - horizon `1200`
- swarm sizes:
  - `1`, `3`, `6`, `10`, `15`, `20`, `30`

Repetitions:

- `20` paired seeds for every swarm size and condition
