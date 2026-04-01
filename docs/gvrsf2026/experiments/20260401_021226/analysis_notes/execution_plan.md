# Focused Pheromone Experiment Plan

This campaign is a focused rerun of the GVRSF experiment prompt with a stronger pheromone design.

Why the design changed:

- The previous generic final-stage pheromone toggle test produced only a modest effect.
- That design mainly tested inference-time dependence, not whether pheromone training and trail reuse improve swarm coordination.
- This campaign instead uses a repeated-source foraging environment where trail reuse should matter.

Key design choices:

- matched training conditions:
  - `mappo_gru_pheromone/stage3_full_marl`
  - `mappo_gru_no_pheromone/stage3_full_marl`
- paired evaluation layouts:
  - seeds `100..119`
- multiple swarm sizes:
  - `1`, `3`, `6`
- trail-reuse-friendly task:
  - one active food source
  - source capacity `12`
  - horizon `1200`

Primary claim being tested:

> Pheromone-trained swarms should outperform no-pheromone-trained swarms, especially in post-discovery and late-episode delivery behavior, and the effect should be strongest at larger swarm sizes.
