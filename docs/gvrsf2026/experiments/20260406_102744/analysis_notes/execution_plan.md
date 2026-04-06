# Experiment Execution Notes

Timestamped campaign folder: `20260406_102744`

This campaign reruns the strongest feasible evaluation-heavy subset of the GVRSF plan as part of the 30-agent full-bundle extension:

- curriculum learning vs weaker/simplified training
- pheromone ablation
- swarm-size scaling
- robustness under harder environments
- baseline comparison

Why this design:

- The repository already contains multiple strong and weaker MAPPO checkpoints.
- Full multi-condition retraining would be compute-heavy and would reduce scientific rigor if done underpowered.
- The chosen design reuses existing checkpoints where possible and evaluates each condition over 20 deterministic episodes with matched controls.

Primary current checkpoint:

- `checkpoints/mappo_g/stage3b_full_swarm_final`

Weaker curriculum checkpoint:

- `checkpoints/mappo_full_600k_33/stage3b_full_swarm_final`

Environment matching rule:

- The runner reconstructs `SwarmConfig` from each checkpoint's `metadata.json`.
- This avoids the mismatch risk in the generic CLI evaluators, which do not fully restore stage geometry from metadata.
