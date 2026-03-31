Implement the next MAPPO behavior fix.

Problem:
- In later swarm stages and demos, non-carrying agents can end up circling or clustering around the nest instead of leaving to search.
- This appears to be a learned local attractor, not direct nest-approach reward leakage, because nest-approach shaping only applies while carrying food.
- The likely causes are nest-adjacent pheromone attraction, weak outward-search pressure after delivery, and no anti-loiter / anti-crowding pressure near the nest for empty agents.

Goal:
- Keep the nest strongly attractive for carrying agents returning home.
- Make the nest much less attractive as a resting/exploitation attractor for non-carrying agents.
- Reduce empty-agent nest circling in later swarm stages without breaking delivery.

Required changes:

1. Add non-carrying nest-attractor controls.
- Add config knobs to suppress pheromone-follow reward near the nest for non-carrying agents.
- Add a small non-carrying nest-loiter penalty inside a configurable radius.
- Add a small non-carrying nest-crowding penalty when too many empty agents cluster near the nest.

2. Bias later swarm stages to use those controls.
- Apply the new settings in the pheromone-on swarm/full-swarm stages.
- Keep early single-agent homing lessons unchanged or nearly unchanged.

3. Add logging/metrics.
- Expose nest-loiter / nest-crowding metrics in episode and eval rows so the effect is measurable.

4. Update docs.
- Update README / MAPPO training docs / Q&A / project log to reflect the new nest-circling fix.

Verification:
- Run py_compile on touched files.
- Run a focused later-stage verification and/or full run.
- Confirm later swarm stages keep nonzero delivery while nest-loiter metrics are reduced versus the old behavior.
