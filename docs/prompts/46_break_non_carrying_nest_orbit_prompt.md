Implement the next fix for the remaining non-carrying nest-orbit failure mode.

Problem to address:
- Even after very large non-carrying nest penalties, retrained policies can still cluster near the nest and orbit it when not carrying food.
- This means soft reward shaping alone is not sufficient.
- We need an explicit behavior-mode intervention for empty agents near the nest.

Goal:
- Break the non-carrying nest-centered local optimum decisively.
- Preserve carrying-food return-to-nest behavior.
- Keep late-stage delivery performance intact while forcing empty agents to leave the nest zone and resume exploration.

What to implement:
1. Add an explicit non-carrying “leave nest zone” mode.
   - When an agent is not carrying food and is inside a configurable nest-adjacent radius, treat it as being in a force-explore mode.
   - This mode should be separate from the carrying-food return path and separate from the post-delivery cooldown path.

2. Override orbit-friendly actions in that mode.
   - Do not rely only on penalties.
   - For non-carrying agents inside the force-explore zone, suppress or override actions that keep them orbiting or lingering near the nest.
   - A practical repo-friendly version is:
     - replace the chosen action with a forward-moving action that turns the agent away from the nest
     - clear action-hold state when this override happens so a bad held action does not persist

3. Keep pheromone influence suppressed near the nest for empty agents.
   - The current pheromone suppression for non-carrying exploration near the nest should remain active.
   - The new forced-explore mode should work together with that suppression, not replace it.

4. Make this active only in the late swarm stages where the problem appears.
   - Tune:
     - `stage2d_small_swarm_large`
     - `stage3a_full_swarm_large`
     - `stage3b_full_swarm_final`
   - Do not interfere with the single-agent return-to-nest curriculum.

5. Add metrics for this mode.
   - Log:
     - fraction of steps spent in non-carrying force-explore mode
     - number of forced action overrides
   - Include them in episode/eval CSVs and checkpoint metadata.

6. Update demo metadata restore.
   - If demo loads a MAPPO checkpoint with this mode enabled, it should restore those settings from `metadata.json`.

7. Update docs.
   - Update README / TRAINING_MAPPO if the behavior interpretation changes.
   - Add concise entries to `docs/QandA.md` and `docs/PROJECT_LOG.md`.

Success criterion:
- In late pheromone-on swarm stages, empty agents should stop orbiting the nest and should visibly leave the nest zone to explore.
- Carrying-food return behavior and delivery should not regress materially.
