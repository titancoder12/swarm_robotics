Implement the next fix for the remaining nest-centered local-optimum problem.

Problem to address:
- Even after prompts 42 through 44, demo still shows cases where empty agents cluster near the nest and then stay there.
- The current nest-loiter and post-delivery penalties are only soft shaping terms; they reduce the behavior but do not reliably produce strong outward exploration.
- The desired behavior is:
  - if an agent is carrying food, it should prioritize returning to the nest
  - if an agent is not carrying food, it should actively fan out and explore the environment instead of orbiting or idling near the nest

Goal:
- Make non-carrying agents explore broadly across the map when they are not delivering food.
- Break the nest-centered local optimum more decisively than prompt 44.

What to implement:
1. Add explicit non-carrying outward-search shaping.
   - Introduce a reward path for non-carrying agents that increases with moving away from the nest when they are inside a configurable “nest influence zone.”
   - This should be distinct from the post-delivery cooldown path.
   - It should apply whenever an agent is not carrying food and remains too close to the nest.

2. Add stronger non-carrying idle / low-displacement penalties near the nest.
   - If a non-carrying agent remains near the nest and is barely moving, penalize that explicitly.
   - This should target the observed “cluster and then just stay there” failure mode, not carrying-phase homing.

3. Add a non-carrying explore-mode state if needed.
   - If the existing reward shaping is still too weak, add a simple env-side notion of “non-carrying explore mode” when an agent is empty and near the nest.
   - While in that mode, reduce or ignore nest-adjacent pheromone attraction and bias the reward toward outward displacement / broader coverage.
   - Keep this stage-bounded and physically plausible.

4. Make the later pheromone-on swarm stages explicitly exploration-biased for empty agents.
   - Tune only the late swarm stages where this problem appears:
     - `stage2d_small_swarm_large`
     - `stage3a_full_swarm_large`
     - `stage3b_full_swarm_final`
   - Do not weaken carrying-home behavior.
   - Do not break the delivery path that is now working better than before.

5. Add metrics that prove empty agents are really fanning out.
   - Track non-carrying near-nest idle fraction and non-carrying outward-search reward totals.
   - Log enough information to compare prompt 45 against prompt 44 on:
     - greedy delivery
     - non-carrying nest loiter / crowding
     - exploration coverage
     - new non-carrying outward-search metrics

6. Preserve delivery-first behavior.
   - Carrying agents should still clearly prefer nest return.
   - The new behavior should only make empty agents leave the nest region and explore.

7. Update docs.
   - Update README / TRAINING_MAPPO if behavior interpretation changes.
   - Add concise notes to `docs/QandA.md` and `docs/PROJECT_LOG.md`.

Success criterion:
- In late pheromone-on stages, empty agents should visibly disperse and explore instead of clustering around the nest.
- Greedy delivery should stay at least as good as prompt 44, while non-carrying nest-local behavior decreases further.
