Implement the next nest-orbit fix after prompt 43.

Observed result after prompt 43:
- `stage3a_full_swarm_large` and `stage3b_full_swarm_final` reduced non-carrying nest-loiter somewhat.
- But nest-adjacent crowding is still high, and the post-delivery outward-search handoff is too short-lived.
- The swarm still appears able to fall back into a nest-centered loop after the brief cooldown expires.

What to implement:
1. Make the post-delivery handoff persist until nest exit in the late pheromone-on stages.
   - Do not let the post-delivery mode expire purely on a short countdown if the agent is still inside the configured nest-exit radius.
   - Keep the agent in post-delivery “leave the nest” mode until it actually clears the nest zone, or until a clearly defined stage-safe escape condition is reached.

2. Add post-delivery crowding pressure near the nest.
   - When multiple post-delivery empty agents are still inside the nest-exit radius, add a small penalty so they do not re-form a stable ring around the nest.

3. Keep pheromone suppression stronger while that exit handoff is active.
   - A post-delivery empty agent should not get sucked back into the nest-adjacent pheromone field before it has truly resumed outward search.

4. Tune only the late pheromone-on swarm stages.
   - Keep the early homing / pickup-return stages unchanged.
   - Focus on `stage2d_small_swarm_large`, `stage3a_full_swarm_large`, and `stage3b_full_swarm_final`.

5. Preserve delivery.
   - The goal is not just lower loiter metrics.
   - Greedy delivery should stay at least as good as prompt 43 while non-carrying nest-loiter/crowding decreases further.

6. Update docs and logs.
   - Record the new prompt in `docs/prompts/`.
   - Add concise notes to `docs/QandA.md` and `docs/PROJECT_LOG.md`.
