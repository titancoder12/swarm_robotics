Implement the next fix for the remaining nest-circling failure mode in late pheromone-on stages.

Problem to address:
- After prompt 42, later swarm stages still show high non-carrying nest-loiter and nest-crowding fractions.
- Greedy delivery improved somewhat, but empty agents can still settle into a nest-adjacent orbit after delivery.
- The current system mostly discourages lingering near the nest, but it does not explicitly teach a post-delivery handoff back into outward search.

What to implement:
1. Add a short post-delivery cooldown / handoff mode for non-carrying agents.
   - When an agent completes a delivery, mark it as being in a post-delivery re-exploration window for a configurable number of steps.
   - During this window, the agent should be encouraged to move away from the nest instead of hovering near it.

2. Add explicit post-delivery outward-search shaping.
   - Reward positive increase in distance from the nest for a short window after delivery.
   - Optionally add a small penalty when a post-delivery agent remains inside a configurable nest-exit radius.
   - Keep this shaping modest so it does not interfere with normal carrying-home behavior.

3. Suppress pheromone attraction for post-delivery empty agents near the nest.
   - If an agent is in the post-delivery cooldown, do not let nest-adjacent pheromone gradients keep it trapped near the nest.
   - This should be stronger than the generic non-carrying nest suppression already added in prompt 42.

4. Stage the behavior only where it matters.
   - Apply the new post-delivery outward-search settings to the later pheromone-on swarm stages, not to the early homing curriculum.
   - Tune stage-specific values so `stage3a_full_swarm_large` and `stage3b_full_swarm_final` both use the new behavior.

5. Add metrics so the effect is visible.
   - Track post-delivery cooldown activity in env `info`.
   - Log enough information in MAPPO episode/eval metrics to see whether the new behavior reduces nest-adjacent orbiting.

6. Update documentation.
   - Update README and MAPPO training docs if command behavior or interpretation changes.
   - Append concise notes to QandA and PROJECT_LOG.

Success criterion:
- Later pheromone-on stages should show lower non-carrying nest-loiter/crowding pressure than prompt 42.
- Greedy delivery should not regress while the swarm spends less time trapped in nest-adjacent loops.
