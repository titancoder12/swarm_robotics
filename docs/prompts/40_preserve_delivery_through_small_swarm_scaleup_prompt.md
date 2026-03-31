Implement the next curriculum fix for recurrent MAPPO.

Problem from the latest full run:
- Single-agent delivery is now stable through `stage1g_single_agent_delivery_obstacles`.
- The first real collapse now happens at the first swarm transition.
- In `stage2a_small_swarm_medium`, the run briefly shows nonzero pickup/delivery, then falls into a long zero-pickup, zero-delivery streak across full-length episodes.
- Later eval rows also show that the policy degrades further in `stage3a_full_swarm_large`.

Goal:
- Preserve the learned single-agent pickup -> carry -> return -> deliver loop through the first small-swarm transition.
- Make the first swarm stages teach “multiple agents can still deliver” before reintroducing harder clutter and pheromone/trail exploitation.
- Get the curriculum to a state where a full run can progress past the first small-swarm stages without collapsing into search-only or freeze behavior.

Required implementation changes:

1. Redesign the first small-swarm transition.
- Do not jump directly from the final single-agent obstacle-return stage into the current `stage2a_small_swarm_medium`.
- Insert an explicit small-swarm bootstrap stack after `stage1g_single_agent_delivery_obstacles`.
- At minimum include:
  - a small-swarm carry bootstrap or homing bootstrap stage
  - a small-swarm easy delivery stage with no or minimal clutter
  - then the current medium stage as the first real swarm clutter stage
- These stages must remain pheromone-off until greedy delivery is stable.

2. Make the first swarm stages easier than the current `stage2a_small_swarm_medium`.
- Use fewer targets.
- Use fewer or zero obstacles at first.
- Keep target respawn off in the first swarm delivery lesson.
- Keep action repeat conservative if that helps preserve return behavior.
- Keep reward shaping explicitly delivery-first in these stages.

3. Promotion must require greedy delivery continuity.
- For the new first swarm stages, promotion should require nonzero greedy delivery and nontrivial delivery conversion.
- Do not allow swarm promotion on pickup-only behavior.
- Keep later swarm/full-swarm stages delivery-sensitive as well.

4. Make the curriculum slices reflect the new stage count.
- Update `stage1`, `stage1_to_2`, and `full` selection boundaries accordingly.
- Keep budget allocation sane for the new stage stack.

5. Update docs.
- Update README and MAPPO training docs so the current curriculum stage list matches the code.
- Update Q&A and project log with the new diagnosis and fix.

Verification:
- Run py_compile on the touched files.
- Run a `--curriculum stage1_to_2` or equivalent focused verification that is long enough to reach the new first swarm stages.
- Confirm from logs/CSVs that the first swarm stage no longer immediately collapses into long zero-pickup, zero-delivery streaks.

