# Fix Pheromone Diffusion And Food Metric Bugs
Work in the existing multi-agent swarm reinforcement learning codebase for a stigmergic foraging environment.

Task:
Fix two concrete simulator/training bugs without rewriting the architecture:

1. Fix the pheromone field so it does not keep expanding forever across the whole map.
2. Fix the misleading demo/training food metric mismatch so training reports clearly distinguish:
   - food picked up / discovered
   - food delivered / retrieved

Goals:
- Keep the simulator physically sensible.
- Make pheromone diffusion bounded by the arena instead of wraparound.
- Make training/debugging metrics trustworthy.
- Keep the repo runnable and minimally changed.

--------------------------------
PART 1 — Fix pheromone expansion bug
--------------------------------

Problem:
- In [env/swarm_env.py](../../env/swarm_env.py), `_update_pheromone()` currently diffuses using `np.roll(...)`.
- `np.roll(...)` wraps the grid edges, so pheromone leaving one side of the world reappears on the opposite side.
- The grid also keeps tiny residual values forever because there is no clipping threshold after decay/diffusion.
- With ongoing deposits, this makes the rendered heatmap look like it expands without stopping.

Required fix:
- Replace the wraparound diffusion with bounded-edge diffusion.
- Do NOT let pheromone wrap from one world edge to the opposite edge.
- Use zero-flux or zero-padded edge handling, but keep the implementation simple and efficient.
- Keep the deposit / decay / diffuse structure intact.

Preferred implementation:
- In `_update_pheromone()`:
  - avoid `np.roll(...)` for neighbor lookup
  - either:
    - use padded slices with zeros outside the grid, or
    - update only interior cells and treat boundaries explicitly
- After decay/diffusion, clip very small values to zero using a config threshold such as:
  - `pheromone_min_value`

Add config if needed:
- `pheromone_min_value: float = 1e-3`

Important:
- Do not change the pheromone observation shape.
- Do not change the action space.
- Do not change the renderer except as needed to reflect the corrected field.

--------------------------------
PART 2 — Fix the food metric / demo-training mismatch
--------------------------------

Problem:
- In [train/independent_dqn_pytorch.py](../../train/independent_dqn_pytorch.py), the console print currently says `food {episode_food_retrieved}`.
- But `episode_food_retrieved` is accumulated from `info["food_delivered"]`, not from pickups.
- In [env/swarm_env.py](../../env/swarm_env.py), `step()` separately reports:
  - `targets_collected`
  - `food_delivered`
- This makes training appear to find no food even when the demo visibly reaches and picks up food but does not deliver it yet.

Required fix:
- Make training logs and CSV outputs clearly separate pickup/discovery from delivery.
- Do NOT silently relabel delivery as discovery.
- Keep the environment semantics unchanged unless a real bug is found.

Preferred implementation:
- In [train/independent_dqn_pytorch.py](../../train/independent_dqn_pytorch.py):
  - keep `episode_food_discovered` sourced from `targets_collected`
  - keep `episode_food_retrieved` sourced from `food_delivered`
  - update the console print so both appear explicitly, for example:
    - `food_found {episode_food_discovered} food_delivered {episode_food_retrieved}`
- Ensure CSV columns and summaries use unambiguous naming.
- If any evaluation path has the same ambiguity, fix it there too.

Also check:
- [analysis/evaluate.py](../../analysis/evaluate.py)
- [train/demo.py](../../train/demo.py)
- any run summaries or plots that label food metrics ambiguously

--------------------------------
PART 3 — Validation and observability
--------------------------------

Add small validation help, without changing algorithm behavior:

1. Pheromone validation
- Verify pheromone no longer wraps across edges.
- Verify distant corners stay near zero unless robots actually visit them.
- Keep rendering compatible with the corrected field.

2. Food metric validation
- Ensure logs distinguish:
  - pickup count
  - delivery count
- Make it easy to see whether failure is:
  - no food detection
  - no pickup
  - pickup without return

3. Keep metrics repo-consistent
- Update any relevant CSV headers / summaries / printed episode lines.
- Avoid breaking existing plots if possible.
- If backward compatibility is needed, preserve old columns and add clearer new ones.

--------------------------------
PART 4 — Implementation style
--------------------------------

- Modify existing files directly.
- Keep changes minimal and local.
- Do not rewrite the training stack.
- Do not change the observation contract.
- Do not change the action contract.
- Keep naming consistent with the repo.

--------------------------------
DELIVERABLES
--------------------------------

After implementation, provide:

1. List of modified files
2. Exact bug fixes made
3. Short explanation of:
   - how bounded pheromone diffusion now works
   - how food pickup vs delivery is now reported
4. Example commands to verify both fixes
5. Any caveats or backward-compatibility notes

--------------------------------
EXAMPLE EXPECTED EFFECT
--------------------------------

After the fixes:
- Pheromone trails spread locally and decay, but do not wrap around the arena forever.
- Training logs no longer imply “food = 0 forever” when the agents are actually reaching or picking up food.
- Demo and training become easier to compare because pickup and delivery are reported separately.

--------------------------------
IMPORTANT
--------------------------------

Keep the system runnable.
Implement real code, not pseudocode.
Do not introduce unrelated behavior changes.
