# Reward exploration and modify pheremone rules!
You are working in an existing multi-agent swarm reinforcement learning codebase for a stigmergic foraging environment.

Task:
1) Modify pheromone deposition so agents can ONLY lay pheromone after they have found food (i.e., when carrying food or after a successful food interaction).
2) Add a reward signal that encourages higher exploration coverage of the environment.

Goals:
- Make pheromone meaningful (only encode successful discovery paths).
- Reduce noise in the pheromone field.
- Encourage agents to explore more of the map early in training.
- Keep everything consistent with decentralized, sim-to-real-safe design.

--------------------------------
PART 1 — Gate pheromone deposition
--------------------------------

Find where pheromone deposition is implemented (likely in env/swarm_env.py or similar).

Modify logic so that:

- Agents can deposit pheromone ONLY when:
  - they are carrying food, OR
  - they have just detected/picked up food (depending on current design)

- When NOT carrying food:
  - pheromone deposition should be disabled (set to zero), OR
  - optionally allow a very small “exploration pheromone” but keep it:
    - separate from food pheromone channel (if multi-channel exists), OR
    - extremely weak (e.g., < 10% of normal strength)

Preferred implementation:
- Use existing agent state flag (e.g., `carrying_food`)
- Wrap deposition logic like:

  if agent.carrying_food:
      deposit_pheromone(strong_value)
  else:
      deposit_pheromone(0.0)

- If no such flag exists, implement a minimal one based on existing pickup logic.

Also ensure:
- Pheromone evaporation remains unchanged
- Pheromone grid updates remain efficient

Add a config option:
- `pheromone_requires_food: bool = True`

--------------------------------
PART 2 — Add coverage-based reward
--------------------------------

Goal:
Encourage agents to explore new areas of the environment.

Find where:
- coverage is tracked (e.g., visited cells, exploration map, etc.)

If not already implemented:
- Add a per-episode or per-agent visited grid (boolean or count-based)
- Track newly visited cells

Add reward:

- When an agent visits a cell that has NOT been visited before (by itself or globally depending on design):
    reward += coverage_reward_value

Recommended:
- Use a SMALL reward (e.g., 0.01–0.1 range)
- Avoid making this dominate food reward

Add config:
- `reward_new_cell: float`

Optional refinement:
- Only reward NEW cells (not revisits)
- Optionally decay reward over time to avoid farming behavior

--------------------------------
PART 3 — Reward breakdown logging
--------------------------------

Extend reward logging to include:

- coverage_reward_total
- pheromone_deposit_events (optional)
- existing reward components

Ensure these appear in:
- episode logs
- any existing metrics dicts
- plotting pipeline if present

--------------------------------
PART 4 — Maintain balance
--------------------------------

Important constraints:

- DO NOT remove or weaken main rewards:
  - food pickup
  - food delivery

- Keep shaping rewards SMALL relative to main reward

- Avoid reward hacking:
  - coverage should only reward first-time visits
  - pheromone gating should not break task completion

--------------------------------
PART 5 — Sim-to-real considerations
--------------------------------

- Pheromone gating is physically realistic → good
- Coverage reward is training-only → OK, but:
  - DO NOT add coverage info into observation unless already present
  - Keep it environment-side reward only

Add comments explaining:
- coverage reward is for training acceleration
- pheromone gating enforces meaningful stigmergy

--------------------------------
PART 6 — Implementation style
--------------------------------

- Modify existing files directly (do NOT rewrite architecture)
- Keep naming consistent with repo
- Use config-driven parameters
- Keep code clean and minimal

--------------------------------
DELIVERABLES
--------------------------------

After implementation, provide:

1. List of modified files
2. Exact config fields added:
   - pheromone_requires_food
   - reward_new_cell
3. Short explanation of:
   - how pheromone gating works
   - how coverage reward is computed
4. Example training command
5. Any caveats for learning behavior

--------------------------------
EXAMPLE EXPECTED EFFECT
--------------------------------

After changes:
- Early training: agents explore more (coverage increases)
- First food discovery triggers meaningful pheromone trails
- Later agents follow these trails more reliably
- Less random/noisy pheromone field

--------------------------------
IMPORTANT
--------------------------------

Keep the system runnable and consistent with current training scripts.
Do not introduce breaking changes to observation shapes or action spaces.
Implement real code, not pseudocode.