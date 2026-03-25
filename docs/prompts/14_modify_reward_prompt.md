# Improve reward signal.
You are working in an existing multi-agent swarm reinforcement learning codebase for a stigmergic foraging environment.

Task:
Modify the environment and reward shaping so agents can discover food more reliably during training.

Main goals:
1. Increase the effective food detection radius so agents can sense food from farther away.
2. Add intermediate reward shaping so the task is less sparse.
3. Keep the implementation consistent with the project’s decentralized / stigmergic design and as sim-to-real-safe as possible.

Important design constraint:
Do NOT introduce unrealistic privileged information into the policy observation if that information would not exist at physical runtime.
That means:
- Avoid giving the policy direct global coordinates of food unless the current design already does that intentionally.
- Prefer local, sensor-like signals.
- Reward shaping may use environment-side calculations, but do not silently change the observation contract in a way that breaks sim-to-real assumptions.
- If you add or change observations, document them clearly and keep them physically interpretable.

What to implement:

1. Increase food detection radius
- Find where food detection / food presence / food sensing is computed.
- Increase the detection radius modestly so agents can detect nearby food sooner.
- Keep the value configurable in the environment config rather than hardcoding it if possible.
- Use an explicit config field like:
  - `food_detection_radius`
  - or `food_sensor_radius`
- Update any related comments/docs.

2. Add intermediate rewards for approaching food
- Add a small positive shaping reward when an agent moves closer to detectable food.
- This should preferably be based on local/detectable food, not omniscient global reward logic unless necessary.
- A good pattern is progress-based shaping:
  - reward small positive value when distance to relevant food decreases
  - small negative or zero when it increases
- Keep reward scale small so it does not dominate the real objective of pickup + delivery.

3. Add reward for sensing food
- Add a small reward when food is detected locally by the agent.
- This should encourage agents to enter food-rich regions.
- Avoid making this so large that agents camp near food without completing pickup/delivery.

4. Add reward for following pheromone
- Add a small shaping reward for moving in a direction aligned with stronger pheromone signal when appropriate.
- Use the project’s existing pheromone sampling geometry and observation semantics.
- Reward should be small and should encourage use of stigmergic cues without overpowering the main objective.
- If possible, make this depend on sensible context, such as:
  - not carrying food vs carrying food
  - inbound vs outbound behavior
- If there is no robust context signal already, keep it simple and conservative.

5. Preserve main task rewards
- Do NOT remove existing primary rewards for:
  - finding / picking up food
  - delivering food to nest
  - other existing important task events
- Intermediate shaping should supplement, not replace, the main reward.

6. Make reward weights configurable
Add config fields for the new shaping terms, for example:
- `reward_food_approach`
- `reward_food_detected`
- `reward_pheromone_follow`
- and any threshold/radius parameters needed

Use conservative defaults so the policy still has to solve the real task.

7. Add clear reward breakdown logging
- Extend reward breakdown / episode metrics so I can inspect how much reward came from:
  - food approach shaping
  - food detection
  - pheromone following
  - existing main rewards
- Keep this repo-consistent with the existing metrics/logging style.

8. Keep sim-to-real considerations explicit
- If you use any signal for shaping that is not currently part of the agent observation, add a brief code comment noting that this is environment-side shaping and whether it is intended only for training.
- Prefer shaping that corresponds to signals available to the physical robot, such as local detection and pheromone cues.

Implementation guidance:
- Search for:
  - environment config
  - reward computation
  - food sensing / pickup logic
  - pheromone sampling
  - observation construction
- Modify the existing code directly, do not produce only pseudocode.
- Keep architecture changes minimal.
- Do not rewrite unrelated systems.

Suggested shaping behavior:
- Food approach reward:
  - very small positive reward when distance to the nearest relevant detectable food decreases from previous step
  - optionally very small negative reward when it increases
- Food detected reward:
  - small bonus when food enters local sensing radius
  - optionally only once per encounter or capped to avoid farming
- Pheromone follow reward:
  - small bonus when chosen movement direction aligns with stronger sampled pheromone gradient
  - keep modest to avoid loops

Safety / quality constraints:
- Avoid reward hacking where agents can farm shaping rewards without completing the task.
- Avoid extremely large shaping coefficients.
- If needed, add caps or one-time bonuses.
- Keep the repo runnable.
- Follow existing naming/style conventions.

After making the changes, provide:
1. A short summary of what files were modified
2. The exact config fields added/changed
3. The reward formula / logic in plain English
4. Example training commands to test the new setup
5. Any sim-to-real caveats I should know

Nice-to-have:
- If the repo already has evaluation scripts, ensure they can still run unchanged.
- If there is an experiment config system, add a config preset for “dense reward shaping” or similar.
- If reward breakdown plots exist, make sure the new reward terms appear there too.