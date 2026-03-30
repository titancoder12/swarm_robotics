You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement curriculum-stage-specific environment construction for the training pipeline so that each curriculum stage uses a deliberately different environment difficulty profile instead of only changing agent count.

Goal:
I want the curriculum to shape the environment itself across stages, not just the swarm size. The early stages should be extremely easy and focused, and later stages should progressively introduce scale, obstacles, and coordination difficulty.

Important context:

- The current curriculum path already exists for recurrent MAPPO.
- The current curriculum mainly changes `n_agents`.
- The environment already randomizes layouts at reset, but I want each stage to randomize within a stage-specific difficulty envelope.
- Do not remove randomization. Constrain and shape it by stage.

--------------------------------------------------
PART 1 — TARGET CURRICULUM ENVIRONMENT DESIGN
--------------------------------------------------

Implement curriculum-stage-specific environment profiles with this structure:

Stage 1A — One agent, insanely easy
- `n_agents = 1`
- very small environment
- no swarm coordination pressure
- ideally just the single agent and a single target
- little to no obstacle complexity
- the purpose is to teach:
  - target seeking
  - pickup / interaction
  - basic goal-directed behavior

This stage should make it extremely obvious to the agent that the task is:
- find the target object

Stage 1B — One agent, larger and mildly obstructed
- still `n_agents = 1`
- larger environment than Stage 1A
- add some obstacles
- obstacle density should still be reasonable, not punishing
- target count can remain small
- the purpose is to teach:
  - target search in a larger space
  - path adjustment around obstacles
  - persistence when the target is not trivially nearby

Stage 2A — Small swarm, medium environment
- small swarm, roughly `2` to `5` agents
- medium-sized environment
- obstacle count should increase moderately
- the purpose is to teach:
  - scaling from single-agent behavior to coordination
  - search coverage with a few agents
  - basic interference handling

Stage 2B — Small swarm, large-but-not-final environment
- still small swarm
- larger environment than Stage 2A
- not yet the final hardest setting
- obstacle count should increase again, but still remain learnable
- the purpose is to teach:
  - coordination at larger scale
  - search behavior without trivial local target discovery

Stage 3A — Full swarm, large-but-not-final environment
- full swarm (`5+`)
- use the same large-but-not-final environment size first
- this is an acclimation step for full swarm scaling
- the purpose is to teach:
  - adjusting to high agent count before final difficulty

Stage 3B — Full swarm, final hard environment
- full swarm (`5+`)
- very large environment
- many obstacles
- random wandering should not work well here
- the purpose is to teach:
  - robust multi-agent search
  - scaling and coordination under difficult environment structure
  - solving the real intended task setting

--------------------------------------------------
PART 2 — REQUIRED IMPLEMENTATION CHANGES
--------------------------------------------------

Implement the environment curriculum in code, not just in docs.

Required behavior:

1. Each curriculum stage must specify its own environment profile
   - not only `n_agents`
   - also size / obstacle / target settings as appropriate

2. The trainer must build the environment from the current stage profile
   - stage config should drive env creation directly

3. Randomization must remain inside each stage
   - example: obstacle layouts and target positions should still vary
   - but within the bounds of that stage’s intended difficulty

4. The stage progression should explicitly match the intended sequence:
   - 1A: one agent, tiny easy world
   - 1B: one agent, bigger world with some obstacles
   - 2A: small swarm, medium world
   - 2B: small swarm, large-but-not-final world
   - 3A: full swarm, same large-but-not-final world
   - 3B: full swarm, final hard world

--------------------------------------------------
PART 3 — ENVIRONMENT VARIABLES TO STAGE
--------------------------------------------------

Use stage configs to control at least:

1. `n_agents`
2. environment width
3. environment height
4. `n_obstacles`
5. `n_targets`

Optional if useful:

6. `max_steps`
7. `active_targets`
8. target respawn behavior
9. obstacle size ranges, if the environment currently supports that cleanly

Do not force unnecessary simulator rewrites.
Prefer the smallest clean implementation that makes stage difficulty meaningfully different.

--------------------------------------------------
PART 4 — DESIGN INTENT TO PRESERVE
--------------------------------------------------

The curriculum should reflect these teaching goals:

1. First teach the single agent that target-seeking matters
2. Then teach the single agent to handle obstacles
3. Then teach a small swarm to scale in a moderate world
4. Then teach a small swarm to scale in a larger world
5. Then let the full swarm adapt to that larger world
6. Finally train the full swarm in the true hard environment

This means the stage environments should not be arbitrary.
Each one should have a clear pedagogical role.

--------------------------------------------------
PART 5 — REPO / ARCHITECTURE CONSTRAINTS
--------------------------------------------------

Respect the current repo structure and existing training paths.

Requirements:

1. Do not break the existing custom DQN trainer
2. Do not break the current environment contract unless clearly necessary
3. Do not remove randomization from resets
4. Keep the curriculum implementation modular and readable
5. Keep the current recurrent MAPPO path runnable

Preferred implementation style:

- extend the curriculum stage config object to include env settings
- build stage-aware env configs in one place
- minimize hardcoded branching spread across many files

--------------------------------------------------
PART 6 — METADATA / LOGGING REQUIREMENTS
--------------------------------------------------

Record the curriculum environment settings clearly.

Required:

1. write stage environment settings into `run_config.json`
2. include them in stage metadata/checkpoints if appropriate
3. make it obvious from logs which environment profile each stage used

At minimum, stage metadata should clearly indicate:

- stage name
- number of agents
- width / height
- obstacle count
- target count

--------------------------------------------------
PART 7 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update documentation as part of the work.

Required doc updates:

1. explain the new staged environment profiles
2. explain why the earliest stage is intentionally extremely easy
3. explain how randomization still exists within each stage
4. explain how the curriculum progresses from tiny single-agent world to large full-swarm hard world
5. explain how to run the new curriculum path

Also update:

- `docs/PROJECT_LOG.md`
- `docs/QandA.md` when user-facing codebase questions are naturally answered by the work

--------------------------------------------------
PART 8 — IMPLEMENTATION PRIORITIES
--------------------------------------------------

Implement in this order:

Priority 1:
- represent stage-specific environment profiles in code

Priority 2:
- make the trainer actually build environments from those profiles

Priority 3:
- preserve stage-specific randomization

Priority 4:
- improve metadata/logging so the stage env settings are visible

Priority 5:
- document the resulting curriculum clearly

--------------------------------------------------
PART 9 — WHAT MUST NOT BREAK
--------------------------------------------------

Preserve:

1. existing environment randomization behavior inside stages
2. existing decentralized inference assumptions
3. existing MAPPO training entrypoint
4. existing DQN baseline path
5. existing checkpoint layout expectations within each algorithm family

Do not turn this into:

- a simulator rewrite
- a full procedural-generation overhaul
- a broad training-architecture rewrite

--------------------------------------------------
PART 10 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short smoke test that confirms the trainer enters multiple distinct stage environments
3. show the exact commands used
4. explain any practical limitations

The smoke test should make it clear that the stage env profiles differ in practice.

--------------------------------------------------
PART 11 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the new stage environment profiles
2. where stage environment settings are defined
3. how the trainer applies them
4. what metadata/logging was added
5. exact verification commands run
6. any remaining simplifications or deferred work

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The stage environments should explicitly teach in this order:

1. one agent, tiny easy world, just learn the goal
2. one agent, larger world, then obstacles
3. small swarm, medium world
4. small swarm, large-but-not-final world
5. full swarm, same large-but-not-final world
6. full swarm, final very large difficult world with many obstacles

Random wandering should not be sufficient in the final stage.

Keep the repo runnable.
