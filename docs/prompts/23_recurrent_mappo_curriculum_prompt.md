You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement a curriculum-learning training path centered on **recurrent MAPPO with GRU** as the main algorithm, using the existing swarm environment and preserving decentralized execution at inference time.

Goal:
I want the repo upgraded so training progresses through a staged curriculum:

1. single-agent training
2. small-swarm training (`2` to `5` agents)
3. full multi-agent MARL training (`5+` agents)

The main algorithm should be:

- **parameter-shared recurrent MAPPO**
- **GRU-based actor/critic**
- **centralized training, decentralized execution**

Important context:

- The environment already has built-in domain randomization in practice because each reset generates a randomized environment.
- Treat that as an existing strength of the training setup.
- Do not rewrite the simulator just to add more randomization unless something small and clearly useful is needed.

--------------------------------------------------
PART 1 — TARGET TRAINING DESIGN
--------------------------------------------------

Implement a clean curriculum-learning training flow with these stages:

Stage 1 — Single agent
- train with `n_agents = 1`
- establish basic navigation, target detection, pickup, and return behavior

Stage 2 — Small swarm
- train with a small swarm, roughly `2` to `5` agents
- introduce coordination pressure gradually
- preserve parameter sharing

Stage 3 — Full MARL
- train with the intended full swarm setting, `5+` agents
- use recurrent MAPPO as the main algorithm
- keep centralized critic training and decentralized actor execution

The curriculum should be implemented as a real training workflow, not just a note in the docs.

--------------------------------------------------
PART 2 — MAIN ALGORITHM REQUIREMENTS
--------------------------------------------------

The main algorithm must be:

- recurrent MAPPO
- shared actor across homogeneous agents
- GRU-based recurrence
- centralized critic during training
- local observation only for actor inference

Required MAPPO features:

1. PPO clipping
2. GAE
3. entropy bonus
4. value loss
5. gradient clipping
6. advantage normalization
7. episode masks / hidden-state reset handling
8. minibatch training over recurrent rollout sequences

Required recurrent design:

1. GRU actor
2. GRU critic, or a clean recurrent critic path if the critic also uses sequence state
3. hidden-state reset at episode boundaries
4. clear inference helper path for local actor execution

Do not use frame stacking as a substitute for proper recurrence in the MAPPO path.

--------------------------------------------------
PART 3 — ENVIRONMENT / INTERFACE REQUIREMENTS
--------------------------------------------------

Use the existing environment contract and preserve deployment relevance.

Requirements:

1. Keep decentralized execution
   - the actor used at inference time must consume only local observations and recurrent hidden state

2. Use the existing local observation path
   - do not break the robot-facing observation contract unless absolutely necessary

3. Use or formalize the centralized state path for training
   - if the repo already exposes a centralized training state, use it
   - if it needs refinement, improve it cleanly and document it

4. Preserve the current discrete action semantics unless there is a strong reason to change them

5. Treat environment randomization as already present
   - each run/reset already creates randomized layouts
   - leverage this in the curriculum and training docs rather than pretending it does not exist

--------------------------------------------------
PART 4 — CURRICULUM LEARNING REQUIREMENTS
--------------------------------------------------

Implement curriculum support explicitly.

Required behavior:

1. Curriculum stages must be represented in code/config, not only in prose
2. There must be a clean way to:
   - train only Stage 1
   - train through Stage 1 -> 2
   - train through Stage 1 -> 2 -> 3
3. Curriculum transitions must be checkpoint-aware
   - later stages should be able to initialize from earlier-stage checkpoints
4. Preserve reproducibility
   - record stage configuration and curriculum progression in metadata/logs

Preferred implementation style:

- explicit stage configs
- clear CLI entrypoint or config-driven schedule
- minimal duplication between stages

Good curriculum variables to stage:

- `n_agents`
- possibly `n_obstacles`
- possibly `n_targets`
- possibly max episode horizon

But keep the first implementation disciplined and small.
The essential curriculum variable is:

- swarm size (`1` -> `2-5` -> `5+`)

--------------------------------------------------
PART 5 — REPO / ARCHITECTURE CONSTRAINTS
--------------------------------------------------

Respect the current repo structure and deployment model.

Requirements:

1. Do not break the current custom DQN trainer
   - keep it runnable as a baseline

2. Add modular structure rather than burying MAPPO logic inside unrelated files

3. Preserve firmware independence
   - `firmware/` must not depend on training internals

4. Keep deployment understandable
   - if recurrent inference helpers are added, keep them minimal and clearly separated

5. Do not add a zoo of half-finished algorithms
   - focus on making recurrent MAPPO the main path

--------------------------------------------------
PART 6 — EXPERIMENT / EVALUATION REQUIREMENTS
--------------------------------------------------

Add enough experiment structure to make the curriculum and MAPPO path credible.

Required:

1. deterministic evaluation mode
2. fixed-seed support
3. curriculum-aware run metadata
4. benchmark-ready episode metrics

Track at least:

- episode reward
- food retrieved / delivered
- exploration coverage
- collision count or rate
- pheromone usage
- episode length
- stage identifier

If feasible, also support:

- multi-seed evaluation summaries
- curriculum-stage comparison outputs

--------------------------------------------------
PART 7 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update documentation as part of the implementation.

Required doc updates:

1. how recurrent MAPPO is structured in this repo
2. how the GRU actor/critic work at a high level
3. how the curriculum stages are defined
4. how to train each curriculum stage
5. how to resume from one stage into the next
6. how centralized training differs from decentralized deployment here
7. how existing environment randomization contributes to robustness

Also update:

- `docs/PROJECT_LOG.md`
- `docs/QandA.md` when user-facing codebase questions are naturally answered by the work

--------------------------------------------------
PART 8 — IMPLEMENTATION PRIORITIES
--------------------------------------------------

Implement in this order:

Priority 1:
- recurrent MAPPO training path with GRU

Priority 2:
- curriculum learning workflow for `1 -> small swarm -> full swarm`

Priority 3:
- centralized state integration and rollout storage quality

Priority 4:
- evaluation and metadata improvements for curriculum runs

Priority 5:
- deployment-facing recurrent inference helper path

--------------------------------------------------
PART 9 — WHAT MUST NOT BREAK
--------------------------------------------------

Preserve:

1. existing DQN baseline path
2. existing environment local observation path unless strongly justified
3. existing action semantics unless strongly justified
4. checkpoint compatibility expectations within each algorithm family
5. decentralized deployment assumptions

Do not turn this into:

- a simulator rewrite
- a robotics middleware rewrite
- an algorithm zoo

--------------------------------------------------
PART 10 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short smoke test for:
   - stage 1 single-agent MAPPO
   - a later-stage small-swarm or full-swarm curriculum step if feasible
3. show the exact commands used
4. explain any constraints if a full smoke test is not practical

--------------------------------------------------
PART 11 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the recurrent MAPPO path added
2. how the GRU recurrence is handled
3. how the curriculum stages are configured and advanced
4. what centralized state definition is used
5. how decentralized execution is preserved
6. what evaluation/metadata support was added
7. exact verification commands run
8. remaining limitations or deferred work

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main algorithm should be:

- **recurrent GRU-based MAPPO**

The curriculum should explicitly be:

- **1 agent**
- **small swarm (2-5)**
- **full MARL (5+)**

Treat the environment’s randomized resets as built-in domain randomization and preserve that strength.

Keep the repo runnable.
