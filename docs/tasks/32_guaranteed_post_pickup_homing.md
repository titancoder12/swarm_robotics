Work in an existing multi-agent swarm reinforcement learning codebase.

Task:
Redesign the early MAPPO return curriculum so it teaches **guaranteed post-pickup homing** before expecting the policy to handle cluttered delivery.

Goal:
Make the recurrent MAPPO path reliably learn this sequence in greedy execution:

1. find target
2. pick up target
3. immediately switch into homing mode
4. complete delivery to nest

The key requirement is:

- the curriculum must first make post-pickup homing itself reliable
- only after that should mild clutter be reintroduced

Do not invent a new algorithm.
Do not solve this by simply running longer.

--------------------------------------------------
PART 1 — WHY TASK 31 WAS NOT ENOUGH
--------------------------------------------------

Current evidence from the latest task 31 verification runs:

- `mappo_task31_verify3` completed cleanly
- `stage1a_single_agent_miniscule` still has strong greedy delivery
- but:
  - `stage1d_single_agent_return_medium` ended with greedy pickup `0.0`, greedy delivery `0.0`
  - `stage1e_single_agent_delivery_bridge` ended with greedy pickup `0.0`, greedy delivery `0.0`
  - `stage1f_single_agent_delivery_obstacles` ended with greedy pickup `0.0`, greedy delivery `0.0`

Interpretation:

- Step 31 improved stage structure
- but it still tried to teach too many things at once
- the system still does not robustly learn:
  - “after pickup, go home”

So the next step is not “milder clutter.”
The next step is:

- teach post-pickup homing almost in isolation

--------------------------------------------------
PART 2 — PRIMARY DESIGN CHANGE
--------------------------------------------------

Redesign the single-agent return stages around a stronger two-phase lesson:

Phase A:
- target discovery and pickup

Phase B:
- post-pickup homing and delivery

The curriculum should make Phase B much easier to learn than it is now.

Required direction:

1. at least one stage must strongly bias the episode toward:
   - pickup happening early
   - most of the challenge being the return
2. clutter should come only after greedy homing is reliable
3. the first cluttered return stage should be introduced only after this homing stage is working

--------------------------------------------------
PART 3 — GUARANTEED POST-PICKUP HOMING STAGE
--------------------------------------------------

Implement or redesign a stage so that post-pickup homing is the dominant lesson.

Requirements:

1. single agent
2. exactly one target
3. no respawn
4. no or near-zero clutter
5. geometry that makes pickup likely and leaves most remaining episode difficulty in the return

Good directions:

- compact arena
- moderate nest-target separation
- episode settings that make failed homing obvious
- shaping that strongly rewards the carried return

The stage should not mainly be “search again.”
It should mainly be:

- “once I have the item, I know I should go home”

--------------------------------------------------
PART 4 — POST-PICKUP-SPECIFIC STAGE DESIGN
--------------------------------------------------

The homing stage should reflect the fact that the real bottleneck now starts after pickup.

Required review:

- target placement relative to nest
- episode length
- reward_nest_approach
- reward_nest_delivery
- reward_undelivered_food
- reward_new_cell while carrying
- whether pickup should happen quickly so the stage spends more of its time on return behavior

Preferred direction:

- reduce time spent teaching search in this stage
- increase time spent teaching return completion

--------------------------------------------------
PART 5 — CLUTTER REINTRODUCTION
--------------------------------------------------

After the guaranteed-homing stage, reintroduce clutter more carefully.

Required:

1. the next stage after homing should be a mild clutter-return stage
2. the later stage can remain the true obstacle-return stage
3. clutter should be staged only after homing is already reliable

The intended progression should be:

1. empty or nearly empty homing stage
2. mild clutter return stage
3. true obstacle return stage

Do not collapse these back into one stage.

--------------------------------------------------
PART 6 — PROMOTION LOGIC
--------------------------------------------------

Promotion should explicitly reflect the new post-pickup homing lesson.

Required:

1. the guaranteed-homing stage must require real greedy delivery before promotion
2. the mild clutter-return stage must also require real greedy delivery
3. pickup-only success must not be enough

Use promotion targets that match the redesigned stages.

--------------------------------------------------
PART 7 — LOGGING AND EVAL
--------------------------------------------------

Keep the current greedy-eval discipline and make the homing-stage results easy to read.

Required:

1. keep logging pickup, delivery, and delivery conversion
2. make stage summaries clearly show whether pickup happened but homing failed
3. make it obvious which stage is the guaranteed-homing stage in logs and metadata

--------------------------------------------------
PART 8 — DOCUMENTATION
--------------------------------------------------

Update docs as part of the implementation.

Required:

1. update `README.md`
2. update `docs/TRAINING_MAPPO.md`
3. update `docs/QandA.md`
4. update `docs/PROJECT_LOG.md`

Docs should explain:

1. what Step 31 changed
2. why Step 31 still failed
3. what Step 32 changes in the early return curriculum
4. which stage is now the guaranteed post-pickup homing stage
5. which checkpoint should be demoed after the change

--------------------------------------------------
PART 9 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks
2. run at least one short smoke run
3. verify:
   - early pickup still works
   - the new guaranteed-homing stage shows nonzero greedy delivery
   - the next mild clutter stage is no longer immediately dead
4. show exact commands used
5. explain remaining limitations honestly

--------------------------------------------------
PART 10 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the new early-return curriculum design
2. what changed in the guaranteed-homing stage
3. what changed in the first cluttered return stage
4. what promotion criteria changed
5. what greedy eval now shows
6. exact verification commands run
7. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main objective is now:

- **teach post-pickup homing first**

The next stage after that can teach:

- **homing under mild clutter**

And only after that:

- **true obstacle-return delivery**

Keep the solution tightly focused on curriculum and stage design for return behavior.
