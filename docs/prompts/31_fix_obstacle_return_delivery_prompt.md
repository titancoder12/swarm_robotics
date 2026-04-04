Work in an existing multi-agent swarm reinforcement learning codebase.

Task:
Fix the current MAPPO bottleneck after Step 30: target discovery and pickup are now mostly working in early single-agent stages, but **greedy return-to-nest delivery still collapses once the task becomes less trivial**, especially in the first obstacle-return stage.

Goal:
Make the recurrent MAPPO curriculum reliably learn:

1. discover target
2. pick up target
3. switch into return-to-nest mode
4. complete delivery under mild clutter

before the curriculum advances into harder obstacle or swarm stages.

Current evidence from the latest run:

- `runs/mappo_full_600k_fix30_20260330_213800/episode_metrics.csv` shows:
  - `stage1a_single_agent_miniscule`:
    - mean `food_picked_up ~= 3.459`
    - mean `food_retrieved ~= 3.459`
  - `stage1b_single_agent_tiny`:
    - attempt 1 mean `food_retrieved ~= 0.325`
    - attempt 2 mean `food_retrieved ~= 0.482`
  - `stage1c_single_agent_small`:
    - pickup is almost perfect
    - delivery remains limited:
      - attempt 1 mean `food_retrieved ~= 0.309`
      - attempt 2 mean `food_retrieved ~= 0.329`
  - `stage1d_single_agent_return_medium`:
    - pickup is effectively solved
    - delivery is still weak:
      - attempt 1 mean `food_retrieved ~= 0.203`
      - attempt 2 mean `food_retrieved ~= 0.201`
  - `stage1e_single_agent_delivery_obstacles`:
    - mean `food_picked_up ~= 2.879`
    - mean `food_retrieved ~= 0.021`
    - the stage still produces many full-length no-progress episodes

- `runs/mappo_full_600k_fix30_20260330_213800/eval_metrics.csv` shows:
  - `stage1a` greedy eval is strong:
    - pickup `4.0`
    - delivery `4.0`
  - `stage1b` greedy eval improved:
    - delivery reached `1.2` on attempt 2
  - `stage1c` greedy eval regressed:
    - latest delivery `0.0`
  - `stage1d` greedy eval:
    - both attempts delivery `0.0`
  - the main collapse is now specifically:
    - **greedy carried return-to-nest completion**
    - especially once clutter is introduced

Interpretation:

- Step 29 fixed the old early greedy collapse
- Step 30 improved pickup and some sampled return behavior
- the remaining failure is narrower:
  - agents can often find and pick up the target
  - but they still do not robustly complete the return-to-nest policy in greedy execution
  - the first obstacle-return stage is still too hard relative to what the policy has actually mastered

Do not invent a new algorithm.
Improve the current recurrent MAPPO path and curriculum so greedy return-to-nest delivery becomes reliable before harder stages.

--------------------------------------------------
PART 1 — PRIMARY RECOMMENDATION
--------------------------------------------------

The next improvement should not be “train longer.”

The main recommendation is:

1. add or refine a dedicated **post-pickup homing stage**
2. simplify the first cluttered delivery stage so it teaches mild obstacle return instead of immediate collapse
3. require greedy delivery success before promotion

Treat the current problem as a curriculum-and-delivery-shaping issue, not a compute-budget issue.

--------------------------------------------------
PART 2 — POST-PICKUP HOMING STAGE
--------------------------------------------------

Implement or refine a stage whose main lesson is:

- search briefly
- pick up one target
- switch into homing
- complete delivery to nest

Requirements:

1. make this stage explicitly easier than the first obstacle-return stage
2. keep it more meaningful than the miniscule/tiny stages
3. focus on post-pickup nest completion, not just target discovery

Good directions:

- moderate arena size
- simple geometry
- no or extremely light clutter
- one target
- short enough episodes that failed return is obvious

If a suitable stage already exists, refine it rather than adding unnecessary bloat.

--------------------------------------------------
PART 3 — FIRST OBSTACLE-RETURN STAGE
--------------------------------------------------

The first obstacle-return stage should teach:

- mild clutter navigation while carrying
- nest completion under manageable obstacle pressure

Required changes:

1. reduce the difficulty spike from the homing stage into the first obstacle stage
2. keep the stage scientifically meaningful
3. do not remove obstacles entirely

Preferred directions:

- fewer obstacles than the later true obstacle stage
- more forgiving obstacle placement
- less pathological trap potential
- arena size and spacing that still allow clear return routes

This stage should be a bridge, not the final challenge.

--------------------------------------------------
PART 4 — CARRYING-STATE SHAPING
--------------------------------------------------

The policy still needs a cleaner switch from search mode to return mode.

Required review:

- `reward_nest_approach`
- `reward_nest_delivery`
- `reward_undelivered_food`
- `reward_new_cell`
- `carrying_reward_new_cell_scale`
- any search-oriented shaping that may still compete after pickup

Preferred direction:

- before pickup:
  - exploration and target acquisition can matter
- after pickup:
  - homing and delivery completion should dominate

Improve carrying-state shaping if needed, but do not merely inflate all rewards.
Use targeted shaping that specifically improves return completion.

--------------------------------------------------
PART 5 — PROMOTION LOGIC
--------------------------------------------------

The curriculum should not promote stages that still fail greedy delivery.

Required changes:

1. make promotion criteria for the homing and early obstacle-return stages more delivery-sensitive
2. ensure stage promotion reflects visible greedy completion, not just pickup
3. preserve existing repeat-limit behavior or refine it if needed

Preferred direction:

- homing stage should require nontrivial greedy delivery
- first obstacle-return stage should also require nontrivial greedy delivery before promotion

The key is to stop weak return policies from slipping into later stages.

--------------------------------------------------
PART 6 — LOGGING AND EVAL
--------------------------------------------------

Keep the current greedy-eval discipline and make it even more useful for this failure mode.

Required:

1. continue logging pickup, delivery, and delivery conversion
2. make runtime summaries clearly expose the pickup-to-delivery dropoff in homing and obstacle stages
3. keep checkpoint selection aligned with greedy delivery behavior

Helpful additions if small:

- stage summaries that emphasize `delivery_conversion`
- explicit notice when a stage fails promotion due to delivery target

--------------------------------------------------
PART 7 — DOCUMENTATION
--------------------------------------------------

Update docs as part of the implementation.

Required:

1. update `README.md`
2. update `docs/TRAINING_MAPPO.md`
3. update `docs/QandA.md`
4. update `docs/PROJECT_LOG.md`

Docs should explain:

1. what Step 30 improved
2. what remained broken after Step 30
3. what Step 31 changes in the curriculum and promotion logic
4. why the first obstacle-return stage was simplified
5. which checkpoint should be demoed after the new changes

--------------------------------------------------
PART 8 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks
2. run at least one short smoke run
3. verify all of the following:
   - early pickup behavior still works
   - the homing stage shows better greedy delivery than before
   - the first obstacle-return stage no longer immediately collapses to near-zero delivery
4. show exact commands used
5. explain remaining limitations honestly

--------------------------------------------------
PART 9 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the curriculum changes
2. what changed in the homing stage
3. what changed in the first obstacle-return stage
4. what promotion criteria changed
5. what greedy eval now shows
6. exact verification commands run
7. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main goal now is:

- **reliable greedy return-to-nest delivery before harder obstacle and swarm stages**

Do not solve this by:

- simply running longer
- removing the obstacle challenge entirely
- inventing a new RL algorithm

Keep the solution focused on:

- curriculum refinement
- carrying-state return shaping
- delivery-sensitive promotion
