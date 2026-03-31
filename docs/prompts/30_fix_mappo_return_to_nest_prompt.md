You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Fix the remaining MAPPO failure mode after prompt 29: early-stage greedy pickup formation improved substantially, but the policy still fails to reliably **return to the nest and complete delivery**, especially once obstacles, larger spaces, or multiple agents are introduced.

Goal:
Make the recurrent MAPPO path learn reliable **carrying-food return-to-nest behavior** that survives into greedy evaluation, instead of only learning target discovery and pickup.

Current evidence from the latest run:

- `runs/mappo_full_600k_fix29_20260330_212303/episode_metrics.csv` shows:
  - `stage1a_single_agent_miniscule` attempt 1 is now strong:
    - mean `food_picked_up ~= 3.785`
    - mean `food_retrieved ~= 3.785`
  - `stage1b_single_agent_tiny` attempt 1 and 2 are also much improved on pickup:
    - mean `food_picked_up ~= 3.6+`
    - but mean `food_retrieved ~= 0.19`
  - `stage1c_single_agent_small` attempt 1 and 2:
    - mean `food_picked_up ~= 3.8+`
    - mean `food_retrieved ~= 0.34 to 0.38`
  - `stage1d_single_agent_delivery_obstacles` attempt 1 and 2:
    - pickup drops sharply
    - delivery is effectively zero
  - `stage2a_small_swarm_medium` attempt 1 and 2:
    - some pickup remains
    - delivery is still zero

- `runs/mappo_full_600k_fix29_20260330_212303/eval_metrics.csv` shows:
  - `stage1a` greedy eval is now genuinely strong:
    - pickup `4.0`, delivery `4.0`
  - `stage1b` greedy eval:
    - pickup is strong
    - delivery stays around `0.0 to 0.4`
  - `stage1c` greedy eval:
    - pickup is strong
    - delivery improves somewhat, but is still limited
  - `stage1d` greedy eval:
    - delivery is still zero
  - `stage2a` greedy eval:
    - delivery is still zero

Interpretation:

- prompt 29 fixed the old “early greedy collapse” problem
- the main failure is now more specific:
  - agents can find and pick up the target
  - but they still do not robustly learn the **return-to-nest completion policy**
- the current system is still weak at:
  - carrying-food navigation back to nest
  - maintaining nest-seeking behavior while carrying under obstacle clutter
  - turning pickup into completed delivery in multi-step episodes

Do not invent a new algorithm.
Improve the current MAPPO path so return-to-nest delivery becomes reliable.

--------------------------------------------------
PART 1 — PRIMARY DIAGNOSIS
--------------------------------------------------

Treat the current failure mainly as:

1. pickup has become learnable
2. return-to-nest behavior while carrying is still weak
3. obstacle-cluttered return and multi-agent scaling still break delivery

The new work should focus on delivery completion, not rediscovering the target.

--------------------------------------------------
PART 2 — RETURN-TO-NEST LEARNING
--------------------------------------------------

The carrying-food policy must become more nest-directed and completion-oriented.

Required improvements:

1. strengthen the learning signal for **successful carried return**
2. make the carrying-food state more behaviorally distinct
3. help the policy maintain nest-seeking once food is picked up

Review together:

- `reward_nest_approach`
- `reward_nest_delivery`
- `reward_undelivered_food`
- carrying-food observation path
- nest-direction observation usefulness
- whether carrying-food should change other shaping terms

Preferred direction:

- once carrying food, the policy should have a clearer reason to switch from search mode to return mode
- successful return should dominate “wander while carrying”

--------------------------------------------------
PART 3 — STAGE DESIGN FOR DELIVERY
--------------------------------------------------

The curriculum should teach return-to-nest more directly.

Required improvements:

1. create or refine a stage whose main lesson is:
   - find one target
   - pick it up
   - bring it back to nest
2. do not jump too quickly from easy pickup into hard obstacle return
3. ensure there is at least one stage that isolates delivery completion before multi-agent clutter dominates

Acceptable approaches:

- adjust stage geometry
- adjust nest/target spacing
- add an intermediate “return training” stage
- refine obstacle stage difficulty progression

Do not bloat the curriculum unnecessarily.
Keep it focused on the delivery bottleneck.

--------------------------------------------------
PART 4 — CARRYING-STATE-SPECIFIC SHAPING
--------------------------------------------------

The current failure strongly suggests the policy needs better carrying-specific incentives.

Required improvements:

1. inspect how shaping changes after pickup
2. make sure search-oriented shaping does not keep competing after pickup
3. ensure nest-seeking while carrying is rewarded more clearly than continued exploration

Review together:

- `reward_food_approach`
- `reward_food_detected`
- `reward_new_cell`
- pheromone following
- whether some shaping should weaken while carrying

Preferred direction:

- before pickup: exploration and target acquisition matter
- after pickup: return-to-nest completion matters much more

--------------------------------------------------
PART 5 — OBSTACLE-STAGE DELIVERY ROBUSTNESS
--------------------------------------------------

The obstacle delivery stage is currently the first serious collapse point.

Required improvements:

1. make obstacle-stage return learning more tractable
2. improve nest return under clutter without making the task trivial
3. keep the stage scientifically meaningful for route-finding

Possible directions:

- refine obstacle density or arrangement in the first delivery-obstacle stage
- reduce unnecessary difficulty spikes in that stage
- strengthen carrying-food nest guidance there
- improve stage-wise control/reward settings specifically for obstacle return

Do not solve this by removing the obstacle challenge entirely.

--------------------------------------------------
PART 6 — GREEDY EVAL AND CHECKPOINT SIGNALS
--------------------------------------------------

Greedy evaluation should keep driving decisions.

Required improvements:

1. continue prioritizing greedy delivery, not only pickup
2. make stage summaries explicitly show pickup-vs-delivery conversion quality
3. keep best-checkpoint selection aligned with visible completed-delivery behavior

Helpful additions:

- delivery conversion summaries
- carrying-food completion ratios
- clearer runtime prints for pickup-to-delivery drop-off

--------------------------------------------------
PART 7 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update docs as part of the implementation.

Required:

1. update `README.md`
2. update `docs/TRAINING_MAPPO.md`
3. update `docs/QandA.md`
4. update `docs/PROJECT_LOG.md`

Documentation must explain:

1. what prompt 29 fixed
2. what still remained broken after prompt 29
3. what prompt 30 changes about return-to-nest and delivery completion
4. how stage design or carrying-state shaping now works
5. which checkpoint should be demoed after the new changes

--------------------------------------------------
PART 8 — VERIFICATION REQUIREMENTS
--------------------------------------------------

After implementation:

1. run syntax/import checks
2. run at least one short smoke run
3. show that:
   - pickup still works in early stages
   - greedy delivery improves in the post-pickup stages
   - obstacle-stage delivery no longer immediately collapses to zero
4. show exact commands used
5. explain remaining limitations honestly

--------------------------------------------------
PART 9 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the new return-to-nest changes
2. what the fix29 run artifacts showed
3. what carrying-state or nest-return changes were made
4. what curriculum/stage changes were made
5. what greedy-eval delivery signals were improved
6. which checkpoint is now recommended for demo
7. exact verification commands run
8. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Prioritize in this order:

1. make carrying-food return-to-nest behavior reliable
2. improve delivery completion in greedy eval
3. keep pickup behavior intact while fixing return behavior
4. keep the changes disciplined and well documented

Do not solve this by focusing only on pickup.
Use the new run evidence and fix the return-to-nest delivery bottleneck directly.
