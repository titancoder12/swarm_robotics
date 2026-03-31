You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Fix the remaining MAPPO failure mode after prompt 28: the trainer now has better stage-transfer stability and greedy-eval-aware progression, but the policy still learns behavior that occasionally succeeds under stochastic training-time sampling and then collapses to near-zero pickup/delivery under greedy evaluation.

Goal:
Make the recurrent MAPPO policy learn behavior that survives into greedy execution in early curriculum stages before expecting good final-stage behavior.

Current evidence from the latest run:

- `runs/mappo_full_600k_fix28_20260330_210820/episode_metrics.csv` shows:
  - `stage1a_single_agent_miniscule` attempt 1 had many sampled successes (`food_picked_up` and `food_retrieved` both nonzero on average)
  - but `stage1a` greedy eval still ended at zero pickup / zero delivery
  - `stage1a` repeat attempt 2 largely collapsed
- `stage1b_single_agent_tiny` attempt 1 and attempt 2 both stayed at:
  - `food_picked_up = 0`
  - `food_retrieved = 0`
  - reward about `-1.2`
- `stage1c_single_agent_small` showed only tiny sampled pickup rates and still zero greedy eval success
- `stage1d_single_agent_delivery_obstacles` currently shows zero pickup and zero delivery

Interpretation:

- prompt 28 fixed important structural issues
- but the remaining main problem is now **greedy-behavior formation**
- the policy is still relying too much on stochastic action sampling during training
- reward / exploration / control pressure in early stages still does not produce robust greedy pickup-return-delivery behavior

Do not invent a new algorithm.
Improve the current MAPPO path so early-stage greedy behavior becomes reliable.

--------------------------------------------------
PART 1 — PRIMARY DIAGNOSIS
--------------------------------------------------

Treat the current failure mainly as:

1. poor alignment between sampled training behavior and greedy evaluation behavior
2. insufficiently learnable early-stage reward/control/exploration balance
3. curriculum stages that still allow “sampled lucky success” without robust policy formation

The new work should focus on making early stages produce stable greedy behavior first.

--------------------------------------------------
PART 2 — EXPLORATION / ENTROPY BEHAVIOR
--------------------------------------------------

The current MAPPO path likely relies too much on action sampling noise.

Required improvements:

1. revisit entropy usage and scheduling
2. reduce the chance that the policy depends on stochastic exploration forever
3. make the training path more likely to converge toward useful greedy actions

Preferred direction:

- support entropy decay or stage-wise entropy settings
- earlier stages may tolerate more exploration
- later within-stage training should become more deterministic

Do not simply set entropy to zero everywhere.
Make the exploration pressure deliberate.

--------------------------------------------------
PART 3 — EARLY-STAGE REWARD / CONTROL REBALANCING
--------------------------------------------------

The first stages must produce reliable greedy pickup and delivery.

Current issue:

- even the easiest stages still end in zero greedy pickup/delivery during evaluation
- timeout-like no-progress behavior remains too attractive

Required improvements:

1. make useful movement in early stages more attractive than freezing
2. keep delivery dominant
3. do not let exploration reward or penalties overwhelm the basic task

Review together:

- `reward_step`
- `reward_collision`
- `reward_pickup`
- `reward_nest_delivery`
- `reward_undelivered_food`
- `reward_new_cell`
- pheromone-related shaping in stages where it may not yet help
- action-repeat settings for the earliest stages

Preferred direction:

- early stages should be explicitly optimized for robust greedy `pickup -> return -> deliver`
- later stages can restore more of the full trail-building complexity

--------------------------------------------------
PART 4 — STAGE-SPECIFIC SIMPLIFICATION
--------------------------------------------------

The earliest stages should teach one thing at a time.

Required improvements:

1. make sure the very earliest stages are not carrying unnecessary distractions
2. if pheromone or other shaping is not helping in stage 1A/1B, reduce or defer it there
3. keep stage 1 focused on making greedy pickup/delivery robust

Examples of acceptable changes:

- stage-wise pheromone disable/enable if justified
- stage-wise simpler reward settings
- stage-wise lower penalties if they are suppressing useful motion
- stage-wise control responsiveness

Do not make the curriculum huge.
Keep the changes disciplined and well explained.

--------------------------------------------------
PART 5 — GREEDY-EVAL AS A FIRST-CLASS SIGNAL
--------------------------------------------------

Greedy evaluation should be more central than it is now.

Required improvements:

1. make early-stage greedy eval a stronger decision-making signal
2. ensure the best checkpoint is not chosen by lucky sampled behavior
3. keep checkpoint selection aligned with visible behavior quality

Possible improvements:

- stronger reliance on greedy eval for best-checkpoint selection
- stage-level summaries that clearly show sampled success vs greedy success
- clearer runtime prints that call out when training success is not surviving into greedy eval

--------------------------------------------------
PART 6 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update docs as part of the implementation.

Required:

1. update `README.md`
2. update `docs/TRAINING_MAPPO.md`
3. update `docs/QandA.md`
4. update `docs/PROJECT_LOG.md`

Documentation must explain:

1. what prompt 28 fixed structurally
2. what remained broken afterward
3. what prompt 29 changes about early-stage greedy-behavior formation
4. how entropy / exploration / reward settings now behave
5. which checkpoint should be demoed after the new changes

--------------------------------------------------
PART 7 — VERIFICATION REQUIREMENTS
--------------------------------------------------

After implementation:

1. run syntax/import checks
2. run at least one short smoke run
3. show that:
   - early-stage greedy eval is being measured clearly
   - checkpoint selection still works
   - early stages are less likely to collapse immediately into zero-greedy-pickup behavior
4. show exact commands used
5. explain remaining limitations honestly

--------------------------------------------------
PART 8 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the new early-stage anti-collapse changes
2. what the latest run artifacts showed
3. what entropy / exploration changes were made
4. what reward/control changes were made
5. what stage-specific simplifications were made, if any
6. which checkpoint is now recommended for demo
7. exact verification commands run
8. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Prioritize in this order:

1. make early-stage greedy behavior reliable
2. reduce dependence on stochastic sampled success
3. tune reward/control pressure only as needed
4. keep the changes understandable and documented

Do not solve this with cosmetic demo changes.
Do not assume prompt 28 was enough.
Use the new run evidence and fix the remaining greedy-behavior failure directly.
