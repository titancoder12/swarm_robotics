Work in an existing multi-agent swarm reinforcement learning codebase.

Task:
Fix the current MAPPO failure mode after Step 32: the new guaranteed-homing stage now produces meaningful **sampled** pickup and delivery, but the behavior still does not survive into **greedy** evaluation.

Goal:
Make the recurrent MAPPO path convert post-pickup homing behavior from:

- “works sometimes under stochastic training-time sampling”

into:

- “works reliably under greedy evaluation and demo”

The main problem is now:

- sampled behavior is improving
- greedy behavior is still collapsing to zero

Do not invent a new algorithm.
Do not solve this by simply training longer.

--------------------------------------------------
PART 1 — CURRENT EVIDENCE
--------------------------------------------------

From the latest task 32 verification run:

- `runs/mappo_task32_verify_20260330_215803/episode_metrics.csv` shows:
  - `stage1d_single_agent_guaranteed_homing`
    - mean `food_picked_up ~= 2.727`
    - mean `food_retrieved ~= 0.273`
  - `stage1e_single_agent_delivery_bridge`
    - mean `food_picked_up ~= 2.643`
    - mean `food_retrieved ~= 0.036`
  - `stage1f_single_agent_delivery_obstacles`
    - mean `food_picked_up ~= 1.647`
    - mean `food_retrieved ~= 0.0`

- but `runs/mappo_task32_verify_20260330_215803/eval_metrics.csv` still shows:
  - greedy `food_picked_up = 0.0`
  - greedy `food_retrieved = 0.0`
  - across the logged stages

Interpretation:

- Step 32 improved the stage design in the right direction
- the policy is now capable of some sampled post-pickup homing
- but the behavior is still not becoming robust enough to survive greedy execution

So the next step is:

- fix the sampled-to-greedy gap directly

--------------------------------------------------
PART 2 — PRIMARY RECOMMENDATION
--------------------------------------------------

Treat the current bottleneck mainly as:

1. sampled behavior exists
2. deterministic/greedy policy quality is still poor
3. stage progression and checkpoint selection should prioritize greedy homing much more aggressively

The next implementation should focus on:

- reducing the sampled-vs-greedy gap
- making greedy post-pickup homing visible and persistent

--------------------------------------------------
PART 3 — GREEDY-HOMING TRAINING PRESSURE
--------------------------------------------------

Required direction:

1. make the trainer pay more attention to greedy homing success during the early return stages
2. do not rely only on stochastic rollout outcomes as evidence of learning
3. bias checkpoint selection and promotion toward visible greedy return behavior

Good directions include:

- stronger use of stage-end greedy eval in checkpoint selection
- more delivery-sensitive best-checkpoint scoring in the homing stages
- stronger penalties for sampled/good but greedy/bad stage outcomes

The key principle is:

- if homing only works while sampling, the stage is not learned yet

--------------------------------------------------
PART 4 — ENTROPY / EXPLORATION REVIEW
--------------------------------------------------

The current gap may still reflect too much stochastic dependence.

Required review:

- stage entropy schedules
- early return-stage entropy decay
- whether the guaranteed-homing and bridge stages remain too stochastic too long

Preferred direction:

- exploration should still help discovery
- but the homing stages should become deterministic enough, early enough, that greedy behavior can form

If needed, make the guaranteed-homing and bridge stages more aggressive about entropy decay than the current defaults.

--------------------------------------------------
PART 5 — PROMOTION AND REPEATS
--------------------------------------------------

Promotion logic should become even more aligned with the actual user-visible goal.

Required:

1. if a homing stage shows sampled success but greedy failure, it should clearly fail promotion
2. stage repeat behavior should remain compatible with this
3. runtime summaries should make this failure mode obvious

Helpful additions:

- clearer “sampled good / greedy bad” summary prints
- explicit stage-failure reason for homing stages

--------------------------------------------------
PART 6 — BEST CHECKPOINT SELECTION
--------------------------------------------------

The trainer should save and recommend checkpoints that reflect actual visible homing behavior.

Required:

1. review how `best_greedy_eval` is currently scored
2. make sure early homing stages weight completed delivery strongly enough
3. keep the recommended demo checkpoint aligned with the strongest greedy homing behavior

If current scoring is too permissive for pickup-only or reward-only results, tighten it.

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

1. what Step 32 improved
2. what still remained broken after Step 32
3. what Step 33 changes in trainer pressure / greedy alignment
4. which checkpoint should be demoed after the new changes

--------------------------------------------------
PART 8 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks
2. run at least one short smoke run
3. verify:
   - sampled homing is still present
   - greedy delivery in the guaranteed-homing stage is improved relative to Step 32
   - bridge-stage greedy behavior is not immediately zero if feasible
4. show exact commands used
5. explain remaining limitations honestly

--------------------------------------------------
PART 9 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of what changed in trainer pressure / greedy alignment
2. what changed in entropy or checkpoint scoring, if anything
3. what changed in promotion logic, if anything
4. what the new verification run shows
5. exact verification commands run
6. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main objective is now:

- **convert sampled post-pickup homing into greedy post-pickup homing**

The next fix should be primarily about:

- trainer pressure
- entropy / determinism timing
- greedy-aligned checkpointing and promotion

not another broad curriculum rewrite.
