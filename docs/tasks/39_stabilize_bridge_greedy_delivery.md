Work in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement the next focused fix for recurrent MAPPO so the bridge stage
(`stage1f_single_agent_delivery_bridge`) maintains greedy delivery
consistently enough that stage 1 is no longer the main blocker and the project
can move into real full training runs.

Context:
- Step 37 was the first pass that produced nonzero greedy delivery in the
  dedicated homing stack.
- Step 38 improved `stage1f_single_agent_delivery_bridge` materially:
  - sampled bridge delivery is nonzero
  - at least one bridge eval row showed strong greedy delivery
  - but later in the same short run, greedy bridge delivery still fell back to
    zero
- So the remaining bottleneck is now:
  - not “can mild clutter ever work?”
  - but “how do we make greedy bridge delivery stable enough to trust the stage
    and proceed to real training?”

Do not broaden scope back into generic RL tuning. Focus on stabilizing
bridge-stage greedy delivery so the stage stops oscillating between “works” and
“collapses.”

--------------------------------------------------
PART 1 — TARGET OUTCOME
--------------------------------------------------

After this change, short verification should show that:

1. `stage1d_single_agent_carry_bootstrap`
   still has nonzero greedy delivery

2. `stage1e_single_agent_guaranteed_homing`
   still has nonzero greedy delivery

3. `stage1f_single_agent_delivery_bridge`
   now maintains nonzero greedy delivery across its short verification run
   instead of producing one good eval row and then falling back to zero

The bridge stage should become stable enough that:
- it no longer looks like a coin flip
- it can be used as a trustworthy handoff into `stage1g`
- stage 1 is no longer the main blocker before a real full training run

--------------------------------------------------
PART 2 — IMPLEMENTATION PRIORITIES
--------------------------------------------------

Priority 1:
- stabilize greedy delivery in `stage1f_single_agent_delivery_bridge`

Priority 2:
- keep `stage1d` and `stage1e` healthy

Priority 3:
- preserve the task 37 and task 38 improvements

Priority 4:
- improve stage-end checkpoint selection / promotion behavior only if it helps
  bridge-stage stability directly

--------------------------------------------------
PART 3 — RECOMMENDED FIX DIRECTION
--------------------------------------------------

Implement a focused bridge-stability solution, not a broad redesign.

Preferred directions:

1. Bridge-stage consistency over peak performance
- optimize for “stable nonzero greedy delivery across evals”
- not just “one impressive bridge eval row”

2. Reduce bridge-stage within-run regression
- inspect why `stage1f` can show a good greedy eval mid-run and then degrade by
  the later eval
- likely causes to consider:
  - too much continued entropy / exploration after the policy already found a
    usable homing strategy
  - policy drift during later updates in the bridge stage
  - bridge-stage geometry still varying too much between resets
  - trainer keeping training after the stage’s best greedy behavior has already
    appeared, then unlearning it

3. Preserve the best bridge behavior instead of letting it drift away
- if needed, make the bridge stage more “best-greedy-behavior preserving”
- examples:
  - stronger best-checkpoint retention within the stage
  - stage-end restore of the best bridge checkpoint before promotion decision
  - stronger delivery-first greedy checkpoint selection specifically for
    `stage1f`
- do this only if it materially supports stage stability

4. Make bridge resets less volatile if needed
- if the bridge stage still varies too much per reset, reduce that volatility
  in a targeted way
- examples:
  - narrower target-distance band
  - slightly safer bridge obstacle placement
  - stronger corridor preservation
- but do not remove clutter entirely

5. Make the bridge stage “lock in” homing once learned
- if useful, make late-stage bridge entropy even lower
- or add stage-local logic that reduces policy drift after nonzero greedy
  delivery appears
- do not make this a global change unless clearly justified

--------------------------------------------------
PART 4 — WHAT TO INSPECT FIRST
--------------------------------------------------

Before editing, inspect:

1. `algorithms/mappo/curriculum.py`
   - `stage1f_single_agent_delivery_bridge`
   - compare it again with `stage1e` and `stage1g`

2. `train/mappo_gru.py`
   - bridge-stage promotion logic
   - greedy checkpoint scoring
   - whether the trainer can drift away from the best bridge policy later in
     the same stage

3. `runs/mappo_task38_verify2_*/eval_metrics.csv`
   - compare the good bridge eval row with the later failed one
   - reason from that actual pattern, not just from assumptions

4. `env/swarm_env.py`
   - only for targeted bridge-stage reset / geometry stability if needed

--------------------------------------------------
PART 5 — REQUIRED CHANGES
--------------------------------------------------

Implement a compact bridge-stability fix that does all of the following:

1. reduce the chance that `stage1f` regresses from nonzero greedy delivery back
   to zero within the same stage

2. preserve or restore the best greedy bridge behavior when appropriate

3. keep delivery-first promotion and checkpointing for the bridge stage

4. avoid regressing `stage1d` and `stage1e`

5. keep the repo runnable

--------------------------------------------------
PART 6 — VALIDATION REQUIREMENTS
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified Python files

2. run at least one focused verification for:
   - `--curriculum stage1`
   - enough total steps to exercise:
     - `stage1d_single_agent_carry_bootstrap`
     - `stage1e_single_agent_guaranteed_homing`
     - `stage1f_single_agent_delivery_bridge`

3. inspect the resulting metrics and report specifically:
   - sampled pickup / delivery / conversion for `stage1f`
   - greedy pickup / delivery / conversion for `stage1f`
   - whether bridge-stage greedy delivery remained nonzero across the short run

4. if `stage1f` still is not stable enough, explain exactly what improved and
   what remains unstable

Show the exact commands used.

--------------------------------------------------
PART 7 — DOCS
--------------------------------------------------

Update the docs after implementation:

1. `README.md`
2. `docs/TRAINING_MAPPO.md`
3. `docs/QandA.md`
4. `docs/PROJECT_LOG.md`

Requirements:
- describe the bridge-stability intent clearly
- record what verification actually showed
- keep the stage descriptions current

--------------------------------------------------
PART 8 — CONSTRAINTS
--------------------------------------------------

Do not:
- remove `stage1d_single_agent_carry_bootstrap`
- remove the task 37 carrying-state bug fix
- weaken the bridge stage back into a pickup-only success criterion
- remove clutter from the bridge stage entirely
- turn this into a generic algorithm rewrite

Prefer:
- small, legible changes
- targeted bridge-stage stabilization
- delivery-first reasoning

--------------------------------------------------
DELIVERABLES
--------------------------------------------------

After implementation, report:

1. what was changed to stabilize bridge-stage greedy delivery
2. why that should reduce within-stage regression
3. the verification command(s) run
4. the actual `stage1d` / `stage1e` / `stage1f` sampled and greedy outcomes
5. whether stage 1 now looks stable enough to move to real full training
6. remaining limitations, if any
