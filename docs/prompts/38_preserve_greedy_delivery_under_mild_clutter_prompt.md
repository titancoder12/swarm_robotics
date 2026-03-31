You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement the next focused fix for recurrent MAPPO so that the policy maintains
greedy delivery behavior once mild clutter is reintroduced after the dedicated
homing lessons.

Context:
- Prompt 37 was the first pass that produced nonzero greedy delivery in:
  - `stage1d_single_agent_carry_bootstrap`
  - `stage1e_single_agent_guaranteed_homing`
- The remaining bottleneck is now narrower:
  - `stage1f_single_agent_delivery_bridge` still collapses back to zero greedy
    delivery in the short verification run
- So the next problem is no longer “can the agent ever learn greedy homing?”
- The new problem is:
  - “how do we preserve that greedy homing behavior when mild clutter and a
    small amount of geometric ambiguity return?”

Do not broaden scope back into generic reward tuning. Focus on the bridge-stage
failure specifically.

--------------------------------------------------
PART 1 — TARGET OUTCOME
--------------------------------------------------

After this change, the training path should:

1. keep the strong greedy delivery behavior already achieved in:
   - `stage1d_single_agent_carry_bootstrap`
   - `stage1e_single_agent_guaranteed_homing`
2. preserve nonzero greedy delivery when advancing into:
   - `stage1f_single_agent_delivery_bridge`
3. only then hand off to:
   - `stage1g_single_agent_delivery_obstacles`

The bridge stage should act like a true continuity stage:
- not a fresh collapse point
- not a near-reset of return behavior
- not just a pickup-rich / delivery-zero stage with slightly more clutter

--------------------------------------------------
PART 2 — IMPLEMENTATION GOALS
--------------------------------------------------

Implement changes that directly support greedy delivery continuity under mild
clutter.

Preferred directions:

1. Bridge-stage geometry continuity
- make `stage1f_single_agent_delivery_bridge` a closer continuation of
  `stage1e_single_agent_guaranteed_homing`
- reintroduce clutter gradually rather than sharply
- if helpful, constrain obstacle placement so the first bridge layouts do not
  create avoidable trap pockets or severe route ambiguity
- the bridge stage should still contain clutter, but it should teach “same
  return behavior, slightly harder geometry,” not “new task”

2. Carrying-phase route continuity
- when the agent is carrying, preserve the homing signal under mild clutter
- if needed, strengthen carrying-only route-follow / nest-approach supervision
  specifically in the bridge stage
- avoid paying the agent for aimless local wandering while carrying

3. Stage-specific bridge simplification
- if necessary, simplify only the bridge stage control/task conditions rather
  than the whole curriculum
- examples:
  - fewer obstacles than now
  - safer obstacle spacing
  - easier nest-target corridor geometry
  - slightly shorter target-to-nest distance band
- but do not make the bridge stage identical to the no-clutter homing stage

4. Greedy-delivery-first promotion
- ensure bridge-stage promotion is driven by greedy delivery continuity
- pickup-only success must not be enough
- sampled delivery alone must not be enough

5. Keep the good parts of prompt 37
- preserve:
  - `start_carrying_food`
  - `stage1d_single_agent_carry_bootstrap`
  - the carrying-state bug fix
  - per-stage repeat-floor behavior for return-critical lessons

--------------------------------------------------
PART 3 — WHAT TO INSPECT FIRST
--------------------------------------------------

Before editing, inspect:

1. `algorithms/mappo/curriculum.py`
   - especially:
     - `stage1e_single_agent_guaranteed_homing`
     - `stage1f_single_agent_delivery_bridge`
     - `stage1g_single_agent_delivery_obstacles`
   - compare how large the jump really is between `stage1e` and `stage1f`

2. `env/swarm_env.py`
   - target placement
   - obstacle placement
   - carrying-phase reward / progress logic
   - any geometry or collision behavior that could disproportionately break
     greedy homing under mild clutter

3. `train/mappo_gru.py`
   - bridge-stage promotion targets
   - greedy checkpoint scoring
   - repeat handling

4. current verification artifacts
   - use the existing prompt-37 verification outputs to reason from the actual
     failure mode, not only from code assumptions

--------------------------------------------------
PART 4 — REQUIRED CHANGES
--------------------------------------------------

Implement a compact, focused bridge-stage fix.

At minimum, do all of the following:

1. Adjust the stage-1 bridge curriculum so `stage1f_single_agent_delivery_bridge`
   is a better continuation from `stage1e`

2. Adjust bridge-stage reward/control/placement settings if needed so carrying
   agents keep homing instead of reverting to zero greedy delivery

3. Keep `stage1g_single_agent_delivery_obstacles` as the next harder stage, but
   make sure the bridge is actually a bridge

4. Keep trainer-side promotion/checkpoint logic aligned so nonzero greedy
   delivery is the success criterion for the bridge stage

--------------------------------------------------
PART 5 — VALIDATION REQUIREMENTS
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files

2. run at least one short focused verification for:
   - `--curriculum stage1`
   - enough steps to exercise:
     - `stage1d_single_agent_carry_bootstrap`
     - `stage1e_single_agent_guaranteed_homing`
     - `stage1f_single_agent_delivery_bridge`

3. inspect the generated metrics and report specifically:
   - sampled pickup / delivery for `stage1f`
   - greedy pickup / delivery for `stage1f`
   - whether `stage1f` still collapses to zero greedy delivery

4. if the bridge stage still does not fully pass, explain what improved and
   what remains broken

Show the exact commands used.

--------------------------------------------------
PART 6 — DOCS
--------------------------------------------------

Update the docs to reflect the current system after this change:

1. `README.md`
2. `docs/TRAINING_MAPPO.md`
3. `docs/QandA.md`
4. `docs/PROJECT_LOG.md`

Requirements:
- document the bridge-stage intent clearly
- make sure stage names and descriptions are current
- describe the new bridge-stage continuity logic
- record what verification actually showed

--------------------------------------------------
PART 7 — CONSTRAINTS
--------------------------------------------------

Do not:
- remove the dedicated carry-bootstrap lesson
- revert the prompt-37 carrying-state bug fix
- weaken success criteria back to pickup-only
- turn this into a generic simulator rewrite
- add a large new algorithm family

Prefer:
- targeted curriculum and env/trainer adjustments
- small, legible changes
- preserving repo runnability

--------------------------------------------------
DELIVERABLES
--------------------------------------------------

After implementation, report:

1. what changed in the bridge stage
2. why those changes should preserve greedy delivery better
3. what verification command(s) were run
4. what the short verification actually showed for:
   - `stage1d`
   - `stage1e`
   - `stage1f`
5. remaining limitations, if any
