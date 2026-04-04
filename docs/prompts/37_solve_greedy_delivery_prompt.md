Work in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement the next iteration of changes specifically to solve the remaining
**greedy delivery** failure.

Current situation after Steps 34-36:

- the trainer now enforces a hard total-step cap
- delivery conversion is tracked and used for promotion/checkpoint scoring
- carrying-phase penalties and carrying-phase metrics exist
- sustained carrying-progress shaping exists
- the homing stages now use lower entropy and stronger delivery pressure

But the short stage-1 verification still shows the same core failure:

- sampled pickup can happen
- sampled delivery can happen in the easier homing stages
- greedy pickup/delivery in `stage1d`, `stage1e`, and `stage1f` still stays at `0.0`

So this step should not merely add more diagnostics or minor reward tweaks.
It must make the training path much more likely to produce a policy that
actually delivers under greedy evaluation.

--------------------------------------------------
PART 1 — PRIMARY OBJECTIVE
--------------------------------------------------

Implement a stronger solution so the agent learns a stable post-pickup
return-to-nest policy that survives greedy evaluation.

The main goal is:

- in the return-focused single-agent stages, greedy delivery must become nonzero
- and the policy should not leave those stages until that happens, unless the hard global step cap is reached

This step should build on Steps 34-36, not revert them.

--------------------------------------------------
PART 2 — CORE RECOMMENDATION TO IMPLEMENT
--------------------------------------------------

The current evidence suggests that reward shaping alone is not enough.
Implement a more explicit training design for the carrying phase.

Required directions:

1. Add a dedicated carrying-phase subtask lesson
- create or refine a stage where the agent starts in a state that is already very close to pickup or already carrying food if needed
- the lesson should isolate the homing problem much more directly than the current guaranteed-homing stage
- the purpose is to teach:
  - “when carrying, go to nest”
  - not “search, maybe pick up, maybe return”

2. Make the homing lesson hard to exit without greedy success
- for this dedicated carrying-phase lesson and the next bridge stage:
  - tighten promotion so nonzero greedy delivery is mandatory
  - do not allow sampled-only success to count as meaningful progress
- if needed, use stage repetition more aggressively for these stages

3. Reduce unnecessary ambiguity during the isolated homing lesson
- if necessary, simplify the earliest homing lesson in a stage-bounded way so the agent can actually learn a deterministic greedy return behavior
- examples:
  - smaller state-space for the homing-only lesson
  - fewer obstacles or none at first
  - shorter target-to-nest path
  - stage-specific control simplification while carrying

Important:
- do not script the solution
- do not hardcode a hand-authored controller
- do not globally change action semantics unless clearly justified

4. Keep the bridge back to the real task
- after the isolated homing lesson succeeds, the curriculum must still:
  - reintroduce normal pickup
  - reintroduce clutter
  - preserve the real return-to-nest task

--------------------------------------------------
PART 3 — TRAINER / EVALUATION REQUIREMENTS
--------------------------------------------------

Make the trainer more aligned with the real target:

1. Greedy-delivery-first promotion
- the key homing stages should prioritize:
  - greedy delivery
  - greedy conversion
  - only then sampled performance

2. Better stage-end decision logic
- later promotion should not happen simply because sampled behavior looks promising
- if greedy homing is still zero, the trainer should make that explicit

3. Best checkpoint selection should remain delivery-first
- do not let pickup-rich / delivery-zero checkpoints become the “best” checkpoints

--------------------------------------------------
PART 4 — CURRICULUM REQUIREMENTS
--------------------------------------------------

Refine the curriculum around a more explicit progression:

1. Search/pickup lesson
2. Dedicated homing-only or near-homing lesson
3. Bridge lesson with normal pickup + mild clutter
4. Obstacle-return lesson
5. Then swarm scaling

This progression should be real in code/config, not just documentation.

--------------------------------------------------
PART 5 — METRICS / LOGGING REQUIREMENTS
--------------------------------------------------

Add enough logging to show whether the new homing lesson is actually working.

At minimum, make it easy to inspect:

- greedy delivery in the isolated homing stage
- greedy delivery in the next bridge stage
- sampled vs greedy delivery gap
- carrying-phase stall / low-progress metrics
- whether promotion happened because greedy delivery criteria were met

Keep runtime output lightweight.

--------------------------------------------------
PART 6 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update:

1. `README.md`
2. `docs/TRAINING_MAPPO.md`
3. `docs/QandA.md`
4. `docs/PROJECT_LOG.md`

The docs should explain:

- why the isolated homing lesson was added
- how it differs from the previous guaranteed-homing stage
- what command to run for a focused stage-1 verification

--------------------------------------------------
PART 7 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short stage-1 smoke test focused on the homing stages
3. show the exact commands used
4. summarize:
- whether greedy delivery improved at all
- whether the dedicated homing lesson produced clearer greedy behavior
- whether the bridge stage still fails or improves

--------------------------------------------------
PART 8 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of what was changed
2. what new homing-specific stage or lesson was added
3. what trainer/promotion logic changed
4. exact verification commands run
5. whether greedy delivery improved in the smoke test
6. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The current bottleneck is no longer:

- “can the policy ever pick up?”

It is:

- “can the policy learn a deterministic return-to-nest policy that survives greedy evaluation?”

So this step must focus on a stronger structural solution for that problem,
not just more small reward nudges.
