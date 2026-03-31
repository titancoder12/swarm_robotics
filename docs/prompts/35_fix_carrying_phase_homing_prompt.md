You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement the next training/system changes needed to fix the remaining MAPPO failure mode after prompt 34:

- agents can often discover and pick up targets
- but they still become pickup-rich and delivery-zero
- the main remaining problem is the **carrying phase**
- after pickup, agents do not reliably switch into a strong return-to-nest policy under clutter

Do not solve this by weakening the task definition or by making delivery optional.
The goal is to make the learned behavior explicitly become:

1. discover target
2. pick up target
3. switch into homing mode
4. return to nest efficiently
5. complete delivery
6. only then move on to the next target/trail task

--------------------------------------------------
PART 1 — MAIN OBJECTIVE
--------------------------------------------------

Implement the next iteration of training and environment shaping so that:

- once an agent is carrying food, the policy is strongly biased toward nest return
- pickup without delivery is no longer behaviorally attractive
- carrying-phase wandering is actively discouraged
- carrying-phase obstacle traps and dithering are reduced
- greedy evaluation reflects true carrying-to-delivery competence

This prompt should build on the work from prompts 30 through 34, not replace it.

--------------------------------------------------
PART 2 — REQUIRED TRAINING / REWARD CHANGES
--------------------------------------------------

Implement carrying-phase-focused improvements.

Required directions:

1. Stronger carrying-only nest-progress shaping
- when an agent is carrying food, progress toward the nest should dominate the shaping signal
- this should be materially stronger than the current weak/ambiguous pressure
- keep the shaping disciplined and local; do not introduce reward hacks that trivially solve the task

2. Suppress exploration reward while carrying
- while carrying food, exploration-style bonuses such as new-cell reward should be heavily reduced or disabled
- the agent should stop being rewarded for roaming once it has already secured a target

3. Penalize carrying without progress
- if an agent is carrying food but:
  - makes little displacement, or
  - makes little/no progress to the nest, or
  - remains in a repeated local area for too long
  then apply a small but meaningful penalty
- this should target dithering and pickup-without-delivery loops

4. Carrying-phase anti-stuck / anti-dithering support
- add clean carrying-specific shaping or metrics for:
  - low-progress while carrying
  - repeated stall while carrying
  - obstacle-adjacent delivery failure patterns
- do not implement brittle scripted behavior; prefer shaping and curriculum support

5. Delivery-first checkpoint scoring
- later-stage greedy checkpoint scoring should continue to prioritize delivery conversion
- if needed, strengthen that further so pickup-heavy / zero-delivery checkpoints are never preferred

--------------------------------------------------
PART 3 — CURRICULUM REQUIREMENTS
--------------------------------------------------

Refine the curriculum to explicitly teach the carrying phase better.

Required behavior:

1. Add or refine a stage that isolates post-pickup homing
- after pickup, the agent should spend most of the stage learning return-to-nest completion
- keep this stage easier than the true obstacle-delivery stages

2. Keep clutter reintroduction gradual
- do not jump directly from guaranteed homing to hard obstacle return
- use a clean bridge stage if needed

3. Promotion must remain delivery-sensitive
- do not allow advancement based only on pickup
- promotion should continue to require actual delivery performance and delivery conversion

4. Carrying-phase metrics should be visible per stage
- make it easy to see whether a stage failed because:
  - pickup was weak
  - delivery conversion was weak
  - carrying-phase stalling was high

--------------------------------------------------
PART 4 — METRICS / LOGGING REQUIREMENTS
--------------------------------------------------

Add or improve metrics that make the carrying-phase failure explicit.

Track at least, where practical:

- delivery conversion
- time/steps spent carrying before delivery
- carrying-phase no-progress or stall count
- carrying-phase low-displacement fraction
- carrying-phase nest-progress efficiency if practical
- pickup-to-delivery latency

Runtime prints should stay lightweight.
The goal is insight, not noisy per-step spam.

--------------------------------------------------
PART 5 — WHAT NOT TO DO
--------------------------------------------------

Do not:

- make delivery optional
- remove the need to return to the nest
- add brittle hardcoded scripted return behavior
- overfit only to one toy layout
- revert the hard total-step cap from prompt 34
- revert delivery-sensitive promotion / checkpoint scoring

--------------------------------------------------
PART 6 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update documentation to reflect the new carrying-phase training design.

Required doc updates:

1. `README.md`
- explain the carrying-phase / homing emphasis at a high level if user-facing behavior changed meaningfully

2. `docs/TRAINING_MAPPO.md`
- explain how carrying-phase shaping and curriculum now work

3. `docs/QandA.md`
- add concise entries for user-facing questions naturally answered by this work

4. `docs/PROJECT_LOG.md`
- log the implementation and verification work

Also review nearby docs touched recently and fix stale wording if needed.

--------------------------------------------------
PART 7 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short stage-1 smoke test focused on the carrying / homing stages
3. show the exact commands used
4. summarize whether:
- sampled carrying-to-delivery improved
- greedy carrying-to-delivery improved
- the carrying-phase metrics became more informative

--------------------------------------------------
PART 8 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the carrying-phase changes
2. what reward/curriculum changes were made
3. what new carrying-phase metrics were added
4. exact verification commands run
5. whether greedy delivery improved in the smoke test
6. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main remaining failure is:

- pickup happens
- delivery still collapses

So this prompt should focus on:

- stronger post-pickup homing pressure
- less reward for wandering while carrying
- better visibility into carrying-phase failure
- keeping delivery, not pickup, as the true success condition
