You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement the next iteration of changes needed to produce **greedy delivery**, not just sampled delivery.

Current situation:

- prompts 30-35 improved pickup behavior, carrying-phase shaping, and trainer discipline
- sampled pickup and sampled delivery can now occur in the easier return stages
- but greedy evaluation still remains at or near zero delivery in the critical homing / bridge / obstacle-return stages

The problem is no longer just “can the policy ever deliver?”
The problem is:

- the policy still does not form a stable greedy post-pickup homing behavior
- it can still rely on stochastic behavior during training without turning that into reliable deterministic return-to-nest execution

The goal of this prompt is to make the early carrying-to-delivery path become:

1. discover
2. pick up
3. switch into a stable homing mode
4. return to nest under greedy evaluation
5. only then move on to harder clutter and swarm stages

--------------------------------------------------
PART 1 — PRIMARY OBJECTIVE
--------------------------------------------------

Implement changes that directly target the remaining sampled-to-greedy gap in the carrying phase.

The main desired outcome is:

- in the early return-focused curriculum stages, greedy pickup and greedy delivery should become nonzero and meaningfully stable

Do not solve this by weakening the task or by hiding the problem behind checkpoint selection alone.
The learned policy itself must become more reliably greedy-delivery-capable.

--------------------------------------------------
PART 2 — REQUIRED TRAINING CHANGES
--------------------------------------------------

Implement the following directions.

1. Make the early homing stages more deterministic
- reduce entropy pressure more aggressively in the return-focused stages once the agent has learned pickup
- the goal is to stop preserving stochastic pickup/delivery behavior that never survives into greedy eval
- be disciplined; keep exploration where it is still needed, but lower it materially in the homing-critical stages

2. Strengthen greedy-delivery gating
- advancement through the key homing / bridge / obstacle-return stages must depend more strongly on greedy delivery, not sampled delivery
- if needed, tighten stage targets so nonzero greedy delivery is required before promotion
- do not let pickup-only or sampled-only success masquerade as learning progress

3. Reward sustained nest progress while carrying
- extend the carrying-phase shaping beyond one-step signed progress
- add a clean short-horizon or persistent-progress signal that rewards sustained movement toward the nest while carrying
- the agent should be pushed toward forming a coherent return trajectory, not just isolated local progress steps

4. Preserve hard global budget discipline
- do not undo prompt 34’s hard total-step cap
- do not undo delivery-conversion-aware promotion or later-stage delivery penalties

--------------------------------------------------
PART 3 — CARRYING-PHASE CONTROL / BEHAVIOR REQUIREMENTS
--------------------------------------------------

Implement a cleaner path for discovering a repeatable carrying-phase control policy.

Required directions:

1. Simplify the first carrying-return control problem if needed
- in the earliest homing-focused stages only, reduce control ambiguity while carrying if that materially helps discover stable greedy homing
- examples could include:
  - stage-specific action persistence changes
  - stage-specific carrying control simplification
  - other disciplined changes that make carrying-phase control less noisy

Important:
- do not hardcode a scripted homing controller
- do not break the existing action semantics globally
- if you simplify control, do it in a minimal and stage-bounded way

2. Keep full complexity for later stages
- once greedy homing is established, later stages should still restore the intended difficulty and behavior

--------------------------------------------------
PART 4 — CURRICULUM REQUIREMENTS
--------------------------------------------------

Adjust the curriculum so the policy does not leave the homing lesson too early.

Required behavior:

1. Keep the guaranteed-homing stage long enough and important enough
- if stage budget or weighting should change, change it
- if a repeat or stricter promotion target is needed, implement it

2. Ensure the bridge stage is still a bridge
- it should remain harder than guaranteed homing, but not so hard that greedy delivery collapses immediately

3. Only promote after real greedy progress
- if greedy delivery remains zero, the trainer should not treat the stage as behaviorally solved

--------------------------------------------------
PART 5 — METRICS / LOGGING REQUIREMENTS
--------------------------------------------------

Keep runtime prints lightweight, but make greedy-delivery progress easier to inspect.

Track or surface, where practical:

- greedy delivery in the homing stages
- sampled vs greedy delivery gap
- carrying sustained-progress metrics
- whether a stage passed due to greedy delivery or merely exhausted retries

If you add new metrics, wire them into:

- episode/eval CSVs where appropriate
- runtime stage/eval summaries
- checkpoint metadata if useful

--------------------------------------------------
PART 6 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update documentation so it reflects the new greedy-homing emphasis.

Required updates:

1. `README.md`
2. `docs/TRAINING_MAPPO.md`
3. `docs/QandA.md`
4. `docs/PROJECT_LOG.md`

Also fix any nearby stale wording if the implementation changes the effective training path.

--------------------------------------------------
PART 7 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short stage-1 smoke test focused on the homing stages
3. show the exact commands used
4. summarize:
- whether sampled delivery changed
- whether greedy delivery changed
- whether the homing stages became more deterministic

--------------------------------------------------
PART 8 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of what was changed
2. how the homing stages were made more deterministic
3. what was changed in promotion / curriculum behavior
4. exact verification commands run
5. whether greedy delivery improved in the smoke test
6. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main remaining failure is:

- sampled homing can happen
- greedy homing still does not reliably happen

So this prompt should specifically focus on:

- converting sampled delivery into greedy delivery
- making the carrying-phase homing policy more stable and deterministic
- not just improving trainer diagnostics or stage bookkeeping
