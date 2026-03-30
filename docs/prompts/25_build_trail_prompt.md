You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement the training-side changes needed to make the swarm learn the full stigmergic loop described in [docs/trail.md](../trail.md):

- explore to find a target
- pick up or interact with the target
- return to the nest
- deposit pheromone on the return route
- let later agents exploit the resulting trail

Goal:
I want the system to learn real trail formation rather than only “reach the target.” The desired outcome is that the first successful discoveries are expensive and exploration-heavy, but repeated successful returns gradually create a pheromone route that makes later target acquisition more efficient.

Important context:

- The environment already supports pheromone, target pickup, nest delivery, and randomized resets.
- The recurrent MAPPO curriculum path already exists.
- The environment curriculum from prompt 24 already makes the stages progressively harder.
- Do not rewrite the simulator from scratch.
- Do not invent a novel RL algorithm.
- Focus on reward design, training logic, curriculum alignment, metrics, and evaluation so the intended trail behavior can actually emerge.

--------------------------------------------------
PART 1 — TARGET BEHAVIOR
--------------------------------------------------

The desired behavior is the full loop:

1. agents explore to discover a target
2. an agent reaches and picks up the target
3. the agent returns to the nest
4. the agent deposits pheromone on the successful return path
5. later agents use that pheromone trail to reach the target more efficiently

The system should not stop at:

- “agent touched target”

It should learn:

- “successful discovery produces a reusable route”

This should support the project’s unknown-environment / disaster-zone story:

- no trusted predefined map
- the first route is expensive to find
- later routes become easier because successful agents wrote route hints into the environment

--------------------------------------------------
PART 2 — CORE TRAINING PRINCIPLE
--------------------------------------------------

Implement the training design so that it teaches:

- find target
- pick up target
- return to nest
- deposit meaningful pheromone
- exploit useful pheromone without eliminating exploration

The system should explicitly support the RL transition from:

- exploration-heavy early behavior

to:

- exploitation-heavy later behavior

But in this project, exploitation is not only inside policy weights. It also happens through:

- shared environmental memory via pheromone

Preserve that interpretation in the implementation and docs.

--------------------------------------------------
PART 3 — REWARD / INCENTIVE REQUIREMENTS
--------------------------------------------------

Implement or refine reward design so the full loop is reinforced correctly.

Required reward priorities:

1. Target discovery / local detection
   - small shaping signal only
   - enough to encourage search
   - not so large that “seeing target” becomes the main objective

2. Pickup
   - meaningful reward
   - enough to teach that the target matters
   - but smaller than successful return / delivery

3. Return to nest / delivery
   - strongest task reward
   - this is the key signal for route completion
   - stronger than pickup

4. Pheromone following
   - small supportive reward only
   - enough to help agents exploit useful trails
   - not so strong that agents just chase pheromone blindly

5. Pheromone deposit
   - neutral or slightly costly per deposit
   - enough to discourage meaningless spam
   - not so costly that successful return trails are suppressed

Required design intent:

- pickup teaches “this object matters”
- delivery teaches “the full route matters”
- pheromone should support the route, not replace the task objective

If the current defaults do not respect that ordering, adjust them.

--------------------------------------------------
PART 4 — PHEROMONE BEHAVIOR REQUIREMENTS
--------------------------------------------------

Pheromone should represent successful return-route information, not arbitrary wandering.

Implement or tighten the logic so that pheromone deposition is tied to meaningful behavior.

Preferred behavior:

1. pheromone is deposited primarily or only when the agent is carrying the target
2. deposition is associated with return-to-nest behavior
3. meaningless exploration-time deposition should be minimized or removed

The trail should mean:

- “this route helped produce a successful return”

not:

- “an agent happened to move here”

If the repo already has partial gating for pheromone deposition, strengthen and clarify it rather than duplicating overlapping mechanisms.

--------------------------------------------------
PART 5 — CURRICULUM ALIGNMENT REQUIREMENTS
--------------------------------------------------

Use the existing curriculum path, but align it with the trail objective rather than only generic progression.

The training design should teach the full behavior progressively:

Stage family 1:
- teach target seeking and pickup in easy single-agent worlds

Stage family 2:
- teach target finding plus return behavior in more structured single-agent and small-swarm worlds

Stage family 3:
- teach multi-agent reuse of successful routes
- make pheromone exploitation increasingly valuable in larger harder worlds

The important training logic is:

1. first the agent learns that targets matter
2. then it learns that returning to nest matters
3. then it learns that successful return paths can help future agents

Do not assume trail formation will emerge if all of that is trained at once with no structure.

If needed, make stage-specific reward emphasis or stage-specific pheromone settings configurable, but keep the implementation disciplined and small.

--------------------------------------------------
PART 6 — EXPLORATION / EXPLOITATION REQUIREMENTS
--------------------------------------------------

Preserve enough exploration for trail discovery, while allowing later trail exploitation.

Required design goals:

1. Early training must still allow discovery
   - avoid making pheromone following dominate too early

2. Later training should allow useful trail reuse
   - agents should benefit from established pheromone structure

3. Some residual exploration should remain
   - otherwise the swarm may over-commit to an early suboptimal route
   - this matters in randomized environments and obstacle-rich worlds

Implement this using the repo’s existing algorithm and training structure where possible.
Do not replace the system with a fundamentally different exploration framework unless there is a strong, minimal reason.

--------------------------------------------------
PART 7 — METRICS / EVALUATION REQUIREMENTS
--------------------------------------------------

Do not judge success only by raw reward.

Add or improve metrics that can demonstrate whether a real trail is forming and helping.

Track at least:

1. food/target picked up
2. food/target delivered to nest
3. pheromone deposit count or rate
4. pheromone-follow usage or related signal
5. episode length
6. exploration coverage
7. collision count/rate

Also add metrics or summaries that help answer:

1. How long does first discovery take?
2. Do later discoveries happen faster than earlier ones?
3. Does route efficiency improve after successful returns?
4. Does pheromone use increase in a meaningful way after initial discovery?
5. Does the swarm become less dependent on random search over time?

If the environment/logging cannot measure all of these directly, implement the strongest practical subset and document the limitation clearly.

--------------------------------------------------
PART 8 — BENCHMARK / ABLATION REQUIREMENTS
--------------------------------------------------

Add enough experiment structure to show that trail formation is real and not just storytelling.

At minimum, support comparison between:

1. pheromone enabled
2. pheromone disabled

And, if practical:

3. freer deposition vs return-gated deposition
4. easier early curriculum vs harder direct training

The goal is to show whether the intended trail mechanism actually improves:

- search efficiency
- delivery efficiency
- reuse of successful routes

Do not overbuild a benchmark zoo.
Add a small, credible evaluation path that directly tests the trail hypothesis.

--------------------------------------------------
PART 9 — REPO / ARCHITECTURE CONSTRAINTS
--------------------------------------------------

Respect the current repo architecture.

Requirements:

1. Do not break the existing DQN baseline path
2. Keep the recurrent MAPPO path runnable
3. Preserve decentralized execution at inference time
4. Do not make firmware depend on training internals
5. Do not rewrite the simulator unnecessarily
6. Keep action semantics stable unless a change is strongly justified

Preferred implementation style:

- use existing config plumbing where possible
- keep trail-related training controls explicit and documented
- keep stage-specific behavior understandable from logs and metadata

--------------------------------------------------
PART 10 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update documentation as part of the implementation.

Required doc updates:

1. explain how the repo now trains for:
   - find -> pick up -> return -> deposit -> exploit

2. explain why delivery reward is stronger than pickup reward

3. explain how pheromone deposition is tied to meaningful return behavior

4. explain how the curriculum supports trail formation progressively

5. explain how to evaluate whether a real trail is forming

6. explain the role of exploration early and exploitation later

Also update:

- `docs/PROJECT_LOG.md`
- `docs/QandA.md` when user-facing codebase questions are naturally answered by the work

--------------------------------------------------
PART 11 — IMPLEMENTATION PRIORITIES
--------------------------------------------------

Implement in this order:

Priority 1:
- make the reward / incentive structure match the full loop

Priority 2:
- tie pheromone deposition to meaningful successful behavior

Priority 3:
- align the curriculum with target -> return -> trail-use learning

Priority 4:
- improve metrics and evaluation so trail formation can be measured

Priority 5:
- document the design clearly

--------------------------------------------------
PART 12 — WHAT MUST NOT HAPPEN
--------------------------------------------------

Do not allow the implementation to collapse into:

1. “touching the target is enough”
2. dense pheromone spam during aimless exploration
3. pheromone-following so strong that agents stop exploring
4. a documentation-only change with no real training impact
5. a broad simulator rewrite unrelated to the trail objective

--------------------------------------------------
PART 13 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short training smoke test
3. run at least one short comparison that helps show trail-related metrics are being recorded
4. show the exact commands used
5. explain any limitations if longer training is needed to observe the full effect

Verification should make it clear that:

- the implementation now explicitly trains the full loop
- trail-related metrics exist
- pheromone behavior is meaningfully connected to successful return behavior

--------------------------------------------------
PART 14 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the training changes made
2. how reward structure now supports the full loop
3. how pheromone deposition/following is handled
4. how the curriculum supports trail formation
5. what new metrics/evaluation support was added
6. exact verification commands run
7. remaining limitations or deferred work

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The main objective is not merely:

- reach target

It is:

- discover target
- return successfully
- leave a reusable trail
- let later agents exploit that trail

Treat pheromone as shared environmental memory and make the training path reflect that clearly.
