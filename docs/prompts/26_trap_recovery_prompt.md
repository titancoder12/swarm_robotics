You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement agent-side trap-recovery improvements so agents can learn to escape local jams, obstacle pockets, and stuck clusters instead of repeatedly oscillating or remaining trapped near a target or obstacle.

Goal:
I want the system to address “stuck near a target / stuck in a cluster” from the agent’s perspective rather than by only making target placement easier. The agent should become better at recognizing non-progress, escaping local traps, avoiding crowded jams, and resuming the main task.

Important context:

- Do not treat this as only an environment-spawn problem.
- The agents are supposed to eventually reach the target even under awkward placement.
- The recurrent MAPPO path already exists and is the main training path.
- The environment already has obstacles, discrete control, action repeat, pheromone, target pickup, and nest delivery.
- Do not rewrite the simulator from scratch.

--------------------------------------------------
PART 1 — TARGET BEHAVIOR
--------------------------------------------------

Implement training-side and policy-relevant changes so agents can:

1. recognize that they are stuck or making no meaningful progress
2. break out of local wall/obstacle traps
3. avoid piling into already-stuck clusters of agents
4. recover and resume the main task afterward

The desired behavior is not:

- sit in one place turning or pushing uselessly
- reinforce a bad local loop
- blindly follow other agents or pheromone into a jam

It is:

- detect stagnation
- escape
- spread out when congested
- continue target / return behavior

--------------------------------------------------
PART 2 — CORE DESIGN PRINCIPLE
--------------------------------------------------

Treat trap recovery as a learnable part of the task.

The system should explicitly teach:

- “not making progress is bad”
- “escaping stagnation is good”
- “joining a dense stuck cluster is bad”
- “local escape logic should override blind repetition”

Do not rely only on easier spawn logic.
The training path should make the policy more robust in difficult local geometry.

--------------------------------------------------
PART 3 — REQUIRED AGENT-SIDE SIGNALS
--------------------------------------------------

Add or formalize the signals needed for the policy to detect being stuck.

At least some of the following should be exposed cleanly through observation, recurrent state usage, reward logic, or logging:

1. recent displacement magnitude
2. recent target-progress signal
3. recent nest-progress signal while carrying
4. recent collision count or collision streak
5. local crowding / nearby-agent density

You do not necessarily need to add every one directly to the observation vector if recurrence and training logic can cover the behavior cleanly, but the system must have enough information to distinguish:

- moving usefully
- moving in place
- repeatedly colliding
- being trapped in a crowded area

--------------------------------------------------
PART 4 — REWARD / INCENTIVE REQUIREMENTS
--------------------------------------------------

Implement reward shaping that teaches trap recovery without destabilizing the main task.

Required behavior:

1. Small stagnation penalty
   - penalize the agent when it remains locally stationary or nearly stationary for too long
   - only apply this when the task is unfinished
   - avoid making it so strong that agents jitter to dodge the penalty

2. Small escape reward
   - if the agent was recently stuck and then breaks out with meaningful displacement or renewed task progress, give a small reward
   - this teaches recovery, not just motion

3. Small crowding / congestion penalty
   - discourage agents from piling into dense local clusters, especially when progress is poor
   - this should help with multi-agent jams near the same target pocket

4. Preserve main-task priority
   - pickup and delivery must still matter more than trap-recovery shaping
   - do not let anti-stuck shaping replace the actual foraging objective

The intended ordering is:

- full task completion first
- recovery shaping second

--------------------------------------------------
PART 5 — ACTION / BEHAVIOR REQUIREMENTS
--------------------------------------------------

The policy must have a believable way to escape local traps.

Required design intent:

1. If the agent is repeatedly making no progress, it should become less willing to repeat the same failing local behavior.
2. Recovery behavior should favor escape-capable actions such as stronger turning, reversing, or backing out when appropriate.
3. Trap recovery should work in both single-agent and multi-agent settings.

Do not add a giant hand-coded behavior tree.
Prefer:

- reward-backed learning
- clean auxiliary signals
- minimal action-space or control adjustments only if they are strongly justified

If you conclude that the current action space is insufficient for reliable trap recovery, document the exact reason and make the smallest justified change.

--------------------------------------------------
PART 6 — RECURRENCE REQUIREMENTS
--------------------------------------------------

Use recurrence intentionally.

The recurrent MAPPO path should support the recovery behavior by allowing the policy to learn patterns like:

- “I have been seeing the same geometry repeatedly”
- “my recent actions are not changing my state enough”
- “I should switch to escape behavior”

Do not treat the GRU as incidental.
Make the implementation and docs reflect that recurrent state is useful for trap detection and recovery.

--------------------------------------------------
PART 7 — PHEROMONE INTERACTION REQUIREMENTS
--------------------------------------------------

Trap recovery must be able to override bad pheromone-following behavior.

Required behavior:

1. Agents should not blindly keep following pheromone into a deadlock.
2. Local non-progress signals should be able to beat weak pheromone attraction.
3. Pheromone-following reward should remain supportive, not dominant.

The system should avoid:

- self-reinforcing congestion where agents keep entering the same trap because the trail or nearby swarm activity makes the area look attractive

--------------------------------------------------
PART 8 — CURRICULUM REQUIREMENTS
--------------------------------------------------

Train trap recovery deliberately, not accidentally.

Extend or refine the curriculum so that later stages include situations such as:

1. targets near corners
2. targets near obstacle pockets
3. narrow approach corridors
4. local congestion near a goal
5. multi-agent crowding around the same reachable target

The goal is to teach:

- escape from local traps
- recovery under crowding
- not just generic navigation

Do not make every stage punishing from the start.
Add these difficulties in a controlled later-stage way.

--------------------------------------------------
PART 9 — METRICS / EVALUATION REQUIREMENTS
--------------------------------------------------

Do not judge success only by reward or delivery count.

Add or improve metrics that measure trap behavior directly.

Track at least a strong practical subset of:

1. fraction of steps with very low displacement
2. stuck-event count
3. average stuck duration
4. collision streaks or repeated-contact patterns
5. local crowding / congestion near targets
6. successful escape count
7. pickup and delivery after congestion events

Also make it possible to answer:

1. Are agents getting stuck less often?
2. Are stuck episodes shorter?
3. Are agents better at escaping crowded pockets?
4. Does performance improve after trap-recovery training is added?

--------------------------------------------------
PART 10 — BENCHMARK / ABLATION REQUIREMENTS
--------------------------------------------------

Add a small, credible comparison path.

At minimum, support comparison between:

1. baseline behavior without trap-recovery changes
2. trap-recovery-enhanced behavior

If practical, also compare:

3. easier placement only vs true agent-side recovery training
4. single-agent recovery vs multi-agent congestion recovery

The point is to show whether the improvement comes from better agent behavior rather than only easier environments.

--------------------------------------------------
PART 11 — REPO / ARCHITECTURE CONSTRAINTS
--------------------------------------------------

Respect the current architecture.

Requirements:

1. Do not break the custom DQN baseline path
2. Keep the recurrent MAPPO path runnable
3. Preserve decentralized execution at inference time
4. Do not make firmware depend on training internals
5. Do not rewrite the simulator broadly

Preferred implementation style:

- use existing config plumbing where possible
- keep trap-recovery controls explicit and documented
- keep the implementation understandable

--------------------------------------------------
PART 12 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update documentation as part of the implementation.

Required doc updates:

1. explain the trap-recovery objective
2. explain what “stuck” means in the system
3. explain the reward / signal changes used to teach recovery
4. explain how recurrence helps trap recovery
5. explain how trap scenarios are introduced in the curriculum
6. explain how to evaluate whether trap recovery actually improved
7. update `README.md` with copy-paste-ready commands for:
   - main training
   - short smoke-test training
   - demo playback
   - any trap-recovery evaluation/comparison run
8. review the existing docs and update stale statements so the docs reflect the current system

Also update:

- `README.md`
- `docs/PROJECT_LOG.md`
- `docs/QandA.md` when user-facing codebase questions are naturally answered by the work

--------------------------------------------------
PART 13 — IMPLEMENTATION PRIORITIES
--------------------------------------------------

Implement in this order:

Priority 1:
- give the system a clean way to detect stagnation / being stuck

Priority 2:
- add small reward shaping for stagnation, recovery, and congestion

Priority 3:
- make sure local recovery can override bad repeated behavior and blind trail-following

Priority 4:
- train on deliberate trap / crowding situations in later curriculum stages

Priority 5:
- add metrics and docs proving the behavior improved

--------------------------------------------------
PART 14 — WHAT MUST NOT HAPPEN
--------------------------------------------------

Do not let the implementation collapse into:

1. easier placement only, without real agent-side recovery learning
2. giant rewards that make agents jitter constantly
3. hand-coded escape logic so heavy that the policy stops being the main decision-maker
4. pheromone-following that still dominates local recovery signals
5. a broad simulator rewrite unrelated to trap recovery

--------------------------------------------------
PART 15 — VERIFICATION
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short smoke test
3. run at least one short trap-recovery-oriented scenario or comparison
4. show the exact commands used
5. explain any limitations if longer training is required to observe the full effect

Verification should make it clear that:

- the policy can detect or react to non-progress more intelligently
- trap-related metrics are being recorded
- recovery behavior is being trained rather than only assumed

--------------------------------------------------
PART 16 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the trap-recovery changes made
2. how stagnation / stuck state is represented or detected
3. how the reward structure teaches recovery
4. how recurrence is being used for recovery behavior
5. how the curriculum now trains trap scenarios
6. what new metrics/evaluation support was added
7. exact verification commands run
8. copy-paste-ready commands for training, smoke test, demo, and comparison
9. remaining limitations or deferred work

--------------------------------------------------
IMPORTANT
--------------------------------------------------

The goal is not just:

- make placement easier

It is:

- make the agents better at escaping traps, jams, and non-progress situations

Treat trap recovery as a learnable part of decentralized swarm intelligence.
