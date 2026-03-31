You are working in an existing multi-agent swarm reinforcement learning codebase.

Task:
Implement targeted training and configuration improvements that directly address the current weaknesses in the MAPPO trail-learning and delivery-learning setup, especially for the goals described in prompt 25 (trail formation) and prompt 26 (target delivery and carrying behavior).

Goal:
I want the current MAPPO path to train more effectively toward:

- reliable `pickup -> return -> deliver`
- meaningful return-path pheromone deposition
- later trail reuse by other agents
- less wasted budget in the curriculum

The focus is not to invent a new RL algorithm. The focus is to fix the current training setup so the existing recurrent MAPPO path has a much better chance of learning the intended behavior.

Important context:

- The recurrent MAPPO path already exists.
- The current system already has carrying state, nest delivery, pheromone deposition, and repeated-use food sources.
- The goal is to improve the training design and configuration, not to rewrite the simulator from scratch.
- Keep decentralized execution intact.

--------------------------------------------------
PART 1 — HIGHEST-PRIORITY ISSUE
--------------------------------------------------

First, fix the curriculum step-allocation problem in the current code.

The current curriculum implementation does not appear to allocate stage budgets correctly:

- `_split_stage_steps(...)` is called with more weights than are meaningfully consumed
- several early stages reuse the same stage-step bucket
- later stages likely receive less budget than intended

Required outcome:

1. the curriculum step-allocation logic must be internally consistent
2. each actual curriculum stage must receive an explicitly intended budget
3. the code should be easy to inspect and reason about
4. the final harder stages should not be starved accidentally

Do not just patch the symptom.
Make the stage-budget logic clear and correct.

--------------------------------------------------
PART 2 — DELIVERY-FOCUSED REWARD IMPROVEMENTS
--------------------------------------------------

Adjust reward design so delivery is more strongly preferred over pickup-only behavior.

Current design direction is good, but it should be made stronger for training.

Required design goals:

1. successful nest delivery must remain clearly more important than pickup
2. ending an episode while still carrying food should be more clearly bad
3. the agent should have less incentive to treat pickup as “good enough”

Implementation guidance:

- keep pickup meaningful
- keep delivery dominant
- strengthen the undelivered-food penalty relative to the current value if appropriate

Do not create giant unstable penalties.
Keep the shaping disciplined and clearly documented.

--------------------------------------------------
PART 3 — EXPLORATION-REWARD TUNING
--------------------------------------------------

Reduce the chance that generic exploration reward competes with the true objective in later stages.

Current issue:

- small exploration bonuses are useful early
- but later they can still reward wandering when the real objective should be:
  - pick up
  - return
  - deliver
  - reinforce useful routes

Required outcome:

1. exploration bonuses should help early learning
2. they should not dominate later delivery/trail learning
3. if stage-wise reward tuning is the cleanest solution, implement it in a disciplined way

Preferred direction:

- weaker exploration incentive in later curriculum stages than in the very earliest ones

--------------------------------------------------
PART 4 — CURRICULUM SHAPE IMPROVEMENTS
--------------------------------------------------

Refine the curriculum so it teaches the right subskills in the right order.

Current issue:

- the early tiny stages are easy, but the later jump into multi-target / respawned behavior may be too abrupt
- the system should learn one complete delivery loop before being asked to manage richer multi-source behavior

Required improvement:

Add or adjust stage definitions so that there is a cleaner “single-source with obstacles, full pickup-return-delivery loop” stage before the harder multi-source respawned stages dominate.

Design intent:

1. first learn to find food
2. then learn to carry it home
3. then learn to repeat that in obstacle-rich single-agent settings
4. then scale to small swarm
5. then scale to full swarm
6. only after that should richer trail exploitation across multiple productive sources become the main challenge

Do not overcomplicate the curriculum.
Prefer a small, defensible change over a large zoo of stages.

--------------------------------------------------
PART 5 — ACTION-REPEAT / CONTROL-TUNING REQUIREMENTS
--------------------------------------------------

Reassess action-repeat for delivery learning.

Current issue:

- `action_repeat_steps = 2` can help with jitter
- but it can also make precise nest-entry, pickup alignment, and obstacle escape harder

Required outcome:

1. evaluate whether one global action-repeat setting is hurting early delivery learning
2. if justified, add clean support for stage-wise or easier-stage control settings
3. keep the change small and understandable

Preferred direction:

- earlier delivery-learning stages may use more responsive control
- later swarm stages may still use stronger smoothing

If you decide the current single setting is already best, document why.

--------------------------------------------------
PART 6 — TRAINING-BUDGET / DEFAULT COMMAND REQUIREMENTS
--------------------------------------------------

Improve the repo’s default guidance so users are less likely to undertrain the model.

Current problem:

- short runs such as `30k`, `180k`, or `200k` total steps can complete successfully
- but they may still be far from sufficient for the current full curriculum and hard final stage

Required outcomes:

1. update the recommended full training command to reflect a more realistic budget
2. document why the shorter command is still likely undertrained
3. keep the recommended command practical, not absurdly large

Preferred direction:

- several hundred thousand total steps
- enough budget that the late full-swarm stages are not only brief fine-tuning

--------------------------------------------------
PART 7 — DOCUMENTATION REQUIREMENTS
--------------------------------------------------

Update the docs as part of the implementation.

Required updates:

1. update `README.md`
2. update `docs/TRAINING_MAPPO.md`
3. update `docs/CONFIG_REFERENCE.md` if config defaults or meanings change
4. update `docs/QandA.md` where user-facing questions are naturally answered
5. update `docs/PROJECT_LOG.md`

Documentation must explain:

1. what curriculum-budget bug was fixed
2. what reward changes were made and why
3. what curriculum-stage changes were made and why
4. whether action-repeat handling changed
5. what the new recommended training command is
6. why that command is more realistic for the current setup

--------------------------------------------------
PART 8 — VERIFICATION REQUIREMENTS
--------------------------------------------------

After implementation:

1. run syntax/import checks on modified files
2. run at least one short smoke test to confirm the new curriculum path still runs
3. show the exact commands used
4. explain any remaining limitations

Verification should demonstrate that:

- curriculum stage budgets are now being assigned coherently
- the training path still runs
- the docs now reflect the new recommendations

--------------------------------------------------
PART 9 — DELIVERABLES
--------------------------------------------------

After implementation, provide:

1. concise summary of the fixes made
2. what was wrong with the old curriculum-budget logic
3. what reward/config changes were made for delivery/trail learning
4. what changed in the curriculum stages
5. whether action-repeat handling changed
6. the updated recommended training command
7. exact verification commands run
8. remaining limitations

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Prioritize:

1. fixing the curriculum-budget bug
2. improving delivery-over-pickup learning pressure
3. making the early-to-late curriculum progression more learnable
4. improving the recommended training defaults and docs

Do not solve this by inventing a new RL algorithm.
Do not solve this by broadly rewriting unrelated parts of the simulator.
