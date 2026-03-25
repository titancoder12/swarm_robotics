# Make Food Discovery Learnable
You are working in an existing multi-agent swarm reinforcement learning codebase for a stigmergic foraging environment.

Task:
Improve the training setup so agents achieve their first successful food interaction much earlier and more reliably.

Context:
The current training signals suggest the policy is learning movement and exploration, but not yet completing the first meaningful foraging event:
- agents are moving
- coverage is improving
- reward is becoming less negative
- but `food_retrieved` / successful food interaction remains at or near zero

This usually means the problem is still too hard as an end-to-end learning task, not simply that the reward is slightly weak.

Your job is to modify the codebase so the system can bootstrap successful food discovery earlier, while staying consistent with the project’s decentralized and sim-to-real-aware design.

Important constraint:
Do NOT add privileged global food information to the policy observation beyond what the current code already intentionally exposes.
You may:
- change training environment difficulty
- add curriculum stages
- add environment-side debug metrics
- adjust local sensing radii and interaction thresholds
- add training-only reward shaping

But do NOT silently turn the task into one that assumes unrealistic robot knowledge at runtime.

--------------------------------------------------
PART 1 — Diagnose the current failure chain explicitly
--------------------------------------------------

Before changing behavior, inspect the existing code paths around:
- food detection / food presence
- nearest-target logic
- pickup interaction
- carrying-food state
- nest delivery
- reward breakdown logging

Relevant existing areas likely include:
- `env/config.py`
- `env/swarm_env.py`
- `train/independent_dqn_pytorch.py`
- `train/evaluate.py`
- any logging / plotting helpers already used by training

The current code already has some shaping/configuration support such as:
- `food_detection_radius`
- `reward_food_approach`
- `reward_food_detected`
- `reward_new_cell`
- `pheromone_requires_food`

Build on those rather than replacing them.

--------------------------------------------------
PART 2 — Add food-debug metrics to find the real bottleneck
--------------------------------------------------

Right now, `food=0` is too coarse to diagnose.

Add metrics so we can tell where the chain is breaking.

Add per-step / per-episode metrics such as:
- `food_detected_events`
  - count how often food enters local detection radius
- `min_distance_to_food`
  - minimum nearest-food distance reached in the episode
- `pickup_events`
  - successful food pickup count
- `carrying_food_steps`
  - number of steps any agent spends carrying food
- `delivery_events`
  - successful food deliveries (if not already clearly logged)
- optional:
  - `pickup_attempt_zone_entries`
  - number of times an agent enters pickup radius even if pickup fails

Requirements:
- wire these into the environment `info` / episode metrics path
- include them in the existing reward/episode logging style
- keep names clear and flat, consistent with existing CSV fields
- do not add them to the observation unless already present

Goal:
Make it possible to distinguish:
- no detection at all
- detection but no pickup
- pickup but no return
- return but no delivery

--------------------------------------------------
PART 3 — Make the task easier through a training curriculum
--------------------------------------------------

Add an “easy mode” or curriculum-friendly configuration path for training.

This should make the task dramatically easier at first, not just slightly easier.

For the easy phase, consider:
- fewer obstacles
- fewer agents by default for debugging
- more food items
- food placed closer to the nest / spawn area
- larger food detection radius
- larger pickup interaction radius
- shorter required return path

Implementation guidance:
- use config-driven parameters
- do not hardcode one-off logic in the main training loop
- prefer either:
  - new config fields, or
  - a small curriculum preset / helper, or
  - a staged difficulty schedule that changes config across training phases

Strong recommendation:
Support at least a simple debug/easy setup like:
- 1 agent
- 3 to 5 food items
- no or few obstacles
- larger detection radius
- larger pickup radius

This is not the final environment.
It is a bootstrap phase to prove the foraging loop is learnable.

--------------------------------------------------
PART 4 — Verify and, if needed, relax food interaction mechanics
--------------------------------------------------

Carefully inspect the pickup and delivery mechanics.

Possible failure modes:
- target radius too small
- pickup overlap too strict
- movement step size jumps past food
- collision logic interferes near food
- carrying-food / delivery state transitions not happening when expected

If needed:
- add a configurable pickup radius or pickup margin
- make food interaction slightly more forgiving in the easy/debug curriculum
- keep default/full environment behavior intact if possible

Do NOT rewrite the environment architecture.
Keep this as a minimal, config-driven improvement.

--------------------------------------------------
PART 5 — Strengthen learning support around first success
--------------------------------------------------

The current code already includes shaping and exploration reward support.
Build on that conservatively.

Focus on getting the first success early.

You may:
- slightly strengthen food detection and food approach shaping
- ensure new-cell coverage reward is meaningful but still small
- slow exploration decay in recommended commands or training presets

But:
- do NOT overpower pickup / delivery rewards
- do NOT make the agent farm shaping forever
- do NOT introduce unrealistic observation features

If useful, add a curriculum-oriented preset or helper comments recommending:
- longer epsilon decay
- higher warmup steps
- 1-agent debugging before multi-agent scaling

--------------------------------------------------
PART 6 — Preserve stigmergy and sim-to-real alignment
--------------------------------------------------

Keep these principles explicit:
- `pheromone_requires_food=True` is realistic and should stay consistent
- coverage reward is training-only and should remain environment-side
- local food cues are acceptable; privileged omniscient observation is not
- the task should still ultimately require pickup and delivery, not just wandering near food

Add short code comments where appropriate explaining:
- which signals are training-only shaping
- which design choices are meant to preserve physical plausibility

--------------------------------------------------
PART 7 — Logging and plotting
--------------------------------------------------

If the training pipeline already writes:
- `episode_metrics.csv`
- `eval_metrics.csv`
- training graphs

then extend the logging minimally so the new food-debug metrics appear there too where appropriate.

If plotting is easy to extend cleanly, include the new metrics in plots.
If not, at minimum ensure they are present in CSV logs.

--------------------------------------------------
PART 8 — Deliverables
--------------------------------------------------

After implementation, provide:

1. A short summary of what files were modified
2. The exact config fields added or changed
3. A plain-English explanation of:
   - what made the environment easier
   - what new food-debug metrics were added
   - whether pickup/delivery mechanics changed
4. Example commands for:
   - easy/debug training
   - normal training
   - any curriculum mode you add
5. A short note on sim-to-real caveats

--------------------------------------------------
Implementation style requirements
--------------------------------------------------

- Modify the existing code directly
- Keep architecture changes minimal
- Follow existing naming/style conventions
- Keep the repo runnable
- Do not change observation shape unless absolutely necessary
- Prefer config fields and small helpers over ad hoc hacks
- Implement real code, not pseudocode

--------------------------------------------------
What “success” looks like
--------------------------------------------------

After the changes, an early/debug run should show:
- nonzero food detection events
- decreasing minimum food distance
- nonzero pickup events
- eventually nonzero carrying-food steps and deliveries

That is the goal.
First prove the loop works.
Then increase difficulty again.
