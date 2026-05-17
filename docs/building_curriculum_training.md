# Building Curriculum Training

This document explains how the MAPPO curriculum-training path was improved over a long sequence of iterations, starting from commit `83954933b517e17baf64681231ec1d3424165fab` and focusing especially on tasks 27 through 47. It is meant to be readable by someone who is comfortable with code, but not necessarily a reinforcement learning specialist.

The short version is:

- the project started with a recurrent MAPPO trainer that could sometimes show success during training, but often failed in greedy evaluation and demo
- we then improved the system step by step, treating each failure mode as a curriculum-learning problem
- the fixes were not one big rewrite; they were a sequence of focused changes to:
  - the curriculum
  - the trainer
  - the reward structure
  - the environment
  - the evaluation and checkpoint-selection logic

If you want to understand how curriculum learning works in this repository, the most important files are:

- [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py)
- [train/mappo_gru.py](../train/mappo_gru.py)
- [env/config.py](../env/config.py)
- [env/swarm_env.py](../env/swarm_env.py)
- [train/experiment_utils.py](../train/experiment_utils.py)

## What Curriculum Learning Means Here

In this repository, curriculum learning means that the model is not trained on the hardest problem first. Instead, it moves through a sequence of stages defined in [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py).

Each [`CurriculumStage`](../algorithms/mappo/curriculum.py) defines:

- how many agents are active
- how large the world is
- how many targets and obstacles exist
- how long an episode is
- what reward settings are used
- whether pheromone is enabled
- how exploratory or deterministic the policy should be

The trainer in [train/mappo_gru.py](../train/mappo_gru.py) then:

1. picks a stage from the curriculum
2. builds a stage-specific environment in [`_build_env(...)`](../train/mappo_gru.py)
3. collects rollouts
4. updates the MAPPO actor and critic
5. runs greedy evaluation
6. decides whether the stage is good enough to advance

That is the central curriculum loop.

## The Training Problem We Started With

The early version of this MAPPO path had a common RL failure pattern:

- the policy could sometimes do useful things during sampled training rollouts
- but greedy evaluation was near zero
- demos therefore looked frozen, timid, or obviously wrong

The best summary of the early diagnosis came right after task 27:

- eval stages from `stage1a` through `stage3b` showed:
  - `food_picked_up = 0`
  - `food_retrieved = 0`
  - `pheromone_deposit_events = 0`
- episode reward looked close to timeout cost, not successful task completion

In plain language, the model had not learned robust behavior. It had learned something that only occasionally looked useful under training-time stochasticity.

The rest of the work was a careful attempt to answer:

- why was the policy collapsing?
- why did pickup happen without delivery?
- why did behavior disappear in greedy demo?
- why did scaling from one agent to many agents break?

## The Core Idea Behind the Improvements

The improvements followed one simple rule:

> Do not ask the model to solve the whole task at once. Teach one missing subskill at a time, verify it in greedy evaluation, then move on.

That is why the history matters. The training pipeline got better because each task isolated one concrete failure mode:

- stage budgets were wrong
- the critic was resetting across stages
- sampled behavior did not survive into greedy behavior
- pickup worked but return-to-nest failed
- homing worked in easy cases but failed under clutter
- single-agent delivery worked but small-swarm scaling failed
- later swarm stages got trapped in nest-centered local optima

The result is a training system that is much easier to reason about and much closer to how curriculum learning is described in RL textbooks: start easy, verify the subskill, then scale the difficulty.

## Phase 1: Fix the Curriculum Budget and Reward Pressure

The first major improvement was task 27.

### What was wrong

The original curriculum budget allocation was internally inconsistent. Stage budgets were being split in a way that did not clearly map to the actual selected stages. That meant later, harder stages could be starved accidentally.

This mattered because curriculum learning only works if later lessons actually get enough practice.

### What changed

In [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py):

- `_split_stage_steps(...)` and `_assign_stage_steps(...)` now allocate budget cleanly
- each stage has an explicit `budget_weight`
- the selected curriculum slice is what gets budgeted

In [train/mappo_gru.py](../train/mappo_gru.py):

- stage-specific `action_repeat_steps` and `reward_new_cell` are applied in [`_build_env(...)`](../train/mappo_gru.py)

In [env/config.py](../env/config.py) and [train/experiment_utils.py](../train/experiment_utils.py):

- delivery-oriented defaults were strengthened:
  - `reward_pickup = 6.0`
  - `reward_nest_delivery = 30.0`
  - `reward_undelivered_food = -10.0`
  - `reward_new_cell = 0.01`
  - `reward_target = 0.0`

### Why it helped

This fixed two basic learning problems:

- the later stages finally got the budget they were supposed to get
- delivery became more important than pickup-only behavior

This was not flashy, but it was foundational. Curriculum learning fails quickly if either the stage budgets or the reward priorities are misaligned.

## Phase 2: Fix Structural MAPPO Instability

The second major improvement was task 28.

### What was wrong

The actor was being carried forward across stages, but the centralized critic often was not. Because the global state shape changed from stage to stage, the critic frequently had to restart.

That is a serious problem for PPO-style methods like MAPPO. The critic is the value estimator that stabilizes policy updates. If it keeps resetting, the policy can drift badly.

This logic lived in [train/mappo_gru.py](../train/mappo_gru.py). The trainer was effectively saying:

- keep the actor
- rebuild the critic
- hope the next stage still works

### What changed

Task 28 introduced a fixed padded critic-state path:

- [`_estimate_stage_state_dim(...)`](../train/mappo_gru.py)
- [`_critic_state_dim(...)`](../train/mappo_gru.py)
- [`_pad_state(...)`](../train/mappo_gru.py)

The trainer now computes one fixed critic input size for the selected curriculum and pads smaller states to match it.

This allowed:

- critic continuity across stages
- optimizer continuity across stages
- much less destabilization during handoff

Task 28 also made stage advancement and checkpointing smarter:

- stage-end greedy evaluation became mandatory
- promotion targets were introduced through [`StagePromotionTarget`](../train/mappo_gru.py) and [`_stage_promotion_target(...)`](../train/mappo_gru.py)
- greedy checkpoint quality became central through [`_greedy_eval_score(...)`](../train/mappo_gru.py)
- a separate `best_greedy_eval/` checkpoint was introduced

### Why it helped

This phase did not solve behavior by itself, but it fixed the trainer so that it stopped fighting the curriculum.

That is an important lesson in RL engineering:

- reward shaping alone cannot rescue a structurally unstable training loop
- trainer continuity and checkpoint selection matter just as much as reward tuning

## Phase 3: Make Greedy Behavior Real, Not Just Sampled

After task 28, a clearer problem emerged:

- sampled training behavior improved a little
- greedy evaluation still failed badly

That led to tasks 29 through 39.

### Task 29: teach greedy pickup and return first

Task 29 changed the earliest stages so the model could learn the basic loop more deterministically.

In [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py):

- early single-agent stages were kept pheromone-off
- movement penalties were softened
- pickup and delivery cues were strengthened
- entropy became stage-specific through `entropy_start` and `entropy_end`

In [train/mappo_gru.py](../train/mappo_gru.py):

- the trainer began reporting sampled-vs-greedy gaps
- [`_stage_entropy_coef(...)`](../train/mappo_gru.py) made entropy decay stage-aware

This produced one of the first clearly positive results:

- `stage1a_single_agent_miniscule` reached greedy pickup and delivery of `4.0 / 4.0`

That was the first real proof that the pipeline could produce stable greedy behavior in at least the easiest lesson.

### Task 30: focus on return-to-nest, not just pickup

Once pickup became reliable in the easiest stage, the next failure mode became obvious:

- the policy could find food
- but it still often failed to carry it home

Task 30 addressed that by making the carrying phase more important than generic exploration.

In [env/config.py](../env/config.py):

- `carrying_reward_new_cell_scale` was introduced so exploration reward could be suppressed while carrying
- `reward_nest_approach` became a stage-relevant knob

In [env/swarm_env.py](../env/swarm_env.py):

- return-to-nest shaping is applied in [`_apply_nest_return_shaping(...)`](../env/swarm_env.py)

In [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py):

- the curriculum added more explicit return-focused stages

The central idea was simple:

- while empty, exploration matters
- while carrying food, exploration should stop being the main objective

That is a very common RL design principle: reward the subskill that matters now, not the subskill that mattered one minute ago.

### Tasks 31–38: build the return stack lesson by lesson

Tasks 31 through 38 were the most educational part of the whole training story.

They show what curriculum learning looks like when a single subskill is still too hard and has to be split into smaller lessons.

The return stack gradually became:

- `stage1d_single_agent_carry_bootstrap`
- `stage1e_single_agent_guaranteed_homing`
- `stage1f_single_agent_delivery_bridge`
- `stage1g_single_agent_delivery_obstacles`

These stages live in [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py).

The new lessons used several environment controls added during tasks 32, 36, 37, and 38:

- `target_nest_distance_min`
- `target_nest_distance_max`
- `target_nest_corridor_clearance`
- `agent_spawn_near_target_radius`
- `start_carrying_food`
- `reward_nest_approach_sustained`
- `carrying_progress_streak_threshold`

These controls are defined in [env/config.py](../env/config.py) and applied in [env/swarm_env.py](../env/swarm_env.py), especially:

- [`_sample_target_position(...)`](../env/swarm_env.py)
- [`_apply_nest_return_shaping(...)`](../env/swarm_env.py)

The curriculum was effectively turned into a homing course:

1. start already carrying food
2. learn to go home
3. then add normal pickup again
4. then add mild clutter
5. then add real obstacles

That is a textbook example of shaping the task, not just the reward.

### The most important bug fix in the whole sequence

Task 37 exposed a real environment bug:

- in [env/swarm_env.py](../env/swarm_env.py), the tank and hover movement drivers were dropping `carrying_food` during normal motion updates

That meant an agent could pick something up, move, and effectively lose the state that said it was carrying food.

This bug directly broke the return-to-nest loop.

This is worth calling out because it shows a major truth about RL debugging:

- sometimes the model is not failing because the reward is wrong
- sometimes the environment is silently erasing the state the policy depends on

After that bug was fixed, task 37 produced the first short verification run with nonzero greedy delivery in the dedicated homing stack.

### Task 39: stabilize the bridge stage

Even after tasks 37 and 38, another subtle problem remained:

- the bridge stage could produce a good greedy evaluation mid-run
- then drift away from it before the final stage-end evaluation

Task 39 fixed that in [train/mappo_gru.py](../train/mappo_gru.py):

- stage-end evaluation now restores the best within-stage actor, critic, and optimizer state before the final promotion check

This sounds technical, but the idea is easy:

- if the trainer finds the best version of the policy halfway through the stage, do not let later updates ruin the handoff

That stabilization step made `stage1f_single_agent_delivery_bridge` stay nonzero on greedy delivery across the whole verification run, including the final evaluation row.

At that point, stage 1 finally looked stable enough for full training.

## Phase 4: Scale from One Agent to a Swarm

Once single-agent delivery was stable, the next collapse happened when the curriculum first introduced multiple agents.

That led to tasks 40 and 41.

### What was wrong

The first jump into swarm behavior was too abrupt. Single-agent delivery was stable, but multi-agent delivery was not.

The trainer was effectively asking:

- can one robot deliver?
- okay, now can several robots deliver in a harder world?

That is too big a jump for many RL systems.

### What changed

New small-swarm stages were inserted into [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py):

- `stage2a_small_swarm_carry_bootstrap`
- `stage2b_small_swarm_delivery_easy`
- `stage2c_small_swarm_medium`
- `stage2d_small_swarm_large`

These stages let the curriculum teach:

1. small-swarm homing
2. small-swarm delivery in a simple world
3. medium clutter
4. larger clutter plus pheromone

Task 41 then reduced unnecessary repeat pressure and budget drain in already-solved stage-1 lessons so the trainer could actually reach the new swarm stack consistently.

### Why it helped

This made the jump from one agent to many agents a curriculum step, not a cliff.

The full-run result after task 41 was a major milestone:

- `stage2c_small_swarm_medium` stayed strong with repeated `4/4` deliveries
- `stage2d_small_swarm_large` kept nonzero greedy delivery
- `stage3a_full_swarm_large` completed with greedy pickup `1.00`, delivery `0.60`
- `stage3b_full_swarm_final` completed with greedy pickup `1.20`, delivery `0.80`

That was the first point where full training looked satisfactory instead of merely promising.

## Phase 5: Remove Nest-Centered Local Optima

After full training became viable, a new behavioral problem showed up in demos:

- non-carrying agents could cluster around the nest
- they could orbit the nest instead of fanning out to explore

This led to tasks 42 through 47.

### Why this mattered

For the intended swarm behavior, the policy should have two distinct modes:

- carrying food: go home
- not carrying food: go out and search

If empty agents stay near the nest, the swarm wastes time and coverage.

### Tasks 42–44: soft penalties and post-delivery handoff

These tasks introduced:

- non-carrying nest loiter penalties
- non-carrying nest crowding penalties
- nest-adjacent pheromone suppression
- post-delivery cooldown and outward-search shaping
- post-delivery exit requirements

These are configured in [env/config.py](../env/config.py) and applied in [env/swarm_env.py](../env/swarm_env.py), especially:

- [`_apply_non_carrying_nest_penalties(...)`](../env/swarm_env.py)
- [`_apply_post_delivery_outward_shaping(...)`](../env/swarm_env.py)

These tasks improved late-stage delivery while reducing nest crowding, but they did not eliminate the local optimum completely.

### Task 45: make empty agents fan out

Task 45 added a more explicit outward-search path for empty agents.

In [env/config.py](../env/config.py), new controls were added for:

- `non_carrying_explore_radius`
- `non_carrying_outward_reward`
- `non_carrying_idle_near_nest_penalty`
- `non_carrying_low_displacement_threshold`
- `non_carrying_explore_ignore_pheromone`

In [env/swarm_env.py](../env/swarm_env.py):

- [`_apply_non_carrying_outward_shaping(...)`](../env/swarm_env.py) rewards empty agents for moving away from the nest

This was an important conceptual step:

- instead of only saying “do not stay near the nest”
- the environment also started saying “go outward and explore”

### Task 46: explicit force-explore mode

Even stronger penalties were still not always enough, so task 46 moved from reward shaping to explicit behavior intervention.

In [env/config.py](../env/config.py):

- `non_carrying_force_explore_mode`
- `non_carrying_force_explore_radius`

In [env/swarm_env.py](../env/swarm_env.py):

- empty agents near the nest can have orbit-friendly actions overridden with outward-moving actions
- action-hold state is cleared so a bad orbit action does not persist

This is one of the clearest examples in the repo of switching from:

- soft learning pressure

to:

- environment-side behavior constraints

That is sometimes the right choice in RL systems when a local optimum is too strong.

### Task 47: more randomness while exploring, but only then

Task 47 added controlled exploratory randomness for empty agents only.

In [env/config.py](../env/config.py):

- `non_carrying_explore_random_action_prob`

In [env/swarm_env.py](../env/swarm_env.py):

- [`_apply_non_carrying_exploration_randomness(...)`](../env/swarm_env.py) injects movement-producing random exploratory actions for non-carrying agents only

The important design constraint was:

- randomness helps exploration
- but it should not interfere with carrying-food return behavior

That is why the randomness was added only to the non-carrying mode.

This is a good example of a targeted exploration mechanism that respects the task structure instead of adding noise everywhere.

## Trainer Improvements That Made the Whole System More Trustworthy

Several trainer changes in [train/mappo_gru.py](../train/mappo_gru.py) were especially important:

### Stage-sensitive entropy

[`_stage_entropy_coef(...)`](../train/mappo_gru.py) lets the trainer decay entropy differently by stage. This made it easier to:

- keep early exploration alive
- but make homing stages more deterministic

### Greedy-first scoring

[`_greedy_eval_score(...)`](../train/mappo_gru.py) heavily rewards delivery and conversion, especially in return-critical stages. It also penalizes pickup-without-delivery in those stages.

This matters because raw reward alone can hide bad behavior. A policy that picks up many targets but never delivers them is not solving the task.

### Promotion targets

[`_stage_promotion_target(...)`](../train/mappo_gru.py) and [`_meets_stage_promotion(...)`](../train/mappo_gru.py) turned the curriculum into a real lesson plan. Stages now have explicit success criteria.

### Hard global step cap

Task 34 fixed another trainer discipline issue:

- before the fix, a run with `--total-steps 600000` could still overshoot badly while finishing stages and repeats
- after the fix, the hard cap is respected

That made experiments easier to reason about and compare.

## Environment Improvements That Made the Behavior Learnable

The environment in [env/swarm_env.py](../env/swarm_env.py) was not just a passive simulator. It gradually became an active teaching partner for the curriculum.

The key shaping paths now include:

- [`_apply_nest_return_shaping(...)`](../env/swarm_env.py)
  - rewards carrying agents for reducing nest distance
- [`_apply_non_carrying_nest_penalties(...)`](../env/swarm_env.py)
  - discourages empty agents from lingering or clustering near the nest
- [`_apply_non_carrying_outward_shaping(...)`](../env/swarm_env.py)
  - rewards empty agents for moving outward and exploring
- [`_apply_post_delivery_outward_shaping(...)`](../env/swarm_env.py)
  - pushes agents to leave the nest zone after delivering
- [`_apply_non_carrying_exploration_randomness(...)`](../env/swarm_env.py)
  - adds targeted randomness to empty-agent exploration

The spawn logic also became more curriculum-aware through [`_sample_target_position(...)`](../env/swarm_env.py), which can enforce:

- a target-to-nest distance band
- a clearer corridor between nest and target

These environment changes matter because RL is not just about the neural network. The environment defines what can be learned easily and what remains unnecessarily ambiguous.

## What the Current Curriculum Looks Like

The current curriculum in [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py) now looks much more like a deliberate lesson sequence:

### Single-agent lesson stack

- `stage1a_single_agent_miniscule`
- `stage1b_single_agent_tiny`
- `stage1c_single_agent_small`
- `stage1d_single_agent_carry_bootstrap`
- `stage1e_single_agent_guaranteed_homing`
- `stage1f_single_agent_delivery_bridge`
- `stage1g_single_agent_delivery_obstacles`

### Small-swarm stack

- `stage2a_small_swarm_carry_bootstrap`
- `stage2b_small_swarm_delivery_easy`
- `stage2c_small_swarm_medium`
- `stage2d_small_swarm_large`

### Full-swarm stack

- `stage3a_full_swarm_large`
- `stage3b_full_swarm_final`

This is a much more coherent curriculum than the earlier versions, because each block teaches one distinct capability:

- find and deliver with one agent
- preserve delivery under mild clutter
- preserve delivery when more agents arrive
- then add pheromone, crowding, and harder clutter

## The Most Important Measured Improvements

The training story is convincing because the changes produced concrete behavioral improvements.

Some of the clearest examples were:

### Task 29

- `stage1a_single_agent_miniscule` reached greedy pickup and delivery of `4.0 / 4.0`

Meaning:

- the earliest curriculum stage finally produced real greedy behavior instead of sampled-only success

### Task 37

- the first short smoke run with nonzero greedy delivery in the dedicated homing stack

Meaning:

- the model could finally return food under greedy evaluation once the carrying-state bug and the homing lesson design were fixed

### Task 39

- `stage1f_single_agent_delivery_bridge` stayed nonzero on all logged eval rows
- final bridge eval included `pickup = 2.0`, `delivery = 2.0`, `conversion = 0.5`

Meaning:

- mild clutter no longer erased greedy delivery by the end of the stage

### Task 41

- `stage2c_small_swarm_medium` stayed strong with repeated `4/4` deliveries
- `stage3b_full_swarm_final` completed with greedy pickup `1.20`, delivery `0.80`

Meaning:

- the full curriculum finally made it through to the hard late stages without collapsing

### Tasks 42–44

- `stage3a` improved from greedy `1.0 / 1.0` to `2.0 / 2.0`
- `stage3b` improved from `1.0 / 0.8` to `1.2 / 1.0`
- nest crowding dropped

Meaning:

- reducing nest-centered local optima improved late-stage swarm behavior without destroying delivery

## What This Teaches About RL and Curriculum Learning

This history is useful beyond this repository because it illustrates several general RL lessons.

### 1. A curriculum is not just “easy to hard”

A good curriculum is:

- easy to inspect
- explicit about subskills
- verified by stage
- willing to add bridge lessons when the jump is too large

That is exactly what happened here.

### 2. Greedy evaluation matters

If training uses stochastic policy sampling but demo uses greedy actions, you must measure both.

Otherwise you can fool yourself into thinking the policy is good when it only works under training-time randomness.

### 3. Reward shaping cannot fix every problem

Several times, the right fix was not “more penalty” or “more reward.” It was:

- preserve critic continuity
- change promotion logic
- fix a state bug
- add a curriculum bridge
- constrain the environment geometry
- explicitly override bad local behavior

### 4. Environment design is part of learning design

Target placement, corridor clearance, carrying-state handling, and post-delivery behavior all changed what the policy could learn.

That is not a side detail. In RL, environment design is part of the algorithmic system.

## If You Want to Read the Code in a Useful Order

For a reader trying to understand both the repo and curriculum RL, this is a good reading order:

1. [algorithms/mappo/curriculum.py](../algorithms/mappo/curriculum.py)
   - read the stage list and see how the lesson plan is structured
2. [train/mappo_gru.py](../train/mappo_gru.py)
   - focus on:
     - [`_build_env(...)`](../train/mappo_gru.py)
     - [`_critic_state_dim(...)`](../train/mappo_gru.py)
     - [`_stage_promotion_target(...)`](../train/mappo_gru.py)
     - [`_greedy_eval_score(...)`](../train/mappo_gru.py)
     - the stage loop
3. [env/config.py](../env/config.py)
   - read the reward and behavior knobs
4. [env/swarm_env.py](../env/swarm_env.py)
   - read the shaping functions:
     - [`_apply_nest_return_shaping(...)`](../env/swarm_env.py)
     - [`_apply_non_carrying_nest_penalties(...)`](../env/swarm_env.py)
     - [`_apply_non_carrying_outward_shaping(...)`](../env/swarm_env.py)
     - [`_apply_post_delivery_outward_shaping(...)`](../env/swarm_env.py)
     - [`_apply_non_carrying_exploration_randomness(...)`](../env/swarm_env.py)

That reading order gives you:

- the lesson plan
- the trainer
- the knobs
- the environment mechanics

## Final Summary

The MAPPO curriculum path improved because we stopped treating “training” as one black box and instead broke it into concrete learnable behaviors:

- first fix the stage budget
- then fix trainer instability
- then make greedy behavior real
- then teach return-to-nest explicitly
- then preserve it under clutter
- then preserve it under scale-up to many agents
- then stop empty agents from wasting time around the nest

That is the main takeaway from this whole history.

The final system is still an RL system with all the usual challenges, but it is now much more interpretable:

- the curriculum stages are explicit
- the trainer knows what success means
- the environment teaches the intended subskills more directly
- the evaluation path is much more trustworthy

That makes the repository a much better place both to train the model and to learn how curriculum learning works in practice.
