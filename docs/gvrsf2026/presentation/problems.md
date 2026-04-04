# Problems I Had To Solve During Development

This is the part of the project I think matters most for judges in computer science.

The final result was not produced by training one model and hoping it worked. I had to repeatedly identify a concrete failure mode, figure out whether it was an algorithm problem, an environment problem, an evaluation problem, or a measurement problem, and then redesign the system around that diagnosis.

In other words, the real work was not just "building a swarm simulator." The real work was solving a sequence of technical problems in reinforcement learning, multi-agent systems, evaluation design, and debugging.

## How I derived solutions

I did not derive the fixes by guessing.

I used the same loop over and over:

1. run training or evaluation and inspect the metrics,
2. compare sampled behavior against greedy behavior,
3. watch the demo to see what the agents were actually doing,
4. inspect the code path responsible for that behavior,
5. change the smallest thing that matched the diagnosis,
6. rerun focused verification before scaling back up.

That process is important because many RL failures look similar from the outside. An agent that fails to deliver food might be failing because:

- the reward function is wrong,
- the curriculum is wrong,
- the evaluation is misleading,
- the checkpoint selection logic is bad,
- or the environment itself has a bug.

One of the main computer science lessons from this project is that those are different classes of failure, and they need different kinds of solutions.

## The problem-solution progression

### Step 1: The original simulator was not enough for a real science question

**Problem**

At the beginning, I had a runnable swarm RL environment, but not yet a system strong enough to answer a real research question about collective intelligence. It could produce motion, but it did not yet support strong experiments, strong baselines, or a full foraging loop.

**Solution**

I expanded the environment, training code, evaluation code, and experiment structure so the project could support controlled comparisons instead of just demos.

**How I derived it**

The early docs and task archive make it clear that the first bottleneck was not model quality. It was scientific infrastructure. Before I could test stigmergy seriously, I needed:

- reusable evaluation,
- repeatable runs,
- saved checkpoints,
- and comparable outputs.

**Computer science relevance**

This was a systems-design problem. Good ML results depend on software architecture, reproducibility, and instrumentation.

### Step 2: I needed observability, not just code

**Problem**

When a learned policy behaved strangely, I often could not tell why. Raw reward numbers were not enough.

**Solution**

I added better documentation, policy probing, debug logging, and later more informative training and evaluation outputs.

**How I derived it**

The project history shows that as soon as the system grew more complex, I started adding tools that made the internal behavior visible. That happened before the hardest MAPPO debugging, and it turned out to be necessary. Later, when sampled behavior looked good but greedy behavior failed, those visibility tools became essential.

**Computer science relevance**

This is an observability problem. In intelligent systems, debugging requires introspection tools, not just final metrics.

### Step 3: The reward signal was too weak and too ambiguous

**Problem**

In the DQN phase, the agents often explored but did not learn the real task well. Part of the issue was sparse reward. Another part was that some signals encouraged behavior without clearly aligning to the final goal.

**Solution**

I improved food detectability, intermediate shaping, and exploration incentives, while making the task loop more explicit.

**How I derived it**

I compared what the agents were doing in demo against what the training logs suggested. The mismatch showed that the reward design was not yet communicating the right priorities.

**Computer science relevance**

This is a credit-assignment problem. In RL, if the objective is delayed and sparse, the software designer must shape the signal carefully without creating fake success.

### Step 4: My pheromone signal was not semantically meaningful enough

**Problem**

At first, pheromone could easily become "just a trail of movement" instead of a useful coordination signal tied to successful foraging.

**Solution**

I changed deposition logic and pheromone-related settings so the signal was more closely tied to meaningful task phases, especially successful carrying and return behavior.

**How I derived it**

I realized that if agents deposit pheromone at the wrong times, then the environment is storing noisy or misleading information. That weakens the stigmergy claim because the field stops being a shared memory of success.

**Computer science relevance**

This is a representation-design problem. Communication channels only help if the information they encode is useful.

### Step 5: The simulator itself had bugs that could fake "learning failure"

**Problem**

Some failures were not caused by the policy at all. They were caused by environment bugs. Two early important examples were:

- pheromone diffusion wrapping around arena edges and spreading unrealistically,
- metrics treating pickup and delivery too ambiguously.

Later, an even more serious bug appeared: movement updates could silently drop the `carrying_food` state.

**Solution**

I fixed the diffusion logic, cleaned up the metrics, and repaired the carrying-state bug in movement updates.

**How I derived it**

I compared visible behavior, logged metrics, and the actual code path. When those did not agree, I stopped treating the problem as "bad learning" and treated it as a simulator-debugging problem.

The `carrying_food` bug was especially important. The policy looked incompetent, but the environment was actually breaking the state transition needed for delivery.

**Computer science relevance**

This is a classic verification problem. In RL, a faulty simulator can make a correct learning algorithm look incorrect.

### Step 6: I needed better experiment control and better output structure

**Problem**

As the number of experiments grew, outputs became harder to compare, rerun, and interpret. This included naming, checkpoint layout, and evaluation organization.

**Solution**

I added better CLI control, structured output folders, comparison runners, fixed checkpoint naming, and cleaner saved metadata.

**How I derived it**

The task archive from this period shows that I was hitting workflow bottlenecks, not just model bottlenecks. I could not iterate efficiently if runs were hard to identify or if outputs were scattered.

**Computer science relevance**

This is a reproducibility and experiment-management problem. Good research software must support controlled iteration.

### Step 7: DQN was useful, but it was not strong enough for the real task

**Problem**

The project needed a decentralized policy that could handle partial observability, delayed rewards, and multi-agent coordination. The DQN path helped me build infrastructure, but it was not the right long-term algorithmic path for the strongest version of the problem.

**Solution**

I pivoted to recurrent MAPPO with curriculum learning.

**How I derived it**

The repo history shows a clear conclusion: after improving the DQN stack, the remaining difficulty was not just tuning. The problem itself needed a stronger algorithmic framework with recurrence, centralized training, and a staged learning process.

**Computer science relevance**

This is an algorithm-selection problem. Choosing the right learning architecture is part of solving the problem.

### Step 8: Increasing the number of agents was not a real curriculum

**Problem**

At first, I could have treated the curriculum as "start with fewer agents, then add more." That turned out to be too simple.

**Solution**

I built stage-specific environments that could change:

- world size,
- obstacle count,
- target count,
- episode length,
- and other difficulty variables.

**How I derived it**

The failures showed that difficulty was multi-dimensional. An agent could fail because of clutter, path length, return geometry, or interference, not just swarm size.

**Computer science relevance**

This is a curriculum-design problem. Difficulty in sequential decision systems is not one-dimensional, so the training sequence must reflect that.

### Step 9: The project needed a true delivery task, not just target contact

**Problem**

A policy that can reach a target is not the same as a policy that can complete foraging. I needed the full loop:

`explore -> find -> pick up -> return -> deliver -> reinforce route`

**Solution**

I made carrying state, delivery at the nest, and return-path behavior explicit parts of the environment and training objective.

**How I derived it**

This came directly from the research claim. If I wanted to study stigmergic swarm foraging, then the environment had to represent actual loop closure, not just "touch target."

**Computer science relevance**

This is an objective-definition problem. The software task must match the scientific claim.

### Step 10: The policy kept collapsing even after I had the right algorithm family

**Problem**

Once recurrent MAPPO existed, the system still failed in a more subtle way. Stage transfer was unstable, and the policy could appear to improve during sampled rollouts but still collapse under greedy evaluation.

**Solution**

I made structural training fixes, especially:

- padded critic-state handling across stages,
- stronger greedy-eval-aware promotion,
- and checkpoint logic centered on actual greedy competence.

**How I derived it**

I compared training behavior against evaluation behavior and realized the problem was not just reward tuning. It was training stability and checkpoint selection.

**Computer science relevance**

This is an optimization-stability problem. In actor-critic systems, architectural continuity matters, not just reward values.

### Step 11: Sampled success was lying to me

**Problem**

A major failure mode was that the agent looked promising during stochastic training rollouts but failed when evaluated greedily. That meant it had not learned a robust policy, only a fragile one that depended on randomness.

**Solution**

I changed early stages to teach stable greedy behavior by:

- turning pheromone off in early stages,
- softening some penalties,
- strengthening pickup and delivery cues,
- decaying entropy within stages,
- and explicitly reporting sampled-vs-greedy gaps.

**How I derived it**

I did not just look at average training reward. I looked at the difference between sampled and greedy execution. That exposed the real failure mode.

**Computer science relevance**

This is an evaluation-design problem. If the evaluation protocol does not match the intended deployment behavior, it can hide failure.

### Step 12: Rewards that help before pickup can hurt after pickup

**Problem**

The agents often found food but still failed to return it. After pickup, exploration reward was still encouraging outward wandering instead of homing.

**Solution**

I introduced carrying-aware shaping so that once an agent had food, the reward structure shifted toward nest approach and away from exploration.

**How I derived it**

I broke the task into phases and asked what the agent should be rewarded for in each phase. That changed the problem from "one reward function for everything" to "reward pressure should follow task mode."

**Computer science relevance**

This is a task-phase modeling problem. Good sequential policies often need phase-specific incentives.

### Step 13: Return-to-nest under clutter had to be decomposed into subskills

**Problem**

The hardest repeated failure was not finding food. It was turning pickup into reliable delivery, especially under clutter. The agent could partially solve easy homing but collapse as soon as the geometry became harder.

**Solution**

I split the homing problem into a sequence of stages:

- carry bootstrap,
- guaranteed homing,
- bridge stage,
- obstacle-return stage.

I also tightened promotion criteria and delivery-sensitive evaluation.

**How I derived it**

I kept narrowing the failure. Each time a stage failed, I asked what exact skill was missing. That led me to stop treating "delivery" as one skill and instead treat it as a chain of smaller skills that needed to be isolated.

**Computer science relevance**

This is hierarchical decomposition. A hard sequential behavior became learnable only when decomposed into testable subproblems.

### Step 14: I had to enforce honest progress, not fake progress

**Problem**

The system could sometimes look like it was improving because pickup counts were rising, even while delivery stayed near zero. Also, the training budget could drift in ways that made stage progress misleading.

**Solution**

I enforced a harder global step cap, added stronger delivery-conversion requirements, and penalized pickup-without-delivery behavior in later stages.

**How I derived it**

I looked at where the policy was cheating the intended objective. The answer was that pickup alone was being treated as too much progress.

**Computer science relevance**

This is a metric-integrity problem. If the progress metric is weak, optimization will exploit it.

### Step 15: One of the biggest breakthroughs came from treating a "learning failure" as a software bug

**Problem**

Even after many curriculum and reward fixes, delivery still failed in a way that looked almost mysterious.

**Solution**

I traced the environment logic more carefully and found that movement updates could drop `carrying_food`. I fixed that, then added an even more explicit carry-bootstrap lesson.

**How I derived it**

This came from refusing to assume the model was the only thing at fault. I checked whether the simulator was preserving the state that the policy depended on. It was not.

**Computer science relevance**

This is a state-consistency problem. In interactive systems, learning quality depends on correct state transition logic.

### Step 16: Solving the one-agent case did not automatically solve the swarm case

**Problem**

After the single-agent delivery stages improved, the first real swarm stages collapsed again. The transition from one agent to several agents was another curriculum wall.

**Solution**

I inserted a small-swarm bootstrap stack before the full swarm stages and reduced wasted training budget on already-solved earlier stages.

**How I derived it**

The metrics showed that the first scale-up stage was the new cliff. That told me scaling itself had to be taught as a skill, not assumed as a side effect.

**Computer science relevance**

This is a transfer-learning and scaling problem. Competence at one scale does not automatically generalize to another.

### Step 17: Empty agents developed a bad local optimum around the nest

**Problem**

Once delivery improved, a new failure mode appeared: non-carrying agents clustered and orbited near the nest instead of leaving to search.

**Solution**

I addressed this in stages:

- reduce nest circling,
- add post-delivery outward handoff,
- hold that mode until real nest exit,
- explicitly fan out non-carrying agents,
- and finally add force-explore logic and targeted randomness for empty agents.

**How I derived it**

I observed the demos directly. The agents were no longer failing at delivery in the old way. They were getting stuck in a new locally stable but useless behavior.

**Computer science relevance**

This is a local-optimum problem in sequential control. Solving one subproblem can create a new attractor state elsewhere in the policy.

### Step 18: Even rendering became a computer science problem

**Problem**

Large-world PyGame rendering did not reliably show the simulation at the correct scale, which made visual diagnosis and presentation weaker.

**Solution**

I added proper off-screen world rendering and scaled display output.

**How I derived it**

This was not cosmetic. If the display lies, then human debugging and scientific communication both become weaker.

**Computer science relevance**

This is a visualization-correctness problem. Good tooling is part of reliable system development.

## What I think the judges should take from this

The most important thing about this project is not just that I trained a swarm policy.

It is that I had to solve a chain of real computer science problems:

- simulator design,
- reward design,
- experiment design,
- observability,
- evaluation validity,
- state-transition debugging,
- curriculum design,
- optimization stability,
- and scaling failure modes.

The final performance came from that engineering process.

## Short board-ready version

If I need to compress this for the board, this is the shortest honest version:

1. **The first problem was not just training. It was measurement.**
   I had to build evaluation, checkpointing, logging, and comparison infrastructure before I could trust my results.

2. **Then I discovered that many "AI failures" were actually software failures.**
   I fixed simulator bugs, especially ambiguous pickup/delivery metrics and a major carrying-state bug.

3. **Then I discovered that the main RL problem was not pickup. It was reliable greedy delivery.**
   I solved that by redesigning the curriculum into smaller subskills and making evaluation delivery-sensitive.

4. **Then I discovered that single-agent success did not transfer automatically to swarm success.**
   I added explicit small-swarm bootstrap stages and later solved nest-orbit local optima for non-carrying agents.

5. **The final result is a stronger computer science project because the solutions are algorithmic and systems-based, not just trial and error.**

