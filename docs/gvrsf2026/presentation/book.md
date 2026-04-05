# Stigmergic Swarm Intelligence: A Textbook-Style Guide to the Project

## Preface

This document is meant to be read from start to finish. It explains the project as both a scientific system and a software-engineering journey.

It has four jobs:

1. teach the theory behind the project deeply enough that a new reader can learn the main technologies,
2. explain the actual system built in this repository,
3. reconstruct the research-and-development process from the archived tasks and git history,
4. show the results, the failures, and the reasoning that turned those failures into progress.

This reconstruction is grounded in the repository itself:

- the code in `env/`, `train/`, and `algorithms/`,
- the task archive in `docs/tasks/`,
- the retrospective docs such as `docs/TRAINING_MAPPO.md`, `docs/building_curriculum_training.md`, and the integrated GVRSF experiment paper,
- and the git commit history and per-commit file diffs.

Where the repository does not preserve every thought process directly, this book makes careful inferences from tasks, code changes, commit messages, and experiment artifacts.

## Chapter 1. The Big Question

This project asks a classical scientific question in a modern computational form:

> Can collective intelligence emerge from many simple agents that only sense locally and communicate indirectly through the environment?

The biological inspiration is ant foraging. Individual ants have narrow perception and simple action rules. Yet ant colonies can still discover food sources, reinforce good routes, adapt to disruption, and scale their labor. One of the central mechanisms behind this is **stigmergy**: agents change the environment, and those environmental changes become information for later agents.

In this repository, that question becomes a swarm-robotics and reinforcement-learning problem:

- agents move in a 2D world,
- they search for food,
- they pick it up,
- they return it to a nest,
- and they can deposit and sense pheromone-like trails.

The project is not satisfied by “interesting motion.” The full target behavior is the foraging loop:

`explore -> find target -> pick up -> return -> deliver -> reinforce route`

That loop is what turns wandering into task completion, and it is why this project became much more than a basic simulator. The hard part was not making agents move. The hard part was making them learn a stable, greedy, scalable delivery policy.

## Chapter 2. Theoretical Foundations

### 2.1 Swarm intelligence

Swarm intelligence studies how group-level capability can arise from local interaction rather than central control. In computer science, this idea matters because centralized systems are often fragile:

- a central planner can become a bottleneck,
- communication links can fail,
- the world can change faster than a single global model can update,
- and scaling the number of agents can make centralized optimization expensive or brittle.

Swarm systems instead rely on:

- local sensing,
- local action,
- distributed state,
- and often indirect coordination.

The strongest swarm systems are not merely “many robots.” They are systems where coordination makes the group more effective than the naive sum of independent individuals.

### 2.2 Stigmergy

**Stigmergy** is coordination through environmental traces. An agent leaves information in the world. Another agent later reads that information and changes its own behavior.

Ant pheromone trails are the classic example. The important idea is not chemistry by itself. The deeper computational idea is **externalized memory**:

- the environment stores partial information,
- that information persists across time,
- and agents do not need direct peer-to-peer negotiation to benefit from it.

In this project, the pheromone field plays that role. A successful return path can deposit a trail; later agents can sense the trail and use it as a bias during their own navigation. This turns the world into a shared, low-bandwidth coordination medium.

### 2.3 Reinforcement learning

Reinforcement learning (RL) studies how an agent learns behavior by trial and error. At each step:

- the agent receives an observation,
- chooses an action,
- receives reward,
- and transitions to a new state.

The objective is not to maximize immediate reward alone. It is to maximize **expected cumulative discounted reward**:

`G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...`

The discount factor `gamma` makes future reward matter while keeping the total objective bounded.

This project needs RL because the desired behavior is long-horizon:

- exploration may not pay off immediately,
- pickup may be separated from delivery by many steps,
- pheromone can help only after a useful route has already been discovered,
- and the “best next action” depends on what phase of the task the agent is currently in.

### 2.4 Bellman reasoning

The Bellman idea says that the value of a state depends on immediate reward plus the value of what follows:

`V(s) = E[r + gamma V(s')]`

This matters because the project is full of delayed consequences. A move that slightly reduces nest distance while carrying food may not give a large immediate reward, but it makes future delivery more likely. Likewise, laying a pheromone trail can help later agents rather than the current agent immediately.

The Bellman view is how the system learns that some actions are valuable because they create good futures, not because they pay instantly.

### 2.5 Partial observability and Dec-POMDP structure

The agents in this repository do not observe the full world state during inference. Each one acts from local observation. That makes the problem a decentralized partially observable multi-agent problem, often described conceptually as a **Dec-POMDP**.

The consequences are important:

- agents cannot rely on full global maps at execution time,
- observations are ambiguous,
- memory matters,
- and coordination must emerge under uncertainty.

This is why the project uses recurrent policies and why centralized training with decentralized execution is such an important design choice.

### 2.6 Centralized training with decentralized execution

The project’s main RL path uses **CTDE**:

- **centralized training**: the learner may use richer training-time state,
- **decentralized execution**: each deployed agent acts only from its local observation and its recurrent hidden state.

Why CTDE is useful here:

- the critic can stabilize learning by evaluating richer joint context,
- but the deployed policy remains realistic for physical robots,
- and the inference path stays decentralized.

This is a strong match for swarm robotics, where training can use extra structure, but real robots cannot assume an omniscient controller.

### 2.7 PPO and MAPPO

The final main algorithm is recurrent **MAPPO**: Multi-Agent Proximal Policy Optimization.

At a high level:

- an **actor** outputs action preferences from local observations,
- a **critic** estimates value,
- advantages measure whether outcomes were better or worse than expected,
- and PPO updates the actor with a clipped objective so the policy does not change too violently in one step.

The PPO ratio is conceptually:

`ratio = pi_new(a|o) / pi_old(a|o)`

The clipped objective prevents the new policy from drifting too far from the old one during a single update. This is critical in complex multi-agent tasks, where unstable policy jumps can destroy useful behavior.

MAPPO extends this style of optimization to multi-agent settings. In this project, the actor is parameter-shared across homogeneous agents, which improves sample efficiency and respects the idea that each robot follows the same learned policy class.

### 2.8 Advantage estimation and GAE

A policy should not be updated directly from raw return alone. That signal can be noisy. The project therefore uses **Generalized Advantage Estimation (GAE)**.

Advantages answer:

> Was this action better or worse than the critic expected?

GAE blends short-horizon and long-horizon credit assignment:

- low bias from looking ahead,
- lower variance than full Monte Carlo returns,
- more stable updates than purely one-step targets.

This is one of the key reasons PPO-style actor-critic methods work well on long-horizon tasks like explore-find-return-deliver.

### 2.9 Recurrent memory

The project uses GRU-based recurrence because local observations are not enough by themselves. An agent may need to remember:

- whether it recently saw a target,
- whether it is currently in a carrying phase,
- which direction previously seemed promising,
- and local motion context across time.

Without memory, partial observability becomes much harsher. Recurrence lets the policy build an internal belief-like summary of recent history.

### 2.10 Reward shaping

Pure sparse reward would make this task much harder. Delivery is the ultimate goal, but waiting until delivery to provide signal creates a difficult credit-assignment problem. The repository therefore uses reward shaping carefully.

Important principle:

> reward the subskill that matters now, without letting shaping replace the real task.

Examples from the project include:

- pickup reward,
- strong nest-delivery reward,
- carrying-phase nest-approach shaping,
- exploration reward for discovering new cells,
- and later penalties for empty-agent loitering near the nest.

The challenge of reward shaping is that it can create false local optima. Much of the repository’s later history is a story of discovering and removing those local optima.

### 2.11 Curriculum learning

Curriculum learning means starting with easier tasks and progressively increasing difficulty. In this repository it became essential.

The central curriculum idea is:

1. teach simple single-agent success,
2. isolate the return-to-nest subskill,
3. reintroduce clutter gradually,
4. scale to a small swarm,
5. then scale to the full swarm and full pheromone-enabled setting.

This is not just pedagogical style. It is a practical stability tool. The project repeatedly showed that directly asking the model to solve the hardest version of the task led to collapse.

## Chapter 3. The System Implemented in the Repository

### 3.1 Environment

The environment lives primarily in `env/swarm_env.py`, with configuration in `env/config.py`.

The simulated world includes:

- a nest,
- food sources,
- rectangular obstacles,
- a pheromone field,
- and a configurable swarm of agents.

The environment eventually became PettingZoo Parallel API native, which aligned the simulator with modern multi-agent tooling and made training and demo paths more consistent.

### 3.2 Observation and action interfaces

The architecture docs describe the current default observation contract as:

- 23 features per observation frame,
- 3-frame history,
- flattened observation dimension `69`.

Features include:

- obstacle-ray sensing,
- nearest-food vector in agent-local coordinates,
- nest direction,
- nearest-agent vector,
- heading encoding,
- speed,
- local food presence,
- carrying-food flag,
- and pheromone samples.

The action space is `Discrete(18)`, built as combinations of:

- throttle in `{-1, 0, 1}`,
- turn in `{-1, 0, 1}`,
- deposit flag in `{0, 1}`.

This is a compact but expressive control space. It is simple enough for learning, but rich enough to support movement, turning, and stigmergic behavior.

### 3.3 Pheromone field

The pheromone grid acts as the project’s stigmergic medium. Over time it is updated by:

- deposition,
- evaporation,
- and optional diffusion.

Conceptually:

`P_next = (1 - evaporation) * P + deposits + diffusion`

Several later tasks focused on making this field scientifically meaningful:

- prevent uncontrolled spread,
- tie deposition to useful task phases,
- and evaluate whether pheromone really improves scaling.

### 3.4 Training backends

The repository supports multiple learning backends:

- a custom PyTorch DQN path,
- SB3 DQN,
- RLlib DQN,
- and finally recurrent MAPPO with curriculum learning.

This matters for two reasons:

1. it provided baseline and comparison infrastructure,
2. it shows the project did not jump directly to its final form.

The system started with simpler value-based baselines and grew toward a stronger actor-critic MARL design as the scientific demands became clearer.

### 3.5 The final MAPPO path

The final main training path is implemented in:

- `train/mappo_gru.py`,
- `algorithms/mappo/networks.py`,
- `algorithms/mappo/curriculum.py`,
- `algorithms/mappo/inference.py`.

Key design choices:

- parameter-shared recurrent actor,
- centralized recurrent critic,
- fixed padded critic-state dimension across curriculum stages,
- greedy-eval-aware promotion and checkpointing,
- stage-specific entropy schedules,
- and stage-specific environment construction.

This design is the technical heart of the mature system.

## Chapter 4. The Research Method

The project’s development process was unusually explicit because the repository preserves a task archive. That means the project was not built as one large opaque implementation. It was built as a sequence of concrete hypotheses.

The method looked like this:

1. observe a failure in metrics or demo,
2. isolate the likely mechanism,
3. write a precise task that changes one part of the system,
4. implement and verify,
5. inspect new evidence,
6. use the new evidence to write the next task.

This is very similar to real research:

- formulate a diagnosis,
- intervene on the system,
- evaluate,
- then refine the theory.

That is why the development story reads like an adventure. Every improvement revealed a deeper obstacle behind it.

## Chapter 5. The Adventure Begins: From Skeleton to Experiment Platform

### 5.1 The opening act

The earliest commits built the project skeleton:

- the initial environment, config, DQN trainer, demo, and random rollout,
- several early architecture, documentation, and screenshot passes,
- several comment and readability commits in early February.

The first important architectural transition came when the environment was converted to a PettingZoo Parallel API native interface. This was a foundational compatibility move. It did not change the scientific question, but it made the codebase more structurally mature.

The next step was broadening the training stack:

- SB3 and RLlib backends were added alongside a training dispatcher,
- later commits refined demo and backend behavior.

At this stage, the repository was becoming a platform rather than just a script.

### 5.2 Tasks 00 to 09: establish the lab

The early task archive builds the project’s base research infrastructure.

`00_INITIAL.md` asked for the initial swarm RL repository skeleton.

Tasks `01` to `07` then escalated the codebase systematically:

- audit the existing repo,
- upgrade the environment for science-fair-grade stigmergic experiments,
- upgrade the training pipeline,
- add experiment infrastructure,
- add the main collective-intelligence scaling experiment,
- compare RL algorithms,
- and generate documentation.

Task `08` focused on API documentation, while task `09` added a policy probing script to inspect learned model behavior in isolation.

The corresponding commits show a pattern:

- environment expansion,
- experiment framework,
- evaluation infrastructure,
- documentation,
- and debugging tools.

Before the project could prove anything, it needed to become inspectable and reproducible.

## Chapter 6. Mission Control, Sim-to-Real Thinking, and Instrumentation

The task archive then turned briefly toward hardware-facing support and system instrumentation.

### 6.1 Mission control

Task `10` asked for a command center, later renamed `mission_control`.

The development sequence around that subsystem shows repeated refinement of the request and then implementation of a separate subsystem:

- thread-safe world state,
- digital pheromone rendering,
- transport interfaces,
- a PyGame view,
- and a fake robot harness.

This part of the story matters because it shows the project was thinking beyond pure simulation. Even when the main scientific arc later centered on simulation and curriculum-trained MAPPO, the system remained anchored to physical deployment concerns and observability.

### 6.2 Bluetooth path

Task `11` extended the hardware communication story through Bluetooth transport, again without collapsing subsystem boundaries. The repository’s tasks insist repeatedly on preserving low coupling, which is good systems thinking: deployment plumbing should not contaminate training logic.

### 6.3 Additional tasks and debug visibility

Task `12` reconstructed incremental development history after an earlier commit range. Task `13` added policy-debug logging. These are signs of a maturing lab notebook:

- not just building features,
- but making the build process legible.

That visibility became crucial later, when the main challenge was understanding why learned behavior looked good during training but failed under greedy evaluation.

## Chapter 7. The DQN Era and Its Limits

Before MAPPO became the main path, the project spent significant effort improving the custom DQN baseline and the evaluation pipeline.

### 7.1 Reward shaping and pheromone discipline

Tasks `14`, `15`, and `16` focused on the early environment and reward problems:

- improve detectability and reward signal,
- make pheromone deposition meaningful,
- encourage exploration,
- and fix concrete bugs in pheromone diffusion and food metrics.

These tasks taught several important lessons:

- a simulator can silently mislead you if its metrics are ambiguous,
- indirect communication only matters if the signal is semantically meaningful,
- and exploration incentives can easily dominate or distort the real task if not staged carefully.

### 7.2 Experiment and output infrastructure

Tasks `17`, `18`, `19`, `20`, `21`, and `22` focused on reproducibility and scale:

- better CLI/config control,
- recording comparison data,
- organizing output directories,
- fixing checkpoint layout,
- validating workflow end to end,
- and improving training efficiency.

By the time this phase ended, the repository had become a serious experimental platform:

- named runs,
- structured outputs,
- comparison conditions,
- fixed evaluation procedures,
- cleaner checkpoint organization,
- and better performance.

But a deeper algorithmic issue remained: the strongest question in the project demanded stronger multi-agent learning than the DQN path seemed likely to provide.

## Chapter 8. The Great Shift: Recurrent MAPPO and Curriculum Learning

Task `23` is one of the decisive turning points in the entire repository.

It asked for:

- recurrent MAPPO with GRU,
- curriculum learning,
- decentralized execution,
- and a staged path from one agent to many.

The first runnable recurrent MAPPO curriculum path was then implemented. This was a major expansion:

- new `algorithms/mappo/` modules,
- new training script `train/mappo_gru.py`,
- curriculum definition,
- inference path,
- and a dedicated training document.

This commit changed the project from “swarm RL platform with several baselines” into “swarm MARL project with a serious main research path.”

### 8.1 Task 24: shape the environment, not just the agent count

Task `24` recognized that simply increasing the number of agents was not a sufficient curriculum. The implementation added stage-specific environment construction so each curriculum stage could manipulate:

- world size,
- number of obstacles,
- number of targets,
- episode length,
- and other difficulty knobs.

This is a deep curriculum-learning insight:

> difficulty is not one-dimensional.

Teaching only by agent count would have been too crude.

### 8.2 Task 25: teach the full stigmergic loop

Task `25` insisted that the objective was not “touch the target” but the full stigmergic loop. The implementation altered training, environment behavior, and docs accordingly. This was the moment the repository’s training objective became fully aligned with the scientific story.

### 8.3 Task 26: make delivery explicit

Task `26` made the delivery mechanic explicit:

- one carried food item at a time,
- visible carrying state,
- delivery at the nest,
- pheromone on the return path.

This delivery upgrade is important because it transformed the environment from a vague collection task into a more physically intelligible foraging loop.

## Chapter 9. The Real Battle: Why the Policy Kept Failing

By task `27`, the project had the right algorithmic direction and the right task structure. Yet the learned policy still struggled. This is where the development history becomes most educational.

The failures were not random. They formed a ladder:

1. stage budgets were wrong,
2. critic continuity across stages was unstable,
3. sampled behavior did not survive into greedy behavior,
4. pickup improved before delivery,
5. homing worked in easy settings but failed under clutter,
6. single-agent skill collapsed when scaling to swarms,
7. later swarm stages developed nest-centered local optima.

This sequence is exactly why the repo’s later history is worth reading closely. It is a case study in diagnosing multi-agent RL failure modes one layer at a time.

## Chapter 10. Tasks 27 to 39: Building the Delivery Brain

This is the most important intellectual arc in the repository.

### 10.1 Task 27: budgets and pressure

Task `27` tightened curriculum budgeting and reward priorities. Its implementation reworked stage allocations and made delivery-oriented rewards more central.

Why it mattered:

- later stages finally received real budget,
- pickup was no longer treated as almost equivalent to delivery,
- and the curriculum became more honest.

### 10.2 Task 28: structural MAPPO stability

Task `28` attacked policy collapse. Its implementation heavily rewrote `train/mappo_gru.py`, including padded critic-state handling and stronger greedy-eval-aware progression logic.

This was a structural fix, not a reward tweak. It recognized that:

- actor continuity without critic continuity is unstable,
- stage transfer must preserve value-learning structure,
- and checkpoint quality must be judged by greedy behavior, not just sampled training episodes.

### 10.3 Task 29: make early greedy behavior real

Task `29` narrowed the problem further. If sampled behavior improved but greedy behavior still died, then the system had not truly learned stable policy logic. The implementation changed early single-agent stages:

- pheromone off,
- softer motion penalties,
- stronger pickup/delivery cues,
- within-stage entropy decay,
- sampled-versus-greedy reporting.

This produced the first strong sign that the pipeline could truly learn deterministic early success.

### 10.4 Task 30: carrying phase over exploration

Task `30` discovered that the learned policy could find food but often not bring it back. The implementation added carrying-aware exploration suppression and stronger nest-approach structure.

This is a subtle but central RL lesson:

> a reward that is good before pickup can become harmful after pickup.

The agent should not keep being rewarded to wander once it is carrying food.

### 10.5 Tasks 31 to 38: split the homing problem into lessons

These tasks gradually built the return stack:

- task `31`: obstacle-return bottleneck,
- task `32`: guaranteed post-pickup homing,
- task `33`: convert sampled homing into greedy homing,
- task `34`: enforce hard global step cap and punish pickup-without-delivery,
- task `35`: diagnose and penalize carrying-phase dithering,
- task `36`: force more deterministic greedy homing,
- task `37`: solve the remaining greedy-delivery failure,
- task `38`: preserve greedy delivery under mild clutter.

The curriculum stages that emerged from this sequence are among the clearest pieces of design in the repo:

- `stage1d_single_agent_carry_bootstrap`,
- `stage1e_single_agent_guaranteed_homing`,
- `stage1f_single_agent_delivery_bridge`,
- `stage1g_single_agent_delivery_obstacles`.

This is textbook curriculum design. A hard skill was decomposed into subskills with controlled geometry, spawn logic, clutter level, and reward pressure.

### 10.6 The bug that mattered most

Task `37` also exposed one of the most important bugs in the whole repo: movement updates could drop the `carrying_food` state during normal motion.

This is the kind of bug that makes RL feel mystical until it is found. The model looked incompetent, but part of the problem was that the environment was breaking the very state transition the policy needed in order to complete delivery.

This is one of the book’s strongest lessons:

> in RL, some “learning failures” are actually simulator-state bugs in disguise.

### 10.7 Task 39: stabilize the bridge stage

Task `39` found another subtle issue. A stage could discover a good greedy policy and then drift away from it before final evaluation. The implementation fixed stage-end restoration of the best within-stage model state.

This was the moment the single-agent return stack finally became stable enough for real full-run training.

## Chapter 11. Tasks 40 to 47: Scaling Up Without Collapsing

Once the single-agent delivery problem was largely solved, the next enemy appeared: scaling.

### 11.1 Tasks 40 and 41: the first swarm cliff

The project learned that the jump from one agent to several agents was not a minor extension. It was another curriculum wall.

Task `40` inserted small-swarm bootstrap stages. Task `41` reduced wasted budget on already-solved earlier stages so the trainer could reach those new lessons reliably.

The new stages included:

- `stage2a_small_swarm_carry_bootstrap`,
- `stage2b_small_swarm_delivery_easy`,
- `stage2c_small_swarm_medium`,
- `stage2d_small_swarm_large`.

This transformed the jump to swarm behavior from a cliff into a staircase.

### 11.2 Tasks 42 to 47: the nest-orbit dragon

After scaling became viable, the demos revealed a new pathology: empty agents clustering and circling near the nest.

This is a classic learned local optimum:

- the nest is behaviorally salient,
- the agents are no longer carrying food,
- but instead of leaving to search, they exploit a useless stable region.

The tasks then escalated logically:

- task `42`: reduce nest circling,
- task `43`: add post-delivery outward handoff,
- task `44`: hold that handoff until actual nest exit,
- task `45`: explicitly force empty agents to fan out and explore,
- task `46`: if shaping still fails, add environment-side force-explore mode,
- task `47`: add randomness only for non-carrying exploration.

This sequence is another miniature textbook. It moves from:

- negative shaping,
- to positive shaping,
- to mode-specific control,
- to direct environment-side intervention.

The repo learned that some local optima are too strong to solve with scalar rewards alone.

## Chapter 12. Task 48 and the Importance of Seeing Correctly

Task `48` may look cosmetic compared with the training tasks, but it matters. It fixed PyGame render scaling so large worlds could fit on screen without changing simulation coordinates.

Why this matters scientifically:

- if rendering lies, human diagnosis suffers,
- if screenshots distort the world, communication suffers,
- and if demo tooling is weak, it becomes harder to tell whether a policy is actually improving.

The follow-up fix to draw pheromone onto the off-screen world surface is a good reminder that visualization systems also need rigorous debugging.

## Chapter 13. The Final Curriculum and Why It Works

The mature `full` curriculum now contains thirteen stages:

1. `stage1a_single_agent_miniscule`
2. `stage1b_single_agent_tiny`
3. `stage1c_single_agent_small`
4. `stage1d_single_agent_carry_bootstrap`
5. `stage1e_single_agent_guaranteed_homing`
6. `stage1f_single_agent_delivery_bridge`
7. `stage1g_single_agent_delivery_obstacles`
8. `stage2a_small_swarm_carry_bootstrap`
9. `stage2b_small_swarm_delivery_easy`
10. `stage2c_small_swarm_medium`
11. `stage2d_small_swarm_large`
12. `stage3a_full_swarm_large`
13. `stage3b_full_swarm_final`

This works better than earlier versions because it respects task structure.

The curriculum teaches:

- basic single-agent competence,
- explicit post-pickup homing,
- clutter-robust return,
- small-swarm preservation of the learned loop,
- then full swarm coordination with pheromone-enabled behavior.

The mature system is not just a stronger model. It is a better lesson plan.

## Chapter 14. Experimental Results

The strongest integrated results are summarized in the integrated GVRSF experiment paper included with the repository.

### 14.1 Broad campaign

The broad campaign showed:

- stronger curriculum-trained MAPPO beating a weaker old checkpoint,
- MAPPO outperforming rule-based and random baselines,
- meaningful scaling with swarm size,
- robustness under harder environments.

Representative headline:

- current curriculum checkpoint: mean `food_delivered = 1.25`
- weaker older checkpoint: mean `food_delivered = 0.00`
- Welch’s `p = 0.000828`

That is important because it shows the curriculum work was not cosmetic. It changed actual task completion.

### 14.2 Killer stigmergy-scaling experiment

The stronger causal stigmergy result came from the matched repeated-source scaling experiment.

Primary `6`-agent result:

- pheromone-trained swarm with pheromone at eval: `1.32` mean deliveries,
- no-pheromone-trained swarm without pheromone: `0.00`,
- paired t-test `p = 0.00653`,
- Wilcoxon `p = 0.00364`.

Late-episode deliveries were also significantly higher for the pheromone-trained condition.

This matters because broad scaling alone can be dismissed as “more robots do more work.” The killer experiment was designed specifically to test whether stigmergy contributes real scaling benefit under matched conditions.

## Chapter 15. What the Project Teaches

The deepest lessons of this repository are not limited to swarm robotics.

### 15.1 RL systems fail in layers

A poor result may come from:

- wrong objective structure,
- wrong training budget,
- unstable value learning,
- ambiguous metrics,
- hidden environment bugs,
- or a real policy failure.

The project improved because it did not treat all failure as “train longer.”

### 15.2 Good curriculum learning is a form of software design

The curriculum did not emerge automatically. It had to be engineered:

- stage geometry,
- stage difficulty,
- entropy schedule,
- promotion criteria,
- checkpoint logic,
- and environment shaping all mattered.

### 15.3 Evaluation must be greedy and mechanism-aware

Sampled training success can lie. This repo repeatedly found that behavior which looked alive under stochastic rollouts collapsed under greedy evaluation.

That is why:

- stage-end greedy evaluation,
- delivery conversion metrics,
- sampled-vs-greedy gap reporting,
- and best-greedy checkpoints

became central to the mature system.

### 15.4 Indirect communication is powerful but easy to fake badly

Pheromone-like communication only means something if:

- deposition is semantically tied to success,
- sensing affects behavior usefully,
- and experiments separate “more agents” from “better coordination.”

The project’s later experiment design shows strong awareness of this distinction.

## Chapter 16. Task-by-Task Chronicle

This appendix walks through the task archive in order. Each task is summarized as:

- its intent,
- the main implementation direction,
- and the development lesson it contributed.

| Task | Intent | Main diff / implementation story | Development lesson |
|---|---|---|---|
| 00 | Initialize the swarm RL repo | Created the skeleton: env, config, demo, DQN trainer, docs structure | Start with a runnable core |
| 01 | Audit before editing | Established repo-aware incremental work | Diagnosis before intervention |
| 02 | Upgrade environment for science-fair stigmergy | Expanded simulator capability and experiment realism | Better science needs better simulation |
| 03 | Upgrade RL training pipeline | Added stronger training/eval infrastructure | Training code must be inspectable |
| 04 | Add reusable experiment framework | Added experiment running and aggregation structure | Reproducibility is part of the product |
| 05 | Build the main scaling experiment | Added flagship collective-intelligence evaluation path | Hypotheses need dedicated experiments |
| 06 | Compare algorithms | Extended experiments across control strategies | Baselines matter |
| 07 | Generate documentation | Produced user-facing system docs | A science project must explain itself |
| 08 | Generate API docs from real code | Forced documentation to track implementation reality | Documentation must be grounded |
| 09 | Add policy probing script | Added direct model inspection tooling | Visibility prevents magical thinking |
| 10 | Build command center / mission control | Added desktop physical-swarm visualization subsystem | Sim-to-real observability matters |
| 11 | Add Bluetooth transport | Extended mission-control communication path | Keep deployment layers modular |
| 12 | Reconstruct incremental tasks | Documented missing development history | Preserve the lab notebook |
| 13 | Add debug logging | Logged per-agent outputs and actions | Fine-grained visibility speeds diagnosis |
| 14 | Improve reward signal | Increased food detectability and intermediate shaping | Sparse tasks often need shaping |
| 15 | Restrict pheromone deposition and reward exploration | Made pheromone more meaningful and exploration measurable | Communication signals need semantics |
| 16 | Fix diffusion and metric bugs | Prevented infinite-like spread and clarified pickup vs delivery metrics | Bad metrics create false conclusions |
| 17 | Add output/config arguments | Better run naming, pheromone toggles, fixed-time eval | Experimental control matters |
| 18 | Record comparison data | Added random baseline and stricter metric semantics | Comparisons need consistent data |
| 19 | Organize comparison outputs | Cleaned experiment directory layout | Analysis should not fight file chaos |
| 20 | Fix checkpoint folder naming | Cleaned checkpoint hierarchy | Output structure affects usability |
| 21 | Validate new checkpoint layout | End-to-end workflow testing | Infrastructure deserves verification |
| 22 | Improve DQN training efficiency | Reduced performance bottlenecks | Speed improves iteration quality |
| 23 | Add recurrent MAPPO curriculum path | Introduced GRU MAPPO, curriculum, new trainer | This was the algorithmic pivot |
| 24 | Build stage-specific environments | Curriculum now changed worlds, not just swarm size | Difficulty is multi-dimensional |
| 25 | Teach full stigmergic trail loop | Aligned training with the true task loop | Objectives must match claims |
| 26 | Upgrade explicit delivery mechanics | Added carrying state, delivery, visible loop closure | Task semantics must be concrete |
| 27 | Improve training budget and config | Fixed stage budgeting and delivery pressure | Curriculum budget is a real algorithmic variable |
| 28 | Fix MAPPO policy collapse | Added critic-state padding and stronger greedy-aware promotion | Structural stability beats blind reward tuning |
| 29 | Fix sampled-to-greedy early behavior | Early stages now teach deterministic pickup/return better | Sampled success is not enough |
| 30 | Fix return-to-nest failure | Suppressed exploration reward while carrying | Reward should follow task phase |
| 31 | Fix obstacle-return delivery | Tightened delivery-sensitive progression in harder early stages | Mild clutter is a real barrier |
| 32 | Add guaranteed post-pickup homing | Controlled target distance/spawn to isolate homing | Isolate the subskill before generalizing it |
| 33 | Convert sampled homing to greedy homing | Sharper entropy and scoring discipline | Determinism must be trained explicitly |
| 34 | Fix pickup-without-delivery and hard cap | Respected total steps and penalized fake progress | Honest budgets and honest scoring matter |
| 35 | Fix carrying-phase control | Added anti-dithering penalties and diagnostics | If behavior stalls, instrument the stall |
| 36 | Force greedy homing | Lowered entropy and strengthened sustained homing pressure | Carrying mode needed clearer control pressure |
| 37 | Solve greedy delivery | Added carry bootstrap and fixed a carrying-state bug | Some training failures are simulator bugs |
| 38 | Preserve delivery under mild clutter | Strengthened bridge-stage continuity | The bridge between easy and hard often decides success |
| 39 | Stabilize bridge-stage greedy delivery | Restored best within-stage model before final eval | Preserve discovered competence |
| 40 | Preserve delivery through first swarm scale-up | Inserted small-swarm bootstrap stages | Scaling must be taught, not assumed |
| 41 | Reach swarm bootstrap reliably | Reduced wasted repeats in already-solved stages | Curriculum time is finite |
| 42 | Reduce nest circling | Added penalties and shaping around empty-agent nest behavior | New capabilities reveal new local optima |
| 43 | Reduce post-delivery orbit | Added post-delivery outward-search handoff | Delivery must hand off to exploration cleanly |
| 44 | Hold exit until nest is actually left | Persisted post-delivery leave-nest mode | Temporary fixes fail if the agent never exits the trap |
| 45 | Force non-carrying fan-out | Added explicit outward exploration shaping | Reward “go do the right thing,” not just “don’t do the wrong thing” |
| 46 | Break nest orbit decisively | Added env-side force-explore mode | Sometimes behavior constraints beat soft shaping |
| 47 | Add exploration randomness for empty agents | Injected targeted randomness only when not carrying | Exploration noise should be phase-specific |
| 48 | Fix PyGame render scaling | Added off-screen world rendering with scaled display | Good visualization is part of rigorous debugging |

## Chapter 17. Milestone-by-Milestone Development Chronicle

This appendix compresses the repository history into reader-friendly milestones instead of raw revision identifiers.

| Milestone | What changed relative to the previous stage |
|---|---|
| Initial repository build | Created the first runnable environment, config, DQN trainer, demo, and random rollout scripts. This is the project’s foundation. |
| Early documentation and screenshots | Added architecture notes, walkthrough material, screenshot support, and clearer README explanations. |
| Readability pass | Added comments and cleanup in the simulator, demo, and DQN trainer to make the code easier to inspect. |
| PettingZoo API transition | Converted the environment and surrounding scripts to a Parallel API style with dict-based observations and actions. This was a major architectural milestone. |
| Multi-backend training support | Added SB3 and RLlib DQN backends plus a training dispatcher, then refined imports, demo behavior, and backend integration. |
| Integration-task expansion | Added the first audited task artifacts and used them to drive large environment and training upgrades, including evaluation utilities and experiment helpers. |
| Experiment framework phase | Added reusable experiment-running infrastructure and expanded it to support collective-intelligence scaling and RL algorithm comparisons. |
| Documentation and API phase | Built task-driven user docs, API docs, and policy-probing tools so the system could be understood as well as run. |
| Mission-control phase | Refined the command-center concept, renamed it to `mission_control`, and implemented the subsystem for physical-swarm visualization and transport handling. |
| Instrumentation and history preservation | Added observation history, training graph outputs, reconstructed missing task history, and introduced policy-debug logging. |
| Reward and pheromone repair phase | Reworked reward shaping, pheromone semantics, and several concrete simulator bugs, including diffusion and metric inconsistencies. |
| Experiment-output hardening | Improved run naming, evaluation recording, comparison output layout, checkpoint organization, validation tasks, and training efficiency. |
| Late DQN-era refinements | Improved performance, repeat-step control, observation design, and sim-to-real alignment while learning the limits of the DQN path. |
| Recurrent MAPPO introduction | Added the first runnable GRU-based recurrent MAPPO curriculum path, including new algorithms, trainer, inference path, and documentation. |
| Early MAPPO stabilization | Cleaned up merge issues, refined demo support, improved collisions and reward settings, and added control flags around the new training path. |
| Stage-specific curriculum environments | Implemented curriculum stages that changed world difficulty, geometry, and clutter rather than only changing swarm size. |
| Trail-learning alignment | Reworked training to teach the full stigmergic trail loop rather than just target contact. |
| Explicit delivery mechanics | Added carrying state, target delivery semantics, and clearer delivery-focused environment behavior. |
| Pre-collapse MAPPO tuning | Iterated on curriculum settings, obstacle layouts, demo overrides, and model tuning before the main task-loop debugging phase. |
| Training-improvement phase | Added stronger curriculum budgeting and training configuration so later stages received real learning time and delivery carried more weight. |
| Policy-collapse repair | Restructured the MAPPO trainer to preserve critic continuity, improve stage transfer, and make greedy evaluation central to promotion and checkpointing. |
| Greedy-behavior repair | Adjusted early single-agent lessons so useful behavior survived greedy execution instead of only appearing under training-time randomness. |
| Return-to-nest repair | Added carrying-aware shaping and curriculum changes so pickup behavior could turn into actual delivery behavior. |
| Return-stack decomposition | Split homing and delivery into multiple staged lessons, then kept refining those lessons under clutter and bridge-stage conditions. |
| Carrying-state bug fix | Fixed the crucial simulator bug where movement updates could silently drop the carrying-food state during normal motion. |
| Bridge-stage stabilization | Restored the best within-stage model state before final stage evaluation so discovered good behavior was not lost before promotion. |
| Small-swarm scale-up | Added small-swarm bootstrap stages and reduced wasted repeats in earlier stages so the curriculum could reach true multi-agent learning reliably. |
| Nest-orbit mitigation | Applied tasks 42 through 45 to reduce empty-agent circling near the nest through penalties, post-delivery handoff, and explicit fan-out shaping. |
| Force-explore intervention | Added an environment-side force-explore mode because soft shaping alone was still not strong enough to break the local optimum. |
| Exploration-randomness phase | Added targeted randomness for non-carrying exploration only, preserving carrying-food return behavior. |
| Render-scaling and demo polish | Fixed PyGame scaling, then improved the demo with seed display, HUD, seed cycling, pause/resume, and click-to-inspect panels. |
| Retrospective and presentation docs | Added the curriculum-building retrospective, refreshed current-facing docs, imported selected GVRSF experiment bundles, and built the presentation materials. |

## Chapter 18. Conclusion

The project began as a swarm RL sandbox. It became:

- a multi-backend swarm-learning platform,
- a curriculum-trained recurrent MAPPO system,
- an experiment suite for testing collective intelligence,
- a sim-to-real-aware robotics project,
- and a detailed case study in how real RL systems are debugged.

The final scientific claim is not that the swarm moves impressively. The stronger claim is that under controlled experiments:

- curriculum learning materially improved the decentralized policy,
- the learned policy beat weaker and simpler baselines,
- performance scaled with swarm size,
- and pheromone-mediated stigmergy provided evidence of real coordination benefit.

The final engineering claim is equally important:

> progress came from disciplined iteration, not from one clever trick.

The archived tasks and git diffs show the project learning how to teach the swarm. In that sense, the adventure story has two protagonists:

- the agents inside the simulator,
- and the development process that gradually learned how to train them.
