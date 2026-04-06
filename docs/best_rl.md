# Best-of-Breed RL Upgrade Plan

This document outlines a practical path from the current DQN-based baseline to a stronger existing RL approach for this project: a partially observable, cooperative, homogeneous multi-robot swarm with decentralized execution and a real deployment target.

## 1. Target End State

The strongest practical target for this repo is:

- **Primary training approach:** parameter-shared **recurrent MAPPO** with **centralized training and decentralized execution (CTDE)**
- **Primary deployment policy:** one shared recurrent actor running on each robot
- **Primary comparison baseline:** a strengthened value-based baseline, not the current plain DQN

Why this target fits the problem:

- The task is **cooperative**, not competitive.
- Each robot has only **local partial observations**.
- The robots are **homogeneous**, so parameter sharing is natural.
- Deployment needs **local execution** on the robot, so a centralized planner is the wrong inference model.
- The environment already exposes enough structure that a centralized critic can be added during training without changing the runtime policy interface.

If only one algorithm family is taken to a high standard, it should be **recurrent MAPPO**. For this type of task, that is a stronger and more standard modern choice than plain DQN.

## 2. Recommended Algorithm Stack

### Tier 1: Strong baseline before changing paradigm

Before replacing the current trainer, upgrade the value-based baseline to:

- Double DQN
- Dueling network
- Prioritized replay
- N-step returns
- Noisy networks or a carefully tuned epsilon schedule
- Distributional Q-learning such as QR-DQN

This gives a much stronger baseline than the current plain DQN and helps separate "algorithm family improvement" from "basic implementation improvement."

### Tier 2: Main target algorithm

Implement **recurrent MAPPO** with:

- shared actor weights across all agents
- centralized critic during training
- local observations for the actor
- global state or concatenated multi-agent features for the critic
- GRU or LSTM memory in the actor and critic
- PPO clipping, entropy bonus, GAE, value-loss clipping, advantage normalization

This should become the main research/deployment path once it outperforms the improved DQN baseline.

### Tier 3: Secondary comparison algorithm

Add one strong cooperative value-decomposition baseline:

- **QMIX** if a usable centralized state is available
- **QPLEX** if more expressive mixing is desired later

This gives a clean comparison between:

- value-based decentralized control
- CTDE actor-critic control

## 3. Why the Current Repo Is Not There Yet

The current repo is a reasonable DQN baseline, but it is not yet best-of-breed because it lacks:

- recurrence for partial observability
- centralized training structure
- stronger value-learning improvements
- disciplined benchmark comparisons across algorithms
- stronger experiment reproducibility and ablation structure

Frame stacking helps, but it is still weaker than a recurrent policy for this problem.

## 4. Phase Plan

### Phase A: Lock down benchmark discipline

Before changing algorithms, define the benchmark clearly.

Required work:

- freeze 2 to 4 benchmark environment configurations
- freeze evaluation seeds and episode counts
- define the primary metrics:
  - food retrieved
  - delivery rate
  - exploration coverage
  - collision rate
  - pheromone usage efficiency
  - sample efficiency
  - inference latency on Pi hardware
- define success criteria for research evaluation and deployment runs
- add run manifests so every experiment records:
  - code revision
  - config
  - checkpoint
  - seed
  - metrics

Why:

- Without this, "better RL" will be hard to prove.

### Phase B: Make the current baseline strong

Upgrade the current custom trainer before introducing a new family.

Recommended order:

1. Double DQN
2. Dueling head
3. N-step returns
4. Prioritized replay
5. Distributional Q-learning
6. Noisy exploration or better exploration schedule management

Expected result:

- a much more defensible DQN-family baseline
- better sample efficiency
- better stability
- a stronger comparison point for MAPPO/QMIX later

### Phase C: Add recurrent observation support cleanly

Move from ad hoc frame stacking to explicit recurrent training support.

Required work:

- define sequence-based replay/rollout storage
- add episode masks and hidden-state resets
- keep deployment inference API simple:
  - local observation in
  - hidden state in/out
  - action out
- document exactly how hidden state resets on episode start and robot reconnect

Expected result:

- a cleaner foundation for recurrent policies
- less dependence on manually chosen history-window size

### Phase D: Implement recurrent MAPPO

This is the main algorithmic step.

Actor:

- shared across robots
- consumes local observation history or current observation plus recurrent hidden state
- outputs action logits over the 18-action space

Critic:

- centralized during training only
- consumes richer global information such as:
  - all agents' observations
  - all agents' positions/headings if available
  - pheromone-field summary
  - food/nest/task state

Training details:

- PPO clipping
- GAE
- entropy regularization
- advantage normalization
- gradient clipping
- mini-batch sequence training

Expected result:

- better handling of partial observability
- better cooperative behavior
- smoother policy improvement than plain DQN

### Phase E: Add QMIX as the cooperative value baseline

If the environment can expose a stable global state for training, add QMIX.

Why:

- It is a standard cooperative MARL baseline.
- It helps answer whether the task benefits more from:
  - actor-critic CTDE
  - or value decomposition

Expected result:

- stronger research credibility
- cleaner comparison against MAPPO

### Phase F: Sim-to-real hardening

Once the training algorithm is strong, improve transfer quality.

Required work:

- domain randomization for:
  - lidar noise
  - heading drift
  - wheel slip
  - action latency
  - communication delay
  - pheromone sampling noise
- randomized robot geometry and motion calibration bounds
- curriculum over obstacle density, target placement, and swarm size
- train/eval splits that include transfer-stress scenarios

Expected result:

- policies that survive deployment better
- fewer Mission Control/runtime surprises

## 5. Architecture Changes Needed

The repo should evolve toward these training abstractions:

- `algorithms/dqn/`
- `algorithms/mappo/`
- `algorithms/qmix/`
- shared rollout storage utilities
- shared evaluation harness
- shared experiment config schema

Key design rule:

- keep `firmware/` independent from training internals
- keep deployment consuming only:
  - checkpoint
  - observation contract
  - action contract
  - optional recurrent hidden state

That preserves the current good property of low coupling between training code and robot runtime.

## 6. Environment Improvements Needed

To support stronger MARL methods well, the environment should expose:

- a clear **local observation** function for decentralized actors
- a clear **global state** function for centralized critics or mixers
- consistent action masking if some actions are invalid
- more explicit task-level metrics per episode
- deterministic evaluation mode separate from stochastic training mode

The environment does not need to change its runtime-facing observation contract for deployment, but training should gain a cleaner centralized-state view.

## 7. Experiment Standard

To claim "best existing RL approach implemented well," the repo should satisfy this minimum standard:

- compare at least:
  - improved DQN baseline
  - recurrent MAPPO
  - QMIX
- run multiple seeds
- report mean and variance
- keep benchmark configs fixed
- track sample efficiency and final asymptotic performance
- include ablations for:
  - recurrence
  - centralized critic
  - pheromone observation
  - explicit deposit action
  - domain randomization

Without ablations, it will be hard to say which change actually helped.

## 8. Recommended Order of Execution

The most efficient order is:

1. Strengthen the current DQN baseline
2. Clean up benchmark/evaluation discipline
3. Add recurrent rollout infrastructure
4. Implement recurrent MAPPO
5. Add QMIX for comparison
6. Add transfer/domain-randomization hardening
7. Pick the best performer for deployment

This order avoids jumping to a new algorithm before the baseline and evaluation stack are trustworthy.

## 9. What "Best-of-Breed" Should Mean Here

For this repo, "best-of-breed existing RL approach" should mean:

- not inventing a novel algorithm
- implementing the strongest established family that fits the task constraints
- evaluating it rigorously against strong baselines
- preserving decentralized robot execution
- making the result deployable on the Pi

That points most strongly to:

- **final target:** recurrent MAPPO with parameter sharing and CTDE
- **best value-based baseline:** improved distributional Double/Dueling DQN with prioritized replay and n-step returns
- **best cooperative comparison:** QMIX

## 10. Practical Recommendation

If time is limited, do not try to build every MARL paper baseline.

The highest-value path is:

1. upgrade the current DQN into a strong Rainbow-style baseline
2. implement recurrent MAPPO well
3. compare those two carefully
4. deploy whichever wins under both simulator metrics and Pi runtime constraints

That is the most credible path to a strong research result and a strong deployment-ready system without turning the repo into an unfocused benchmark zoo.
