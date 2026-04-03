# Prompt — Upgrade This Repo to a Best-of-Breed Existing RL Stack

You are working in the existing repository and must upgrade the RL/training side of the project from a practical DQN baseline into a stronger, more modern, better-structured existing RL stack for a **cooperative, partially observable, homogeneous multi-robot swarm** problem with **decentralized execution** on physical robots.

This prompt is based on the plan in [docs/best_rl.md](../best_rl.md). Follow that direction, but implement the work in a repo-aware, incremental, well-documented way.

## High-level goal

Turn the project from:

- a plain value-based DQN-style training baseline with some recent improvements

into:

- a **stronger DQN-family baseline**
- a **parameter-shared recurrent MAPPO implementation** with **centralized training and decentralized execution (CTDE)**
- and a **cooperative comparison baseline** such as **QMIX** if the environment can support a stable centralized training state

The end result should be a repo that is much stronger for:

- RL learning
- controlled experimentation
- sim-to-real transfer
- and science-fair-quality demonstration

without breaking the current deployment architecture.

---

## Core recommendation you must follow

The **main target algorithm** is:

> **parameter-shared recurrent MAPPO with centralized training and decentralized execution**

The current custom DQN trainer should still remain in the repo, but it must be upgraded into a stronger baseline before or alongside MAPPO so comparisons are fair.

If time and code complexity allow, also add:

> **QMIX** as a cooperative MARL comparison baseline

Do **not** invent a novel RL algorithm. Use strong, established existing methods and implement them well.

---

## Repo-specific problem framing

This repo already has the important ingredients:

- a multi-agent swarm environment in `env/`
- current DQN-family training code in `train/`
- model definitions in `models/`
- a physical robot runtime in `firmware/`
- Mission Control / desktop command center integration
- an existing action/observation contract already used by deployment

The upgraded RL stack must respect those realities.

Important implications:

- The training code may change significantly.
- The deployment/runtime interface should change only when there is a strong reason.
- `firmware/` must remain **independent from training internals**.
- Deployment should remain **decentralized**: each robot runs its own local actor.

---

## Architectural constraints

1. **Do not break or bloat the current firmware runtime.**

   - Do not make `firmware/` depend on training internals.
   - Do not make `firmware/` import `algorithms/` modules directly unless absolutely necessary for checkpoint loading or minimal inference helpers.
   - Keep the deployment-facing contract compact and stable.

2. **Preserve decentralized execution.**

   - Even if training uses centralized critics or joint state, robot inference must still be local.

3. **Keep code modular.**

   The repo should evolve toward a structure along the lines of:

   ```text
   algorithms/
     dqn/
     mappo/
     qmix/
     common/
   train/
   models/
   env/
   firmware/
   ```

   You may adapt naming, but preserve separation of concerns.

4. **Do not remove the current custom DQN path.**

   - Upgrade it.
   - Keep it runnable as a baseline.

5. **Do not make this a benchmark zoo.**

   - Focus on a few strong algorithms implemented well.
   - Prioritize correctness, reproducibility, and deployment relevance over adding many methods.

---

## Required algorithm roadmap

Implement the following in a sensible phased order.

### Phase 1: Strengthen the current value-based baseline

Upgrade the existing DQN-family implementation to include, where appropriate:

- Double DQN
- Dueling network heads
- N-step returns
- Prioritized replay
- Better exploration support
- Preferably a distributional variant such as QR-DQN if feasible

This upgraded DQN baseline must remain easy to train from the CLI and must become the repo’s new strong baseline for comparison.

### Phase 2: Add recurrent sequence support

The current repo has frame stacking, but that is not enough.

Add clean recurrent support for:

- sequence-aware rollout storage
- hidden-state reset handling
- episode masks
- batched recurrent training
- recurrent inference helpers

This should support a future recurrent actor and critic cleanly.

### Phase 3: Implement recurrent MAPPO

Implement a proper recurrent MAPPO training path with:

- parameter sharing across homogeneous agents
- centralized critic during training
- decentralized actor during inference
- PPO clipping
- GAE
- entropy bonus
- value loss
- gradient clipping
- advantage normalization
- recurrent hidden state support

The actor should use local observations only at inference time.

The critic may use a centralized state or richer joint information at training time.

### Phase 4: Add QMIX if the environment can support it cleanly

If the environment can expose a suitable centralized state, add:

- QMIX as a cooperative comparison baseline

If that turns out to be too invasive or low-value relative to MAPPO, document the limitation clearly and avoid forcing a poor implementation.

---

## Environment and interface requirements

The environment must be improved so stronger MARL methods are well supported.

Add or formalize the following:

1. **Local observation path**

   - the actor should have a clearly defined local observation interface

2. **Centralized state path**

   - the critic / mixer should have a clearly defined training-time state interface
   - this must be documented separately from the deployment observation

3. **Evaluation mode**

   - deterministic evaluation behavior
   - fixed seed support
   - benchmark-ready metrics

4. **Episode/task metrics**

   At minimum track:

   - episode reward
   - food retrieved
   - delivery efficiency
   - exploration coverage
   - collision count/rate
   - pheromone usage
   - episode length
   - sample efficiency

5. **Action semantics**

   - preserve the existing 18-action semantics unless a change is strongly justified
   - if action semantics do change, explain why and update every affected doc/runtime path carefully

---

## Experiment discipline requirements

You must improve experiment quality, not just algorithm complexity.

Implement or improve:

- fixed benchmark environment configs
- multi-seed evaluation
- mean/variance reporting
- reproducible run manifests
- consistent checkpoint metadata
- stronger evaluation scripts
- fair comparison across algorithms

Add at least a small benchmark protocol that answers:

- does improved DQN beat the old baseline?
- does recurrent MAPPO beat improved DQN?
- does QMIX help relative to MAPPO or not?

Do not rely on one cherry-picked run.

---

## Sim-to-real requirements

The upgraded RL stack must remain relevant to deployment.

Improve training-time transfer support with:

- lidar noise randomization
- heading drift / pose noise
- wheel slip / motion uncertainty
- control latency
- communication delay
- pheromone-sensing uncertainty
- randomized geometry / calibration bounds where appropriate

Do not turn this into a full robotics simulator rewrite, but add the domain-randomization pieces that most directly support Mission Control + physical robot transfer.

---

## Firmware and deployment constraints

The physical robot path must remain usable.

Requirements:

- preserve decentralized inference
- keep the robot-side runtime interface understandable
- if recurrent inference is added, provide a minimal hidden-state handling path for deployment
- document exactly how hidden state is initialized and reset
- do not entangle `firmware/` with training-only code

If MAPPO becomes the recommended deployment path, provide the minimal loading/inference support needed for the robot runtime without turning `firmware/` into a training package.

---

## Documentation requirements

You must update documentation as part of the work.

Add or update docs covering:

1. the new RL architecture
2. the algorithm choices and why they fit this problem
3. benchmark/evaluation procedure
4. centralized training vs decentralized execution
5. deployment implications for recurrent policies
6. how to train each supported algorithm from the CLI
7. how to compare results fairly

Also update:

- `docs/PROJECT_LOG.md`
- `docs/QandA.md` when user-facing codebase questions are naturally answered by the work

---

## Deliverables

When implementing this prompt later, the repo should end up with:

1. **A stronger DQN-family baseline**

   - clearly better than the current plain baseline in implementation quality

2. **A working recurrent MAPPO training path**

   - parameter shared
   - CTDE
   - deployable actor interface

3. **Optionally QMIX**

   - if supported cleanly by the environment

4. **Better evaluation tooling**

   - benchmark runs
   - multi-seed summaries
   - result artifacts

5. **Clear documentation**

   - architecture
   - training commands
   - deployment implications
   - benchmark methodology

6. **A concise summary of what changed and why**

---

## Implementation priorities

Prioritize in this order:

1. correctness
2. benchmark discipline
3. modular architecture
4. strong recurrent MAPPO implementation
5. stronger DQN baseline
6. sim-to-real relevance
7. optional QMIX comparison

If tradeoffs are needed, prefer:

- one excellent MAPPO implementation over several weak algorithms
- one strong DQN baseline over many half-finished value methods
- clean evaluation over flashy but unreliable results

---

## Non-goals

Do **not**:

- invent a novel RL algorithm
- replace the project with an end-to-end robotics stack rewrite
- make Mission Control depend on training internals
- turn firmware into a monolithic training/runtime hybrid
- add many weak baselines just to look comprehensive

---

## Definition of done

This task is complete when:

- the repo has a much stronger DQN-family baseline
- the repo has a working recurrent MAPPO implementation with CTDE
- benchmark/evaluation discipline is clearly improved
- results can be compared across algorithms in a fair, documented way
- deployment remains decentralized and compatible with the robot runtime model
- documentation clearly explains the architecture and usage

When done, explain:

- what algorithm became the recommended primary path and why
- whether QMIX was added or intentionally omitted
- what central-state definition was used for training
- what changed in deployment, if anything
- what tradeoffs remain
