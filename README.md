# Artificial Collective Intelligence: Carpenter Ant-Inspired Stigmergic Swarm Robotics for Decentralized Systems Using Deep Reinforcement Learning

Individual ants possess limited cognitive capacity. Yet when thousands interact, colonies exhibit coordinated collective intelligence. This intelligence is not centralized in a single agent but emerges from decentralized interactions shaped by evolutionary processes. A key coordination mechanism is stigmergy—indirect communication through environmental modification (e.g., pheromone trails) that agents can sense and exploit.

Inspired by carpenter ants, this research investigates artificial collective intelligence arising from decentralized interactions with the environment, developing a computational framework for swarm robotics that contrasts with prevailing models of intelligence relying on large-scale centralized computing. To enable scalable experimentation, a custom multi-agent reinforcement learning simulation environment was created for decentralized policy learning. A physical swarm robotics platform was then constructed to evaluate the learned behaviours in the real world.

By coupling learned behaviours with shared “digital pheromone” fields, the system demonstrates how collective intelligence emerges from distributed agents, contributing to the emerging field of Physical AI, where intelligent algorithms interact directly with and control physical systems rather than operating only in digital environments. The robotic platform used to evaluate the system provides an accessible testbed for future swarm robotics research and is fully open-source, including algorithms, models, mechanical designs, firmware, component specifications, and assembly documentation. 

Decentralized swarm systems have applications in environments where communication infrastructure is unreliable or centralized control is fragile, such as planetary exploration and disaster response. By leveraging local decision-making and redundancy, stigmergic swarm systems provide resilience, adaptability, and robustness under uncertainty.

## Documentation Guide

This repo has a large `docs/` tree. Use this table of contents to jump to the right document quickly.

### Core Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
  - High-level system structure. Use this first if you want to understand how environment, training, evaluation, and deployment pieces fit together.
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
  - Implementation-grounded API reference for the environment and surrounding tooling. Best when you need exact interfaces and object behavior.
- [docs/CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md)
  - Reference for `SwarmConfig` and the shared CLI-to-config mapping. Use this when changing environment or reward settings.
- [docs/TRAINING_MAPPO.md](docs/TRAINING_MAPPO.md)
  - The current main training-path document. Explains the recurrent MAPPO curriculum, stage design, recommended commands, and checkpoint usage.
- [docs/GRUMMAPO.md](docs/GRUMMAPO.md)
  - Introductory, implementation-grounded guide to GRU MAPPO in this repo. Best when you want both intuition and a code walkthrough.
- [docs/TRAINING_CUSTOM.md](docs/TRAINING_CUSTOM.md)
  - The older custom DQN training path. Useful if you want the baseline trainer rather than the current MAPPO path.
- [docs/EVALUATE.md](docs/EVALUATE.md)
  - Practical evaluation guide for single-checkpoint and comparison runs. Use this for `analysis/evaluate.py` and `analysis/evaluate_comparison.py`.
- [docs/EVALUATION_CUSTOM.md](docs/EVALUATION_CUSTOM.md)
  - Older evaluation notes focused on the custom DQN workflow. Useful when working with legacy baseline runs.
- [docs/EXPERIMENT_API.md](docs/EXPERIMENT_API.md)
  - Reference for the experiment framework and benchmark entry points. Use this when running repeatable sweeps.
- [docs/ONBOARDING.md](docs/ONBOARDING.md)
  - Main “how do I run this repo?” guide. Best starting point for a new developer who wants current commands and key files.
- [docs/QandA.md](docs/QandA.md)
  - Rolling question-and-answer file. Newer entries are current; older lower entries are historical context.
- [docs/PROJECT_LOG.md](docs/PROJECT_LOG.md)
  - Chronological change log and work journal. Useful for understanding why the current system looks the way it does.
- [docs/curriculum_training.md](docs/curriculum_training.md)
  - Walkthrough of training in general and the curriculum path specifically. Good for understanding the control flow in code.
- [docs/trail.md](docs/trail.md)
  - Conceptual note on the intended trail behavior: discover, return, deposit, and exploit. Use this for the high-level research idea.
- [docs/best_rl.md](docs/best_rl.md)
  - Higher-level RL strategy discussion. Useful if you want to compare the current stack with stronger alternatives.
- [docs/CTDE_STATE.md](docs/CTDE_STATE.md)
  - Notes on centralized-training/decentralized-execution state design. Helpful when reasoning about the MAPPO critic state.
- [docs/OBSERVATION_SPEC.md](docs/OBSERVATION_SPEC.md)
  - Observation layout reference. Use this if you need the exact policy input contract.
- [docs/ACTION_SPEC.md](docs/ACTION_SPEC.md)
  - Action-space reference. Use this when checking how discrete action ids map to simulator behavior.
- [docs/UML.md](docs/UML.md)
  - UML-style diagrams for the environment and system structure. Useful when you want a visual mental model.
- [docs/DQN_EXPLAINED.md](docs/DQN_EXPLAINED.md)
  - Plain-language DQN walkthrough grounded in this repo’s code. Best for learning the baseline trainer.
- [docs/RESOURCES.md](docs/RESOURCES.md)
  - External learning and reference links. Useful when you want supporting RL or framework material.
- [docs/audit.md](docs/audit.md)
  - Historical review and audit-style notes. This is more for project history than day-to-day use.
- [docs/ToDo.md](docs/ToDo.md)
  - Open tasks and future work ideas. Use it as a backlog rather than a source of truth for current behavior.

### Deployment And Hardware

- [docs/SimToReal.md](docs/SimToReal.md)
  - Sim-to-real deployment design notes. Useful when mapping the simulator policy contract onto a physical robot.
- [docs/PI_MIGRATION.md](docs/PI_MIGRATION.md)
  - Raspberry Pi migration notes. Best when moving from the preserved rule-based runtime toward model-driven control.
- [docs/BLUETOOTH_FIRMWARE_NOTES.md](docs/BLUETOOTH_FIRMWARE_NOTES.md)
  - Notes on the BLE / firmware-side communication path. Useful for the live robot stack.
- [docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md)
  - Mission Control system overview. Use this when working on desktop-side multi-robot coordination infrastructure.
- [docs/CAMERA_SETUP.md](docs/CAMERA_SETUP.md)
  - Practical Raspberry Pi camera bring-up and recovery notes, including Picamera2 setup, dependency fixes, tested HSV bounds, and a known-good launch command.
- [docs/MISSION_CONTROL_FIRMWARE_PSEUDOCODE.md](docs/MISSION_CONTROL_FIRMWARE_PSEUDOCODE.md)
  - Pseudocode-level view of the Mission Control firmware path. Helpful before editing the runtime code.
- [docs/CAMERA_PROPOSAL.md](docs/CAMERA_PROPOSAL.md)
  - Camera-based extension idea. This is a design proposal, not current runtime behavior.

### Manuals

- [docs/manual/QUICK_START.md](docs/manual/QUICK_START.md)
  - Minimal command-focused quick start. Best if you want the shortest path to a working run.
- [docs/manual/PROJECT_STRUCTURE.md](docs/manual/PROJECT_STRUCTURE.md)
  - Short project layout guide. Useful when you need to orient yourself in the repo.
- [docs/manual/EXPERIMENT_GUIDE.md](docs/manual/EXPERIMENT_GUIDE.md)
  - Manual for running experiments and interpreting experiment outputs. Good companion to `train/run_experiments.py`.
- [docs/manual/RESULTS_INTERPRETATION.md](docs/manual/RESULTS_INTERPRETATION.md)
  - How to read metrics and results artifacts. Useful after training or evaluation finishes.

### Presentation And Misc

- [docs/swarm_robotics_presentation.md](docs/swarm_robotics_presentation.md)
  - Presentation-style summary material. Useful for talks or science-fair communication.

### Task Archive

- [docs/tasks/](docs/tasks/)
  - Implementation task history used to iteratively build the current system. These files are historical build records, not the current source of truth for behavior.


# Swarm RL PyGame Environment (Stigmergy)

A multi-agent PyGame environment for swarm RL with pheromone stigmergy, using
the PettingZoo Parallel API. The repo currently contains:

- a custom DQN baseline
- a recurrent GRU MAPPO training path with CTDE
- demo and evaluation utilities
- a robot-facing runtime that preserves decentralized inference

## Screenshots

<p>
  <img src="docs/images/swarm_default.png" width="480" />
  <br />
  <em>Default scene: tank dynamics with pheromone heatmap enabled.</em>
</p>
<p>
  <img src="docs/images/swarm_no_pheromone.png" width="480" />
  <br />
  <em>No pheromone rendering: same environment without the heatmap overlay.</em>
</p>
<p>
  <img src="docs/images/swarm_hover.png" width="480" />
  <br />
  <em>Hovercraft dynamics: agents drift slightly due to inertia/noise.</em>
</p>
<p>
  <img src="docs/images/swarm_dense.png" width="480" />
  <br />
  <em>Dense swarm: more agents, targets, and obstacles.</em>
</p>

Generate the screenshots locally:

```bash
python train/capture_screenshots.py
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For robot deployment with [firmware/run.py](firmware/run.py), `pyserial` is now included in `requirements.txt` because [firmware/ant.py](firmware/ant.py) depends on it.

## Random rollout (rendered)

```bash
python train/random_rollout.py
```

## Quickstart: Trail Learning (Recommended Path)

The current recommended research path is recurrent MAPPO with the staged
curriculum. The intended behavior is:

- explore to discover a target
- pick it up
- visibly switch into a carrying-food state
- return to the nest
- deposit pheromone on the successful return route
- complete one delivery by reaching the nest while carrying
- let later agents exploit that trail

The current default task settings now make that loop more explicit:

- there are `3` food sources in play
- each source has `4` uses
- a source loses one use on pickup
- when a source is exhausted, it respawns somewhere else if target respawn is enabled
- agents can carry only one food item at a time
- carrying agents render in a distinct green-highlighted color in demo mode

Recommended full training run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --stage-repeat-limit 1 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_full_run
```

Recommended demo checkpoint after training:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --max-steps 300 --render-scale 0.75
```

Current single-agent return stack:

- `stage1a_single_agent_miniscule`
- `stage1b_single_agent_tiny`
- `stage1c_single_agent_small`
- `stage1d_single_agent_carry_bootstrap`
  - the agent starts already carrying food and learns pure homing first
- `stage1e_single_agent_guaranteed_homing`
  - the first normal pickup-plus-delivery homing stage
- `stage1f_single_agent_delivery_bridge`
  - the first mild clutter return stage
- `stage1g_single_agent_delivery_obstacles`
  - the first true single-agent obstacle-return stage

Prompt 38 makes the bridge stage more continuity-preserving instead of letting it become an abrupt collapse point:

- `stage1f_single_agent_delivery_bridge` now uses a smaller arena jump from `stage1e`
- its single obstacle is intentionally smaller than the later obstacle-return stage
- target placement now tries to preserve a clear return corridor from nest to target in the bridge stage
- the bridge still contains clutter, but it is meant to keep greedy homing alive rather than replace it with a new task

Prompt 39 then stabilizes bridge-stage greedy delivery at the trainer level:

- stage-end evaluation now restores and evaluates the best within-stage policy instead of the last drifted one
- that keeps `stage1f_single_agent_delivery_bridge` from ending on a worse policy than the one it already discovered earlier in the stage
- this is the first configuration that looks credible for a real full training run rather than only more stage-1 debugging

Prompt 40 then adds an explicit small-swarm bootstrap stack so the first scale-up does not wipe out the learned delivery loop:

- `stage2a_small_swarm_carry_bootstrap`
  - small swarm, already carrying, pure homing
- `stage2b_small_swarm_delivery_easy`
  - small swarm, easy pickup-plus-delivery, no clutter, pheromone off
- `stage2c_small_swarm_medium`
  - first real small-swarm medium stage, still delivery-first and pheromone off
- `stage2d_small_swarm_large`
  - larger small-swarm stage where trail behavior can return

Prompt 41 then reduces repeat and budget pressure in the already-solved single-agent carry/bootstrap stages so `stage1_to_2` and full runs reach the new swarm bootstrap stages instead of spending too much budget re-proving stage-1 lessons.

Prompt 44 and prompt 45 then strengthen the late swarm behavior around the nest:

- post-delivery outward pressure stays active until agents actually leave the nest zone
- non-carrying agents are pushed to fan out and explore instead of orbiting the nest
- late swarm stages now use strong non-carrying loiter, crowding, idle, and no-outward-progress penalties near the nest

Prompt 46 then goes beyond reward shaping and adds an explicit env-side “leave the nest zone” mode for empty agents in the late swarm stages:

- if a non-carrying agent remains inside the configured nest-adjacent force-explore radius, the env can override its chosen action with an outward-moving action
- this is meant to break the specific orbiting / turn-in-place / local-circling failure mode that larger scalar penalties alone did not eliminate
- carrying-food return behavior is unchanged; this mode is only for empty agents near the nest

Prompt 37 also fixed a real environment bug in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py): the tank and hover movement drivers were dropping `carrying_food` during normal movement updates, which could silently break return-to-nest lessons immediately after the first move.

What the main arguments mean:

- `--backend mappo`
  - use the recurrent MAPPO trainer instead of the DQN/SB3/RLlib paths

- `--headless`
  - run without opening a PyGame window so long training is faster and more stable

- `--curriculum full`
  - train through the full staged curriculum rather than only the early stages

- `--n-agents 6`
  - target six agents for the later full-swarm curriculum stages

- `--total-steps 600000`
  - total training budget across the whole curriculum
  - this budget is distributed across the curriculum slice you selected, not across omitted stages
  - this is much more serious than a short `30k`–`200k` run because the later stages need real time

- `--rollout-steps 128`
  - collect on-policy rollouts in chunks of 128 steps before PPO-style updates

- `--update-epochs 4`
  - run four optimization passes over each collected rollout batch

- `--minibatch-size 256`
  - minibatch size used during PPO optimization

- `--eval-every 10000`
  - run evaluation every 10,000 training steps

- `--eval-episodes 5`
  - use five episodes for each scheduled evaluation so eval is less noisy

- `--stage-repeat-limit 1`
  - allow one retry when a stage still fails its minimum greedy pickup/delivery target
  - this helps prevent weak early stages from being silently promoted

- `--reward-pickup 6`
  - keep pickup meaningful, but not as important as completed delivery

- `--reward-nest-delivery 30`
  - make successful return-to-nest delivery the strongest core task reward

- `--reward-undelivered-food -10`
  - penalize ending an episode while still carrying food
  - this helps discourage “pick up but never bring it home”

- `--folder-name mappo_full_run`
  - base name for checkpoints and run outputs

Why this is the recommended starting point:

- the current curriculum spreads learning across many stages
- the early stages now use more responsive `action_repeat_steps = 1`, while later stages keep smoother `action_repeat_steps = 2`
- the early stages now deliberately simplify the task: pheromone is disabled in stage 1, movement penalties are softened, and pickup/delivery cues are stronger so greedy `pickup -> return -> deliver` behavior can form first
- prompts 30, 31, and 32 now suppress exploration reward while carrying in the return-focused stages, add a dedicated guaranteed-homing stage with controlled target/agent placement, and then reintroduce clutter through a bridge stage before the true obstacle-return stage
- prompt 33 now tightens trainer pressure on those early return stages: entropy decays faster there, greedy checkpoint scoring weights completed delivery and conversion more heavily, and stage summaries explicitly print sampled-vs-greedy pickup/delivery gaps
- prompt 34 makes `--total-steps` a real hard global cap in the trainer and tightens later-stage scoring/promotion so pickup-without-delivery is treated as failure rather than progress
- prompt 35 now adds carrying-phase stall penalties and metrics, so once an agent is carrying food the trainer can measure and penalize no-progress / low-displacement return behavior instead of only noticing pickup and delivery endpoints
- prompt 36 now pushes the homing stages further toward deterministic return behavior by adding sustained carrying-progress shaping, lowering return-stage entropy more aggressively, and tightening early return-stage delivery/conversion promotion targets
- the trainer now decays entropy within each stage instead of keeping one fixed exploration pressure forever, so early rollouts can explore while later updates in the same stage become more deterministic
- the early stages keep a slightly stronger exploration bonus, and later stages reduce `reward_new_cell` so delivery and trail reuse compete less with wandering
- the trainer now keeps a fixed padded centralized critic state dimension across the selected curriculum so the critic can carry across stages instead of resetting whenever the stage shape changes
- stage progression is now greedy-eval-aware, with optional repeats when pickup/delivery remain below minimum promotion targets
- the trainer now prints sampled-vs-greedy pickup/delivery gaps and saves `best_greedy_eval/` so demo can use the strongest greedy checkpoint instead of assuming `latest/` is best
- stage summaries now also print pickup-to-delivery conversion, which is the main signal for whether return-to-nest behavior is actually forming
- episode/eval CSVs now also include carrying-phase signals such as stall events, low-progress fraction, low-displacement fraction, and carrying-penalty totals
- checkpoint metadata now also records the sustained carrying-progress reward coefficient used for the stage
- the final stage is still large and hard: `1400x950`, `18` obstacles, `6` agents
- a shorter run can finish, but often leaves the later full-swarm stages undertrained
- `600k` is not guaranteed to be optimal, but it is a practical strong starting point for the current repo

Short smoke test:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --n-targets 3 --active-targets 3 --food-source-capacity 4 --target-respawn --folder-name mappo_trail_smoke
```

Render the trained MAPPO policy:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --max-steps 300 --render-scale 0.75
```

In demo mode, agents now switch to a distinct carrying-food color after pickup
and return to the normal agent color after a completed nest delivery.
The PyGame window title also shows the current reset seed, and the HUD shows
step, pickups, deliveries, pheromone drops, and carrying-agent count.
You can press `Space` to pause/resume and click an agent to open an inspector
panel showing that agent's current inputs and model outputs.

Cycle demo resets through only specific seeds:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --max-steps 0 --render-scale 0.75 --seed-list 45,40,58
```

Headless MAPPO evaluation:

```bash
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --n-agents 6 --episodes 10 --headless --output-dir runs/eval --filename mappo_full_run_eval --active-targets 3 --food-source-capacity 4
```

Pheromone comparison example:

```bash
python analysis/evaluate_comparison.py --policy-kind mappo_gru --checkpoint-with-pheromone checkpoints/mappo_full_run/best_greedy_eval --checkpoint-without-pheromone checkpoints/mappo_no_pher_run/best_greedy_eval --agent-min 1 --agent-max 6 --episodes-per-agent 3 --headless --output-dir experiments/experiment_data/trail_compare
```

## Quickstart: DQN Baseline

Train headless and save checkpoints:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000 --save-dir checkpoints --folder-name demo_run --save-every 2000
```

Render the trained policy:

```bash
python train/demo.py --checkpoint-dir checkpoints/demo_run/full_policy
```

## Train (headless)

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000
```

Use a shared policy:

```bash
python train/independent_dqn_pytorch.py --shared-policy --headless
```

Save checkpoints during training:

```bash
python train/independent_dqn_pytorch.py --headless --save-every 2000 --save-dir checkpoints --folder-name demo_run
```

Train with an explicit epsilon schedule:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 30000 --save-dir checkpoints --folder-name demo_run --epsilon-start 1.0 --epsilon-final 0.05 --epsilon-decay-steps 20000 --warmup-steps 2000
```

If you run that command manually, keep it on one shell line or use `\` line continuations exactly. Entering `>`-prefixed continuation lines or isolated flag lines can make the shell create empty files such as `--epsilon-final`, `--epsilon-decay-steps`, or `--warmup-steps` in the repo root.

## Training Backends (Optional)

Default (custom DQN):
```bash
python train/train.py --backend custom --headless --total-steps 10000
```

Stable-Baselines3 DQN (shared policy):
```bash
python train/train.py --backend sb3 --headless --total-steps 10000 --save-path checkpoints/sb3_dqn.zip
```

RLlib DQN (shared policy):
```bash
python train/train.py --backend rllib --headless --total-steps 10000 --save-dir checkpoints/rllib_dqn
```

If Ray warns about `/tmp` being full or socket path length, point it to a different temp dir:
```bash
python train/train.py --backend rllib --headless --total-steps 10000 --save-dir checkpoints/rllib_dqn --ray-tmpdir /Users/christopherlin/.ray_tmp
```

## Demo (rendered)

```bash
python train/demo.py --checkpoint-dir checkpoints
```

Shared-policy demo:

```bash
python train/demo.py --checkpoint-dir checkpoints --shared-policy
```

SB3 demo:

```bash
python train/demo.py --backend sb3 --sb3-model checkpoints/sb3_dqn.zip
```

RLlib demo:

```bash
python train/demo.py --backend rllib --rllib-checkpoint checkpoints/rllib_dqn
```

MAPPO demo:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --max-steps 300 --render-scale 0.75
```

With a custom Ray temp dir:
```bash
python train/demo.py --backend rllib --rllib-checkpoint checkpoints/rllib_dqn --ray-tmpdir /Users/christopherlin/.ray_tmp
```

To auto-exit after N steps (useful for smoke tests):
```bash
python train/demo.py --backend custom --max-steps 200
```

## Robot Deployment

The current robot runtime lives in [firmware/](firmware).

Main files:

- [firmware/ant.py](firmware/ant.py)
  - serial client and low-level ESP32 command helpers
- [firmware/run.py](firmware/run.py)
  - direct checkpoint inference loop for the physical robot
- [firmware/ant.service](firmware/ant.service)
  - example systemd unit

Typical deployment flow:

```bash
python firmware/run.py --checkpoint-dir checkpoints --shared-policy
```

This runtime loads [models/q_network.py](models/q_network.py), reads scan lines from `ant.py`, normalizes the observation locally, predicts a discrete action, and sends `turn`, `move`, or `stop` commands directly to the robot.

## Command Center

The repo now includes a separate live operator subsystem in [mission_control/](mission_control). It is a PyGame command center for:

- receiving live robot `POS`, `PHER`, and `SENSE` messages
- visualizing robot positions and trails
- maintaining the authoritative digital pheromone field
- returning simulator-compatible `PHER_RESP` samples back to robots

Run it locally:

```bash
python -m mission_control.main --tcp-host 127.0.0.1 --tcp-port 8765
```

Run a local fake robot against it:

```bash
python -m mission_control.fake_robot --robot-id robot_0 --port 8765
```

Documentation:

- [docs/MISSION_CONTROL.md](docs/MISSION_CONTROL.md)
- [docs/MISSION_CONTROL_FIRMWARE_PSEUDOCODE.md](docs/MISSION_CONTROL_FIRMWARE_PSEUDOCODE.md)

## Environment API (PettingZoo Parallel API)

`SwarmEnv` implements the PettingZoo Parallel API.

Methods:
- `reset(seed=None, options=None) -> (obs_dict, info_dict)`
- `step(action_dict) -> (obs_dict, rewards_dict, terminations, truncations, infos)`
- `render(mode="human", fps=60)`
- `close()`

### Actions
Discrete action space with 18 actions:
`{throttle ∈ [-1,0,1]} × {turn ∈ [-1,0,1]} × {deposit ∈ [0,1]}`.
Provide `actions` as a dict keyed by agent id (e.g., `agent_0`) with values in `[0, 17]`.

### Observations
Each agent gets a local observation history vector; `reset`/`step` return a
dict of `agent_id -> obs`.

Current default per-frame features:

- 9 lidar rays
- 2 nearest detectable target features: distance and relative angle
- 2 nest-direction features
- 2 nearest-neighbor features
- 2 heading features: `sin(theta)`, `cos(theta)`
- 1 normalized speed feature
- 1 food-presence flag
- 1 carrying-food flag
- 3 pheromone samples

Current default observation size:

- 23 features per frame
- `observation_history_steps = 3`
- flattened `obs_dim = 69`

### Stigmergy (pheromone)
The environment maintains a pheromone grid with:

- explicit deposit vs no-deposit action choice
- decay
- bounded diffusion
- carrying-food scaling
- optional gating so deposition only happens while carrying food and making return-to-nest progress

The intended training story is trail formation:

- discovery is expensive early
- successful returns write route hints into the environment
- later agents can exploit those hints

## Files
- `env/swarm_env.py` : environment implementation
- `env/config.py` : configuration dataclass
- `train/random_rollout.py` : random policy sanity check
- `train/independent_dqn_pytorch.py` : independent or shared DQN training
- `train/demo.py` : load and render trained checkpoints
- `firmware/run.py` : physical robot policy runtime
- `firmware/ant.py` : direct serial robot interface
- `docs/ARCHITECTURE.md` : detailed functionality and architecture
- `docs/PROJECT_LOG.md` : decisions, notes, and next steps to resume later

## Notes
- Episode ends when all targets are collected or `max_steps` reached.
- Dynamics are pluggable via `dynamics_mode`: `"tank"`, `"hover"`, or `"mixed"`.

## Beginner Walkthrough (PyGame + RL)

If you’re new to PyGame and RL, this section gives a quick mental model and a practical path to running the project.

### What this project does
- Simulates a swarm of agents in a 2D PyGame world.
- Exposes an RL-style API (`reset`, `step`) with multi-agent observations and rewards.
- Adds stigmergy via a pheromone grid that agents can sense and write to.
- Supports a trail-formation objective where agents learn `discover -> return -> deposit -> exploit`.

### The fastest way to see it working
1) Random sanity check (renders a window):
```bash
python train/random_rollout.py
```

2) Train a basic model (headless, no window):
```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000 --save-dir checkpoints --save-every 2000
```

3) Run the demo with the trained model (renders a window):
```bash
python train/demo.py --checkpoint-dir checkpoints
```

### How the RL loop works (simple view)
Each step:
1) You give actions for each agent.
2) The environment moves agents, handles collisions, and collects targets.
3) You receive rewards + new observations.

### What the agents are learning
- **+8** when an agent reaches a target
- **-0.01** each step (encourages speed)
- **-0.2** for collisions with walls/obstacles

So the learned behavior should be: “find targets quickly without crashing.”

### How PyGame fits in
- PyGame is only used for rendering and window events.
- If you run with `--headless`, no window is opened and PyGame doesn’t render.

### Where to change behavior
- `env/config.py` controls most parameters:
  - number of agents/targets/obstacles
  - rewards
  - lidar rays
  - pheromone settings
  - non-carrying exploration controls, including env-side random exploration overrides
  - render scaling via `render_scale`
  - dynamics mode (`tank`, `hover`, `mixed`)

### Want deeper details?
See `docs/ARCHITECTURE.md` for a full breakdown of modules and data flow.
