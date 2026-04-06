# Step 1 - Initialize Code For A Swarm Reinforcement Learning Project

Build a new reinforcement learning repository for a 2D swarm robotics simulation.

Build the initial project skeleton from scratch with these goals:

- a clean Python package layout for environment, training, docs, and checkpoints
- a local-venv friendly workflow
- minimal dependencies for simulation and custom PyTorch training
- code that is readable by a student learning RL for the first time

Create this initial structure:

- `env/`
- `train/`
- `docs/`
- `checkpoints/`

Add the first version of:

- `env/config.py`
- `env/swarm_env.py`
- `train/random_rollout.py`
- `requirements.txt`
- `README.md`

Requirements for `env/config.py`:

- define a central configuration object for the environment
- include world size, agent count, target count, obstacle count, episode length, render settings, movement settings, sensor settings, reward settings, pheromone settings, and seed
- keep field names explicit and beginner-readable
- make it easy for future scripts to override config values from the command line

Requirements for `env/swarm_env.py`:

- create the first version of a multi-agent swarm environment
- keep it simple but structured for later RL work
- include:
  - environment reset
  - environment step
  - agent state tracking
  - target spawning
  - obstacle spawning
  - basic collision handling
  - a local observation vector per agent
  - a discrete action table that maps action ids to movement commands
  - reward for reaching targets
  - optional pheromone deposit and decay support, even if basic at first
  - PyGame rendering for agents, targets, obstacles, and a visible world boundary

Requirements for `train/random_rollout.py`:

- instantiate the environment
- sample random actions for all active agents
- step the environment for a short episode
- optionally render unless a headless flag is provided
- print basic episode statistics

Requirements for `README.md`:

- explain the project goal in plain language
- describe the top-level directories
- show how to create a venv, install requirements, and run the random rollout

Constraints:

- use Python and PyTorch
- keep the action space discrete
- keep the environment code ready for future RL integration
- avoid premature abstractions

At the end, provide:

- files created
- the exact command to run the first random rollout
- any assumptions you made about default hyperparameters


# Step 2 - Implement A Proper Multi-Agent Swarm Environment With Stigmergy

Extend the existing project into a more complete swarm RL environment.

Work inside the current files instead of replacing them.

Goals:

- make the environment suitable for DQN training
- keep the simulation visually understandable
- model swarm coordination with local sensing and pheromone-based stigmergy

Environment requirements:

- maintain multiple agents in a shared 2D world
- keep a discrete 9-action interface
- implement an action table that covers combinations of forward/reverse throttle and left/right turn
- support randomized targets and obstacles on reset
- include per-agent heading and speed
- implement local sensing with lidar-style obstacle rays
- include nearest-target information in the observation vector
- include nearest-neighbor information in the observation vector
- include heading as `sin(theta), cos(theta)`
- include speed in the observation vector
- include local pheromone samples in the observation vector
- keep observations normalized or clipped consistently

Pheromone requirements:

- keep a world-grid pheromone field
- support deposit, decay, and optional diffusion
- allow the environment to update the pheromone map every step
- render the pheromone field as a heatmap overlay when rendering is enabled

Reward and episode requirements:

- reward target collection
- include a small step penalty so agents are encouraged to act efficiently
- support episode truncation by max steps
- return useful per-agent info dictionaries

Rendering requirements:

- show agents with visible heading
- show obstacles and targets clearly
- show pheromone heatmap clearly but without hiding the agents

Implementation constraints:

- preserve compatibility with the existing project structure
- keep the code self-contained in `env/`
- do not introduce a training library yet

At the end, provide:

- files modified
- the final observation layout
- the final 9-action mapping
- the command to run the updated random rollout


# Step 3 - Add A Custom PyTorch DQN Trainer For The Swarm Environment

The environment is ready. Now implement a custom DQN training pipeline for it.

Do not introduce Stable-Baselines3 or RLlib yet. Start with a clear custom trainer so the learning pipeline is easy to understand and modify.

Add or update these files:

- `train/independent_dqn_pytorch.py`
- `train/demo.py`
- update `requirements.txt` if needed
- update `README.md`

Requirements for the trainer:

- use PyTorch
- build a small MLP Q-network
- support replay buffer training
- support target network updates
- support epsilon-greedy exploration
- handle multiple agents in the same environment
- support two training modes:
  - independent per-agent Q-networks
  - shared-policy mode using one shared Q-network
- save checkpoints to `checkpoints/`
- keep CLI flags for key hyperparameters such as:
  - `--total-steps`
  - `--batch-size`
  - `--lr`
  - `--gamma`
  - `--epsilon-start`
  - `--epsilon-end`
  - `--epsilon-decay-steps`
  - `--target-update-every`
  - `--save-every`
  - `--save-dir`
  - `--headless`
  - `--shared-policy`

Requirements for the demo script:

- load saved checkpoints
- run inference only
- render the environment
- support both shared and independent checkpoints
- use greedy action selection

Important implementation details:

- derive observation size and action count from the environment rather than hard-coding them where possible
- keep checkpoint naming simple and predictable
- keep the network definition readable, not over-engineered
- keep the demo path consistent with the custom trainer’s checkpoint format

At the end, provide:

- files modified or added
- exact commands to train and demo
- a brief explanation of how independent-policy and shared-policy training differ in this codebase


# Step 4 - Make The Environment PettingZoo Parallel API Compatible

Refactor the current swarm environment so it follows the PettingZoo Parallel API cleanly.

Goals:

- preserve the existing environment behavior and action semantics
- make the environment usable by external RL libraries later
- update the custom trainer and demo scripts to work with the new dict-based multi-agent API

Requirements:

- convert `env/swarm_env.py` to a PettingZoo Parallel-style environment
- keep agent ids stable and explicit, such as `agent_0`, `agent_1`, etc.
- make `reset()` return observation dictionaries and info dictionaries
- make `step()` accept a dict of actions keyed by agent id
- return dicts for observations, rewards, terminations, truncations, and infos
- keep rendering functional after the refactor

Update all affected training and inference scripts so they work with the new API:

- `train/independent_dqn_pytorch.py`
- `train/demo.py`
- `train/random_rollout.py`

Constraints:

- do not change the core task unless required by the API change
- preserve the same discrete 9-action control interface
- keep the custom trainer fully functional after the conversion

At the end, provide:

- files modified
- a short explanation of the new reset/step signatures
- commands to smoke test the environment and trainer after the refactor


# Step 5 - Add Backend Flexibility With SB3, RLlib, And A Unified Training Entry Point

The custom DQN trainer is working and the environment now follows the PettingZoo Parallel API.

Add optional library backends while preserving the custom pipeline.

Create or update these files:

- `train/train.py`
- `train/sb3_dqn.py`
- `train/rllib_dqn.py`
- `train/demo.py`
- `requirements.txt`

Goals:

- keep the custom PyTorch DQN trainer as the default learning reference
- add Stable-Baselines3 DQN as a library baseline
- add Ray RLlib DQN as another backend
- make backend selection happen through one dispatcher script

Requirements:

- `train/train.py` should dispatch based on a `--backend` flag with values `custom`, `sb3`, and `rllib`
- keep CLI arguments reasonably aligned across backends where practical
- document any backend-specific limitations clearly in code comments or help text

Requirements for SB3 integration:

- wrap the environment as needed without rewriting the environment itself
- support training a shared policy
- save checkpoints in a predictable subdirectory

Requirements for RLlib integration:

- register the environment explicitly
- include any compatibility shims needed for checkpoint loading and demo use
- handle likely RLlib demo edge cases such as checkpoint URI normalization and config validation quirks
- add support for a custom temporary directory flag if RLlib needs it

Requirements for the demo path:

- `train/demo.py` should be able to run checkpoints from custom, SB3, and RLlib backends
- include a `--max-steps` flag for smoke-test friendly demos

Constraints:

- do not remove or degrade the custom trainer
- do not fork the environment into backend-specific copies
- keep all three backends using the same environment contract and action semantics

At the end, provide:

- files modified or added
- exact train commands for `custom`, `sb3`, and `rllib`
- exact demo commands for all supported backends
- any backend-specific caveats


# Step 6 - Add Screenshot Tooling And Beginner-Friendly Documentation

The project is now functionally useful, but it needs documentation and lightweight presentation tooling.

Create or update these files:

- `train/capture_screenshots.py`
- `docs/ONBOARDING.md`
- `docs/UML.md`
- `docs/QandA.md`
- `docs/RESOURCES.md`
- `docs/DQN_EXPLAINED.md`
- `docs/ToDo.md`

Goals:

- make the codebase understandable to a beginner
- capture recurring technical explanations in persistent docs
- support generating screenshots for documentation and evaluation

Requirements for `train/capture_screenshots.py`:

- render several representative environment scenes
- save screenshots under `docs/images/`
- keep the script deterministic enough to reproduce useful images

Requirements for `docs/ONBOARDING.md`:

- walk through the repo from environment to training to demo
- explain the role of `env/swarm_env.py`, `train/independent_dqn_pytorch.py`, `train/train.py`, and `train/demo.py`
- include a beginner-friendly RL-to-code mapping

Requirements for `docs/UML.md`:

- include a readable text-based architecture view and at least one environment/MDP style diagram

Requirements for `docs/QandA.md`:

- start a running Q&A reference for repeated questions
- include concise entries explaining concepts such as replay buffer vs trajectory, PettingZoo vs PyTorch, Q-function vs policy, and what PettingZoo Parallel means in this repo

Requirements for `docs/RESOURCES.md`:

- link to official PettingZoo and Gymnasium docs
- include PyTorch and RL learning resources suitable for a student

Requirements for `docs/DQN_EXPLAINED.md`:

- explain the repo’s Q-network, Bellman target, and one training iteration in practical terms

Requirements for `docs/ToDo.md`:

- capture future work, especially around sim-to-real transfer and safety

Constraints:

- keep docs grounded in actual code that exists
- do not invent features not present in the repository

At the end, provide:

- files created or updated
- the command to regenerate screenshots
- a short summary of the beginner documentation coverage


# Step 7 - Add A Sim-To-Real Deployment Skeleton Without Changing The Learning Contract

Add the first real deployment architecture for moving the trained policy from simulation onto a physical robot.

The target hardware split is:

- Raspberry Pi as the high-level runtime and inference node
- Arduino or similar microcontroller as the low-level sensing and actuation layer

Create or update these files:

- `docs/SimToReal.md`
- `robot/messages.py`
- `robot/sensor_bridge.py`
- `robot/observation_builder.py`
- `robot/policy_runner.py`
- `robot/action_bridge.py`
- `robot/runtime.py`

Core design rule:

- keep the policy boundary unchanged
- the trained model should still consume the same normalized observation vector and emit the same discrete action id
- real-world adaptation should happen in the sensor and action plumbing, not by changing the learned policy interface

Requirements for `docs/SimToReal.md`:

- explain the Pi/Arduino split of responsibilities
- describe the end-to-end data flow from sensor packet to observation to policy to action command to motor controller
- define an observation compatibility contract
- define an action compatibility contract
- explain why safety must remain outside the policy

Requirements for `robot/messages.py`:

- define clear message/data structures for sensor packets and action commands

Requirements for `robot/observation_builder.py`:

- convert a generic sensor packet into the same observation vector expected by the policy
- preserve feature order, normalization assumptions, clipping behavior, and missing-data behavior as much as possible

Requirements for `robot/policy_runner.py`:

- load trained checkpoints
- run inference on one observation
- keep the API simple for deployment use

Requirements for `robot/action_bridge.py`:

- map a policy action id into a hardware-agnostic action command

Requirements for `robot/runtime.py`:

- wire together sensor bridge, observation builder, policy runner, and action bridge into a deployment loop skeleton

Constraints:

- do not rewrite the trainer
- do not rewrite the environment
- keep deployment code separate from `env/` and `train/`

At the end, provide:

- files created
- a one-paragraph summary of the deployment architecture
- what still remains hardware-specific


# Step 8 - Extract The Shared Q-Network And Reduce Robot Runtime Dependencies

The deployment skeleton should not depend on importing the training script just to get the Q-network definition.

Refactor the repository so the network architecture is shared cleanly across training, demo, and deployment.

Create or update these files:

- `models/q_network.py`
- `train/independent_dqn_pytorch.py`
- `train/demo.py`
- `robot/policy_runner.py`

Requirements:

- move the `QNetwork` definition into `models/q_network.py`
- update the custom trainer to import the network from there
- update the demo code to import the network from there
- update the robot deployment inference path to import the network from there
- keep checkpoint compatibility intact

Goal:

- after this refactor, the intended minimal robot-side runtime dependency for inference should be `robot/`, `models/q_network.py`, `env/config.py`, and a checkpoint

Constraints:

- do not change the model architecture unless absolutely necessary
- do not break existing checkpoints if they were trained with the same observation/action contract
- keep the import structure simple

At the end, provide:

- files modified or created
- a brief explanation of why this refactor matters for deployment
- confirmation that the custom trainer, demo path, and robot runtime all share the same network definition


# Step 9 - Add A Raspberry Pi Reference Integration Based On A Serial ESP32 Control Pattern

Now add a concrete Raspberry Pi reference integration that shows how a real Pi-side robot script can reuse the deployment modules in this repository.

Assume there is an existing external pattern with:

- a Python script on the Pi
- a `systemd` service
- serial communication over `/dev/serial0`
- an ESP32 that accepts high-level commands and streams scan data as JSON lines

Create this new `pi/` package and add:

- `pi/__init__.py`
- `pi/esp32_robot.py`
- `pi/esp32_sensor_adapter.py`
- `pi/esp32_action_bridge.py`
- `pi/run_policy.py`

Requirements for `pi/esp32_robot.py`:

- implement a thin serial client for the ESP32 protocol
- include connect, close, line reading, raw command send, JSON parsing, ack/err waiting, and high-level helpers such as `turn`, `move`, `stop`, and `brake`

Requirements for `pi/esp32_sensor_adapter.py`:

- convert serial scan lines into the generic `SensorPacket`
- bucketize angle-distance scan data into the fixed lidar-style observation channels expected by `robot/observation_builder.py`
- use placeholder values for sensor channels that are not available yet

Requirements for `pi/esp32_action_bridge.py`:

- convert generic `ActionCommand` objects into the current ESP32 command vocabulary
- support configurable turn step size and move distance

Requirements for `pi/run_policy.py`:

- create a CLI for running a checkpoint on the Pi
- instantiate `SwarmConfig`
- connect the serial client, sensor adapter, observation builder, policy runner, command mapper, and action bridge
- read real sensor data, build an observation, predict an action, and send a hardware command in a loop
- support flags for checkpoint path, serial port, baud rate, loop rate, scan duration, and basic motion tuning

Constraints:

- do not remove the generic `robot/` abstraction layer
- keep the Pi code as a reference integration, not a platform-wide rewrite
- preserve the action semantics used during training

At the end, provide:

- files created
- the end-to-end Pi control path
- what assumptions the Pi integration makes about the ESP32 protocol


# Step 10 - Preserve The Existing Rule-Based Pi Workflow And Document The Migration Path

The repository now has a generic deployment layer and a Pi-side learned-policy path. Add compatibility-first Pi artifacts so a robot developer can switch to this repository without immediately changing field behavior.

Create or update these files:

- `pi/ants.py`
- `pi/ants.service`
- `docs/PI_MIGRATION.md`
- update `docs/SimToReal.md`
- update `docs/QandA.md`

Requirements for `pi/ants.py`:

- preserve the current Raspberry Pi rule-based control pattern
- open `/dev/serial0`
- read JSON scan lines from the ESP32
- choose a movement direction with simple handcrafted free-space logic
- send high-level movement commands back to the ESP32
- keep the code close to the baseline behavior rather than refactoring it into a perfect architecture

Requirements for `pi/ants.service`:

- provide a compatibility-first `systemd` service matching the current Raspberry Pi deployment style
- run as user `pi`
- assume a checkout under `/home/pi/ant`
- restart automatically on failure
- make it clear in docs that the service paths should be updated if the repo is cloned elsewhere

Requirements for `docs/PI_MIGRATION.md`:

- explain that `pi/ants.py` preserves today’s rule-based behavior
- explain that `pi/run_policy.py` is the migrated learned-policy target path
- define a phased migration plan:
  1. validate the preserved baseline on hardware
  2. move only the serial transport into reusable helpers
  3. build real `SensorPacket` objects
  4. build and inspect real observations
  5. run model inference in shadow mode
  6. map learned actions into the existing command vocabulary
  7. only then switch control from rule-based to model-based
- include sections on service compatibility, required components, and hardware-specific tuning still needed

Requirements for `docs/SimToReal.md` and `docs/QandA.md`:

- update them so the Pi-side reference integration is discoverable
- explain what the existing Pi rule-based script does
- explain that the repository now includes both a preserved baseline path and a learned-policy path

Constraints:

- preserve the rest of the codebase
- keep the docs specific to the actual files and runtime flow
- present the migration conservatively, not as production-complete autonomy

At the end, provide:

- files created or updated
- a concise summary of the compatibility path versus the migrated path
- the key command a Raspberry Pi user would run to start experimenting with the learned policy
