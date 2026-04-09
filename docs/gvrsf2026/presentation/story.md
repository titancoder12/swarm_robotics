# Code Story

This project trains decentralized swarm behavior in simulation, then preserves
the same observation and action contract when running the policy on a physical
robot with Mission Control acting as shared environmental memory.

## Chapter 1: The Big Picture

The codebase has four main characters:

1. the simulator
2. the trainer
3. Mission Control
4. the robot runtime

They all orbit around the same idea:

- each agent acts locally
- no agent has global control
- coordination happens through the environment, especially digital pheromones

The best first map is [ARCHITECTURE.md](./ARCHITECTURE.md), but the real story
starts in code.

## Chapter 2: The World Is Born

The heart of the project is [swarm_env.py](../env/swarm_env.py).

This file creates the swarm world. It defines:

- the agents
- the nest
- the food targets
- the obstacles
- the pheromone field
- the observation space
- the action space
- the reward logic
- the step-by-step simulation loop

The main class is `SwarmEnv`.

When `SwarmEnv` is created, it sets up the world dimensions, builds the action
table, allocates storage for agent states, targets, obstacles, pheromone grid,
and observation history, and defines the PettingZoo-compatible observation and
action spaces.

This is important to explain:

- the environment is not just graphics
- it is the full scientific testbed

If a judge asks, "Where is the actual problem defined?", the answer is:

- [SwarmEnv](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)

## Chapter 3: What An Agent Knows

The most important design decision in the project is the observation contract.

That is documented in [OBSERVATION_SPEC.md](./OBSERVATION_SPEC.md), but the
real source of truth is in [swarm_env.py](../env/swarm_env.py).

Each agent does not get the full world state.

Instead, each agent receives a local observation frame containing:

- lidar rays
- target distance and angle
- nest direction
- nearest neighbor vector
- heading
- speed
- food-presence flag
- carrying-food flag
- pheromone samples

One frame is 23 numbers. The system stacks 3 recent frames, so the default
policy input is 69 numbers.

That number matters because it appears again later in the robot runtime. The
simulator and the physical robot are tied together by this observation shape.

If a judge asks, "What exactly does the neural network see?", this is one of
your strongest answers:

- not pixels
- not a global map
- a structured local observation vector

## Chapter 4: What An Agent Can Do

The action space is deliberately simple.

The environment uses 18 discrete actions. Conceptually, each action is a
combination of:

- throttle: reverse, stop, forward
- turn: left, straight, right
- deposit: no or yes

That means the network does not output motor voltages directly. It outputs a
small symbolic action choice, and the environment interprets that choice into
movement and possible pheromone deposition.

This makes the policy easier to train and easier to port to a physical robot.

If a judge asks, "Why discrete actions?", a good answer is:

- it keeps the policy compact
- it aligns well with simple robot commands
- it makes simulator-to-robot transfer more manageable

## Chapter 5: How One Simulation Step Works

This is the core loop story.

At each step in [swarm_env.py](../env/swarm_env.py), the environment does the
following:

1. receives one action per agent
2. converts each action ID into throttle, turn, and deposit intent
3. updates agent motion through the selected dynamics driver
4. checks collisions, pickups, nest return, and delivery events
5. updates the pheromone grid
6. computes rewards
7. builds the next observations
8. returns observations, rewards, termination flags, truncation flags, and info

So when someone asks, "Where does the learning signal come from?", the answer
is:

- the reward is created inside the environment after the consequences of the
  action are simulated

## Chapter 6: Motion Is Abstracted

The environment separates policy choice from physics.

This is handled by the `DynamicsDriver` abstraction in
[swarm_env.py](../env/swarm_env.py).

There are multiple motion models, including:

- `TankKinematicsDriver`
- `HovercraftDriver`

The policy does not know which one is active. It only chooses the abstract
action. The driver converts that into motion.

That is a good example of clean architecture:

- policy layer decides
- dynamics layer executes

## Chapter 7: Training Begins At The Dispatcher

The main training entrypoint is [train.py](../train/train.py).

This file is intentionally small. It does not do the training itself. Instead,
it acts like a switchboard.

Depending on the `--backend` flag, it dispatches to:

- the custom DQN trainer
- the SB3 trainer
- the RLlib trainer
- the MAPPO trainer

Today, the main research path is MAPPO.

So in practice, the most important path is:

- [train.py](/Users/christopherlin/dev/cwsf2026/sim/train/train.py)
- then [mappo_gru.py](/Users/christopherlin/dev/cwsf2026/sim/train/mappo_gru.py)

If a judge asks, "Where do you actually start training?", that is the answer.

## Chapter 8: The Main Learning Brain

The central modern training path is [mappo_gru.py](../train/mappo_gru.py).

This file does the heavy lifting for recurrent MAPPO training.

Its job is to:

- parse MAPPO-specific training arguments
- build environments for each curriculum stage
- create the actor and critic
- collect rollouts
- compute advantages
- update the policy and value networks
- run evaluation
- save checkpoints

This is where the project becomes a true learning system instead of a hand-made
behavior system.

Two important model ideas live here:

- the actor is decentralized at execution time
- the critic is centralized during training

That is the CTDE idea:

- centralized training
- decentralized execution

If a judge asks, "How can agents learn with more information than they later
use?", the answer is:

- training uses a richer critic state
- deployment uses only local observations

## Chapter 9: Why There Is A GRU

The inference helper is [inference.py](../algorithms/mappo/inference.py).

This file is short but very important.

It loads the trained recurrent actor and performs one inference step.

The actor is a `SharedGRUActor`.

Why use a GRU?

- because the agent does not see the whole world at once
- memory across time helps it behave more intelligently
- the GRU lets the policy integrate recent history, not just the current frame

So if a judge asks, "Why is this not just a feedforward network?", the answer
is:

- the environment is partially observable
- recurrence helps with partial observability

## Chapter 10: The Curriculum Is The Teacher

One of the most important ideas in the code is that the final behavior is too
hard to learn all at once.

That is why training uses curriculum stages in
[mappo_gru.py](../train/mappo_gru.py) and the curriculum definitions it imports.

The code builds easier stages first, then harder ones.

Conceptually, the curriculum teaches:

- pickup
- homing
- delivery
- obstacle handling
- small swarm coordination
- larger swarm trail behavior

This is a strong explanation for judges because it shows the project is not
just "throwing RL at the problem." It is shaping the learning process.

If a judge asks, "How did you stop the policy from getting stuck in bad local
behavior?", one strong answer is:

- curriculum learning

## Chapter 11: Demo Mode Is The Easiest Place To See The Policy

If you want to show where a trained policy is visualized, the answer is
[demo.py](../train/demo.py).

This file:

- loads a checkpoint
- builds the environment
- runs the policy step by step
- renders the world
- can also publish simulator telemetry outward to Mission Control

This is useful because it connects model inference to something visible.

If a judge asks, "How do you inspect a trained model?", the simplest answer is:

- use the demo path in [demo.py](/Users/christopherlin/dev/cwsf2026/sim/train/demo.py)

## Chapter 12: Mission Control Is The Shared Memory Keeper

Mission Control lives in [mission_control/main.py](../mission_control/main.py).

This is the live desktop command center.

It does not drive the robot directly. Instead, it acts as shared infrastructure.

Mission Control:

- receives robot messages
- tracks robot positions
- stores lidar and target markers
- maintains the digital pheromone field
- answers `SENSE` queries with pheromone samples
- renders the whole state in a PyGame window

This is one of the most elegant parts of the architecture because it turns
stigmergy into a real deployment system:

- the environment-like shared field still exists
- but now it exists outside the simulator

If a judge asks, "What exactly is Mission Control?", a good one-line answer is:

- it is the live desktop subsystem that maintains shared digital pheromone
  memory and visualizes the swarm

## Chapter 13: The Protocol Is Simple On Purpose

Mission Control uses a line-based protocol.

The key message types are:

- `POS`
- `PHER`
- `SENSE`
- `PHER_RESP`
- `LIDAR`
- `TARGET`

The beauty here is that the deployment interface is simple text, not a huge
binary protocol.

That makes it:

- debuggable
- portable
- easy to route through TCP, BLE, or relay

If a judge asks, "How do the robot and command center communicate?", the short
answer is:

- newline-delimited text messages

## Chapter 14: The Robot Runtime Preserves The Simulator Contract

The physical robot entrypoint is [run.py](../firmware/run.py).

This is one of the most important files in the entire project.

Why?

Because this file is the sim-to-real bridge.

Its job is to:

- load the trained policy checkpoint
- talk to the real robot hardware
- collect sensor data
- convert that data into the same observation format used in training
- run policy inference
- convert the chosen action back into robot commands
- optionally talk to Mission Control over BLE, TCP, or relay

This is where the project proves it is not just a simulator.

If a judge asks, "How do you actually deploy the learned model?", this file is
the answer.

## Chapter 15: The Robot Does Not See The World The Same Way We Do

The robot runtime uses a `PolicyConfig` in [run.py](../firmware/run.py) to
mirror the simulator observation contract.

That means:

- same action count
- same lidar ray count
- same pheromone sample count
- same observation stacking logic

This is a major design principle:

- do not change the policy interface at deployment unless absolutely necessary

That is why the robot runtime contains many normalization helpers:

- bucketizing scan data
- normalizing ranges
- normalizing XY vectors
- normalizing pheromone samples

The runtime is constantly translating messy physical signals into the clean
training-time representation.

## Chapter 16: The ESP32 Is The Hardware Gatekeeper

The Pi does not directly drive every low-level device. Instead, it talks to an
ESP32 through [ant.py](../firmware/ant.py).

This file is the serial interface layer.

It:

- opens the serial connection
- waits for the stream to become ready
- sends commands like turn, move, stop, and brake
- reads streamed scan lines
- parses JSON-like line data from the robot side

This is the file to mention if a judge asks:

- "How does the Pi actually talk to the robot hardware?"

The answer is:

- through a serial command-and-stream interface handled by
  [ant.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/ant.py)

## Chapter 17: Camera Integration Was Added Carefully

The current physical runtime can optionally use the onboard camera to fill the
target-related observation slots.

The important idea is:

- the camera does not replace the policy
- it fills the same compact target features the policy already expects

That is why the camera integration is compatible with the simulator-trained
model.

The camera path lives in:

- [camera.py](../firmware/camera.py)
- [run.py](../firmware/run.py)

If a judge asks, "Did you retrain the whole model on raw images?", the answer
is:

- no
- the camera is used as a feature extractor that feeds the existing observation
  contract

## Chapter 18: Why Mission Control Matters In Sim-To-Real

In simulation, the pheromone field is just part of the environment.

In the real robot stack, that same idea needs somewhere to live.

Mission Control becomes that place.

The robot sends:

- position
- deposits
- sensing requests

Mission Control returns:

- pheromone samples

So Mission Control acts like the externalized environment memory that makes
stigmergy possible in the physical system.

That is one of the best conceptual bridges in the entire project.

## Chapter 19: The Deep Idea Behind The Code

The codebase is not just "robot code plus neural network code."

It is really an argument expressed in software:

- intelligence does not have to be centralized
- useful coordination can emerge from simple local agents
- environment can function as memory
- shared traces can guide later behavior
- simulation can be made structurally compatible with physical deployment

So when you explain the code, always bring it back to this:

- the architecture is serving the scientific idea

## Chapter 20: The Fastest Judge Answers

Here are a few fast answers you can memorize.

If asked, "Where is the world implemented?"

- [swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)

If asked, "Where do you start training?"

- [train.py](/Users/christopherlin/dev/cwsf2026/sim/train/train.py), then
  [mappo_gru.py](/Users/christopherlin/dev/cwsf2026/sim/train/mappo_gru.py)

If asked, "Where is the trained model loaded?"

- [inference.py](/Users/christopherlin/dev/cwsf2026/sim/algorithms/mappo/inference.py)
  for simulator inference
- [run.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run.py) for robot deployment

If asked, "Where is the sim-to-real bridge?"

- [run.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run.py)

If asked, "Where is Mission Control?"

- [mission_control/main.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/main.py)

If asked, "Where do digital pheromones live in the real system?"

- inside Mission Control

If asked, "Why does the deployment work at all?"

- because the robot runtime preserves the simulator’s observation and action
  contract as closely as possible

## Final One-Minute Version

If you need to explain the whole codebase in one minute, say this:

The simulator in [swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
defines a decentralized swarm world with local observations, discrete actions,
food, obstacles, and a pheromone field. Training starts in
[train.py](/Users/christopherlin/dev/cwsf2026/sim/train/train.py) and mainly
uses recurrent MAPPO in
[mappo_gru.py](/Users/christopherlin/dev/cwsf2026/sim/train/mappo_gru.py),
where curriculum learning gradually teaches pickup, homing, delivery, and swarm
coordination. Mission Control in
[mission_control/main.py](/Users/christopherlin/dev/cwsf2026/sim/mission_control/main.py)
acts as the live shared digital pheromone system for the physical deployment.
Finally, [firmware/run.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run.py)
loads the trained policy, converts real robot sensor data into the same
observation format used in simulation, runs inference, and sends actions to the
physical robot through [ant.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/ant.py).

That is the whole story:

- build the world
- train the policy
- preserve the interface
- deploy the behavior
