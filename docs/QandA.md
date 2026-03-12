# Q&A

This file captures recurring questions and answers discussed during development so future contributors can reference decisions quickly.

## Q: Is `ReplayBuffer` the same as a trajectory?
A: No. The replay buffer stores **individual transitions** `(s, a, r, s', done)` and does not preserve episode order. A trajectory is an **ordered sequence** of transitions. The buffer may contain pieces of trajectories, but it is not itself a trajectory.

## Q: Does PettingZoo replace PyTorch or do learning/inference?
A: No. PettingZoo is only the **environment API**. Learning/inference is done by an RL algorithm (custom DQN, SB3, RLlib) implemented separately, typically using a deep learning framework like PyTorch.

## Q: If I switch to an RL library, does the environment need to change?
A: Usually no. As long as `SwarmEnv` follows the PettingZoo Parallel API, you can swap the learning backend without changing the environment. You may add wrappers if a library expects a different API.

## Q: Is inference the same across RL libraries?
A: No. Each library saves models differently and has its own load/predict API. A unified demo script can route to the correct loader based on `--backend`.

## Q: Is `swarm_env.py` “RL code”?
A: It is the **environment** that defines the MDP: observations, actions, rewards, and termination. It does not learn. The training code is separate.

## Q: In sim-to-real transfer, is `_get_obs()` the key interface?
A: Largely yes: `_get_obs()` maps world state to the observation vector the policy uses. On a robot, the equivalent is the **sensor preprocessing pipeline** that produces the same vector. You also need an **action interface** that maps actions to motor commands.

## Q: Does `_handle_targets` reward agents based on distance, and how would this work in the real world?
A: In the sim, `_handle_targets()` gives a reward **only when a target is collected** (within the target radius). It does **not** reward based on distance. In the real world, you’d need a **detectable target** (e.g., marker/vision/RFID) so the robot can confirm contact, or you’d change the task so rewards come from its own sensors (e.g., signal strength/proximity). The observation pipeline must be built from real sensors, and “target collected” must be detectable by the robot.

## Q: If reward depends on a target radius, does the environment need to know target locations?
A: In simulation, yes—the environment tracks target positions to determine if an agent is within the collection radius. That’s not necessarily distance-shaped reward; it’s a **binary “collected” check**. In the real world, the equivalent is a **target detection signal** from sensors, not a hidden oracle.

## Q: What needs to change for real-world deployment?
A: See `docs/ToDo.md` for a required-change checklist covering sensors/observations, action interface, target detection, reset procedures, safety, domain gap mitigation, runtime loop, and logging.

## Q: What does it mean that the env is PettingZoo-native?
A: It means `SwarmEnv` implements the **PettingZoo Parallel API** directly (dict-based `reset()` and `step()` with per-agent observations/rewards/terminations). You can plug it into PettingZoo-compatible tooling without extra wrappers.

## Q: What does MLP stand for?
A: **Multi-Layer Perceptron** — a standard feedforward neural network with fully connected layers.

## Q: Do SB3/RLlib do the training themselves or use an underlying DL library?
A: They **do the training**, but they are built on top of a deep learning framework. SB3 uses **PyTorch**. RLlib supports **PyTorch** (and sometimes other backends). The library manages the RL algorithm loop; the DL framework handles neural nets and gradients.

## Q: What’s the relationship between PettingZoo, SB3/RLlib, and PyTorch?
A: PettingZoo is the **environment API** (multi-agent interface). SB3/RLlib are **RL training libraries** that implement algorithms and use environments. PyTorch is the **deep learning backend** used by those libraries to build and train neural networks. In short: PettingZoo (env) → SB3/RLlib (RL algorithm) → PyTorch (NNs/gradients).

## Q: Is a Q-function a policy?
A: Not exactly. A **Q-function** estimates how good an action is in a state. A **policy** chooses actions. In DQN, the policy is derived from the Q-function by taking `argmax_a Q(s, a)` (often with epsilon-greedy exploration).

## Q: Are `batch_size`, `lr`, and `epsilon_*` for stochastic batch gradient descent?
A: **Partly.** `batch_size` and `lr` control the **stochastic gradient descent** updates (how many samples per update, how big each step is). `epsilon_*` is **not** about gradient descent; it controls **exploration** in the policy (random action rate), not the optimizer.

## Q: What does "PettingZoo Parallel" mean?
A: It refers to PettingZoo's **Parallel API** for multi-agent environments. All agents act at the same timestep, and `step()` takes a dict of actions for all active agents, then returns dicts of observations, rewards, terminations, truncations, and infos keyed by agent ID.

## Q: How should I learn this project from scratch as a beginner?
A: Start with this sequence: (1) RL basics and terms, (2) environment/MDP in `env/swarm_env.py`, (3) custom DQN training in `train/independent_dqn_pytorch.py`, (4) inference in `train/demo.py`, (5) backend switching via `train/train.py` (custom/SB3/RLlib), then (6) sim-to-real considerations in `docs/ToDo.md`.

## Q: For sim-to-real transfer, what should we pay attention to during training?
A: Focus on **domain gap**: match observation normalization and action scaling, add sensor/dynamics noise, include delays, and train with randomized environments. Consider safety penalties and curriculum training. These items were added to `docs/ToDo.md`.

## Q: Can you teach me the code and the RL flow in this repo?
A: Start from `env/swarm_env.py`: that file is the environment and defines the RL problem through `reset()`, `step()`, observations, rewards, and episode endings. Then read `train/independent_dqn_pytorch.py`: that file is the learning loop that collects transitions, stores them in replay buffers, samples minibatches, computes Bellman targets, and updates Q-networks. `train/train.py` is only a dispatcher that picks the backend (`custom`, `sb3`, or `rllib`), and `train/demo.py` is inference-only for running trained checkpoints.

## Q: Can you explain RL concepts for a beginner by mapping them directly to this repo?
A: Yes. In this repo, the **environment/MDP** is `env/swarm_env.py`, the **state/observation** is the vector built by `_get_obs()`, the **action space** is the 9-way discrete table from `_build_action_table()`, the **reward** is computed in `step()` and `_handle_targets()`, the **policy/value model** is `QNetwork` in `train/independent_dqn_pytorch.py`, and the **learning loop** is `train()` in that same file. The easiest way to learn RL here is to follow one cycle: observation → action selection → `env.step()` → reward/next observation → replay buffer → Bellman update.

## Q: How should we architect sim-to-real deployment on a Raspberry Pi with an Arduino sensor/motor layer while minimizing code changes?
A: Keep the learned policy and observation/action shapes unchanged as much as possible. Put a **real-world adapter** on the Raspberry Pi that replaces the simulator’s `_get_obs()` and `step()` side effects: it should read fused sensor data from the Arduino and Pi-side estimators, build the same normalized observation vector the policy expects, run inference, and send a high-level action command to a low-level controller. The Arduino should remain a real-time I/O layer for sensors and motor control, while the Pi stays the policy/runtime “brain.” Use a small message contract such as `sensor_packet -> observation_builder -> policy_inference -> action_command -> motor_controller`, with safety overrides outside the policy.

## Q: What files implement the first sim-to-real deployment skeleton in this repo?
A: The detailed design is documented in `docs/SimToReal.md`. The Pi-side runtime skeleton lives in `robot/`: `sensor_bridge.py` reads `SensorPacket` data, `observation_builder.py` converts packets into policy observations, `policy_runner.py` loads the trained DQN checkpoint and runs inference, `action_bridge.py` maps actions into high-level commands, and `runtime.py` connects the loop together. This keeps `env/` and `train/` largely unchanged.
