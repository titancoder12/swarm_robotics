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

## Q: Can you explain `QNetwork`, the Bellman update, and one full training iteration in this repo together?
A: Yes. `QNetwork` in `train/independent_dqn_pytorch.py` is a small MLP that maps one observation vector to one Q-value per discrete action. The Bellman target in `train()` is `reward + gamma * max_a' Q_target(next_obs, a')` unless the episode ended, and training minimizes the gap between that target and the current `Q(obs, action)`. One full training iteration is: build actions with epsilon-greedy, call `env.step()`, store transitions in replay, sample random minibatches after warmup, compute Bellman targets, update the online network, and periodically copy weights to the target network.

## Q: Where is the standalone doc that explains `QNetwork`, the Bellman update, and one full training iteration?
A: See `docs/DQN_EXPLAINED.md`. It consolidates the explanation of tensor shapes, the Bellman target used in the custom trainer, and the step-by-step control flow of one training iteration in `train/independent_dqn_pytorch.py`.

## Q: What are `custom`, `sb3`, and `rllib`, and when should I pick each one?
A: They are the three training backends selected by `train/train.py`. `custom` runs the repo’s own PyTorch DQN trainer in `train/independent_dqn_pytorch.py`, which is the easiest option for learning and modifying the algorithm. `sb3` runs Stable-Baselines3 DQN through `train/sb3_dqn.py`, which is simpler than RLlib and useful when you want a standard library implementation with fewer moving parts. `rllib` runs Ray RLlib DQN through `train/rllib_dqn.py`, which is heavier but more suitable when you want richer multi-agent/distributed tooling. In this repo, start with `custom` for understanding and experiments, use `sb3` when you want a cleaner library baseline, and reach for `rllib` only if you specifically need Ray/RLlib capabilities.

## Q: Is the sim-to-real runtime basically the same as the custom demo inference loop?
A: Yes, for the policy portion it is very close. The key pieces in `train/demo.py` are loading the model, building `obs_tensor`, running the network to get Q-values, taking `argmax` to choose the action, and then handing that action to a downstream executor. In sim, the downstream executor is `env.step(...)`; on the robot, it should be a real-world action bridge that translates the chosen action into physical commands. The main extra requirement is that the Pi must build observations from real sensors with the same feature order and normalization the model was trained on.

## Q: Is the `robot/` directory meant to support that sim-to-real inference loop?
A: Yes. `robot/policy_runner.py` handles checkpoint loading and inference, `robot/observation_builder.py` turns sensor packets into policy observations, `robot/sensor_bridge.py` reads incoming sensor data, `robot/action_bridge.py` maps discrete actions into high-level robot commands, and `robot/runtime.py` wires the loop together. It is the Pi-side deployment skeleton for the same inference pattern used in the custom demo.

## Q: Can the checkpoint and a small inference snippet be used directly instead of the full `robot/` package?
A: Yes, that can be a reasonable integration strategy. The key requirement is interface compatibility: the checkpoint format, observation feature order and normalization, and action semantics must match training exactly. Also, large checkpoints usually should not be committed directly to Git unless you intentionally use an artifact mechanism such as Git LFS or release assets.

## Q: If we use the `robot/` package on the Raspberry Pi, do we still need `env/`, `train/`, or `swarm_env.py`?
A: You do not need `env/swarm_env.py` for robot inference. After refactoring, the intended minimal runtime dependency is `robot/`, `env/config.py`, `models/q_network.py`, and the checkpoint. The Raspberry Pi no longer needs `train/independent_dqn_pytorch.py` just to load the custom DQN model.

## Q: Does the current `ant.py` / `ant.service` setup look like Raspberry Pi-side robot code?
A: Yes, it looks like Raspberry Pi-side runtime code rather than microcontroller firmware. The Python script opens `/dev/serial0` with `pyserial`, imports `cv2`, and sends high-level commands to an ESP32, while the service file is a standard Linux `systemd` unit. That is consistent with Linux userspace code running on a Raspberry Pi and delegating low-level control to a microcontroller.

## Q: Is there an example in this repo showing how the Raspberry Pi code can run the learned policy?
A: Yes. The `pi/` folder is the Raspberry Pi reference integration. `pi/esp32_robot.py` wraps the ESP32 serial protocol, `pi/esp32_sensor_adapter.py` converts scan lines into `SensorPacket`, `pi/esp32_action_bridge.py` maps policy actions back into the ESP32 command set, and `pi/run_policy.py` connects those pieces to `robot/`, `models/q_network.py`, and `env/config.py` so the checkpoint can drive the physical robot.

## Q: What does the current `ant.py` Raspberry Pi control loop do?
A: It is a Raspberry Pi-side Python control loop that connects to an ESP32 over `/dev/serial0`, reads JSON sensor scan lines, picks a simple handcrafted movement direction based on free space, and sends high-level serial commands like turn, move, stop, and brake to the ESP32. It is not running the learned model; it is running a rule-based obstacle-avoidance style behavior.

## Q: Is there a Pi script in this repo that preserves the current behavior?
A: Yes. `pi/ants.py` preserves the current Raspberry Pi control loop. It keeps the same rule-based serial behavior while the Pi runtime transitions gradually toward the shared deployment modules and the learned model.

## Q: Is there also a systemd service file in this repo to mirror the current Raspberry Pi deployment setup?
A: Yes. `pi/ants.service` matches the current Raspberry Pi launch pattern. It is meant to start `pi/ants.py` at boot, but `WorkingDirectory` and `ExecStart` should be updated if the repo is cloned somewhere other than `/home/pi/ant`.

## Q: Is `pi/run_policy.py` the intended end result of migrating from the rule-based Raspberry Pi controller to the learned model?
A: Yes. `pi/ants.py` preserves the current rule-based behavior, while `pi/run_policy.py` is the intended migrated runtime that uses `SensorPacket -> ObservationBuilder -> PolicyRunner -> CommandMapper -> ESP32ActionBridge` to drive the robot from the learned policy. It is the architectural end state of the migration, though it may still need hardware-specific calibration before production use.

## Q: Is the current code backward compatible with commit `8ac33636556db0af11cd7c09b9b57e03d94a40d1`?
A: Not fully. The main breaking change is the environment and learning contract: that commit still used the older 19-dimensional observation space, while the current code uses a 23-dimensional observation that adds nest-aware foraging features such as nest direction, local food presence, and carrying-food state. As a result, old checkpoints from that era are not compatible with the current trainer/evaluator without retraining. The `Discrete(9)` action interface remains compatible, and the `pi/` code added in that commit still exists, but overall the current codebase should be treated as source-compatible for many commands, not checkpoint-compatible or behavior-identical with that commit.

## Q: If `ants.py` becomes the main Raspberry Pi runtime, are the `esp32_*.py` files still needed?
A: Not strictly. `pi/esp32_robot.py` is a serial client wrapper, `pi/esp32_sensor_adapter.py` converts raw scan lines into the generic sensor packet format used by `robot/`, and `pi/esp32_action_bridge.py` maps generic action commands back into the ESP32 serial protocol. If all sensor reading, observation building, model inference, and action sending are implemented directly inside `pi/ants.py`, those helper modules are no longer required at runtime. They are still useful as reference code or as a cleaner modularization path if the Pi runtime grows and you want to split hardware IO, observation translation, and action translation into separate files later.

## Q: What files changed since revision `8ac33636556db0af11cd7c09b9b57e03d94a40d1`?
A: Relative to that revision, the current branch differs in these files: `.gitignore`, `README.md`, `analysis/plot_metrics.py`, `docs/ARCHITECTURE.md`, `docs/ONBOARDING.md`, `docs/PI_MIGRATION.md`, `docs/PROJECT_LOG.md`, `docs/QandA.md`, `docs/SimToReal.md`, `docs/manual/EXPERIMENT_GUIDE.md`, `docs/manual/PROJECT_STRUCTURE.md`, `docs/manual/QUICK_START.md`, `docs/manual/RESULTS_INTERPRETATION.md`, `docs/prompts/Integration_prompt4.txt`, `docs/prompts/integration_prompt1.txt`, `docs/prompts/integration_prompt2.txt`, `docs/prompts/integration_prompt3.txt`, `docs/prompts/integration_prompt5.txt`, `docs/prompts/integration_prompt6.txt`, `docs/prompts/integration_prompt7.txt`, `env/config.py`, `env/swarm_env.py`, `experiments/benchmark_configs.py`, `models/rule_based_policy.py`, `requirements.txt`, `robot/observation_builder.py`, `train/evaluate.py`, `train/experiment_utils.py`, `train/independent_dqn_pytorch.py`, and `train/run_experiments.py`.

## Q: How do I run the policy probing script?
A: From the repo root: `python train/policy_probe.py --list-cases` shows the available hand-written observation cases. To probe a shared-policy checkpoint, run `python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead`. To probe an independent checkpoint, run `python train/policy_probe.py --checkpoint-dir checkpoints --case wall_ahead --agent-index 0`.

## Q: What does `--headless` do in training?
A: It runs the environment without opening a PyGame window. In `SwarmEnv.__init__`, `headless=True` sets `render_mode=None`, `render()` becomes a no-op, and screenshot saving is disabled. The environment logic, observations, rewards, and stepping still run normally; only on-screen rendering is skipped. This is the normal mode for faster training and evaluation.

## Q: Why save checkpoints during training?
A: Checkpoints let you keep intermediate versions of the learned policy instead of only the final one. In this repo that is useful for at least four reasons: you can resume or inspect a run if training stops early, you can demo or evaluate earlier models without retraining, you can compare model quality across training stages, and you avoid losing all progress from a long run if something crashes. They are especially useful now that training/evaluation and experiment scripts can load saved models later.

## Q: In `train/independent_dqn_pytorch.py`, what does `--save-dir checkpoints` do, and what happens if I do not pass it?
A: `--save-dir` tells the trainer where to write checkpoint files. In the current code its default is already `checkpoints`, so if you do not pass `--save-dir`, it still saves to the `checkpoints/` folder. The separate `--save-every` flag controls whether intermediate checkpoints are also written during training. Even if `--save-every` is left at its default `0`, the trainer still saves one final checkpoint at the end into `args.save_dir`.

## Q: Can you walk me through training in `train/independent_dqn_pytorch.py`?
A: Yes. The file follows a standard DQN structure. `parse_args()` defines the CLI, `train(args)` builds the environment and run directory, infers `obs_dim` from `env.reset()`, creates either independent or shared `QNetwork` instances plus matching target networks, optimizers, and replay buffers, then enters the main loop. Each loop iteration computes epsilon, selects one action per agent with epsilon-greedy, steps the environment, stores `(obs, action, reward, next_obs, done)` into replay, and after warmup samples minibatches to apply the Bellman update with Huber loss. Every `target_update` steps it syncs target networks, optionally runs `_evaluate_policy()`, logs episode metrics when an episode ends, optionally saves checkpoints during training, and always saves final checkpoints plus `summary.json` at the end.

## Q: What does `train/evaluate.py` do?
A: It runs headless evaluation episodes and writes evaluation metrics; it does not train. It builds a `SwarmEnv`, infers `obs_dim` from `env.reset()`, then either loads custom DQN checkpoints with `_load_models(...)` or builds rule-based policies with `_build_rule_based_policies(...)`. For each evaluation episode it resets the env with a deterministic seed, runs greedy action selection (`argmax` over Q-values for DQN, `.act(obs)` for rule-based), steps the env until termination/truncation, accumulates metrics like mean reward, food retrieved, exploration coverage, pheromone usage, episode length, and swarm efficiency, logs one row per episode to `eval_metrics.csv`, and finally writes aggregated means to `eval_summary.json`.
