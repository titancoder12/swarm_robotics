# Q&A

This file captures recurring questions and answers discussed during development so future contributors can reference current decisions quickly.

## Q: Is `ReplayBuffer` the same as a trajectory?
A: No. The replay buffer stores individual transitions `(s, a, r, s', done)` and does not preserve episode order. A trajectory is an ordered sequence of transitions.

## Q: Does PettingZoo replace PyTorch or do learning/inference?
A: No. PettingZoo is only the environment API. Learning and inference are handled by the RL algorithm implementation, such as the custom DQN trainer, SB3, or RLlib.

## Q: If I switch to an RL library, does the environment need to change?
A: Usually no. As long as [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py) keeps the PettingZoo Parallel API contract, you can swap the learning backend without changing the environment.

## Q: Is inference the same across RL libraries?
A: No. Each library has its own checkpoint format and predict/load API, so inference code differs by backend even if the environment stays the same.

## Q: Is `swarm_env.py` “RL code”?
A: It is the environment that defines the MDP: observations, actions, rewards, and termination. It does not learn. The training code lives in [train/](/Users/christopherlin/dev/cwsf2026/sim/train/).

## Q: In sim-to-real transfer, is `_get_obs()` the key interface?
A: Largely yes. `_get_obs()` defines the policy input contract. On the robot, the equivalent is the sensor preprocessing pipeline that must produce the same feature layout and normalization.

## Q: Does `_handle_targets()` reward agents based on distance?
A: No. In simulation it rewards collection events, not shaped distance progress. In the real world you would need a real detection signal for “target collected” or redesign the reward around directly measurable events.

## Q: If reward depends on a target radius, does the environment need to know target locations?
A: In simulation, yes. The environment tracks target positions to decide whether collection happened. In the real world, the equivalent should come from sensors rather than hidden ground-truth coordinates.

## Q: What needs to change for real-world deployment?
A: See [docs/ToDo.md](/Users/christopherlin/dev/cwsf2026/sim/docs/ToDo.md) for the current checklist covering sensors, observations, action interface, target detection, reset procedures, safety, and domain gap issues.

## Q: What does it mean that the env is PettingZoo-native?
A: It means [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py) implements the PettingZoo Parallel API directly with per-agent dict I/O.

## Q: What does MLP stand for?
A: Multi-Layer Perceptron.

## Q: Do SB3 and RLlib do the training themselves or use an underlying DL library?
A: They do the RL training loop but use a deep learning backend for neural networks. In this repo, that backend is PyTorch.

## Q: What’s the relationship between PettingZoo, SB3/RLlib, and PyTorch?
A: PettingZoo is the environment API, SB3/RLlib are RL libraries, and PyTorch is the neural-network backend used by those libraries.

## Q: Is a Q-function a policy?
A: Not exactly. A Q-function scores actions. In DQN, the deployed policy is usually the greedy action `argmax_a Q(s, a)`.

## Q: Are `batch_size`, `lr`, and `epsilon_*` all optimizer settings?
A: No. `batch_size` and `lr` affect gradient updates. `epsilon_*` controls exploration in the policy.

## Q: What does “PettingZoo Parallel” mean?
A: All active agents act at the same timestep, and `step()` consumes one action per active agent and returns dicts keyed by agent ID.

## Q: How should I learn this project from scratch as a beginner?
A: Start with [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py), then [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py), then [train/demo.py](/Users/christopherlin/dev/cwsf2026/sim/train/demo.py), then [train/train.py](/Users/christopherlin/dev/cwsf2026/sim/train/train.py), and finally the sim-to-real notes in [docs/SimToReal.md](/Users/christopherlin/dev/cwsf2026/sim/docs/SimToReal.md).

## Q: What should we pay attention to during training for sim-to-real transfer?
A: Focus on domain gap: observation normalization, action scaling, noise, delay, environment randomization, and safety constraints.

## Q: Can you explain RL concepts by mapping them directly to this repo?
A: Yes. The environment is [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py), the policy/value model is [models/q_network.py](/Users/christopherlin/dev/cwsf2026/sim/models/q_network.py), the custom learning loop is [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py), and the simulator demo path is [train/demo.py](/Users/christopherlin/dev/cwsf2026/sim/train/demo.py).

## Q: How should we architect sim-to-real deployment on a Raspberry Pi with an Arduino or ESP32 motor/sensor layer?
A: Keep the learned observation and action contracts stable. The Pi should build the same observation vector the model expects, run inference, and send a discrete action to a lower-level controller on the microcontroller side.

## Q: Can you explain `QNetwork`, the Bellman update, and one full training iteration in this repo together?
A: Yes. See [docs/DQN_EXPLAINED.md](/Users/christopherlin/dev/cwsf2026/sim/docs/DQN_EXPLAINED.md) and [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py). The short version is: collect transitions, store replay, sample minibatches, compute Bellman targets with the target network, update the online network, and periodically sync the target network.

## Q: What are `custom`, `sb3`, and `rllib`, and when should I pick each one?
A: They are the three training backends dispatched by [train/train.py](/Users/christopherlin/dev/cwsf2026/sim/train/train.py). Use `custom` for learning and code changes, `sb3` for a lighter library baseline, and `rllib` when you specifically need Ray/RLlib tooling.

## Q: Is the sim-to-real runtime basically the same as the custom demo inference loop?
A: For the policy step, yes. Both build an observation, run the network, choose an action, and hand that action to an executor. In sim the executor is `env.step(...)`; on hardware it is the robot command layer.

## Q: Can the checkpoint and a small inference snippet be used directly instead of a larger deployment framework?
A: Yes. The current robot runtime in [firmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run_policy.py) follows that approach.

## Q: What is the current minimal robot-side runtime dependency set?
A: The minimum live path is [firmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run_policy.py), [firmware/ant.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/ant.py), [models/q_network.py](/Users/christopherlin/dev/cwsf2026/sim/models/q_network.py), the checkpoint files, and runtime dependencies such as `torch`, `numpy`, and `pyserial`.

## Q: Does the current `ant.py` / `ant.service` setup look like Raspberry Pi-side robot code?
A: Yes. It is Linux userspace code running on the Pi and sending higher-level commands to an ESP32 over serial.

## Q: Is there an example in this repo showing how the Raspberry Pi code can run the learned policy?
A: Yes. The current example is [firmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run_policy.py).

## Q: What does the current `ant.py` Raspberry Pi control loop do?
A: It is a rule-based serial control loop that reads scans and issues turn, move, stop, or brake commands. It is not the learned-policy runner.

## Q: Is there a script in this repo that preserves the direct serial robot behavior?
A: Yes. [firmware/ant.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/ant.py) is the direct serial control helper and runtime.

## Q: Is there also a systemd service file in this repo?
A: Yes. See [firmware/ant.service](/Users/christopherlin/dev/cwsf2026/sim/firmware/ant.service).

## Q: What is the intended learned-policy runtime on the Raspberry Pi now?
A: [firmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run_policy.py) is the intended robot inference runtime.

## Q: How do I run the policy probing script?
A: From the repo root, run `python train/policy_probe.py --list-cases` to list cases. To probe a shared policy, run `python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead`. For an independent policy, use `--agent-index`.

## Q: What does `--headless` do in training?
A: It disables PyGame window rendering. Environment stepping, observations, rewards, and learning still run normally.

## Q: Why save checkpoints during training?
A: They let you resume runs, compare model quality across training stages, run later demos/evaluations, and avoid losing all progress from a crash.

## Q: In `train/independent_dqn_pytorch.py`, what does `--save-dir checkpoints` do?
A: It sets the checkpoint output directory. The default is already `checkpoints/`, and the trainer still writes a final checkpoint there even if `--save-every` is `0`.

## Q: Can you walk me through training in `train/independent_dqn_pytorch.py`?
A: The file builds the environment, infers `obs_dim`, creates networks, target networks, optimizers, and replay buffers, then loops through epsilon-greedy data collection, replay storage, Bellman updates, target sync, evaluation, logging, and checkpoint writing.

## Q: What does `train/evaluate.py` do?
A: It runs headless evaluation episodes and writes evaluation metrics. It does not train.

## Q: What does `train/policy_probe.py` do?
A: It is a checkpoint inspection tool that feeds hand-written observations into the custom DQN and prints the resulting Q-values, chosen action, and a short interpretation.

## Q: What do the observation values in `train/policy_probe.py` mean?
A: They are the policy input features: lidar readings, target direction, optional nest direction, neighbor direction, heading encoding, speed, optional food/carry flags, and pheromone samples.

## Q: How do I translate the normalized observation values into millimeters?
A: First convert normalized values back into sim units using config scales such as `lidar_max_range` and `max_speed`. Then apply an external calibration like `mm_per_sim_unit`.

## Q: What millimeter scale would you recommend for this sim?
A: A reasonable starting point is `1 sim unit = 10 mm`, but the right calibration should be anchored to your real robot size.

## Q: What scale mapping does `firmware/run_policy.py` use?
A: It uses explicit physical CLI parameters such as `--lidar-max-range-mm`, `--move-distance-mm`, `--reverse-distance-mm`, and `--turn-step-deg`. It does not apply a hidden sim-unit conversion inside the runtime.

## Q: Does `firmware/run_policy.py` treat TOF `-1` as out-of-range / far?
A: Yes. Negative readings are ignored and the bucket remains at its default max-range value, which is then normalized as far.

## Q: What is the mm-to-units relationship in `firmware/run_policy.py`, and how do I tweak it?
A: The runner itself uses explicit millimeter and degree CLI arguments rather than a hidden sim-unit conversion. Adjust the scale by changing those CLI arguments.

## Q: Given a real robot radius of 70 mm, what should the sim-unit scale be?
A: Using `mm_per_sim_unit = real_robot_radius_mm / cfg.agent_radius` and the current default `agent_radius = 10`, the scale would be `7 mm` per sim unit.

## Q: Where do I find the sim-unit calibration info?
A: The current robot radius config lives in [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py), and the calibration guidance is documented in this file and the config docs.

## Q: Would `agent_radius = 7` and `target_radius = 3` work?
A: Yes, if you want to keep a `1 sim unit = 10 mm` calibration. Whether `target_radius = 3` is a good task setting depends on the physical target size and pickup tolerance.

## Q: How do I download the model and run `firmware/run_policy.py` on the Pi?
A: Copy the repo or at least [firmware/](/Users/christopherlin/dev/cwsf2026/sim/firmware/), [models/](/Users/christopherlin/dev/cwsf2026/sim/models/), and [checkpoints/](/Users/christopherlin/dev/cwsf2026/sim/checkpoints/) to the Pi, install runtime dependencies including `pyserial`, then run `python firmware/run_policy.py --checkpoint-dir checkpoints` or add `--shared-policy` to load `shared.pt`.

## Q: If I download the model into the same directory, how do I use it?
A: `firmware/run_policy.py` expects `--checkpoint-dir` to point to a directory, not a direct `.pt` file path. It then looks for `shared.pt` or `agent_0.pt` inside that directory.

## Q: Can I run `firmware/run_policy.py` without using PyTorch, just for runtime inference?
A: Not as currently written. It loads a PyTorch checkpoint into [models/q_network.py](/Users/christopherlin/dev/cwsf2026/sim/models/q_network.py) and runs the forward pass with PyTorch.

## Q: How do I run `agent_0.pt`?
A: You do not run it directly. It is a checkpoint file. Use [firmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run_policy.py) for robot inference or [train/demo.py](/Users/christopherlin/dev/cwsf2026/sim/train/demo.py) for simulator playback.

## Q: Are `agent_0.pt`, `agent_1.pt`, etc. individual agents or training versions?
A: They are per-agent checkpoints for distinct swarm agents. The alternative is `shared.pt`, which is one policy reused by all agents.

## Q: Where do I find the rewards?
A: Reward constants are defined in [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py), and the reward logic is applied in [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py).

## Q: How do I run training?
A: The main custom command is `python train/independent_dqn_pytorch.py --headless --total-steps 10000`. You can also use [train/train.py](/Users/christopherlin/dev/cwsf2026/sim/train/train.py) to dispatch to SB3 or RLlib.

## Q: Is the current RL deterministic or stochastic?
A: Training is stochastic because the custom trainer uses epsilon-greedy exploration and random replay sampling. Inference is deterministic because the deployed action rule uses greedy `argmax`. The environment can also be stochastic depending on reset randomness and dynamics mode.

## Q: Is the deployed robot-side inference model deterministic?
A: Yes. [firmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/firmware/run_policy.py) chooses actions with `torch.argmax(...)`, so the same observation and same checkpoint yield the same action.

## Q: Is it hard to make the deployed robot policy stochastic, and can that be done without changing the interface?
A: No. You can keep the same observation vector, checkpoint format, and discrete action output while changing only the action-selection rule, for example with epsilon-greedy inference, softmax sampling over Q-values, or small noise before `argmax`.
