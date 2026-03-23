# Project Log

Use this file to capture decisions, open questions, and next steps so we can resume smoothly across sessions.

## Summary
- Multi-agent PyGame swarm environment with stigmergy.
- PettingZoo Parallel API for dict-based multi-agent interactions.
- Discrete 9-action joystick interface.
- DQN training with independent or shared policies.
- Optional SB3/RLlib DQN training backends.
- Demo and screenshot tooling included.
 - RLlib backend uses compatibility shims and may require setting `RAY_TMPDIR` if `/tmp` is full.

## Change Log
- Added a repo rule: when Codex updates files, it must log a short summary in `docs/PROJECT_LOG.md` (added to `AGENTS.md`).
- Summary of recent work: converted `SwarmEnv` to PettingZoo Parallel API, updated training/demo scripts to dict-based I/O, added SB3/RLlib backends with dispatcher, expanded dependencies in `requirements.txt`, updated docs, and added RLlib compatibility shims; checkpoints saved for RLlib in `checkpoints/rllib_dqn/`.
- Fixed RLlib demo loading by normalizing the checkpoint path to a `file://` URI in `train/demo.py`.
- Patched RLlib demo to coerce replay buffer type to a string during `Algorithm.from_checkpoint`.
- Registered the `swarm_pz` env in RLlib demo before loading checkpoints.
- Disabled strict RLlib DQN config validation during demo checkpoint load to avoid replay buffer type errors.
- Added `--max-steps` to `train/demo.py` so demos can auto-exit for smoke tests.
- Added `--ray-tmpdir` flag to RLlib training and demo to avoid `/tmp` pressure and socket path length issues.
- Added `.ray_tmp/` to `.gitignore`.
- Added `docs/ONBOARDING.md` with a code-first RL walkthrough for future developers.
- Expanded `docs/ONBOARDING.md` with the DQN trainer mapping (theory → code and key blocks).
- Added `docs/QandA.md` to capture common questions and answers.
- Added an `AGENTS.md` rule to log system/codebase questions into `docs/QandA.md`.
- Added an Environment MDP diagram to `docs/UML.md`.
- Logged a Q&A about `_handle_targets` rewards and sim-to-real implications.
- Logged a Q&A clarifying that target radius checks imply environment target knowledge in sim.
- Added `docs/ToDo.md` with a sim-to-real required-change checklist.
- Logged Q&A about what it means for the env to be PettingZoo-native.
- Logged Q&A defining MLP (Multi-Layer Perceptron).
- Added inline comments to `train/independent_dqn_pytorch.py` and expanded the DQN walkthrough in `docs/ONBOARDING.md`.
- Logged Q&A about SB3/RLlib training vs underlying DL libraries.
- Logged Q&A about PettingZoo vs SB3/RLlib vs PyTorch roles.
- Logged Q&A explaining that a Q-function is not itself a policy.
- Logged Q&A clarifying that `batch_size`/`lr` are optimizer settings while `epsilon_*` is exploration.
- Expanded `docs/ToDo.md` with sim-to-real training checks and logged the related Q&A.
- Added line-by-line annotations to `train/independent_dqn_pytorch.py` and noted this in `docs/ONBOARDING.md`.
- Logged Q&A defining what "PettingZoo Parallel" means.
- Added `docs/RESOURCES.md` with official links (PettingZoo/Gymnasium), GitHub repos, and learning resources.
- Added Stanford CS234 YouTube playlist to `docs/RESOURCES.md`.
- Logged beginner learning path Q&A in `docs/QandA.md`.
- Logged a Q&A summarizing how to learn the repo by following the environment, custom DQN trainer, backend dispatcher, and demo flow.
- Logged a beginner-oriented Q&A mapping core RL concepts directly onto the main environment and DQN trainer files.
- Logged a Q&A describing a minimal-change Raspberry Pi + Arduino sim-to-real architecture with a Pi-side real-world adapter and Arduino low-level control layer.
- Added `docs/SimToReal.md` with a detailed Raspberry Pi + Arduino deployment design and added a `robot/` package containing `SensorBridge`, `ObservationBuilder`, `PolicyRunner`, `ActionBridge`, and a runtime loop skeleton.
- Logged a Q&A summarizing `QNetwork`, the Bellman update, and one full DQN training iteration in the custom trainer.
- Added `docs/DQN_EXPLAINED.md` to capture the custom DQN trainer explanation in one place and logged a Q&A pointing to it.
- Logged a Q&A comparing the `custom`, `sb3`, and `rllib` training backends and when to choose each one.
- Logged a Q&A clarifying that the sim-to-real policy loop can mirror the custom demo inference loop, with sensor-built observations and a real-world action bridge replacing `env.step(...)`.
- Expanded `docs/SimToReal.md` to map the `robot/` package onto the custom demo inference loop and logged a Q&A confirming that `robot/` is the Pi-side sim-to-real deployment skeleton.
- Logged a Q&A covering the tradeoff between sharing only the checkpoint plus a minimal inference snippet versus using this repo’s `robot/` package.
- Logged a Q&A clarifying the actual minimal Pi-side runtime dependencies: `swarm_env.py` is not needed, but the current `robot/` package still depends on `env/config.py` and `train/independent_dqn_pytorch.py` for `QNetwork`.
- Logged a Q&A confirming that the current `ant.py` / `ant.service` setup looks like Raspberry Pi-side runtime code, based on its Python script, systemd service, and serial communication with an ESP32.
- Added a `pi/` reference integration layer based on the existing Raspberry Pi control structure, showing how Raspberry Pi code can reuse `robot/`, `models/q_network.py`, and `env/config.py` to run the trained policy on the physical robot.
- Logged a Q&A summarizing what `ant.py` currently does: serial connection, scan reading, handcrafted free-space action selection, and ESP32 command transmission.
- Added `pi/ants.py` as a preserved Raspberry Pi control script matching the current rule-based behavior and added `docs/PI_MIGRATION.md` to guide gradual migration from that baseline toward model-based control.
- Added `pi/ants.service` as a systemd unit mirroring the current Raspberry Pi deployment pattern and documented the path-adjustment requirement in `docs/PI_MIGRATION.md`.
- Expanded `docs/PI_MIGRATION.md` to state explicitly that `pi/run_policy.py` is the intended migrated end state, while `pi/ants.py` remains the baseline, and logged the matching Q&A.
- Extracted `QNetwork` into `models/q_network.py` and updated training, demo, and robot inference code to share it, removing the robot runtime's dependency on the training script.
- Upgraded the environment for science-fair-grade stigmergic foraging support by adding configurable nest mechanics, food pickup and nest delivery, richer local observations (nest direction, food presence, carrying state), nest rendering, and aligned the robot observation builder to the new observation contract.
- Upgraded the custom training pipeline for research-grade experiments by adding structured episode/evaluation logging, checkpoint metadata, optional deterministic evaluation during training, training plot generation, and standalone evaluation/plot scripts while preserving the existing DQN CLI flow.
- Added a reusable experiment framework on top of the existing trainer/evaluator, including shared env sweep flags, benchmark experiment definitions, a central experiment runner, aggregated trial metrics with mean/std, and publication-style summary plots under the existing `runs/`, `results/`, and `analysis/` structure.

## Key Commands
- Random rollout: `python train/random_rollout.py`
- Train (headless): `python train/independent_dqn_pytorch.py --headless --total-steps 10000 --save-dir checkpoints --save-every 2000`
- Demo: `python train/demo.py --checkpoint-dir checkpoints`
- Screenshot gallery: `python train/capture_screenshots.py`

## Decisions
- Default training uses independent Q-networks unless `--shared-policy` is passed.
- Pheromone grid uses simple deposit + decay + diffusion.

## Open Questions
- Do we want a nest/return task and food-carrying state?
- Should pheromone deposit depend on carrying food?
- Add video capture or logging dashboards?

## Next Steps
- Add carry state + nest reward (optional)
- Add two pheromone channels (food vs nest)
- Improve training stability or add PPO alternative

## 2026-03-18
- Added `collective_intelligence_scaling` to the shared experiment registry with agent-count sweeps for pheromone-on vs pheromone-off conditions and a default 20-trial setup.
- Extended the shared experiment runner to honor per-experiment default trial counts, carry case metadata into trial/aggregate CSVs, and derive `efficiency_per_robot` only in the aggregation layer.
- Extended analysis plotting with grouped error-bar plots and generated collective-scaling outputs for completion time, swarm efficiency, pheromone usage, and efficiency per robot versus swarm size.
- Added a rule-based swarm baseline policy that consumes the standard 23-dim observation and outputs the same 9 discrete actions used by the learned controllers.
- Extended the shared evaluation pipeline and experiment runner to support `rl_algorithm_comparison`, including DQN, shared-policy DQN, and rule-based cases under the same eval/output schema.
- Added analysis-layer derived metrics for `convergence_speed` and `time_to_first_food` plus algorithm-comparison plots for food retrieval, efficiency, convergence speed, and overall efficiency bars.
- Refreshed the top-level architecture and onboarding docs to match the current 23-dim environment, experiment framework, and baseline set.
- Added a focused `docs/manual/` set covering quick start, project structure, experiments, and results interpretation for new users and judges.
- Logged the backward-compatibility assessment for commit `8ac33636556db0af11cd7c09b9b57e03d94a40d1`, noting that action semantics remain stable but the 19-dim observation/checkpoint contract no longer matches the current 23-dim environment.
- Logged the exact changed-file list relative to revision `8ac33636556db0af11cd7c09b9b57e03d94a40d1`.
- Added `docs/API_REFERENCE.md`, a code-grounded developer API/specification reference covering the environment contract, 23-dim observation layout, 9-action mapping, dynamics, rewards, config fields, and training/evaluation/experiment entry points.
- Added focused companion specs `docs/OBSERVATION_SPEC.md` and `docs/ACTION_SPEC.md` so developers can reference the current observation and action contracts without reading the full API reference.
- Added `docs/CONFIG_REFERENCE.md` and `docs/EXPERIMENT_API.md` to split the config surface and experiment framework into focused developer references.
- Added `train/policy_probe.py`, a beginner-friendly single-file script for manually feeding hand-written 23-dim observation vectors into a trained custom DQN checkpoint and inspecting Q-values, chosen action, and a plain-language interpretation.
- Updated onboarding, API reference, and quick-start docs to include how to run `train/policy_probe.py`.
- Fixed `train/policy_probe.py` so it prints observations using the active checkpoint layout and can explain both current 23-D and legacy 19-D checkpoint inputs by inspecting the checkpoint tensor shapes instead of assuming a single observation format.
- Logged a Q&A clarifying that `--headless` disables PyGame rendering and screenshot capture but does not change environment stepping, observations, rewards, or training logic.
- Logged a Q&A explaining why saving checkpoints during training is useful for resuming runs, comparing intermediate policies, running demos/evaluations later, and avoiding loss of progress.
- Logged a Q&A clarifying that `--save-dir` defaults to `checkpoints/` and that the trainer still writes a final checkpoint there even when `--save-every` is left at `0`.
- Logged a Q&A walking through the execution flow of `train/independent_dqn_pytorch.py`, including env setup, epsilon-greedy action selection, replay storage, Bellman updates, target-network sync, evaluation, logging, and checkpoint saving.
- Logged a Q&A explaining that `train/evaluate.py` runs headless deterministic evaluation for DQN or rule-based policies, writes per-episode metrics to `eval_metrics.csv`, and writes aggregated means to `eval_summary.json`.
- Added `docs/TRAINING_CUSTOM.md`, a detailed implementation-grounded walkthrough of the custom DQN trainer, including the main training flow, a line-by-line Bellman update explanation, and a tensor-shape walkthrough of one training step.
- Added `docs/EVALUATION_CUSTOM.md`, a detailed implementation-grounded walkthrough of `train/evaluate.py`, covering policy loading, deterministic evaluation flow, metric logging, summary generation, and the main differences from training.
- Added detailed explanatory comments throughout `train/policy_probe.py` without changing its behavior, clarifying observation layouts, legacy checkpoint handling, checkpoint-shape inspection, the one-step forward pass, and the plain-language action explanation logic.
- Added `docs/CAMERA_PROPOSAL.md` describing how to use an onboard camera as a feature-producing perception sensor, how to simulate camera-derived features in the environment, and a phased observation-contract and training plan for integrating vision without jumping straight to raw-image RL.
- Logged a Q&A explaining what each observation value in `train/policy_probe.py` represents, including lidar, target/nest/neighbor body-frame vectors, heading encoding, normalized speed, food/carry state, pheromone samples, and the legacy 19-D checkpoint difference.
- Logged a Q&A clarifying how to convert normalized observation values back into sim units and then into millimeters by applying `lidar_max_range`, `max_speed`, and an external mm-per-sim-unit calibration.
- Logged a Q&A recommending `1 sim unit = 10 mm` as a practical starting calibration, with rationale based on robot size, lidar range, world size, and the current Pi-side millimeter command conventions.
- Added `AntSwarmFirmware/run_policy.py`, a firmware-side trained-policy runner that reuses the existing sensor/action adapters and commands the robot through `AntSwarmFirmware/ant.py`.
- Updated `AntSwarmFirmware/run_policy.py` to bypass `ESP32ActionBridge` and send actions directly through `ESP32Robot.turn()`, `move()`, and `stop()`.
- Updated `pi/run_policy.py` to remove the `pi.esp32_*` bridge modules and interact directly with `AntSwarmFirmware/ant.py` for both scan reads and motion commands.
- Logged a Q&A noting that `AntSwarmFirmware/run_policy.py` compiles, but a CLI smoke test is currently blocked by a missing `pyserial` dependency in `AntSwarmFirmware/ant.py`.
- Inlined the policy config, scan bucketization, observation building, checkpoint loading, and action prediction logic inside `AntSwarmFirmware/run_policy.py` so it no longer imports `env.config`, `pi.esp32_sensor_adapter`, `robot.observation_builder`, or `robot.policy_runner`.
- Logged a Q&A documenting the current scale mapping in `AntSwarmFirmware/run_policy.py`, including lidar normalization and direct motion-command distances/turn angles.
- Logged a Q&A clarifying that `AntSwarmFirmware/run_policy.py` treats TOF `-1` readings as out-of-range/far by leaving the corresponding lidar bucket at its default max-range value.
- Logged a Q&A explaining that `AntSwarmFirmware/run_policy.py` uses explicit CLI mm/degree parameters rather than a hidden sim-unit conversion, and noted which arguments control those scales.
- Logged a Q&A computing the simulator calibration for a 70 mm radius robot: `1 sim unit = 7 mm` if `agent_radius` remains `10`.
- Logged a Q&A pointing to where the sim-unit calibration is defined and documented (`env/config.py` and `docs/QandA.md`).
- Logged a Q&A confirming that `agent_radius = 7` and `target_radius = 3` are consistent under a `1 sim unit = 10 mm` calibration, with the note that target radius should be chosen to match the physical target and pickup tolerance.
- Logged a Q&A with the deployment steps for copying checkpoints to the Pi, installing dependencies including `pyserial`, running `AntSwarmFirmware/run_policy.py`, and updating the systemd service if needed.
- Logged a Q&A auditing current `pi/` and `robot/` usage, noting that only part of `robot/` is still used by `pi/run_policy.py` while several `pi/esp32_*` and `robot/` integration modules are currently orphaned.
- Logged a Q&A confirming that removing `pi/` and `robot/` is safe if `AntSwarmFirmware/run_policy.py` is the only deployment path, with the caveat that docs and any `pi/run_policy.py` usage must be updated too.
- Logged a Q&A recommending `AntSwarmFirmware/` as the better current robot runtime path when simplicity and direct deployment matter more than preserving the older `pi/` and `robot/` abstraction layers.
- Removed the `pi/` and `robot/` packages, added `pyserial` to `requirements.txt`, and updated the live docs to make `AntSwarmFirmware/` the single documented robot deployment path.
- Logged a Q&A clarifying that `AntSwarmFirmware/run_policy.py` expects `--checkpoint-dir` to be a directory containing `shared.pt` or `agent_0.pt`, not a direct file path.
