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
