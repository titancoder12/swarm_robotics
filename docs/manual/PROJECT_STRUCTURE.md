# Project Structure Guide

This document explains the current directory layout and the role of each major file or folder.

## Top-Level Directories

### `env/`

Core simulation environment.

- [env/config.py](/Users/christopherlin/dev/cwsf2026/sim/env/config.py)
  - `SwarmConfig` dataclass
  - world size, counts, rewards, sensor settings, pheromone parameters, and experiment hooks
- [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
  - environment API
  - reset/step/render logic
  - observations
  - rewards
  - nest, food, obstacles, pheromones

### `train/`

Training, evaluation, and experiment orchestration.

- [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py)
  - main custom DQN training path
- [train/evaluate.py](/Users/christopherlin/dev/cwsf2026/sim/train/evaluate.py)
  - shared evaluation for DQN and rule-based policies
- [train/experiment_utils.py](/Users/christopherlin/dev/cwsf2026/sim/train/experiment_utils.py)
  - CSV logging, JSON writing, config helpers, aggregation helpers
- [train/run_experiments.py](/Users/christopherlin/dev/cwsf2026/sim/train/run_experiments.py)
  - central experiment runner
- [train/train.py](/Users/christopherlin/dev/cwsf2026/sim/train/train.py)
  - backend dispatcher
- [train/demo.py](/Users/christopherlin/dev/cwsf2026/sim/train/demo.py)
  - render trained policies
- [train/random_rollout.py](/Users/christopherlin/dev/cwsf2026/sim/train/random_rollout.py)
  - random-policy sanity check
- [train/capture_screenshots.py](/Users/christopherlin/dev/cwsf2026/sim/train/capture_screenshots.py)
  - screenshot generation
- [train/sb3_dqn.py](/Users/christopherlin/dev/cwsf2026/sim/train/sb3_dqn.py)
  - Stable-Baselines3 backend
- [train/rllib_dqn.py](/Users/christopherlin/dev/cwsf2026/sim/train/rllib_dqn.py)
  - RLlib backend

### `experiments/`

Experiment definitions.

- [experiments/benchmark_configs.py](/Users/christopherlin/dev/cwsf2026/sim/experiments/benchmark_configs.py)
  - named sweeps
  - trial counts
  - case definitions
  - algorithm/condition metadata

### `analysis/`

Plot generation utilities.

- [analysis/plot_metrics.py](/Users/christopherlin/dev/cwsf2026/sim/analysis/plot_metrics.py)
  - training plots
  - grouped error-bar experiment plots
  - bar charts with standard deviation

### `models/`

Shared policy definitions.

- [models/q_network.py](/Users/christopherlin/dev/cwsf2026/sim/models/q_network.py)
  - custom DQN network
- [models/rule_based_policy.py](/Users/christopherlin/dev/cwsf2026/sim/models/rule_based_policy.py)
  - hand-coded baseline policy

### `server/`

Live command-center subsystem for physical robot experiments.

- [server/main.py](/Users/christopherlin/dev/cwsf2026/sim/server/main.py)
  - PyGame application entrypoint
  - receiver wiring
  - operator controls
- [server/core/](/Users/christopherlin/dev/cwsf2026/sim/server/core/)
  - robot registry
  - pheromone field
  - trails
  - thread-safe world state
- [server/comms/](/Users/christopherlin/dev/cwsf2026/sim/server/comms/)
  - line protocol parsing
  - direct TCP / serial receivers
- [server/ui/](/Users/christopherlin/dev/cwsf2026/sim/server/ui/)
  - renderer
  - status panel
  - colors
- [server/fake_robot.py](/Users/christopherlin/dev/cwsf2026/sim/server/fake_robot.py)
  - local fake-message harness for smoke testing without hardware

### `AntSwarmFirmware/`

Current physical robot runtime.

- [AntSwarmFirmware/ant.py](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/ant.py)
  - direct serial interface to the ESP32 robot controller
- [AntSwarmFirmware/run_policy.py](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/run_policy.py)
  - direct learned-policy runtime used on the robot
- [AntSwarmFirmware/ant.service](/Users/christopherlin/dev/cwsf2026/sim/AntSwarmFirmware/ant.service)
  - example systemd service for the robot runtime

### `docs/`

Project documentation.

- top-level docs for architecture, onboarding, sim-to-real, and theory
- `docs/manual/` for user guides
- `docs/prompts/` for integration prompt history

### `checkpoints/`

Default checkpoint output location.

Typical contents:

- `agent_0.pt`, `agent_1.pt`, ...
- `shared.pt`
- `metadata.json`

## Output Directories Created by Runs

### `runs/`

Per-run artifacts such as:

- `episode_metrics.csv`
- `eval_metrics.csv`
- `summary.json`
- `run_config.json`
- checkpoints

### `results/`

Aggregated experiment outputs such as:

- `trial_metrics.csv`
- `aggregate_metrics.csv`

### `analysis/`

Generated plots such as:

- reward curves
- experiment comparison charts

## Recommended Navigation Order

If you are reading the code for the first time:

1. [env/swarm_env.py](/Users/christopherlin/dev/cwsf2026/sim/env/swarm_env.py)
2. [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/sim/train/independent_dqn_pytorch.py)
3. [train/evaluate.py](/Users/christopherlin/dev/cwsf2026/sim/train/evaluate.py)
4. [experiments/benchmark_configs.py](/Users/christopherlin/dev/cwsf2026/sim/experiments/benchmark_configs.py)
5. [train/run_experiments.py](/Users/christopherlin/dev/cwsf2026/sim/train/run_experiments.py)
