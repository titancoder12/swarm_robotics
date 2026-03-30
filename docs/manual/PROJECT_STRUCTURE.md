# Project Structure Guide

This document explains the current directory layout and the role of each major file or folder.

## Top-Level Directories

### `env/`

Core simulation environment.

- [env/config.py](../../env/config.py)
  - `SwarmConfig` dataclass
  - world size, counts, rewards, sensor settings, pheromone parameters, and experiment hooks
- [env/swarm_env.py](../../env/swarm_env.py)
  - environment API
  - reset/step/render logic
  - observations
  - rewards
  - nest, food, obstacles, pheromones

### `train/`

Training, evaluation, and experiment orchestration.

- [train/independent_dqn_pytorch.py](../../train/independent_dqn_pytorch.py)
  - main custom DQN training path
- [analysis/evaluate.py](../../analysis/evaluate.py)
  - shared evaluation for DQN and rule-based policies
- [train/experiment_utils.py](../../train/experiment_utils.py)
  - CSV logging, JSON writing, config helpers, aggregation helpers
- [train/run_experiments.py](../../train/run_experiments.py)
  - central experiment runner
- [train/train.py](../../train/train.py)
  - backend dispatcher
- [train/demo.py](../../train/demo.py)
  - render trained policies
- [train/random_rollout.py](../../train/random_rollout.py)
  - random-policy sanity check
- [train/capture_screenshots.py](../../train/capture_screenshots.py)
  - screenshot generation
- [train/sb3_dqn.py](../../train/sb3_dqn.py)
  - Stable-Baselines3 backend
- [train/rllib_dqn.py](../../train/rllib_dqn.py)
  - RLlib backend

### `experiments/`

Experiment definitions.

- [experiments/benchmark_configs.py](../../experiments/benchmark_configs.py)
  - named sweeps
  - trial counts
  - case definitions
  - algorithm/condition metadata

### `analysis/`

Plot generation utilities.

- [analysis/plot_metrics.py](../../analysis/plot_metrics.py)
  - training plots
  - grouped error-bar experiment plots
  - bar charts with standard deviation

### `models/`

Shared policy definitions.

- [models/q_network.py](../../models/q_network.py)
  - custom DQN network
- [models/rule_based_policy.py](../../models/rule_based_policy.py)
  - hand-coded baseline policy

### `mission_control/`

Live command-center subsystem for physical robot experiments.

- [mission_control/main.py](../../mission_control/main.py)
  - PyGame application entrypoint
  - receiver wiring
  - operator controls
- [mission_control/core/](../../mission_control/core)
  - robot registry
  - pheromone field
  - trails
  - thread-safe world state
- [mission_control/comms/](../../mission_control/comms)
  - line protocol parsing
  - direct TCP / serial receivers
- [mission_control/ui/](../../mission_control/ui)
  - renderer
  - status panel
  - colors
- [mission_control/fake_robot.py](../../mission_control/fake_robot.py)
  - local fake-message harness for smoke testing without hardware

### `firmware/`

Current physical robot runtime.

- [firmware/ant.py](../../firmware/ant.py)
  - direct serial interface to the ESP32 robot controller
- [firmware/run.py](../../firmware/run.py)
  - direct learned-policy runtime used on the robot
- [firmware/ant.service](../../firmware/ant.service)
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

1. [env/swarm_env.py](../../env/swarm_env.py)
2. [train/independent_dqn_pytorch.py](../../train/independent_dqn_pytorch.py)
3. [analysis/evaluate.py](../../analysis/evaluate.py)
4. [experiments/benchmark_configs.py](../../experiments/benchmark_configs.py)
5. [train/run_experiments.py](../../train/run_experiments.py)
