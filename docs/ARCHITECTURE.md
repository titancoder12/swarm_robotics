# Swarm RL System Architecture

This project is a stigmergic swarm robotics simulation and experiment platform. Agents learn or follow policies in a shared 2D world with food, a nest, obstacles, and a pheromone field. The core scientific question is whether indirect communication through the environment improves collective behavior.

## System Overview

The system has five main layers:

1. Environment
   - [env/config.py](../env/config.py) defines the simulation parameters.
   - [env/swarm_env.py](../env/swarm_env.py) implements the PettingZoo Parallel environment, rewards, observations, pheromones, and rendering.

2. Policies and models
   - [models/q_network.py](../models/q_network.py) defines the shared DQN network used by custom checkpoints.
   - [models/rule_based_policy.py](../models/rule_based_policy.py) defines the non-learning rule-based baseline used in comparative experiments.

3. Training and evaluation
   - [train/independent_dqn_pytorch.py](../train/independent_dqn_pytorch.py) is the main custom DQN trainer.
   - [train/evaluate.py](../train/evaluate.py) runs shared evaluation for learned and rule-based policies.
   - [train/train.py](../train/train.py) dispatches between custom, SB3, and RLlib training backends.

4. Experiment framework
   - [experiments/benchmark_configs.py](../experiments/benchmark_configs.py) defines experiment sweeps.
   - [train/run_experiments.py](../train/run_experiments.py) runs trials, aggregates outputs, and generates plots.
   - [train/experiment_utils.py](../train/experiment_utils.py) provides run-directory, CSV, JSON, config, and aggregation helpers.

5. Analysis and deployment
   - [analysis/plot_metrics.py](../analysis/plot_metrics.py) generates training and experiment plots.
   - `robot/` and `pi/` contain sim-to-real and Raspberry Pi integration code.

## Environment Design

The environment is a 2D world containing:

- a nest
- food targets
- rectangular obstacles
- a pheromone field with evaporation and diffusion
- a configurable number of agents

Agents do not communicate directly. Coordination emerges from local sensing and environmental traces.

### Observation Space

Each agent receives a 23-dimensional observation vector built in `_get_obs()` in [env/swarm_env.py](../env/swarm_env.py):

- lidar obstacle rays
- nearest food vector in agent-local coordinates
- nest direction in agent-local coordinates
- nearest-agent vector
- heading as `sin(theta), cos(theta)`
- normalized speed
- local food-presence flag
- carrying-food flag
- pheromone samples

This 23-dimensional contract is the current source of truth for training, evaluation, and sim-to-real integration.

### Action Space

The action space remains `Discrete(9)`. Actions map to a 3x3 grid of `(throttle, turn)` values:

- throttle in `{-1, 0, 1}`
- turn in `{-1, 0, 1}`

This mapping is defined in `_build_action_table()` in [env/swarm_env.py](../env/swarm_env.py).

### Task Mechanics

The current environment supports:

- food pickup
- optional return-to-nest delivery
- obstacle avoidance
- pheromone deposition
- pheromone sensing
- exploration tracking
- failed-agent and observation-noise experiment hooks

### Pheromone Dynamics

The pheromone field is stored as a grid and updated every step using deposit, evaporation, and optional diffusion. Conceptually:

`P(x, y, t + 1) = (1 - evaporation_rate) * P(x, y, t) + deposition + diffusion`

Rendering can show the field as a heatmap overlay.

## Training Architecture

The main research training path is the custom DQN trainer in [train/independent_dqn_pytorch.py](../train/independent_dqn_pytorch.py).

It supports:

- independent Q-networks per agent
- shared-policy DQN with `--shared-policy`
- replay buffers
- epsilon-greedy exploration
- target-network updates
- checkpoint saving
- structured CSV and JSON logging
- periodic evaluation

The trainer reads `obs_dim` dynamically from the environment, so it is aligned with the current 23-dimensional observation space.

Optional comparison backends still exist:

- [train/sb3_dqn.py](../train/sb3_dqn.py)
- [train/rllib_dqn.py](../train/rllib_dqn.py)

## Evaluation and Metrics

Shared evaluation is handled by [train/evaluate.py](../train/evaluate.py). It evaluates:

- DQN checkpoints
- shared-policy DQN checkpoints
- the rule-based baseline

Logged metrics include:

- `mean_episode_reward`
- `food_retrieved`
- `exploration_coverage`
- `pheromone_usage`
- `episode_length`
- `swarm_efficiency`

The experiment runner also derives:

- `efficiency_per_robot`
- `convergence_speed`
- `time_to_first_food`

These derived metrics are computed in the analysis/aggregation layer, not in the trainer.

## Experiment Framework

The shared experiment system uses:

- [experiments/benchmark_configs.py](../experiments/benchmark_configs.py)
- [train/run_experiments.py](../train/run_experiments.py)

Key implemented experiments:

- `collective_intelligence_scaling`
- `rl_algorithm_comparison`

The experiment framework reuses the same trainer and evaluator for consistent metrics across trials.

Outputs are written to:

- `runs/` for per-run logs and checkpoints
- `results/` for aggregated CSV outputs
- `analysis/` for plots

## Rendering and Demo Paths

For interactive visualization:

- [train/random_rollout.py](../train/random_rollout.py) runs a random policy sanity check.
- [train/demo.py](../train/demo.py) renders trained policies.
- [train/capture_screenshots.py](../train/capture_screenshots.py) generates documentation images.

The renderer can show:

- agents
- obstacles
- food
- nest
- pheromone heatmap

## Sim-to-Real Path

The project also contains deployment-oriented code:

- `robot/` provides generic sensor, observation, policy, and action bridges.
- `pi/` provides Raspberry Pi-side runtime examples, including a preserved rule-based runtime and a model-driven runtime.

These modules reuse the same observation and action contracts where possible.

## Recommended Reading Order

For a new reader:

1. [docs/manual/QUICK_START.md](manual/QUICK_START.md)
2. [docs/ONBOARDING.md](ONBOARDING.md)
3. [docs/manual/PROJECT_STRUCTURE.md](manual/PROJECT_STRUCTURE.md)
4. [docs/manual/EXPERIMENT_GUIDE.md](manual/EXPERIMENT_GUIDE.md)
5. [docs/manual/RESULTS_INTERPRETATION.md](manual/RESULTS_INTERPRETATION.md)
