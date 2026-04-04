Step 7 — Documentation & User Manual Generator

Document the reinforcement learning swarm robotics system that has been incrementally developed through multiple stages.

GOAL

Generate a complete, user-friendly, and technically accurate documentation set for the current project, based on the implementations from the previous steps in this session (i.e., the latest six steps).

This documentation should allow a new user (or judge) to:

understand the system architecture

run simulations

train models

run experiments

interpret outputs

CONTEXT (CRITICAL — DO NOT IGNORE)

The codebase has already been upgraded with:

ENVIRONMENT

23-dimensional observation space

Discrete(9) action space

supports:

nest

food

obstacles

pheromone field (evaporation + diffusion)

rendering includes pheromone heatmap

TRAINING

custom DQN pipeline

logging includes:

reward

food retrieval

exploration

pheromone usage

swarm efficiency

EXPERIMENT FRAMEWORK

train/run_experiments.py (central runner)

experiments/benchmark_configs.py

outputs:

runs/

results/

analysis/

EXPERIMENTS IMPLEMENTED

collective_intelligence_scaling

rl_algorithm_comparison

BASELINES

DQN

shared-policy DQN

rule-based policy (models/rule_based_policy.py)

IMPORTANT CONSTRAINTS

DO NOT rewrite or restructure the codebase

DO NOT introduce new systems

Documentation must reflect the CURRENT implementation exactly

Use actual file names and commands

TASK

Generate a complete documentation set.

REQUIRED OUTPUT FILES

IMPORTANT:

Before creating any documentation files, first inspect the existing /docs/ folder and avoid conflicting with or duplicating existing documents.

Existing docs already present include files such as:

ARCHITECTURE.md

ONBOARDING.md

DQN_EXPLAINED.md

PI_MIGRATION.md

QandA.md

PROJECT_LOG.md

RESOURCES.md

SimToReal.md

UML.md

ToDo.md

docs/prompts/

Documentation strategy:

Prefer updating existing docs if an equivalent document already exists.

Only create a new file when there is no suitable existing file.

If multiple new docs are needed, place them under a dedicated subfolder such as:

docs/manual/

to avoid cluttering the top-level docs directory.

In the output, clearly state which files were updated versus newly created.

Recommended mapping:

1. System overview

Prefer updating one of these if appropriate:

ARCHITECTURE.md

UML.md

If neither is suitable, create:

docs/manual/SYSTEM_OVERVIEW.md

Content should explain:

what the project does

key idea: stigmergic swarm intelligence

high-level architecture:

environment

agents

training

experiment framework

2. Setup / running instructions

Prefer updating:

ONBOARDING.md

If ONBOARDING.md is not the right place, create:

docs/manual/SETUP_AND_RUN.md

Explain step-by-step:

environment setup

dependencies

how to run simulation

how to train

how to run experiments

Include exact commands like:

python train/run_experiments.py --experiment collective_intelligence_scaling

python train/run_experiments.py --experiment rl_algorithm_comparison

3. Project structure guide

Prefer updating:

ARCHITECTURE.md

If that would make ARCHITECTURE.md too overloaded, create:

docs/manual/PROJECT_STRUCTURE.md

Explain directory structure:

env/

train/

analysis/

experiments/

models/

docs/

Describe purpose of each key file.

4. Experiment guide

Prefer creating:

docs/manual/EXPERIMENT_GUIDE.md

unless there is already a suitable experiments doc.

Explain each experiment:

collective_intelligence_scaling

rl_algorithm_comparison

For each:

goal

variables

metrics

how to run

what outputs mean

5. Results interpretation guide

Prefer creating:

docs/manual/RESULTS_INTERPRETATION.md

Explain how to read:

CSV outputs

plots

Explain key metrics:

food_retrieved

swarm_efficiency

exploration_coverage

convergence_speed

Explain what patterns indicate:

emergent intelligence

benefit of pheromones

advantage of RL vs rule-based

6. Quick start

Prefer updating:

ONBOARDING.md

or creating a concise separate file:

docs/manual/QUICK_START.md

A short version for judges:

what this project is

how to run one experiment

where to see results

STYLE REQUIREMENTS

clear headings

concise explanations

minimal fluff

readable by:

engineers

science fair judges

students

OUTPUT

Provide:

list of files created/updated

full contents of each document

Ensure documentation is:

accurate

consistent with codebase

immediately usable

Do NOT generate placeholder text.
Do NOT invent features not present in the code.








Output:

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• The docs themselves look consistent with the implemented experiment names, commands, and output paths. I’m adding the final project-
  log entry for the documentation refresh now.

• Edited docs/PROJECT_LOG.md (+2 -0)
    93  - Added analysis-layer derived metrics for `convergence_speed` and `time_to_first_food` plus algorithm-comparison plots for fo
        od retrieval, efficiency, convergence speed, and overall efficiency bars.
    94 +- Refreshed the top-level architecture and onboarding docs to match the current 23-dim environment, experiment framework, and
        baseline set.
    95 +- Added a focused `docs/manual/` set covering quick start, project structure, experiments, and results interpretation for new
        users and judges.

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

• Updated the documentation set in place and added a new docs/manual/ subfolder for focused user guides.

  Files updated

  - docs/ARCHITECTURE.md
  - docs/ONBOARDING.md

  Files created

  - docs/manual/PROJECT_STRUCTURE.md
  - docs/manual/EXPERIMENT_GUIDE.md
  - docs/manual/RESULTS_INTERPRETATION.md
  - docs/manual/QUICK_START.md

  Housekeeping update

  - docs/PROJECT_LOG.md was updated with a short summary of the documentation refresh.

  Below are the full contents of the six user-facing documentation files.

  ———

  docs/ARCHITECTURE.md

  # Swarm RL System Architecture

  This project is a stigmergic swarm robotics simulation and experiment platform. Agents learn or follow policies in a shared 2D world
  with food, a nest, obstacles, and a pheromone field. The core scientific question is whether indirect communication through the
  environment improves collective behavior.

  ## System Overview

  The system has five main layers:

  1. Environment
     - [env/config.py](../../env/config.py) defines the simulation parameters.
     - [env/swarm_env.py](../../env/swarm_env.py) implements the PettingZoo Parallel environment,
  rewards, observations, pheromones, and rendering.

  2. Policies and models
     - [models/q_network.py](../../models/q_network.py) defines the shared DQN network used by custom
  checkpoints.
     - [models/rule_based_policy.py](../../models/rule_based_policy.py) defines the non-learning
  rule-based baseline used in comparative experiments.

  3. Training and evaluation
     - [train/independent_dqn_pytorch.py](../../train/independent_dqn_pytorch.py) is the main custom
  DQN trainer.
     - [train/evaluate.py](../../train/evaluate.py) runs shared evaluation for learned and rule-based
  policies.
     - [train/train.py](../../train/train.py) dispatches between custom, SB3, and RLlib training
  backends.

  4. Experiment framework
     - [experiments/benchmark_configs.py](../../experiments/benchmark_configs.py) defines experiment
  sweeps.
     - [train/run_experiments.py](../../train/run_experiments.py) runs trials, aggregates outputs,
  and generates plots.
     - [train/experiment_utils.py](../../train/experiment_utils.py) provides run-directory, CSV,
  JSON, config, and aggregation helpers.

  5. Analysis and deployment
     - [analysis/plot_metrics.py](../../analysis/plot_metrics.py) generates training and experiment
  plots.
     - `robot/` and `pi/` contain sim-to-real
  and Raspberry Pi integration code.

  ## Environment Design

  The environment is a 2D world containing:

  - a nest
  - food targets
  - rectangular obstacles
  - a pheromone field with evaporation and diffusion
  - a configurable number of agents

  Agents do not communicate directly. Coordination emerges from local sensing and environmental traces.

  ### Observation Space

  Each agent receives a 23-dimensional observation vector built in `_get_obs()` in [env/swarm_env.py](/Users/christopherlin/dev/
  cwsf2026/sim/env/swarm_env.py):

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

  This mapping is defined in `_build_action_table()` in [env/swarm_env.py](../../env/swarm_env.py).

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

  The main research training path is the custom DQN trainer in [train/independent_dqn_pytorch.py](/Users/christopherlin/dev/cwsf2026/
  sim/train/independent_dqn_pytorch.py).

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

  - [train/sb3_dqn.py](../../train/sb3_dqn.py)
  - [train/rllib_dqn.py](../../train/rllib_dqn.py)

  ## Evaluation and Metrics

  Shared evaluation is handled by [train/evaluate.py](../../train/evaluate.py). It evaluates:

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

  - [experiments/benchmark_configs.py](../../experiments/benchmark_configs.py)
  - [train/run_experiments.py](../../train/run_experiments.py)

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

  - [train/random_rollout.py](../../train/random_rollout.py) runs a random policy sanity check.
  - [train/demo.py](../../train/demo.py) renders trained policies.
  - [train/capture_screenshots.py](../../train/capture_screenshots.py) generates documentation
  images.

  The renderer can show:

  - agents
  - obstacles
  - food
  - nest
  - pheromone heatmap

  ## Sim-to-Real Path

  The project also contains deployment-oriented code:

  - `robot/` provides generic sensor, observation, policy, and action bridges.
  - `pi/` provides Raspberry Pi-side runtime examples, including a preserved rule-based
  runtime and a model-driven runtime.

  These modules reuse the same observation and action contracts where possible.

  ## Recommended Reading Order

  For a new reader:

  1. [docs/manual/QUICK_START.md](../manual/QUICK_START.md)
  2. [docs/ONBOARDING.md](../ONBOARDING.md)
  3. [docs/manual/PROJECT_STRUCTURE.md](../manual/PROJECT_STRUCTURE.md)
  4. [docs/manual/EXPERIMENT_GUIDE.md](../manual/EXPERIMENT_GUIDE.md)
  5. [docs/manual/RESULTS_INTERPRETATION.md](../manual/RESULTS_INTERPRETATION.md)

  ———

  docs/ONBOARDING.md

  # Onboarding and Run Guide

  This guide is the main entry point for running the current project. It covers setup, simulation, training, evaluation, and
  experiments using the code as it exists now.

  ## What This Project Does

  This repository studies stigmergic swarm intelligence in a 2D robotics simulation. Agents search for food, avoid obstacles, interact
  through a pheromone field, and optionally return food to a nest. The main research focus is whether indirect communication through
  pheromones improves collective performance.

  ## Quick Start

  If you only want one command to see the current system working:

  ```bash
  python train/run_experiments.py --experiment collective_intelligence_scaling

  That command trains and evaluates the flagship experiment, then writes:

  - raw logs under runs/
  - aggregated CSVs under results/
  - plots under analysis/

  For a short version, see docs/manual/QUICK_START.md.

  ## 1. Setup

  Create and activate a virtual environment:

  python -m venv .venv
  source .venv/bin/activate

  Install dependencies:

  pip install -r requirements.txt

  ## 2. Run a Simulation

  Random rendered rollout:

  python train/random_rollout.py

  Capture example screenshots:

  python train/capture_screenshots.py

  ## 3. Train a Custom DQN Policy

  Headless training:

  python train/independent_dqn_pytorch.py --headless --total-steps 10000

  Training with checkpoints and run outputs:

  python train/independent_dqn_pytorch.py \
    --headless \
    --total-steps 20000 \
    --save-dir checkpoints \
    --save-every 5000 \
    --output-dir runs \
    --experiment-name dqn_foraging \
    --eval-every 5000 \
    --eval-episodes 5

  Important notes:

  - the current observation dimension is 23
  - old 19-dimensional checkpoints are not compatible
  - the current action space is still Discrete(9)

  ## 4. Render a Trained Policy

  Custom DQN demo:

  python train/demo.py --checkpoint-dir checkpoints

  The demo renders the current environment, including pheromone heatmap, food, nest, obstacles, and agents.

  ## 5. Evaluate a Policy

  Evaluate saved DQN checkpoints:

  python train/evaluate.py --checkpoint-dir checkpoints --episodes 10 --output-dir runs/eval

  Evaluate the rule-based baseline directly:

  python train/evaluate.py --policy-kind rule_based --n-agents 5 --episodes 10 --output-dir runs/rule_eval

  Outputs include:

  - eval_metrics.csv
  - eval_summary.json

  ## 6. Run Experiments

  The central experiment entry point is train/run_experiments.py.

  ### Collective Intelligence Scaling

  python train/run_experiments.py --experiment collective_intelligence_scaling

  This experiment compares swarm sizes with pheromone on vs pheromone off.

  ### RL Algorithm Comparison

  python train/run_experiments.py --experiment rl_algorithm_comparison

  This experiment compares:

  - DQN
  - shared-policy DQN
  - rule-based baseline

  across multiple swarm sizes and pheromone conditions.

  ### Run All Registered Experiments

  python train/run_experiments.py --experiment all

  ## 7. Output Locations

  The project uses three main output directories:

  - runs/
      - per-run logs
      - checkpoints
      - training CSVs
      - run configs
  - results/
      - trial-level experiment CSVs
      - aggregate experiment CSVs
  - analysis/
      - plots generated from training or experiments

  ## 8. Key Files to Know

  - env/swarm_env.py
      - environment logic
  - env/config.py
      - simulation configuration
  - train/independent_dqn_pytorch.py
      - custom DQN trainer
  - train/evaluate.py
      - shared evaluation path
  - train/run_experiments.py
      - experiment runner
  - experiments/benchmark_configs.py
      - experiment definitions
  - models/rule_based_policy.py
      - rule-based baseline

  ## 9. Suggested Reading Order

  1. docs/manual/QUICK_START.md
  2. docs/ARCHITECTURE.md
  3. docs/manual/PROJECT_STRUCTURE.md
  4. docs/manual/EXPERIMENT_GUIDE.md
  5. docs/manual/RESULTS_INTERPRETATION.md


  ---

  **[docs/manual/PROJECT_STRUCTURE.md](../manual/PROJECT_STRUCTURE.md)**
  ```md
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
  - [train/evaluate.py](../../train/evaluate.py)
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

  ### `robot/`

  Generic sim-to-real runtime components.

  - sensor bridge
  - observation builder
  - policy runner
  - action bridge
  - deployment runtime

  ### `pi/`

  Raspberry Pi-side integration examples.

  - `pi/ants.py`
    - compatibility-first runtime based on the existing Pi controller style
  - `pi/run_policy.py`
    - model-driven Pi runtime
  - `pi/ants.service`
    - service file example

  ### `docs/`

  Project documentation.

  - top-level docs for architecture, onboarding, sim-to-real, and theory
  - `docs/manual/` for user guides
  - `docs/prompts/` for integration step history

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
  3. [train/evaluate.py](../../train/evaluate.py)
  4. [experiments/benchmark_configs.py](../../experiments/benchmark_configs.py)
  5. [train/run_experiments.py](../../train/run_experiments.py)

  ———

  docs/manual/EXPERIMENT_GUIDE.md

  # Experiment Guide

  This project currently has two primary science-fair experiments implemented in the shared framework.

  The common entry point is:

  ```bash
  python train/run_experiments.py --experiment <name>

  All experiments reuse the same:

  - environment
  - custom DQN training loop
  - shared evaluator
  - logging outputs
  - aggregation path

  ## Shared Output Format

  Every experiment writes:

  - raw run artifacts in runs/<experiment>/...
  - trial-level metrics in results/<experiment>/trial_metrics.csv
  - aggregated metrics in results/<experiment>/aggregate_metrics.csv
  - plots in analysis/<experiment>/

  ## 1. Collective Intelligence Scaling

  Experiment name:

  collective_intelligence_scaling

  ### Goal

  Test whether larger swarms improve performance, and whether that improvement depends on stigmergic communication through pheromones.

  ### Variables

  - swarm sizes: 1, 2, 3, 5, 10
  - condition A: pheromone enabled
  - condition B: pheromone disabled
  - default trials: 20

  ### Run Command

  python train/run_experiments.py --experiment collective_intelligence_scaling

  ### Main Metrics

  - food_retrieved
  - exploration_coverage
  - pheromone_usage
  - episode_length
  - swarm_efficiency
  - mean_episode_reward
  - derived: efficiency_per_robot

  ### Main Plots

  - completion_time_vs_agents.png
  - efficiency_vs_agents.png
  - pheromone_usage_vs_agents.png
  - efficiency_per_robot_vs_agents.png

  ### What It Tests

  This experiment is designed to support the claim that performance improves non-linearly with swarm size when stigmergic
  communication is available.

  ## 2. RL Algorithm Comparison

  Experiment name:

  rl_algorithm_comparison

  ### Goal

  Compare learned and non-learned control strategies under the same environment and metric pipeline.

  ### Algorithms

  - dqn
  - shared_dqn
  - rule_based

  ### Variables

  - swarm sizes: 1, 3, 5, 10
  - pheromone enabled
  - pheromone disabled
  - default trials: 20

  ### Run Command

  python train/run_experiments.py --experiment rl_algorithm_comparison

  ### Main Metrics

  - food_retrieved
  - swarm_efficiency
  - exploration_coverage
  - pheromone_usage
  - episode_length
  - mean_episode_reward
  - derived: convergence_speed
  - derived: time_to_first_food

  ### Main Plots

  - algorithm_food_retrieval.png
  - algorithm_efficiency.png
  - algorithm_convergence_speed.png
  - algorithm_comparison_bar.png

  ### What It Tests

  This experiment is designed to support the claim that learned controllers scale more effectively than the rule-based baseline,
  especially when stigmergic communication is available.

  ## Useful Optional Flags

  The experiment runner supports:

  - --trials
  - --seed
  - --total-steps
  - --eval-every
  - --eval-episodes
  - --runs-dir
  - --results-dir
  - --analysis-dir
  - --save-every
  - --cuda
  - --no-plots

  Example shorter run:

  python train/run_experiments.py \
    --experiment rl_algorithm_comparison \
    --trials 1 \
    --total-steps 100 \
    --eval-every 0 \
    --eval-episodes 1

  ## How the Experiments Stay Comparable

  The experiments are comparable because they share:

  - the same environment implementation
  - the same observation contract
  - the same action space
  - the same evaluator
  - the same result CSV schema

  This means differences in results are attributable to the experiment variables rather than to different logging or measurement
  pipelines.


  ---

  **[docs/manual/RESULTS_INTERPRETATION.md](../manual/RESULTS_INTERPRETATION.md)**
  ```md
  # Results Interpretation Guide

  This guide explains how to read the CSV outputs and plots produced by training and experiments.

  ## Output Files

  ### Training outputs in `runs/`

  Common files:

  - `episode_metrics.csv`
  - `eval_metrics.csv`
  - `summary.json`
  - `run_config.json`
  - checkpoint files

  ### Experiment outputs in `results/`

  Common files:

  - `trial_metrics.csv`
  - `aggregate_metrics.csv`

  ### Plots in `analysis/`

  These include training curves and experiment comparison figures.

  ## Key Metrics

  ### `food_retrieved`

  How much food was successfully delivered or collected during evaluation.

  Higher is better.

  ### `swarm_efficiency`

  Food retrieval normalized by episode length.

  Higher is better. This is a compact measure of how productive the swarm is over time.

  ### `exploration_coverage`

  Fraction of the coverage grid visited during an episode.

  Higher means the swarm explored more of the environment.

  ### `pheromone_usage`

  Mean local pheromone intensity experienced by agents.

  Higher values indicate stronger interaction with the stigmergic field.

  ### `episode_length`

  Length of the episode in steps.

  Lower can be better if the swarm finishes the task quickly. Interpret it together with `food_retrieved`.

  ### `mean_episode_reward`

  Average reward across agents in an episode.

  Useful as a broad training signal, but less interpretable than task-specific metrics like `food_retrieved`.

  ### `efficiency_per_robot`

  Derived in experiment aggregation:

  `food_retrieved / n_agents`

  Useful for checking whether adding robots is producing true collective gains or just adding more total effort.

  ### `convergence_speed`

  Derived from training logs in the experiment layer.

  Lower values mean the reward stabilized earlier in training.

  ### `time_to_first_food`

  Derived from evaluation logs in the experiment layer.

  Lower values mean the policy reached food sooner in evaluation.

  ## How to Read the CSVs

  ### `trial_metrics.csv`

  One row per trial.

  Use this file when you want:

  - raw values for each seed
  - custom statistical analysis
  - to verify variance across repeated runs

  ### `aggregate_metrics.csv`

  One row per experiment case.

  This file contains means and standard deviations, which are the main values used for the published plots.

  ## What Patterns Matter

  ### Signs of emergent collective intelligence

  Look for:

  - higher `food_retrieved` as swarm size grows
  - improved `swarm_efficiency` with more agents
  - stable or improving `efficiency_per_robot`

  If performance rises faster with pheromones than without pheromones, that is evidence for stigmergic coordination rather than simple
  scaling by agent count.

  ### Signs that pheromones help

  Look for:

  - higher `food_retrieved` in pheromone-on runs
  - higher `swarm_efficiency` in pheromone-on runs
  - meaningful `pheromone_usage` only in pheromone-enabled conditions
  - better scaling curves in the collective intelligence experiment

  ### Signs RL beats the rule-based baseline

  In the algorithm comparison experiment, look for:

  - better `food_retrieved`
  - better `swarm_efficiency`
  - faster `time_to_first_food`
  - better scaling across larger swarm sizes

  If DQN or shared-policy DQN improves while the rule-based baseline plateaus, that supports the claim that learning captures
  coordination strategies the heuristic baseline does not.

  ## Important Caveats

  - `mean_episode_reward` is useful, but it is not the main scientific metric.
  - `episode_length` should not be read alone; pair it with `food_retrieved` and `swarm_efficiency`.
  - `convergence_speed` and `time_to_first_food` are derived in the experiment layer, not in the trainer.
  - very short smoke-test runs are useful for validation but not for scientific conclusions.

  ———

  docs/manual/QUICK_START.md

  # Quick Start

  This project simulates a swarm of agents that search for food, interact through pheromones, and learn coordination behaviors.

  ## Fastest Way to See the Project Working

  Run the flagship experiment:

  ```bash
  python train/run_experiments.py --experiment collective_intelligence_scaling

  That command will:

  - train policies
  - evaluate them
  - aggregate metrics
  - generate plots

  ## Where to Look After It Finishes

  - raw run logs: runs/collective_intelligence_scaling/
  - aggregated CSVs: results/collective_intelligence_scaling/
  - plots: analysis/collective_intelligence_scaling/

  ## If You Want the Algorithm Comparison

  python train/run_experiments.py --experiment rl_algorithm_comparison

  That experiment compares:

  - DQN
  - shared-policy DQN
  - rule-based baseline

  ## If You Want a Visual Simulation First

  python train/random_rollout.py

  ## If You Want to Train a Single Policy

  python train/independent_dqn_pytorch.py --headless --total-steps 10000

  ## Important Current Facts

  - observation space: 23
  - action space: Discrete(9)
  - main environment features: nest, food, obstacles, pheromones
  - main experiment runner: train/run_experiments.py


  I did not create duplicate docs for topics already covered by existing files. The main top-level docs were refreshed, and the new
  material was placed under `docs/manual/` to keep `docs/` from getting cluttered.
