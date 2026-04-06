Step 4 — Experiment Framework

Build experiment infrastructure inside the existing swarm robotics reinforcement learning codebase.

GOAL

Add a reusable experiment framework to the existing project so experiments can be run consistently and produce research-quality results.

CONTEXT FROM PROMPT 3 (CRITICAL — DO NOT IGNORE)

The training pipeline and environment have already been upgraded with the following:

ENVIRONMENT

Observation space = 23 dimensions

Includes:

lidar obstacle rays

nearest food / target vector

nest direction

nearest agent vector

heading (sin, cos)

speed

local food presence flag

carrying-food flag

pheromone samples

Action space = Discrete(9)

TRAINING SYSTEM

Custom DQN trainer upgraded and working

Training reads obs_dim dynamically from env.reset()

Logging now includes:

episode reward

food discovered

food delivered to nest

exploration coverage

pheromone usage

collisions

episode length

reward breakdown

swarm efficiency

AVAILABLE INFRASTRUCTURE (MUST REUSE)

train/independent_dqn_pytorch.py

train/evaluate.py

train/experiment_utils.py

analysis/plot_metrics.py

OUTPUT ARTIFACTS ALREADY PRODUCED

episode_metrics.csv

eval_metrics.csv

summary.json

run_config.json

metadata.json (checkpoint compatibility)

IMPORTANT CONSTRAINTS

DO NOT recreate training loops

DO NOT duplicate evaluation logic

Build on top of experiment_utils and evaluate.py

Ensure all experiments use the SAME metrics for comparability

Maintain compatibility with obs_dim = 23

REQUIRED CAPABILITIES

Extend the project to support a structured experiment runner that:

runs multiple trials

sweeps parameters (e.g., number of agents, pheromone on/off)

aggregates metrics across trials

saves results in a consistent format

generates publication-quality plots

REQUIRED EXPERIMENTS TO SUPPORT

Swarm Scaling

Stigmergy Ablation

Baseline Comparison

Robot Failure Test

Noise Robustness

Each experiment must:

call existing training/evaluation pipeline

reuse existing logging outputs

NOT reimplement metrics collection

DESIGN REQUIREMENTS

Experiment Runner

Create or extend a central runner (e.g., run_experiments.py):

loops over configurations

sets seeds for reproducibility

runs training or evaluation

collects metrics from CSV outputs

Metrics Aggregation

Aggregate across trials:

mean

standard deviation

Use metrics already logged:

food_retrieved

food_delivered

exploration_coverage

pheromone_usage

collisions

episode_length

reward

Plotting

Extend existing plotting utilities to generate:

mean curves

error bars

labeled axes

legends

OUTPUT FILE STRUCTURE

Follow existing structure such as:

runs/

results/

analysis/

Do NOT invent a completely new directory layout.

OUTPUT

Modify the existing codebase in place and provide:

files modified

files added

commands to run the experiment framework

explanation of how experiments reuse existing logs and evaluation code

Output:

   - collisions
          - episode_length
          - swarm_efficiency
  - Training compatibility:
      - the custom DQN trainer already reads obs_dim dynamically from env.reset()
      - the experiment framework reuses:
          - train/independent_dqn_pytorch.py
          - train/evaluate.py
          - train/experiment_utils.py
          - analysis/plot_metrics.py

  Files modified

  - env/config.py
  - env/swarm_env.py
  - train/experiment_utils.py
  - train/independent_dqn_pytorch.py
  - train/evaluate.py
  - analysis/plot_metrics.py
  - docs/PROJECT_LOG.md

  Files added

  - experiments/benchmark_configs.py
  - train/run_experiments.py

  Summary of changes

  - Added shared environment sweep flags to the existing training/evaluation pipeline:
      - --n-agents
      - --n-targets
      - --n-obstacles
      - --max-steps-per-episode
      - --dynamics-mode
      - --pheromone-disabled
      - --failed-agent-count
      - --observation-noise-std
  - Added experiment-ready env hooks needed for required experiments:
      - robot failure test via failed_agent_count
      - noise robustness via observation_noise_std
  - Added central experiment runner:
      - loops over configs
      - sets seeds
      - runs existing training
      - runs existing evaluation
      - aggregates metrics across trials
      - writes consistent outputs under runs/, results/, and analysis/
  - Added built-in benchmark sweeps for:
      - swarm_scaling
      - stigmergy_ablation
      - baseline_comparison
      - robot_failure_test
      - noise_robustness
  - Added aggregate plotting support:
      - mean/std curves
      - bar plots with error bars

  How experiments reuse existing logs and evaluation code

  - Training is still done by train/independent_dqn_pytorch.py
  - Final policy evaluation is still done by train/evaluate.py
  - Shared config/log helpers remain in train/experiment_utils.py
  - Plotting extends analysis/plot_metrics.py
  - The experiment runner does not collect metrics independently; it consumes:
      - episode_metrics.csv
      - eval_metrics.csv
      - eval_summary.json
      - summary.json
      - metadata.json

  Commands to run the framework

  Run one experiment family:

  python train/run_experiments.py --experiment swarm_scaling --trials 3 --total-steps 10000 --eval-every 2000 --eval-episodes 3

  Run all supported experiment families:

  python train/run_experiments.py --experiment all --trials 3 --total-steps 10000 --eval-every 2000 --eval-episodes 3

  Fast smoke run without plots:

  python train/run_experiments.py --experiment swarm_scaling --trials 1 --total-steps 10 --eval-every 0 --eval-episodes 1 --no-plots

  Result structure

  - per-run artifacts: runs/...
  - aggregated experiment metrics: results/...
  - summary plots: analysis/...

  Validation

  - End-to-end smoke test passed through the new experiment runner
  - Aggregated outputs were created in results/
  - Syntax check passed with:

  PYTHONPYCACHEPREFIX=/Users/christopherlin/dev/cwsf2026/sim/.pycache python -m compileall train analysis experiments env

  Compatibility notes

  - Observation dim remains 23 throughout the framework
  - Existing training loop was reused, not duplicated
  - Existing evaluation logic was reused, not duplicated
  - Existing checkpoint naming remains unchanged
  - Old 19-dim checkpoints are still incompatible and require retraining
  - Baseline comparison is implemented using existing custom-DQN variants:
      - independent DQN
      - shared-policy DQN
