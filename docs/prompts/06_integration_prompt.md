Step 6 — RL Algorithm Comparison Experiment

Implement a comparative study inside the existing swarm RL project.

GOAL

Evaluate whether stigmergic swarm coordination works across multiple control strategies using the existing experiment framework and training pipeline.

CONTEXT FROM PROMPT 5 (CRITICAL — DO NOT IGNORE)

The system already includes a fully working experiment pipeline and one completed flagship experiment.

ENVIRONMENT + TRAINING

Observation dim = 23

Action space = Discrete(9)

Custom DQN trainer working

Logging includes:

mean_episode_reward

food_retrieved

exploration_coverage

pheromone_usage

collisions

episode_length

swarm_efficiency

EXPERIMENT FRAMEWORK (ALREADY EXISTS — MUST REUSE)

train/run_experiments.py (central runner)

experiments/benchmark_configs.py

train/experiment_utils.py

train/evaluate.py

analysis/plot_metrics.py

AVAILABLE SWEEP FLAGS

--n-agents

--pheromone-disabled

--failed-agent-count

--observation-noise-std

OUTPUT PIPELINE

runs/ → raw logs

results/ → aggregated CSVs

analysis/ → plots

EXISTING EXPERIMENT (REFERENCE IMPLEMENTATION)

collective_intelligence_scaling already implemented

Uses shared framework

Produces:

trial_metrics.csv

aggregate_metrics.csv

plots with mean + std

IMPORTANT CONSTRAINTS

DO NOT create a new experiment runner

DO NOT duplicate training loops

DO NOT duplicate evaluation logic

DO NOT manually compute metrics outside the aggregation layer

MUST integrate into benchmark_configs.py

MUST run via run_experiments.py

MUST produce outputs identical in structure to Step 5

TASK

Implement "RL Algorithm Comparison Experiment" USING the existing experiment framework.

ALGORITHMS TO COMPARE

DQN (existing baseline)

Shared-policy DQN (if already supported in codebase)

Rule-based ant agents (NEW — must be implemented)

NOTE:

PPO is OPTIONAL — only include if trivial to plug in

Do NOT build a full PPO pipeline from scratch

RULE-BASED BASELINE (REQUIRED)

Implement a lightweight non-learning agent policy:

Behavior:

If pheromone detected → follow gradient

If carrying food → move toward nest

If food detected → move toward food

Otherwise → random exploration

Constraints:

Must use same observation inputs (23-dim)

Must use same action space (Discrete(9))

Must plug into evaluation pipeline (train/evaluate.py) without modifying core logic

EXPERIMENT DESIGN

Swarm sizes:

agents = [1, 3, 5, 10]

Conditions:

pheromone enabled

pheromone disabled

Trials:

trials_per_condition = 20

IMPLEMENTATION REQUIREMENTS

Add experiment config to:

experiments/benchmark_configs.py

Example:

rl_algorithm_comparison = {
"algorithms": ["dqn", "shared_dqn", "rule_based"],
"agents": [1, 3, 5, 10],
"pheromone": [True, False],
"trials": 20
}

Extend run_experiments.py to:

switch algorithm via config

reuse existing training pipeline for DQN variants

bypass training for rule-based agents (evaluation only)

Ensure ALL algorithms produce results through:

train/evaluate.py

same logging outputs

METRICS (USE EXISTING LOGS)

Use already available metrics:

food_retrieved

swarm_efficiency

exploration_coverage

pheromone_usage

episode_length

mean_episode_reward

ADDITIONAL DERIVED METRICS (ANALYSIS ONLY)

Compute in aggregation layer only:

convergence_speed (episodes to stable reward)

time_to_first_food

Do NOT modify training logic.

VISUALIZATIONS

Generate:

algorithm_food_retrieval.png

algorithm_efficiency.png

algorithm_convergence_speed.png

algorithm_comparison_bar.png

Requirements:

mean + standard deviation

consistent axes across algorithms

legends (algorithm types)

KEY SCIENTIFIC CLAIM TO ENABLE

"Stigmergic coordination enables learned agents (RL) to outperform rule-based agents and scale more effectively with swarm size."

OUTPUT

Modify the existing codebase and provide:

files modified

files added

exact command to run this experiment

confirmation that ALL algorithms used the SAME evaluation and logging pipeline

confirmation that results are comparable to Step 5 outputs

Output:
Implemented the RL algorithm comparison experiment in the existing framework.

  Files modified

  - experiments/benchmark_configs.py
  - train/evaluate.py
  - train/run_experiments.py
  - docs/PROJECT_LOG.md

  Files added

  - models/rule_based_policy.py

  What changed

  - Added rl_algorithm_comparison to the shared benchmark registry with:
      - algorithms: dqn, shared_dqn, rule_based
      - swarm sizes: 1, 3, 5, 10
      - conditions: pheromone on/off
      - default trials: 20
  - Extended the shared evaluator to support --policy-kind rule_based while keeping the same eval_metrics.csv and eval_summary.json
    outputs.
  - Added a lightweight rule-based baseline that uses the same 23-dim observation and the same Discrete(9) action space.
  - Extended the shared experiment runner so rule-based cases skip training but still run through the same evaluator and result
    aggregation path.
  - Added analysis-layer derived metrics:
      - convergence_speed
      - time_to_first_food
  - Added Step 6 plots:
      - algorithm_food_retrieval.png
      - algorithm_efficiency.png
      - algorithm_convergence_speed.png
      - algorithm_comparison_bar.png

  Exact command

  XDG_CACHE_HOME=/Users/christopherlin/dev/cwsf2026/sim/.cache \
  MPLCONFIGDIR=/Users/christopherlin/dev/cwsf2026/sim/.mplcache \
  python train/run_experiments.py --experiment rl_algorithm_comparison

  Output locations

  - Raw run artifacts: runs/rl_algorithm_comparison/...
  - Aggregated CSVs:
      - results/rl_algorithm_comparison/trial_metrics.csv
      - results/rl_algorithm_comparison/aggregate_metrics.csv
  - Plots:
      - analysis/rl_algorithm_comparison/algorithm_food_retrieval.png
      - analysis/rl_algorithm_comparison/algorithm_efficiency.png
      - analysis/rl_algorithm_comparison/algorithm_convergence_speed.png
      - analysis/rl_algorithm_comparison/algorithm_comparison_bar.png

  Shared-pipeline confirmation
  Yes. All algorithms use the same evaluation and logging path:

  - DQN and shared-policy DQN still use the existing training pipeline
  - all three algorithms, including rule_based, go through train/evaluate.py
  - all results are written into the same trial_metrics.csv and aggregate_metrics.csv schema

  That makes the Step 6 outputs directly comparable to Step 5 outputs. I also smoke-tested the full experiment path with --trials
  1 --total-steps 10 --eval-episodes 1, and it produced both the aggregate CSVs and the four required plots.

 
