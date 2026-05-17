Step 5 — Collective Intelligence Scaling Experiment

Implement the primary flagship experiment inside the existing stigmergic swarm RL codebase.

GOAL

Add the main experiment that proves swarm performance improves through stigmergic coordination rather than simply increasing the number of robots.

CONTEXT FROM TASK 4 (CRITICAL — DO NOT IGNORE)

The experiment framework has already been implemented with the following capabilities:

ENVIRONMENT + TRAINING

Observation dim = 23

Action space = Discrete(9)

Training uses custom DQN and reads obs_dim dynamically

Logging already includes:

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

AVAILABLE CLI FLAGS (MUST USE, DO NOT REIMPLEMENT)

--n-agents

--n-targets

--n-obstacles

--pheromone-disabled

--failed-agent-count

--observation-noise-std

OUTPUT PIPELINE (ALREADY EXISTS)

runs/ → raw training logs

results/ → aggregated experiment metrics

analysis/ → plots

IMPORTANT CONSTRAINTS

DO NOT write a new experiment runner

DO NOT manually collect metrics

DO NOT duplicate logging or evaluation logic

Use run_experiments.py as the execution entry point

Use benchmark_configs.py to define experiment sweeps

TASK

Implement the "Collective Intelligence Scaling Experiment" USING the existing experiment framework.

EXPERIMENT DESIGN

Swarm sizes:

agents = [1, 2, 3, 5, 10]

Conditions:

Condition A:

RL + pheromone communication

Condition B:

RL without pheromone
(use --pheromone-disabled flag)

Trials:

trials_per_condition = 20

IMPLEMENTATION REQUIREMENTS

Add a new experiment configuration in:

experiments/benchmark_configs.py

Example structure:

collective_intelligence_scaling = {
"agents": [1, 2, 3, 5, 10],
"pheromone": [True, False],
"trials": 20
}

Ensure run_experiments.py can run this config WITHOUT modification to core logic

Ensure metrics are read from existing outputs:

episode_metrics.csv

eval_metrics.csv

summary.json

METRICS (USE EXISTING LOGS)

Use already available metrics:

food_retrieved

exploration_coverage

pheromone_usage

swarm_efficiency

episode_length

mean_episode_reward

DERIVED METRIC (ADD IN ANALYSIS LAYER ONLY)

Compute (in aggregation/analysis stage):

efficiency_per_robot = food_retrieved / n_agents

Do NOT modify training code for this.

VISUALIZATIONS

Extend plotting (analysis/plot_metrics.py or new helper) to generate:

completion_time_vs_agents.png

efficiency_vs_agents.png

pheromone_usage_vs_agents.png

efficiency_per_robot_vs_agents.png

Graphs must include:

mean values

standard deviation (error bars)

labeled axes

legend (pheromone vs no pheromone)

KEY SCIENTIFIC CLAIM TO ENABLE

Your implementation must make it possible to demonstrate:

"Performance improves non-linearly with swarm size ONLY when stigmergic communication is enabled."

OUTPUT

Modify the existing codebase and provide:

files modified

files added

exact command to run this experiment (via run_experiments.py)

exact location of generated plots and CSVs

confirmation that results are derived from the shared experiment framework (not custom logic)



Output:
Implemented the collective-scaling experiment inside the existing framework.

  Files modified

  - experiments/benchmark_configs.py
  - train/run_experiments.py
  - analysis/plot_metrics.py
  - docs/PROJECT_LOG.md

  Files added

  - None

  What changed

  - Added collective_intelligence_scaling to the shared benchmark registry with:
      - n_agents = [1, 2, 3, 5, 10]
      - condition = pheromone_on | pheromone_off
      - default_trials = 20
  - Updated the shared runner so experiments can use per-experiment default trial counts and carry case metadata like n_agents and
    condition into the result CSVs.
  - Kept all metric collection in the existing train/eval pipeline.
  - Added the derived metric efficiency_per_robot = food_retrieved / n_agents only in the aggregation layer.
  - Extended plotting with grouped error-bar plots and forced the headless Agg backend so plot generation works in non-GUI runs.

  How to run
  For the full intended experiment:

  XDG_CACHE_HOME=/Users/christopherlin/dev/cwsf2026/sim/.cache \
  MPLCONFIGDIR=/Users/christopherlin/dev/cwsf2026/sim/.mplcache \
  python train/run_experiments.py --experiment collective_intelligence_scaling

  If you want to override the default 20 trials:

  XDG_CACHE_HOME=/Users/christopherlin/dev/cwsf2026/sim/.cache \
  MPLCONFIGDIR=/Users/christopherlin/dev/cwsf2026/sim/.mplcache \
  python train/run_experiments.py --experiment collective_intelligence_scaling --trials 20

  Smoke-tested with:

  XDG_CACHE_HOME=/Users/christopherlin/dev/cwsf2026/sim/.cache \
  MPLCONFIGDIR=/Users/christopherlin/dev/cwsf2026/sim/.mplcache \
  PYTHONPYCACHEPREFIX=/Users/christopherlin/dev/cwsf2026/sim/.pycache \
  python train/run_experiments.py --experiment collective_intelligence_scaling --trials 1 --total-steps 10 --eval-every 0 --eval-
  episodes 1 --runs-dir /tmp/cis_runs3 --results-dir /tmp/cis_results3 --analysis-dir /tmp/cis_analysis3

  Generated outputs

  - Raw per-run logs: runs/collective_intelligence_scaling/...
  - Aggregated CSVs:
      - results/collective_intelligence_scaling/trial_metrics.csv
      - results/collective_intelligence_scaling/aggregate_metrics.csv
  - Plots:
      - analysis/collective_intelligence_scaling/completion_time_vs_agents.png
      - analysis/collective_intelligence_scaling/efficiency_vs_agents.png
      - analysis/collective_intelligence_scaling/pheromone_usage_vs_agents.png
      - analysis/collective_intelligence_scaling/efficiency_per_robot_vs_agents.png

  Shared-framework confirmation
  This experiment reuses the existing framework end to end:

  - training still goes through train/independent_dqn_pytorch.py
  - evaluation still goes through train/evaluate.py
  - metrics still come from the shared outputs:
      - episode_metrics.csv
      - eval_metrics.csv
      - summary.json
      - eval_summary.json
  - no custom training loop, evaluation loop, or manual metric collection was added

  Compatibility is intact with the current obs_dim = 23 pipeline because the runner still delegates to the existing trainer and
  evaluator.

 
