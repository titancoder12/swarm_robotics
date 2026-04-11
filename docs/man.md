# Swarm RL Command Manual

## NAME

`man.md` — command reference for training, experiments, evaluation, demos, and analysis in the swarm robotics simulator.

## SYNOPSIS

Run commands from the simulator repo root:

```bash
cd /Users/christopherlin/dev/cwsf2026/sim
source .venv/bin/activate
```

Primary entry points:

```bash
python train/train.py ...
python train/independent_dqn_pytorch.py ...
python train/mappo_gru.py ...
python train/run_experiments.py ...
python analysis/evaluate.py ...
python analysis/evaluate_comparison.py ...
python train/demo.py ...
python train/policy_probe.py ...
python train/random_rollout.py
```

## DESCRIPTION

This repo has two main training paths:

- recurrent MAPPO, which is the current main path
- custom PyTorch DQN, which is the older baseline path

It also includes:

- a top-level training dispatcher
- an experiment runner for repeatable sweeps
- single-checkpoint evaluation
- comparison evaluation across pheromone conditions
- a visual demo runner
- small utilities for policy inspection and screenshots

This manual focuses on the commands that are actually implemented in the repo today.

## QUICK START

Fastest MAPPO smoke test:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --folder-name mappo_trail_smoke
```

Recommended full MAPPO run:

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --stage-repeat-limit 1 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_full_run
```

Quick DQN baseline run:

```bash
python train/train.py --backend custom --headless --total-steps 10000 --eval-every 2000 --folder-name dqn_smoke
```

Open a trained checkpoint in the simulator:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --max-steps 300 --render-scale 0.75
```

## COMMON PATTERNS

### Output Locations

- training runs: `runs/`
- checkpoints: `checkpoints/`
- experiment aggregates: `results/`
- experiment plots: `analysis/`
- evaluation outputs: usually `runs/eval/` or a custom `--output-dir`

### Frequently Used Shared Environment Flags

Many training, evaluation, and demo commands inherit the shared environment flags from `train/experiment_utils.py`.

Common ones:

- `--folder-name`
- `--filename`
- `--n-targets`
- `--n-obstacles`
- `--max-steps-per-episode`
- `--dynamics-mode tank|hover|mixed`
- `--action-repeat-steps`
- `--use-pheromone` / `--no-use-pheromone`
- `--pheromone-disabled`
- `--failed-agent-count`
- `--observation-noise-std`
- `--target-nest-distance-min`
- `--target-nest-distance-max`
- `--obstacle-layout`
- `--reward-pickup`
- `--reward-nest-delivery`
- `--reward-undelivered-food`
- `--reward-new-cell`
- `--eval-steps`
- `--active-targets`
- `--target-respawn` / `--no-target-respawn`
- `--food-source-capacity`
- `--render-scale`

Example environment overrides:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --total-steps 5000 --n-obstacles 2 --active-targets 1 --no-target-respawn
```

```bash
python analysis/evaluate.py --policy-kind mappo_gru --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --episodes 5 --headless --no-use-pheromone
```

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --render-scale 0.8 --obstacle-layout random
```

## COMMANDS

## `python train/train.py`

### PURPOSE

Top-level dispatcher for training backends.

### BACKENDS

- `custom` — custom PyTorch DQN
- `sb3` — Stable-Baselines3 DQN
- `rllib` — RLlib DQN
- `mappo` — recurrent GRU MAPPO

### MOST IMPORTANT FLAGS

- `--backend`
- `--use-pheromone` / `--no-use-pheromone`

All remaining flags are passed through to the selected backend.

### EXAMPLES

Run the default custom DQN backend:

```bash
python train/train.py --headless --total-steps 10000
```

Run MAPPO:

```bash
python train/train.py --backend mappo --headless --curriculum full --total-steps 600000
```

Run DQN with pheromones forced off:

```bash
python train/train.py --backend custom --headless --total-steps 20000 --no-use-pheromone
```

Run a short MAPPO smoke test with a custom folder:

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --total-steps 2400 --folder-name mappo_smoke
```

## `python train/independent_dqn_pytorch.py`

### PURPOSE

Train the custom PyTorch DQN baseline directly.

### IMPORTANT FLAGS

- `--total-steps`
- `--n-agents`
- `--shared-policy`
- `--headless`
- `--cuda`
- `--seed`
- `--save-dir`
- `--save-every`
- `--experiment-name`
- `--output-dir`
- `--eval-every`
- `--eval-episodes`
- `--epsilon-start`
- `--epsilon-final`
- `--epsilon-decay-steps`
- `--warmup-steps`

### EXAMPLES

Basic single run:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 10000
```

Shared-policy DQN:

```bash
python train/independent_dqn_pytorch.py --headless --shared-policy --total-steps 25000 --folder-name shared_dqn_trial
```

Longer run with periodic evaluation:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 50000 --eval-every 5000 --eval-episodes 5 --save-every 5000 --folder-name dqn_eval_run
```

More exploratory run:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 40000 --epsilon-start 1.0 --epsilon-final 0.1 --epsilon-decay-steps 20000
```

Pheromone-off ablation:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 20000 --no-use-pheromone --folder-name dqn_no_pheromone
```

Small cluttered test:

```bash
python train/independent_dqn_pytorch.py --headless --total-steps 15000 --n-obstacles 2 --active-targets 1 --no-target-respawn --folder-name dqn_easy_world
```

## `python train/mappo_gru.py`

### PURPOSE

Train the recurrent GRU MAPPO path directly.

### IMPORTANT FLAGS

- `--headless`
- `--cuda`
- `--seed`
- `--n-agents`
- `--total-steps`
- `--rollout-steps`
- `--update-epochs`
- `--minibatch-size`
- `--lr`
- `--gamma`
- `--gae-lambda`
- `--clip-ratio`
- `--entropy-coef`
- `--value-coef`
- `--max-grad-norm`
- `--hidden-size`
- `--eval-every`
- `--eval-episodes`
- `--output-dir`
- `--save-dir`
- `--curriculum stage1|stage1_to_2|full`
- `--resume-checkpoint`
- `--stage-repeat-limit`
- `--no-plots`

### EXAMPLES

Short stage-1 smoke test:

```bash
python train/mappo_gru.py --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --folder-name mappo_stage1_smoke
```

Full recommended run:

```bash
python train/mappo_gru.py --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --stage-repeat-limit 1 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_full_run
```

Intermediate curriculum:

```bash
python train/mappo_gru.py --headless --curriculum stage1_to_2 --n-agents 6 --total-steps 120000 --eval-every 5000 --folder-name mappo_stage12
```

Resume from a checkpoint:

```bash
python train/mappo_gru.py --headless --curriculum full --resume-checkpoint checkpoints/mappo_full_run/latest --total-steps 200000 --folder-name mappo_resume
```

Force a smaller rollout / minibatch combo:

```bash
python train/mappo_gru.py --headless --curriculum stage1 --total-steps 12000 --rollout-steps 64 --update-epochs 2 --minibatch-size 128
```

Pheromone-off MAPPO ablation:

```bash
python train/mappo_gru.py --headless --curriculum full --total-steps 80000 --no-use-pheromone --folder-name mappo_no_pheromone
```

## `python train/run_experiments.py`

### PURPOSE

Run repeatable benchmark experiment families using the custom DQN pipeline plus shared evaluation and aggregation.

### AVAILABLE EXPERIMENTS

- `all`
- `swarm_scaling`
- `stigmergy_ablation`
- `baseline_comparison`
- `robot_failure_test`
- `noise_robustness`
- `collective_intelligence_scaling`
- `rl_algorithm_comparison`

### IMPORTANT FLAGS

- `--experiment`
- `--trials`
- `--seed`
- `--total-steps`
- `--eval-every`
- `--eval-episodes`
- `--runs-dir`
- `--results-dir`
- `--analysis-dir`
- `--save-every`
- `--headless`
- `--cuda`
- `--no-plots`

### EXAMPLES

Run everything:

```bash
python train/run_experiments.py --experiment all
```

Run the collective-intelligence experiment:

```bash
python train/run_experiments.py --experiment collective_intelligence_scaling
```

Run the RL algorithm comparison:

```bash
python train/run_experiments.py --experiment rl_algorithm_comparison
```

Cheap test run:

```bash
python train/run_experiments.py --experiment rl_algorithm_comparison --trials 1 --total-steps 100 --eval-every 0 --eval-episodes 1
```

Run with explicit output directories:

```bash
python train/run_experiments.py --experiment collective_intelligence_scaling --runs-dir runs_tmp --results-dir results_tmp --analysis-dir analysis_tmp
```

Save intermediate checkpoints during each trial:

```bash
python train/run_experiments.py --experiment robot_failure_test --trials 3 --total-steps 20000 --save-every 5000
```

## `python analysis/evaluate.py`

### PURPOSE

Evaluate a single checkpoint or rule-based policy and write evaluation metrics.

### POLICY KINDS

- `dqn`
- `mappo_gru`
- `rule_based`

### IMPORTANT FLAGS

- `--checkpoint-dir`
- `--shared-policy`
- `--policy-kind`
- `--episodes`
- `--n-agents`
- `--seed`
- `--output-dir`
- `--headless`
- `--debug-policy`

### EXAMPLES

Evaluate a shared DQN checkpoint:

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/dqn_run --shared-policy --policy-kind dqn --episodes 10 --headless
```

Evaluate a MAPPO checkpoint:

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --policy-kind mappo_gru --episodes 10 --headless
```

Evaluate the rule-based policy:

```bash
python analysis/evaluate.py --policy-kind rule_based --episodes 10 --headless --output-dir runs/eval_rule_based
```

Evaluate without pheromones:

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --policy-kind mappo_gru --episodes 5 --headless --no-use-pheromone
```

Evaluate with more agents than training:

```bash
python analysis/evaluate.py --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --policy-kind mappo_gru --episodes 5 --n-agents 10 --headless
```

## `python analysis/evaluate_comparison.py`

### PURPOSE

Compare checkpoints trained with and without pheromone over a sweep of swarm sizes.

### IMPORTANT FLAGS

- `--checkpoint-with-pheromone`
- `--checkpoint-without-pheromone`
- `--policy-kind dqn|mappo_gru`
- `--shared-policy` / `--no-shared-policy`
- `--agent-min`
- `--agent-max`
- `--agent-step`
- `--episodes-per-agent`
- `--output-dir`
- `--seed`
- `--headless`
- `--max-steps`

### EXAMPLES

Basic comparison:

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/with_pher --checkpoint-without-pheromone checkpoints/without_pher --policy-kind dqn --headless
```

MAPPO comparison:

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/mappo_with_pher/best_greedy_eval --checkpoint-without-pheromone checkpoints/mappo_without_pher/best_greedy_eval --policy-kind mappo_gru --headless
```

Sweep agents from 1 to 12:

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/with_pher --checkpoint-without-pheromone checkpoints/without_pher --agent-min 1 --agent-max 12 --agent-step 1 --episodes-per-agent 5 --headless
```

Short-horizon comparison:

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/with_pher --checkpoint-without-pheromone checkpoints/without_pher --max-steps 300 --episodes-per-agent 3 --headless
```

Write results to a dedicated directory:

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/with_pher --checkpoint-without-pheromone checkpoints/without_pher --output-dir experiments/pheromone_comparison --headless
```

## `python train/demo.py`

### PURPOSE

Run a trained or baseline policy in the interactive simulator.

### BACKENDS

- `custom`
- `sb3`
- `rllib`
- `mappo`
- `random`

### IMPORTANT FLAGS

- `--backend`
- `--checkpoint-dir`
- `--sb3-model`
- `--rllib-checkpoint`
- `--shared-policy`
- `--n-agents`
- `--seed`
- `--seed-list`
- `--headless`
- `--max-steps`
- `--demo-epsilon`
- `--debug-policy`
- `--mc-relay-url`
- `--mc-relay-session`
- `--mc-tcp-host`
- `--mc-tcp-port`
- `--render-scale`

### EXAMPLES

Open the MAPPO demo:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --max-steps 300 --render-scale 0.75
```

Rotate through hand-picked seeds:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --seed-list 45,40,58 --max-steps 0 --render-scale 0.75
```

Run a custom DQN checkpoint:

```bash
python train/demo.py --backend custom --checkpoint-dir checkpoints/dqn_run --shared-policy --max-steps 300
```

Run a random policy:

```bash
python train/demo.py --backend random --max-steps 300
```

Run a headless TCP-connected demo that streams into Mission Control:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --headless --max-steps 200 --mc-tcp-host 127.0.0.1 --mc-tcp-port 8765
```

Run against the HTTP relay:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --headless --max-steps 200 --mc-relay-url https://example.com --mc-relay-session sim_demo
```

Run with slight exploratory noise:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --demo-epsilon 0.05 --max-steps 500
```

## `python train/policy_probe.py`

### PURPOSE

Inspect custom DQN checkpoints on hand-written observation cases.

### IMPORTANT FLAGS

- `--checkpoint-dir`
- `--shared-policy`
- `--agent-index`
- `--case`
- `--list-cases`

### EXAMPLES

List available probe cases:

```bash
python train/policy_probe.py --list-cases
```

Probe a shared checkpoint on `target_ahead`:

```bash
python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead
```

Probe agent 0 from an independent checkpoint set:

```bash
python train/policy_probe.py --checkpoint-dir checkpoints --case wall_ahead --agent-index 0
```

Try a pheromone-following case:

```bash
python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case pheromone_trail
```

Inspect carrying-food behavior:

```bash
python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case carrying_to_nest
```

## `python train/random_rollout.py`

### PURPOSE

Run the environment with random actions for quick visual sanity checks.

### EXAMPLES

Default rollout:

```bash
python train/random_rollout.py
```

Use it after environment changes:

```bash
python train/random_rollout.py
```

Use it before training to verify rendering:

```bash
python train/random_rollout.py
```

## `python train/capture_screenshot.py`

### PURPOSE

Capture one representative simulator screenshot to `docs/images/swarm_demo.png`.

### EXAMPLES

Basic screenshot capture:

```bash
python train/capture_screenshot.py
```

Regenerate docs image after a render change:

```bash
python train/capture_screenshot.py
```

Quick smoke test for rendering:

```bash
python train/capture_screenshot.py
```

## `python train/capture_screenshots.py`

### PURPOSE

Generate the standard screenshot bundle used in the docs.

### OUTPUTS

- `docs/images/swarm_default.png`
- `docs/images/swarm_no_pheromone.png`
- `docs/images/swarm_hover.png`
- `docs/images/swarm_dense.png`

### EXAMPLES

Generate the full screenshot set:

```bash
python train/capture_screenshots.py
```

Refresh screenshots before updating docs:

```bash
python train/capture_screenshots.py
```

Verify rendering modes:

```bash
python train/capture_screenshots.py
```

## LEGACY / OPTIONAL BACKENDS

## `python train/sb3_dqn.py`

### PURPOSE

Train a Stable-Baselines3 DQN baseline.

### EXAMPLES

```bash
python train/sb3_dqn.py --headless --total-steps 10000
```

```bash
python train/sb3_dqn.py --headless --total-steps 25000 --n-agents 6 --save-path checkpoints/sb3_trial.zip
```

```bash
python train/sb3_dqn.py --total-steps 5000 --n-agents 3
```

## `python train/rllib_dqn.py`

### PURPOSE

Train an RLlib DQN baseline.

### EXAMPLES

```bash
python train/rllib_dqn.py --headless --total-steps 10000
```

```bash
python train/rllib_dqn.py --headless --total-steps 25000 --n-agents 6 --save-dir checkpoints/rllib_trial
```

```bash
python train/rllib_dqn.py --headless --total-steps 5000 --ray-tmpdir /tmp/ray_swarm
```

## RECIPES

### Recipe: MAPPO Smoke Test

```bash
python train/train.py --backend mappo --headless --curriculum stage1 --n-agents 6 --total-steps 2400 --rollout-steps 64 --update-epochs 2 --minibatch-size 128 --eval-every 0 --no-plots --folder-name mappo_smoke
```

### Recipe: Full MAPPO Training + Demo

```bash
python train/train.py --backend mappo --headless --curriculum full --n-agents 6 --total-steps 600000 --rollout-steps 128 --update-epochs 4 --minibatch-size 256 --eval-every 10000 --eval-episodes 5 --stage-repeat-limit 1 --reward-pickup 6 --reward-nest-delivery 30 --reward-undelivered-food -10 --folder-name mappo_full_run
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --max-steps 300 --render-scale 0.75
```

### Recipe: DQN Baseline + Evaluation

```bash
python train/train.py --backend custom --headless --total-steps 50000 --eval-every 5000 --folder-name dqn_baseline
python analysis/evaluate.py --checkpoint-dir checkpoints/dqn_baseline --shared-policy --policy-kind dqn --episodes 10 --headless --output-dir runs/eval_dqn_baseline
```

### Recipe: Pheromone Ablation

With pheromone:

```bash
python train/mappo_gru.py --headless --curriculum full --total-steps 80000 --folder-name mappo_with_pher
```

Without pheromone:

```bash
python train/mappo_gru.py --headless --curriculum full --total-steps 80000 --no-use-pheromone --folder-name mappo_without_pher
```

Compare:

```bash
python analysis/evaluate_comparison.py --checkpoint-with-pheromone checkpoints/mappo_with_pher/best_greedy_eval --checkpoint-without-pheromone checkpoints/mappo_without_pher/best_greedy_eval --policy-kind mappo_gru --headless
```

### Recipe: Stream Simulator Into Mission Control

Start Mission Control separately, then:

```bash
python train/demo.py --backend mappo --checkpoint-dir checkpoints/mappo_full_run/best_greedy_eval --headless --max-steps 300 --mc-tcp-host 127.0.0.1 --mc-tcp-port 8765
```

### Recipe: Generate Presentation Assets

```bash
python train/capture_screenshot.py
python train/capture_screenshots.py
```

## NOTES

- The current recommended path is recurrent MAPPO, not the older DQN baselines.
- `train/train.py` is the safest entry point when you want a backend switch without remembering each file name.
- `train/demo.py` now supports Mission Control streaming over TCP or relay mode.
- `train/run_experiments.py` is built around the custom DQN experiment framework, not the MAPPO curriculum trainer.
- For policy inspection of the custom DQN only, use `train/policy_probe.py`.

## SEE ALSO

- [README.md](/Users/christopherlin/dev/cwsf2026/sim/README.md)
- [docs/manual/QUICK_START.md](/Users/christopherlin/dev/cwsf2026/sim/docs/manual/QUICK_START.md)
- [docs/TRAINING_MAPPO.md](/Users/christopherlin/dev/cwsf2026/sim/docs/TRAINING_MAPPO.md)
- [docs/manual/EXPERIMENT_GUIDE.md](/Users/christopherlin/dev/cwsf2026/sim/docs/manual/EXPERIMENT_GUIDE.md)
- [docs/EVALUATE.md](/Users/christopherlin/dev/cwsf2026/sim/docs/EVALUATE.md)
