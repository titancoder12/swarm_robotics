Work with the existing swarm RL codebase.

GOAL

Audit the current repository before making changes.

Important constraints:

Do NOT create a new repository.

Do NOT assume a fresh project structure.

Work inside the existing codebase.

Reuse existing files whenever possible.

Preserve all currently working functionality.

TASKS

Inspect the current repository structure.

Identify the existing files related to:

environment

training

rendering / demo

config

experiment scripts

Summarize what already exists.

Identify missing components needed for a science-fair-grade swarm RL system.

Produce an integration plan that maps the desired features onto the current files.

Recommend the minimal set of file edits and any new files that truly need to be added.

OUTPUT

Return a structured documentation-style report (not code):

a concise audit of the current codebase

a file-by-file integration plan

a list of proposed edits

a list of proposed new files

compatibility notes and assumptions

Format the output clearly using sections and bullet points so it can be read like a technical design document.

Do NOT implement code changes in this step.
Do NOT generate large code blocks.
Only include minimal code snippets if needed to explain compatibility issues.


Output:

# Swarm RL Codebase Audit and Integration Plan

  ## 1. Audit Summary

  ### Current State

  This repository is already a functioning swarm RL codebase. It includes:

  - a custom multi-agent swarm environment with PettingZoo Parallel API
  - PyGame rendering
  - a custom PyTorch DQN trainer
  - SB3 and RLlib training backends
  - random rollout and demo scripts
  - checkpoint loading/inference
  - an initial sim-to-real deployment path via robot/ and pi/

  ### Overall Assessment

  The codebase is strong enough for:

  - simulation demos
  - basic RL training
  - comparative backend experiments
  - early sim-to-real scaffolding

  It is not yet a science-fair-grade RL system in the sense of:

  - rigorous experiment management
  - reproducible evaluation
  - metrics/plots/ablation workflow
  - stronger scientific task design
  - automated validation and regression checks

  ## 2. Existing Repository Structure

  ### Environment

  - env/config.py
      - central dataclass for world, sensors, rewards, dynamics, rendering, and seeding
  - env/swarm_env.py
      - PettingZoo Parallel environment
      - action mapping
      - tank/hover dynamics
      - obstacle/target spawning
      - collision handling
      - pheromone grid update
      - observation construction
      - rendering and screenshots

  ### Training

  - train/independent_dqn_pytorch.py
      - custom DQN implementation
      - replay buffer
      - epsilon-greedy exploration
      - target network sync
      - checkpoint saving
  - train/sb3_dqn.py
      - SB3 DQN backend
  - train/rllib_dqn.py
      - RLlib DQN backend with compatibility shims
  - train/train.py
      - backend dispatcher

  ### Rendering / Demo

  - train/demo.py
      - inference and rendering for custom, SB3, and RLlib checkpoints
  - train/random_rollout.py
      - sanity-check runner
  - train/capture_screenshots.py
      - screenshot generation for docs

  ### Shared Model / Deployment

  - models/q_network.py
      - shared Q-network definition
  - robot/messages.py
  - robot/observation_builder.py
  - robot/policy_runner.py
  - robot/action_bridge.py
  - robot/sensor_bridge.py
  - robot/runtime.py
      - generic sim-to-real runtime skeleton
  - pi/ants.py
      - preserved rule-based Pi runtime
  - pi/run_policy.py
      - Pi runtime using learned policy
  - pi/esp32_robot.py
  - pi/esp32_sensor_adapter.py
  - pi/esp32_action_bridge.py
  - pi/ants.service

  ### Documentation / Experiment Notes

  - README.md
  - docs/ARCHITECTURE.md
  - docs/ONBOARDING.md
  - docs/DQN_EXPLAINED.md
  - docs/SimToReal.md
  - docs/PI_MIGRATION.md
  - docs/ToDo.md
  - docs/PROJECT_LOG.md
  - docs/QandA.md

  ## 3. What Already Exists Functionally

  ### Environment

  Already implemented:

  - multi-agent parallel stepping
  - randomized targets and obstacles
  - reward shaping
  - tank and hover dynamics
  - pheromone-based stigmergy
  - local observation pipeline
  - rendering and screenshot capture

  ### Learning

  Already implemented:

  - custom DQN
  - shared-policy and independent-policy modes
  - checkpoint save/load
  - custom/SB3/RLlib backend comparison path

  ### Deployment

  Already implemented:

  - checkpoint-compatible inference path
  - generic deployment abstractions
  - Pi-side serial integration example
  - preserved rule-based Pi baseline

  ## 4. Missing Components for a Science-Fair-Grade Swarm RL System

  ### Scientific Evaluation Gaps

  Missing or weak:

  - standardized evaluation script separate from training
  - fixed benchmark scenarios and seeds
  - aggregated metrics across multiple runs
  - automatic CSV/JSON result logging
  - plots for learning curves and success metrics
  - baseline comparison workflow
  - ablation workflow

  ### Reproducibility Gaps

  Missing or weak:

  - experiment config files
  - reproducible run manifests
  - central seed management across all backends
  - versioned experiment outputs
  - explicit checkpoint metadata

  ### Task / Research Design Gaps

  Missing or weak:

  - richer swarm task beyond “collect targets”
  - return-to-nest / transport behavior
  - carrying state
  - multi-pheromone channels
  - curriculum or staged difficulty
  - robustness/domain randomization experiments

  ### Engineering Validation Gaps

  Missing or weak:

  - automated tests
  - env API regression tests
  - observation-shape and action-map tests
  - smoke tests for each backend
  - checkpoint round-trip tests

  ### Presentation / Science Fair Gaps

  Missing or weak:

  - explicit hypothesis and experimental questions
  - benchmark tables
  - repeatable demo protocol
  - evaluation visuals and plots
  - “what changed / why it matters” experiment summary

  ## 5. File-by-File Integration Plan

  ### Environment Layer

  #### env/config.py

  Use as the main feature/config expansion point for:

  - curriculum settings
  - optional carrying state flags
  - optional nest behavior
  - observation noise/domain randomization toggles
  - evaluation-specific fixed scenario flags

  Minimal change approach:

  - add new optional config fields
  - keep defaults matching current behavior

  #### env/swarm_env.py

  Use as the integration point for:

  - richer task mechanics
  - additional reward terms
  - optional nest/carry state
  - optional second pheromone channel
  - deterministic eval scenarios

  Minimal change approach:

  - preserve current reset/step/render contract
  - gate new behavior behind config flags
  - do not break obs/action layout unless explicitly versioned

  ### Training Layer

  #### train/independent_dqn_pytorch.py

  Use as the main custom research trainer for:

  - improved logging
  - evaluation hooks
  - curriculum scheduling
  - checkpoint metadata
  - optional reward/episode summaries to file

  Minimal change approach:

  - keep current training loop intact
  - add logging/eval hooks around existing loop
  - avoid changing checkpoint tensor shapes

  #### train/sb3_dqn.py

  Use mainly as:

  - library baseline
  - comparison run path

  Minimal change approach:

  - add seed/logging consistency only
  - do not over-customize unless needed for fair comparison

  #### train/rllib_dqn.py

  Use mainly as:

  - advanced baseline
  - multi-agent/distributed comparison path

  Minimal change approach:

  - keep compatibility shims isolated
  - avoid making RLlib the primary research path unless needed

  #### train/train.py

  Use as:

  - unified experiment entrypoint

  Minimal change approach:

  - add shared options for experiment name/output dir/eval mode
  - keep backend dispatch structure

  ### Rendering / Demo Layer

  #### train/demo.py

  Use for:

  - visual qualitative evaluation
  - demo-video generation hook
  - science-fair presentation demos

  Minimal change approach:

  - add optional overlay text / metrics
  - optionally add deterministic demo seeds
  - preserve current backend support

  #### train/random_rollout.py

  Use for:

  - env smoke test
  - baseline sanity check

  Minimal change approach:

  - keep as-is, maybe add --headless and seed flags later

  #### train/capture_screenshots.py

  Use for:

  - documentation and poster visuals

  Minimal change approach:

  - optionally add more benchmark scenes
  - keep existing image generation behavior

  ### Shared Model / Deployment Layer

  #### models/q_network.py

  Use as:

  - single source of truth for custom checkpoint-compatible model definition

  Minimal change approach:

  - keep architecture stable unless checkpoint versioning is introduced

  #### robot

  Use for:

  - deployment abstractions
  - simulated sensor/action contract
  - eventual real-robot runtime standardization

  Minimal change approach:

  - preserve current observation/action contract
  - add metadata/logging, not structural churn

  #### pi

  Use for:

  - concrete Raspberry Pi runtime
  - current baseline and migrated learned-policy runtime

  Minimal change approach:

  - keep pi/ants.py behavior-preserving
  - evolve pi/run_policy.py as the end-state runtime

  ## 6. Proposed Minimal Set of File Edits

  ### High Priority Edits

  - env/config.py
      - add optional experiment/eval/task-extension flags
  - env/swarm_env.py
      - add optional task extensions behind flags
      - add deterministic eval scenario support
  - train/independent_dqn_pytorch.py
      - add structured metrics logging
      - add evaluation hook
      - add experiment output directories
  - train/train.py
      - unify experiment CLI options across backends
  - train/demo.py
      - optional deterministic demo mode and HUD/metrics overlay
  - README.md
      - add experiment/evaluation workflow
  - docs/ARCHITECTURE.md
      - update once evaluation/task extensions land

  ### Medium Priority Edits

  - robot/runtime.py
      - add logging and health/status output
  - pi/run_policy.py
      - add safer runtime controls and logging hooks

  ## 7. Proposed New Files

  Only add files that fill clear missing system roles.

  ### Recommended New Files

  - train/evaluate.py
      - deterministic evaluation across seeds/checkpoints
      - outputs success rate, reward, collisions, target collection
  - train/experiment_utils.py
      - shared output-dir creation, metric writing, seed setup
  - experiments/default_eval.yaml or experiments/benchmark_configs.py
      - experiment definitions for reproducible runs
  - analysis/plot_metrics.py
      - generate science-fair-grade plots from logged CSV/JSON
  - tests/test_env_smoke.py
      - reset/step/shape/basic reward sanity checks
  - tests/test_checkpoint_inference.py
      - verify saved custom checkpoint reloads and infers
  - docs/EXPERIMENT_PLAN.md
      - hypothesis, baselines, metrics, ablation matrix

  ### Optional New Files

  - env/tasks.py
      - if task logic becomes too large for swarm_env.py
  - env/rewards.py
      - if reward logic becomes more complex
  - env/observation.py
      - if _get_obs() grows substantially
  - train/curriculum.py
      - if curriculum scheduling becomes non-trivial

  ## 8. Compatibility Notes

  ### Preserve These Contracts

  - observation shape/order for existing checkpoints
  - discrete 9-action mapping
  - current checkpoint filenames for custom backend
  - current SwarmEnv Parallel API behavior
  - current pi/ants.py behavior
  - current train/demo.py backend behavior

  ### Be Careful With

  - changing QNetwork architecture
      - this breaks checkpoint compatibility
  - changing observation layout silently
      - this invalidates trained models and deployment assumptions
  - changing action semantics
      - this affects both simulation and robot runtime
  - adding richer tasks directly into default behavior
      - should be config-gated to preserve current demos/training

  ## 9. Assumptions

  - “Science-fair-grade” means:
      - reproducible experiments
      - interpretable metrics
      - baseline comparisons
      - clear task framing and results
  - Existing working functionality must remain the default path.
  - Sim-to-real support should remain compatible with the current observation/action contract unless explicitly versioned.
  - The custom trainer remains the best place for research iteration; SB3 and RLlib should stay comparison backends, not the primary
    integration target.

  ## 10. Recommended Sequence

  1. Add evaluation and experiment logging before adding new swarm mechanics.
  2. Lock down reproducible benchmark runs and plots.
  3. Then extend the task: nest/carry/multi-pheromone/curriculum.
  4. Only after that, adapt the Pi/runtime side to any changed observation or action contracts.

  ## 11. Bottom Line

  This codebase already contains the core of a real swarm RL project:

  - environment
  - trainer
  - baselines
  - demo path
  - deployment scaffolding

  The minimal path forward is not a rewrite. It is:

  - add evaluation infrastructure
  - add reproducibility/metrics
  - add one or two task extensions behind config flags
  - preserve current checkpoints, demos, and Pi baseline while extending the system carefully

  No code changes were made in this step.

 
