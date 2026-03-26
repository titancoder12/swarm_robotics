You are working in an existing multi-agent swarm reinforcement learning codebase for a stigmergic foraging environment.

Task:
Implement support for:
1. a configurable `filename` argument for training/evaluation outputs,
2. training and evaluation with pheromone either enabled or disabled via CLI/config,
3. fixed-time evaluation with 4 active targets that continuously respawn after collection,
4. systematic swarm-scaling experiments using those settings.

Goal:
I want to be able to run experiments cleanly with different names, compare “with pheromone” vs “without pheromone,” and evaluate the trained model in a fixed-time setup where 4 targets are always present via respawning.

--------------------------------------------------
PART 1 — ADD CLI / CONFIG ARGUMENTS
--------------------------------------------------

Add the following arguments to the relevant training and evaluation entry points.

1. `--filename`
Purpose:
- A user-specified experiment/model/output name
- Used to name output files, CSVs, graphs, and optionally checkpoints/log folders

Requirements:
- Accept a string, e.g.:
  - `--filename pheromone_on_run1`
  - `--filename no_pheromone_baseline`
- Sanitize if necessary for filesystem safety
- If not provided, use a sensible default such as:
  - `default_run`
  - or timestamp-based fallback

Use this `filename` value consistently in:
- CSV filenames
- graph filenames or subfolders
- experiment metadata
- optional checkpoint prefixes if appropriate

2. `--use-pheromone`
Purpose:
- Enable or disable pheromone behavior during training/evaluation

Requirements:
- Add a clear boolean-style interface, such as one of the following:
  - `--use-pheromone true/false`
  - or paired flags:
    - `--use-pheromone`
    - `--no-pheromone`
- Follow repo style conventions

Behavior:
- When pheromone is enabled:
  - existing pheromone logic works normally
- When pheromone is disabled:
  - pheromone deposition is disabled
  - pheromone sensing/input should either:
    - be zeroed out, or
    - be disabled in a repo-consistent way
- Training should still run cleanly in no-pheromone mode
- Evaluation should also run cleanly in no-pheromone mode

Important:
- Do not break observation shape unless the repo already supports observation variants cleanly
- Prefer keeping observation shape unchanged and zeroing pheromone channels/signals when pheromone is disabled
- If the repo uses pheromone samples in the observation vector, return zeros for those values when pheromone is disabled
- If pheromone rendering exists, optionally suppress rendering or leave it blank/zero

--------------------------------------------------
PART 2 — INCLUDE FIXED-TIME 4-TARGET RESPAWN SETUP
--------------------------------------------------

Update evaluation logic so that:

- Each evaluation episode starts with exactly 4 active targets/food items
- When a target is collected/removed, immediately respawn a replacement target at a random valid location
- Maintain the active target count at 4 throughout the episode
- Use a fixed episode duration rather than “finish when all targets are collected”

Important:
Because targets respawn, “completion time” is no longer the primary metric.
The main question becomes:
- how many targets are collected within a fixed amount of time/steps?

Add a configurable CLI/config variable such as:
- `--eval-steps`
or
- `--fixed-horizon`

This should control the length of each evaluation episode.

Also add:
- `--active-targets 4`

Default:
- `active_targets = 4`

Respawn requirements:
- Respawn only at valid locations
- Avoid invalid placements such as:
  - inside obstacles
  - inside nest if invalid
  - directly overlapping agents if the environment prevents that
- Reuse existing spawn utilities if available

--------------------------------------------------
PART 3 — TRAINING WITH OR WITHOUT PHEROMONE
--------------------------------------------------

Update training logic so experiments can be run in two modes:

A. Pheromone-enabled training
- standard stigmergic behavior
- pheromone deposition/sensing active

B. Pheromone-disabled training
- no effective pheromone deposition
- no effective pheromone information to the policy
- everything else should remain as consistent as possible

Implementation requirements:
- Make this controlled by the CLI/config argument
- Ensure the environment config reflects the chosen mode
- Ensure logs/metadata clearly indicate whether pheromone was enabled
- Include the pheromone mode in filenames / saved outputs where practical

Examples:
- `pheromone_on_*`
- `pheromone_off_*`

If the codebase already has a config object, add something like:
- `use_pheromone: bool = True`

If pheromone is disabled:
- deposited pheromone should be zero
- pheromone field updates may be skipped or remain zeroed
- pheromone observation channels/signals should be zeroed for the policy
- metrics should still work cleanly

--------------------------------------------------
PART 9 — TRAINING CHECKPOINT MILESTONES
--------------------------------------------------

Update training/checkpoint saving so the following milestone models are saved automatically:

- `checkpoints/1_4_trained`
- `checkpoints/1_2_trained`
- `checkpoints/3_4_trained`
- `checkpoints/full_policy`

Interpretation:
- `1_4_trained` = checkpoint saved at 25% of total training progress
- `1_2_trained` = checkpoint saved at 50% of total training progress
- `3_4_trained` = checkpoint saved at 75% of total training progress
- `full_policy` = final checkpoint at 100%

Also:
- include experiment metadata indicating:
  - filename
  - use_pheromone
- if practical, include pheromone mode in auxiliary metadata/logging, but keep the checkpoint filenames exactly as specified unless repo conventions require extensions

If checkpoints use extensions, keep repo style, e.g.:
- `checkpoints/1_4_trained.pt`
- `checkpoints/1_2_trained.pt`
- `checkpoints/3_4_trained.pt`
- `checkpoints/full_policy.pt`

Milestone save logic should:
- save once when crossing each threshold
- not spam repeated saves
- work with resumed training if practical

--------------------------------------------------
PART 10 — CLI / SCRIPT INTERFACE
--------------------------------------------------

Provide or update command-line entry points for training and evaluation.

Training arguments should include at least:
- `--filename`
- `--use-pheromone` and/or `--no-pheromone`
- existing training args
- checkpoint/output args as needed

Evaluation arguments should include at least:
- `--checkpoint`
- `--filename`
- `--use-pheromone` and/or `--no-pheromone`
- `--agent-min 1`
- `--agent-max 30`
- `--agent-step 1`
- `--episodes-per-agent 10`
- `--output-dir experiment_data`
- `--headless`
- `--eval-steps`
- `--active-targets 4`
- `--seed`

Use clear defaults.

--------------------------------------------------
PART 11 — IMPLEMENTATION GUIDANCE
--------------------------------------------------

Search the repo for:
- existing training entry points
- existing evaluation scripts
- metric collection
- plotting scripts
- environment config
- pheromone logic
- target spawning / respawn logic
- checkpoint saving

Prefer reusing existing utilities and metric code.

Keep implementation minimal and repo-consistent:
- do not rewrite the architecture
- do not break training code
- do not change observation shape unless absolutely necessary
- prefer zeroing pheromone inputs rather than changing observation dimensions
- do not add heavy dependencies

If possible, structure code cleanly with helpers such as:
- argument parsing helper
- pheromone mode config helper
- fixed-target-respawn helper
- CSV export helper
- plot saving helper
- milestone checkpoint helper

--------------------------------------------------
PART 12 — DEFINITIONS / METRIC CONSISTENCY
--------------------------------------------------

Be explicit in code comments and final summary about definitions:

- active_targets:
  - number of simultaneously present targets in the environment
  - default 4

- fixed evaluation time:
  - controlled by `--eval-steps`
  - all episodes run for this many steps unless unavoidable early termination occurs

- targets_collected:
  - total number of targets collected over the fixed evaluation horizon
  - because collected targets respawn, this can exceed 4

- efficiency:
  - `targets_collected / number_of_agents`

- coverage_efficiency:
  - `coverage / total_steps_taken`
  - or repo-consistent equivalent

- time_to_first_discovery:
  - first step at which any target is discovered, detected, picked up, or collected
  - document exactly which event is used

- pheromone disabled mode:
  - no effective pheromone deposition
  - no effective pheromone information to the policy
  - preferably observation shape unchanged with pheromone values zeroed

--------------------------------------------------
PART 13 — QUALITY REQUIREMENTS
--------------------------------------------------

- The code must work with a provided trained checkpoint
- The code must support training with and without pheromone
- The code must use real implementation, not pseudocode
- Save outputs to disk cleanly
- Handle missing metrics gracefully
- Use deterministic seeds where practical
- Avoid crashing if one agent count fails; log and continue if reasonable
- Keep filenames and experiment metadata clear
- Make plots readable and labeled
- Keep reward/metric naming consistent with repo terminology

--------------------------------------------------
DELIVERABLES
--------------------------------------------------

After implementing, provide:

1. List of modified / added files
2. Example commands for:
   - training with pheromone enabled
   - training with pheromone disabled
   - evaluation with pheromone enabled
   - evaluation with pheromone disabled
3. Description of each CSV output
4. Description of each generated plot
5. Explanation of how `filename` is used
6. Explanation of how pheromone ON/OFF mode is implemented
7. Explanation of how 4-target respawning works
8. Explanation of milestone checkpoint thresholds
9. Any assumptions made about discovery, collection, coverage, or early termination

--------------------------------------------------
EXAMPLE COMMANDS
--------------------------------------------------

Examples only; adapt to repo conventions.

Training with pheromone:
- `python train/...py --filename pheromone_on_run1 --use-pheromone`

Training without pheromone:
- `python train/...py --filename pheromone_off_run1 --no-pheromone`

Evaluation with pheromone:
- `python analysis/...py --checkpoint checkpoints/full_policy.pt --filename eval_pheromone_on --use-pheromone --agent-min 1 --agent-max 30 --agent-step 1 --episodes-per-agent 10 --eval-steps 2000 --active-targets 4`

Evaluation without pheromone:
- `python analysis/...py --checkpoint checkpoints/full_policy.pt --filename eval_pheromone_off --no-pheromone --agent-min 1 --agent-max 30 --agent-step 1 --episodes-per-agent 10 --eval-steps 2000 --active-targets 4`

--------------------------------------------------
IMPORTANT
--------------------------------------------------

Implement the feature directly in the existing repo.
Keep it runnable.
Follow the project’s current naming/style conventions.

Be practical and repo-aware:
- Reuse existing training, evaluation, and plotting code where possible
- If “targets collected” is actually called “food delivered” or similar in this repo, adapt to the repo’s terminology
- If exploration snapshots are not already supported, implement the simplest clean visited-cell heatmap possible using existing environment state
- Prefer fixed-time throughput metrics over completion metrics in the respawn-based setup
- Prefer keeping observation dimensions unchanged when disabling pheromone by zeroing pheromone-related inputs