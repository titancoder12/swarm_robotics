# Instructions: Execute the Full GVRSF 2026 Experiment Program

This document is the standing execution brief for the full experiment program in this repository.

The experimental plan is based on:

- [docs/gvrsf2026/experiments/experiments.md](./experiments.md)

The point of this run is not to produce a lightweight demo summary. The point is to execute a full experimental campaign, generate the artifacts, analyze the results, and write a complete final report.

## Overall Goal

Use the current repository to:

1. execute the strongest feasible experiments described in [docs/gvrsf2026/experiments/experiments.md](./experiments.md)
2. generate the relevant raw data and processed artifacts
3. produce clear figures and tables
4. write a complete final report under `docs/gvrsf2026/experiments/`
5. keep the result reproducible from the repo

The final deliverables should include:

- raw outputs
- processed summaries
- graphs
- experiment metadata
- a final written report

The final report should embed the generated graphs inline with Markdown image syntax. The figures should appear directly in the relevant sections instead of being left as file links only.

## Core Principle

Do not run weak, one-off, underpowered comparisons and call them results.

The experiment set needs to be:

- controlled
- repeated
- quantitatively grounded
- tied to the actual metrics the repo logs
- explicit about threats to validity

If full-scale execution of every ideal condition is not practical, then:

1. prioritize the highest-signal experiments first
2. preserve scientific validity
3. document exactly what was run and what was reduced
4. explain the tradeoffs honestly in the report

## Primary Source of Truth

The main experiment design guidance is in:

- [docs/gvrsf2026/experiments/experiments.md](./experiments.md)

The practical evaluation and training tooling is in:

- [analysis/evaluate.py](../../../analysis/evaluate.py)
- [analysis/evaluate_comparison.py](../../../analysis/evaluate_comparison.py)
- [train/train.py](../../../train/train.py)
- [train/mappo_gru.py](../../../train/mappo_gru.py)
- [train/run_experiments.py](../../../train/run_experiments.py)
- [experiments/benchmark_configs.py](../../../experiments/benchmark_configs.py)

Execution should be based on the current code, not on assumptions.

## Required Experiment Set

The strongest feasible version of these experiment families should be attempted:

1. **Curriculum Learning vs Weaker or Simplified Training**
2. **Pheromone Ablation**
3. **Swarm-Size Scaling**
4. **Robustness Under Harder Environments**
5. **Baseline Comparison**

The final experiment bundle should be a true full bundle, not a narrow subset folder that contains only one family.

That means one timestamped campaign folder should include:

- the broad multi-family bundle
  - curriculum vs weaker training
  - baseline comparison
  - robustness
  - general swarm-size scaling
- and a dedicated decisive experiment focused on the strongest stigmergy claim

### Required decisive experiment

**Scaling Laws of Stigmergic Collective Intelligence**

This decisive experiment should test:

- swarm size as an independent variable
  - preferred: `1`, `2`, `3`, `5`, `10`
  - acceptable smaller set if needed: `1`, `3`, `6`, `10`
- communication condition as an independent variable
  - RL + pheromone
  - RL without pheromone
  - optional baseline: rule-based or random without learned coordination

The experiment should be designed to support the following claim if the data allow it:

> As the swarm grows, performance improves much more strongly when stigmergy is enabled than when it is disabled.

The full bundle should not be reduced to only the decisive experiment.
The decisive experiment should not be reduced to only a single swarm size.

Preferred outcome variables for the decisive experiment:

- completion time or time-to-task-threshold
- `food_delivered`
- success rate
- efficiency per robot
- exploration coverage
- pheromone usage
- late or post-discovery deliveries

The strongest result is not merely that larger swarms do more total work. The strongest result is that the **gap between pheromone and no-pheromone conditions widens as swarm size increases**, while per-agent efficiency stays higher under pheromone.

If the current checkpoints or tooling make the exact preferred swarm-size set impractical, a smaller but still scientifically strong set is acceptable, such as:

- `1`, `3`, `6`
- or `1`, `2`, `3`, `6`, `10`

If that happens, the report should explain why the exact preferred set was not used.

If compute or time makes the full matrix unrealistic, preserve scientific quality by:

- fully executing the most important subset first:
  - pheromone ablation
  - swarm-size scaling
  - baseline comparison
  - curriculum-vs-weaker-training if practical
- then add robustness if time allows

Do not silently skip major parts. If something is skipped or reduced, document it clearly.

Preferred bundle structure inside one timestamped run:

1. broad campaign section
2. decisive stigmergy-scaling section
3. integrated final interpretation section explaining how the decisive experiment strengthens the full bundle

For future runs, a full-bundle campaign can be assembled in either of these two valid ways:

1. **Fresh unified execution**
   - rerun every required family directly into the current timestamped folder
2. **Federated full-bundle execution**
   - reuse one or more prior timestamped subcampaigns if they are already scientifically valid, stronger than a rushed rerun, and fully reproducible from the repo

If the federated approach is used, the new timestamped folder still has to be a complete bundle. It needs to include:

- an integrated `report.md`
- organized subfolders or copied artifacts for the broad campaign
- organized subfolders or copied artifacts for the decisive scaling campaign
- a provenance table explaining which source campaign each section came from
- a justification for why reuse was stronger than rerunning

Do not silently rely on old runs. If prior subcampaigns are reused, the report should say so explicitly and treat the new timestamped folder as the self-contained bundle.

## Phase 1: Repository Audit Before Running

Before running experiments:

1. inspect the current MAPPO path and current evaluation tooling
2. inspect what metrics are already logged and which are trustworthy
3. inspect what existing checkpoints are available
4. determine which experiments can reuse existing strong checkpoints
5. determine which experiments require fresh training

Required outcome:

- produce a concise execution plan in the final report
- explain what used existing checkpoints versus what required new training

Do not guess. Read the current code and artifacts first.

## Phase 2: Experimental Design Translation

Translate [experiments.md](./experiments.md) into a concrete runnable matrix.

For each experiment family, define:

- independent variable(s)
- dependent variable(s)
- control variables
- number of training runs
- number of evaluation episodes
- checkpoint source
- exact command pattern
- output directories

At minimum, every experiment section in the final report should state:

- hypothesis
- variables
- method
- metrics
- sample size
- analysis approach

## Phase 3: Artifact Directory Structure

Create a clean experiment artifact structure under:

- `docs/gvrsf2026/experiments/`
- and reusable machine-readable outputs under:
  - `runs/`
  - `results/`
  - `experiments/experiment_data/`

Every full experiment campaign should create a timestamped run folder under:

- `docs/gvrsf2026/experiments/<timestamp>/`

where `<timestamp>` uses a stable sortable format such as:

- `YYYYMMDD_HHMMSS`

All report-facing artifacts for one campaign should live inside that timestamped folder.

Required final folder structure for one experiment campaign should include at least:

- `docs/gvrsf2026/experiments/experiments.md`
- `docs/gvrsf2026/experiments/instruction.md`
- `docs/gvrsf2026/experiments/<timestamp>/report.md`
- `docs/gvrsf2026/experiments/<timestamp>/figures/`
- `docs/gvrsf2026/experiments/<timestamp>/tables/`
- `docs/gvrsf2026/experiments/<timestamp>/metadata/`

If needed, also create:

- `docs/gvrsf2026/experiments/<timestamp>/raw_exports/`
- `docs/gvrsf2026/experiments/<timestamp>/analysis_notes/`
- `docs/gvrsf2026/experiments/<timestamp>/commands/`

All generated report-facing artifacts should have stable names and be easy to inspect.

Additionally:

- the report should state the timestamped run folder it belongs to
- figures and tables referenced in the report should be linked from inside that same timestamped folder
- the report should embed the key figures inline inside the relevant experiment sections
- if summary index files are created at the top level of `docs/gvrsf2026/experiments/`, they should not replace or overwrite per-run artifacts

Important documentation rule for this workflow:

- do **not** update `docs/PROJECT_LOG.md`
- do **not** update `docs/QandA.md`

The timestamped experiment folder under `docs/gvrsf2026/experiments/<timestamp>/` should be treated as the complete self-contained record of the run.

## Phase 4: Experiment Execution Standards

### Repetition

Follow the discipline in [experiments.md](./experiments.md):

- for evaluation-only comparisons: aim for `20` or more episodes per condition
- for training comparisons: aim for multiple independent runs per condition when practical

For the pheromone family specifically, if the first pass is promising but statistically borderline:

- increase the primary pheromone comparison to `50` or more paired episodes
- prioritize the main swarm size where stigmergy should matter most, usually the largest swarm tested
- keep smaller swarm sizes as supporting analyses if compute becomes a constraint

If compute limits force a smaller setup:

- preserve fair controls
- increase evaluation episodes if training repeats are reduced
- explain the limitation honestly

### Fairness

For each comparison:

- keep evaluation horizon fixed unless there is a justified scaling rule
- keep environment settings matched where appropriate
- keep swarm size fixed unless it is the independent variable
- keep checkpoint-selection logic consistent
- for pheromone studies, use paired evaluation layouts:
  - the same seed or layout should be evaluated across all pheromone conditions before moving to the next seed
- do not compare a best-looking demo seed against a random seed average

### Metrics

Use actual repo metrics wherever possible.

Primary metrics:

- `food_delivered` / `food_retrieved`
- `food_picked_up`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`
- `mean_episode_reward`

Secondary diagnostics when useful:

- `carrying_stall_events`
- `carrying_stall_fraction`
- `carrying_low_progress_fraction`
- `non_carrying_nest_loiter_fraction`
- `non_carrying_nest_crowding_fraction`
- `non_carrying_explore_active_fraction`
- `non_carrying_force_explore_fraction`
- `non_carrying_random_explore_fraction`

Do not hide behind reward alone. Task-grounded metrics need to be primary.

## Phase 5: Specific Experiment Requirements

### A. Curriculum Learning vs Weaker or Simplified Training

Goal:

- show that the current curriculum design materially improves learning quality

Requirements:

- compare the current curriculum against at least one weaker or simplified training condition
- use a consistent training budget if practical
- evaluate all resulting checkpoints under the same final evaluation conditions
- focus especially on:
  - `food_delivered`
  - `delivery_conversion`
  - greedy-eval quality

If retraining multiple full conditions is too expensive:

- run the strongest reduced comparison that can still be justified
- explain the reduction clearly

### B. Pheromone Ablation

Goal:

- test the central stigmergy claim directly

This experiment family should be designed so the stigmergy claim is genuinely testable. Do not rely only on a weak same-checkpoint, pheromone-on-vs-off at eval-time comparison.

Required conditions if feasible:

1. trained with pheromone, evaluated with pheromone
2. trained with pheromone, evaluated without pheromone
3. trained without pheromone, evaluated without pheromone

Strongly preferred:

4. trained without pheromone, evaluated in the matched layouts used by the other conditions

Design requirements:

- prioritize matched training conditions over evaluation-only toggles whenever such checkpoints exist or can be trained feasibly
- run the pheromone comparison at multiple swarm sizes if practical:
  - at minimum `1`, `3`, and `6`
- if compute is limited, treat `6` agents as the primary stigmergy condition and `1`/`3` agents as supporting context
- use paired layouts:
  - for each seed, run all pheromone conditions on the same environment seed before moving to the next seed
- choose environments where trail reuse should matter:
  - not trivially easy
  - not so hard that all conditions fail
  - long enough that one agent can discover a route and later agents can exploit it

For the pheromone family, total deliveries alone are not enough. The analysis should also test whether performance improves **after the first discovery**.

Focus metrics:

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`

Additional pheromone-specific metrics:

- `time_to_first_discovery`
- `time_to_first_delivery`
- `pickup_to_delivery_latency`
- delivery count in early vs late episode windows
- post-discovery delivery rate

Statistical requirement for the pheromone family:

- use paired or layout-matched statistical analysis whenever possible
- if the main pheromone effect is borderline at `n=20`, rerun with a larger paired sample before concluding that the effect is weak
- the report should clearly identify:
  - the primary pheromone hypothesis test
  - its sample size
  - its p-value and effect size

Execution requirement for the pheromone family:

- define one **primary pheromone result** ahead of time, preferably:
  - the largest swarm size tested
  - matched training conditions
  - repeated-source or otherwise trail-reuse-friendly environment
- aim to produce a statistically significant primary pheromone result
- if the first pheromone run is not significant or not convincing, strengthen the design and rerun:
  - increase paired sample size
  - prioritize the primary swarm size over secondary sizes
  - use a more trail-reuse-friendly environment such as repeated-source foraging
  - keep the comparison matched and fair
- do not stop at a merely qualitative "looks better" result if a stronger feasible design can still be run

Acceptable stopping rule:

- the pheromone section can stop iterating when the primary pheromone result is both:
  - statistically significant by the stated primary test
  - scientifically interpretable as a stigmergy effect rather than a vague score difference

Interpretation requirement:

- a strong stigmergy result should show either:
  - significantly higher total delivery under pheromone-enabled conditions
  - or significantly faster or more efficient post-discovery exploitation
  - or both

If the pheromone result remains weak, the report should say so plainly and explain whether the likely reason is:

- evaluation-only ablation instead of matched training
- environments that do not strongly reward trail reuse
- insufficient swarm-size-dependent effect
- or another concrete limitation

### C. Swarm-Size Scaling

Goal:

- test collective-intelligence scaling

Recommended swarm sizes:

- `1`
- `2`
- `3`
- `5`
- `6`

Optional:

- `8`
- `10`

Focus metrics:

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `efficiency per robot`

### D. Robustness Under Harder Environments

Goal:

- show that the learned system degrades gracefully rather than collapsing

Vary:

- obstacle density
- environment size
- or both

Focus metrics:

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `carrying_stall_fraction`

### E. Baseline Comparison

Goal:

- compare the current system to simpler alternatives

Required baselines where feasible:

- MAPPO current best checkpoint
- rule-based baseline
- random baseline
- optional DQN baseline

Focus metrics:

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`

## Phase 6: Graph and Table Requirements

Generate report-quality figures.

Required figure qualities:

- readable labels
- axis titles
- units or metric meaning where relevant
- clear legends
- no misleading scales
- export to PNG at minimum

Recommended figure set:

1. curriculum-vs-weaker-training delivery figure
2. curriculum-vs-weaker-training conversion figure
3. pheromone-ablation delivery figure
4. swarm-size-scaling delivery figure
5. swarm-size-scaling efficiency-per-robot figure
6. baseline-comparison delivery figure
7. optional robustness figure

Also generate tables for:

- exact condition definitions
- sample sizes
- summary statistics
- significance tests if used

All figures and tables used in the final report should be saved as separate artifacts.

## Phase 7: Statistical Analysis Requirements

At minimum, report:

- mean
- standard deviation
- sample size `n`

For the strongest claims, also include:

- 95% confidence intervals
- significance testing where appropriate

Good candidates for significance testing:

- pheromone-on vs pheromone-off
- current curriculum vs weaker curriculum

Use simple, defensible methods. Do not use flashy statistics unless justified.

If statistical testing is omitted for any comparison, explain why.

## Phase 8: Final Report Requirements

Write the final report to:

- `docs/gvrsf2026/experiments/<timestamp>/report.md`

The report should be polished and publication-style while still readable to a smart semi-technical reader.

Required sections:

1. Title
2. Abstract
3. Research Question
4. Motivation and Background
5. System Overview
6. Experimental Methodology
7. Experiment 1 Results
8. Experiment 2 Results
9. Experiment 3 Results
10. Experiment 4 Results
11. Experiment 5 Results
12. Discussion
13. Threats to Validity
14. Limitations
15. Conclusion
16. Reproducibility Appendix

The report should:

- cite the relevant repo files
- cite the exact commands or command patterns used
- include links to the generated figures and tables
- state what was actually run
- distinguish between planned and completed experiments if anything was reduced

The tone should be:

- rigorous
- concise
- evidence-driven

It should read like a serious project report, not an internal scratchpad.

## Phase 9: Reproducibility Requirements

The final report should include a reproducibility section stating:

- repo revision used
- important checkpoints used
- key commands used
- random seed strategy
- output directories
- any practical runtime limitations

Also save machine-readable experiment metadata for each major experiment family under:

- `docs/gvrsf2026/experiments/<timestamp>/metadata/`

This metadata should make the final report auditable.

## Phase 10: Documentation Requirements

Do not modify unrelated code unless required to complete the experiments.

Documentation updates may be made under `docs/gvrsf2026/` only.

Do **not** update:

- `docs/PROJECT_LOG.md`
- `docs/QandA.md`

Do not clutter the repo with unnecessary intermediate prose.

## Phase 11: Verification Requirements

Before finalizing:

1. verify that all referenced figures exist
2. verify that all referenced tables exist
3. verify that the report links point to the correct artifacts
4. verify that the conclusions match the actual data
5. verify that no major claim is unsupported

If any experiment family was not completed, the report should say so clearly.
