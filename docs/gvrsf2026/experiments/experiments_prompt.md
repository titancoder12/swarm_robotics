# Prompt: Execute the Full GVRSF 2026 Experiment Program

You are working in an existing multi-agent swarm reinforcement learning codebase.

Your task is to execute the experimental program based on the plan in:

- [docs/gvrsf2026/experiments.md](./experiments.md)

The project is entered in the **computer science** category of the Greater Vancouver Regional Science Fair. The bar is high: the work must be rigorous, reproducible, quantitatively convincing, and presented like a serious applied computer science research project.

This is not a request for a lightweight demo summary.
This is a request to run a full experimental campaign, generate artifacts, analyze the results, and write a polished final report.

The final outcome should be strong enough to support a top-prize science fair presentation.

## Overall Goal

Use the current repository to:

1. execute the strongest feasible experiments described in [docs/gvrsf2026/experiments.md](./experiments.md)
2. generate all relevant raw data and processed artifacts
3. produce clear figures and tables
4. write a complete final report under `docs/gvrsf2026/experiments/`
5. ensure everything is reproducible from the repo

The final deliverables must include:

- raw outputs
- processed summaries
- graphs
- experiment metadata
- a final written report

The final report must **embed** the generated graphs inline with Markdown image syntax.
Do not leave graphs as file links only.
Readers should be able to read the report and see the figures directly in context without opening each image separately.

## Core Principle

Do not run weak, one-off, underpowered comparisons and call them “science.”

The experiment set must be:

- controlled
- repeated
- quantitatively grounded
- tied to the actual metrics the repo logs
- explicit about threats to validity

If full-scale execution of every ideal condition is not practical, you must:

1. prioritize the highest-signal experiments first
2. preserve scientific validity
3. document exactly what was run and what was reduced
4. explain the tradeoffs honestly in the report

## Primary Source of Truth

The experiment design guidance is in:

- [docs/gvrsf2026/experiments.md](./experiments.md)

The practical evaluation/training tooling is in:

- [analysis/evaluate.py](../../analysis/evaluate.py)
- [analysis/evaluate_comparison.py](../../analysis/evaluate_comparison.py)
- [train/train.py](../../train/train.py)
- [train/mappo_gru.py](../../train/mappo_gru.py)
- [train/run_experiments.py](../../train/run_experiments.py)
- [experiments/benchmark_configs.py](../../experiments/benchmark_configs.py)

You must base the execution on the current code, not on assumptions.

## Required Experiment Set

You must attempt the strongest feasible version of the following experiment families from [experiments.md](./experiments.md):

1. **Curriculum Learning vs Weaker/Simplified Training**
2. **Pheromone Ablation**
3. **Swarm-Size Scaling**
4. **Robustness Under Harder Environments**
5. **Baseline Comparison**

The final experiment bundle must be a **true full bundle**, not a narrow subset folder that only contains one family.

That means a satisfactory execution should include, in one timestamped campaign folder:

- the broad multi-family bundle
  - curriculum vs weaker training
  - baseline comparison
  - robustness
  - general swarm-size scaling
- and a dedicated **killer experiment** focused on the strongest stigmergy claim

Required killer experiment:

- **Scaling Laws of Stigmergic Collective Intelligence**

This killer experiment must test:

- swarm size as an independent variable
  - prefer `1`, `2`, `3`, `5`, `10`
  - if needed, justify a slightly smaller set such as `1`, `3`, `6`, `10`
- communication condition as an independent variable
  - RL + pheromone
  - RL without pheromone
  - optional baseline: rule-based or random without learned coordination

The killer experiment must be designed to support the following claim if the data allow it:

> As the swarm grows, performance improves much more strongly when stigmergy is enabled than when it is disabled.

Do not reduce the full bundle to only the killer experiment.
Do not reduce the killer experiment to only a single swarm size.

For the killer experiment, prefer these outcome variables:

- completion time or time-to-task-threshold
- `food_delivered`
- success rate
- efficiency per robot
- exploration coverage
- pheromone usage
- late/post-discovery deliveries

The strongest version of the result is not merely that larger swarms do more total work.
The strongest version is that the **gap between pheromone and no-pheromone conditions widens as swarm size increases**, while per-agent efficiency stays higher under pheromone.

If the repository’s current checkpoints or tooling make the exact preferred swarm-size set impractical, you may use a smaller but still scientifically strong set such as:

- `1`, `3`, `6`
- or `1`, `2`, `3`, `6`, `10`

But the report must explain why the exact preferred set was not used.

If compute/time makes the full matrix unrealistic, preserve scientific quality by:

- fully executing the most important subset first:
  - pheromone ablation
  - swarm-size scaling
  - baseline comparison
  - curriculum-vs-weaker-training if practical
- then add robustness if time allows

Do not silently skip major parts.
If you skip or downscale something, document it clearly.

Preferred bundle structure inside one timestamped run:

1. broad campaign section
2. killer stigmergy scaling section
3. integrated final interpretation section that explains how the killer experiment strengthens the full bundle

For future executions, a full-bundle campaign may be assembled in either of these two valid ways:

1. **Fresh unified execution**
   - rerun every required family directly into the current timestamped folder
2. **Federated full-bundle execution**
   - reuse one or more prior timestamped subcampaigns if they are already scientifically valid, stronger than a rushed rerun, and fully reproducible from the repo

If the federated approach is used, the new timestamped folder must still be a **complete bundle**. It must include:

- an integrated `report.md`
- organized subfolders or copied artifacts for the broad campaign
- organized subfolders or copied artifacts for the killer scaling campaign
- a provenance table explaining which source campaign each section came from
- a justification for why reuse was scientifically stronger than rerunning

Do not silently rely on old runs.
If prior subcampaigns are reused, the report must say so explicitly and treat the new timestamped folder as the self-contained science-fair bundle.

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

Do not guess.
Read the current code and current artifacts.

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

At minimum, every experiment section in the final report must state:

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

Every full experiment campaign must create a timestamped run folder under:

- `docs/gvrsf2026/experiments/<timestamp>/`

where `<timestamp>` is in a stable sortable format such as:

- `YYYYMMDD_HHMMSS`

This is required so the experiment program can be run multiple times without overwriting earlier results.

All report-facing artifacts for one campaign must live inside that timestamped folder.

Required final folder structure for one experiment campaign should include at least:

- `docs/gvrsf2026/experiments/experiments.md` (already exists; treat as the standing plan)
- `docs/gvrsf2026/experiments/experiments_prompt.md` (this file; standing execution prompt)
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

- the report must state the timestamped run folder it belongs to
- figures and tables referenced in the report must be linked from inside that same timestamped folder
- the report must embed the key figures inline inside the relevant experiment sections, not just link to them
- if summary index files are created at the top level of `docs/gvrsf2026/experiments/`, they must not replace or overwrite per-run artifacts

Important documentation rule for this experiment workflow:

- do **not** update `docs/PROJECT_LOG.md`
- do **not** update `docs/QandA.md`

The timestamped experiment folder under `docs/gvrsf2026/experiments/<timestamp>/` should be treated as the complete self-contained record of the run.

## Phase 4: Experiment Execution Standards

### Repetition

Follow the discipline recommended in [experiments.md](./experiments.md):

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
  - the same seed/layout must be evaluated across all pheromone conditions before moving to the next seed
- do not compare “best cherry-picked demo seed” against a random seed average

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

Do not hide behind reward alone.
Task-grounded metrics must be primary.

## Phase 5: Specific Experiment Requirements

### A. Curriculum Learning vs Weaker/Simplified Training

Goal:
- show that the current curriculum design materially improves learning quality

Requirements:
- compare the current curriculum against at least one weaker/simplified training condition
- use consistent training budget if practical
- evaluate all resulting checkpoints under the same final evaluation conditions
- focus especially on:
  - `food_delivered`
  - `delivery_conversion`
  - greedy-eval quality

If retraining multiple full conditions is too expensive:
- run the strongest reduced comparison you can justify
- explain the reduction clearly

### B. Pheromone Ablation

Goal:
- test the central stigmergy claim directly

This experiment family must be redesigned to make the stigmergy claim genuinely testable.
Do not rely only on a weak “same checkpoint, pheromone on vs off at eval time” comparison.

Required conditions if feasible:

1. trained with pheromone, evaluated with pheromone
2. trained with pheromone, evaluated without pheromone
3. trained without pheromone, evaluated without pheromone

Strongly preferred:

4. trained without pheromone, evaluated with pheromone disabled but with the same matched layouts as the other conditions

Design requirements:

- prioritize **matched training conditions** over evaluation-only toggles whenever such checkpoints exist or can be trained feasibly
- run the pheromone comparison at multiple swarm sizes if practical:
  - at minimum `1`, `3`, and `6`
- if compute is limited, treat `6` agents as the primary stigmergy condition and `1`/`3` agents as supporting context
- use paired layouts:
  - for each seed, run all pheromone conditions on the same environment seed before moving to the next seed
- choose environments where trail reuse should matter:
  - not trivially easy
  - not so hard that all conditions fail
  - enough episode length for one agent to discover a route and later agents to exploit it

For the pheromone family, do not treat total deliveries alone as sufficient evidence.
You must also analyze whether performance improves **after the first discovery**.

Focus metrics:

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`

Additional required pheromone-specific metrics:

- `time_to_first_discovery`
- `time_to_first_delivery`
- `pickup_to_delivery_latency`
- delivery count in early vs late episode windows
- post-discovery delivery rate

Statistical requirement for the pheromone family:

- use paired or layout-matched statistical analysis whenever possible
- if the main pheromone effect is borderline at `n=20`, rerun with a larger paired sample before concluding the effect is weak
- the report must clearly identify:
  - the primary pheromone hypothesis test
  - its sample size
  - its p-value and effect size

Execution requirement for the pheromone family:

- define one **primary pheromone result** ahead of time, preferably:
  - the largest swarm size tested
  - matched training conditions
  - repeated-source or otherwise trail-reuse-friendly environment
- the execution should aim to produce a **statistically significant** primary pheromone result
- if the first pheromone run is not significant or not scientifically convincing, you must strengthen the design and rerun
  - increase paired sample size
  - prioritize the primary swarm size over secondary sizes
  - use a more trail-reuse-friendly environment such as repeated-source foraging
  - keep the comparison matched and fair
- do not stop at a merely qualitative “looks better” result if a stronger feasible design can still be run

Acceptable stopping rule:

- the pheromone section may stop iterating when the primary pheromone result is both:
  - statistically significant by the stated primary test
  - scientifically interpretable as a stigmergy effect rather than a vague score difference

Interpretation requirement:

- a strong stigmergy result should show either:
  - significantly higher total delivery under pheromone-enabled conditions
  - or significantly faster/more efficient **post-discovery exploitation**
  - or both

If the pheromone result remains weak, the report must say so plainly and explain whether the likely reason is:

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

Optionally:

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

All figures and tables used in the final report must be saved as separate artifacts.

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

Use simple, defensible methods.
Do not use flashy statistics unless justified.

If statistical testing is omitted for any comparison, explain why.

## Phase 8: Final Report Requirements

Write the final report to:

- `docs/gvrsf2026/experiments/<timestamp>/report.md`

The report must be polished and publication-style, while still accessible to a smart semi-technical science-fair audience.

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

The report must:

- cite the relevant repo files
- cite the exact commands or command patterns used
- include links to the generated figures and tables
- state what was actually run
- distinguish between planned and completed experiments if any were reduced

The tone should be:

- rigorous
- concise
- evidence-driven
- science-fair-ready

Do not write it like an internal scratchpad.
Write it like a serious project report.

## Phase 9: Reproducibility Requirements

The final report must include a reproducibility section that states:

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

You may update documentation under `docs/gvrsf2026/` only.

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

If any experiment family was not completed, the report must say so clearly.

## Deliverables

At the end, you must provide:

1. a concise summary of what experiments were actually run
2. the location of all main artifacts
3. the location of the final report
4. the strongest conclusions supported by the data
5. remaining limitations or missing experiment families, if any

## Important Constraints

Do not:

- fabricate results
- hide incomplete experiments
- rely on screenshots as primary evidence
- claim stigmergy without an ablation
- claim collective intelligence without a scaling experiment
- cherry-pick one demo seed and present it as the main result

Do:

- prioritize scientific validity
- preserve fairness across comparisons
- be explicit about what was measured
- produce a report worthy of a top science fair submission
