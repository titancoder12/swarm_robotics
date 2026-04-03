# GVRSF 2026 Experiment Notes

This document records the experimental program for this project as I want to frame it for GVRSF.

It is based on the repository as it exists now:

- the main research path is recurrent MAPPO with curriculum learning
- the key system behavior is `discover -> pick up -> return -> deliver -> deposit pheromone -> later exploit trail`
- the main measurement paths are:
  - [analysis/evaluate.py](../../analysis/evaluate.py)
  - [analysis/evaluate_comparison.py](../../analysis/evaluate_comparison.py)
  - [train/mappo_gru.py](../../train/mappo_gru.py)
  - [env/swarm_env.py](../../env/swarm_env.py)

The point here is to define the experiments in a way that stands up as real computer-science research, not just a collection of demos.

## Overall Experimental Framing

The project is strongest when I frame it around a clear algorithmic question:

> Can curriculum-trained decentralized reinforcement learning plus stigmergic trail formation produce measurably better multi-agent search-and-return behavior than non-stigmergic or non-curriculum alternatives?

That framing matters because it makes the project:

- algorithmic
- experimentally testable
- practically relevant to unknown-environment navigation
- more substantial than just "I trained a model and it looked cool"

## Main Experimental Story

The strongest version of the research story has three parts:

1. **Curriculum learning is necessary**
   Without the curriculum improvements, the policy collapses or fails to reach robust greedy delivery.
2. **Stigmergy is useful**
   Pheromone-enabled trained swarms outperform comparable no-pheromone or weakened-pheromone conditions.
3. **The behavior scales**
   The trained system keeps meaningful delivery and exploration performance as swarm size increases and environments become harder.

Those three claims together are much stronger than any single experiment in isolation.

## Core Metrics

These are the primary metrics already supported by the current codebase:

- `food_retrieved` / `food_delivered`
- `food_picked_up`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`
- `episode_length`
- `mean_episode_reward`

These are the secondary diagnostic metrics from the current MAPPO path:

- `carrying_stall_events`
- `carrying_stall_fraction`
- `carrying_low_progress_fraction`
- `carrying_low_displacement_fraction`
- `non_carrying_nest_loiter_fraction`
- `non_carrying_nest_crowding_fraction`
- `non_carrying_explore_active_fraction`
- `non_carrying_force_explore_fraction`
- `non_carrying_random_explore_fraction`

The logic behind the metric set is:

- `food_delivered` is the clearest task-success metric
- `delivery_conversion` shows whether pickup turns into completion
- `exploration_coverage` shows whether the swarm is actually searching
- `pheromone_usage` helps support the stigmergy claim
- the carrying and non-carrying diagnostics help explain why performance changes

For a board or report, the four metrics I most likely want to foreground are:

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`

The rest are better used as diagnostic support.

## Experimental Standards

These are the standards I want to hold across the serious experiments.

### Repetition

- Use at least `20` independent evaluation episodes per reported condition.
- For training-comparison experiments, use at least `3` independently trained runs per condition if training time allows.
- If `3` full training runs per condition are too expensive, fall back to:
  - `1` full training run per condition
  - `30` to `50` evaluation episodes per final checkpoint
  - and explicitly state that training variance remains a limitation

### Randomness Control

- Fix the code and hyperparameters.
- Vary only the independent variable being tested.
- Record the exact training seed(s) and evaluation seed ranges.

### Statistics

At minimum, report:

- mean
- standard deviation
- sample size `n`

For the strongest experiments, also report:

- 95% confidence intervals
- an effect size
- a significance test for two-condition comparisons

The statistics do not need to be unusually advanced. They do need to be disciplined.

Practical choices:

- Welch's t-test is acceptable for two-condition comparisons
- Mann-Whitney U is acceptable if the distributions are clearly non-normal
- ANOVA or Kruskal-Wallis is useful for `3+` condition comparisons

### Fairness

For every comparison, keep fixed:

- the evaluation horizon
- the environment difficulty
- the agent count, unless swarm size is the variable
- the checkpoint-selection rule

Otherwise the comparison gets weak quickly.

## Main Experiment Set

These are the five experiments that make the strongest overall package.

---

## Experiment 1: Curriculum Learning vs Simpler Training

### Purpose

Show that the curriculum improvements are not cosmetic. They are necessary for the system to learn robust greedy delivery.

### Hypothesis

The current curriculum-trained MAPPO pipeline should produce significantly better greedy delivery and delivery conversion than a weaker or simplified training setup.

### Independent Variable

Training setup:

- full current MAPPO curriculum
- ablated curriculum or weaker training setup

The main ablation options are:

1. full current curriculum
2. no stage-repeat logic or reduced curriculum stability
3. simplified curriculum with fewer bridge/bootstrap stages

If I only run one ablation, the cleanest comparison is:

- current curriculum
- simplified older-style curriculum

### Dependent Variables

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `mean_episode_reward`

### Controls

Keep fixed:

- same total training budget
- same MAPPO network architecture
- same evaluation horizon
- same final evaluation environment

### Why This Matters

This is not just a performance comparison. It isolates the real algorithmic contribution in the project:

- the training design itself is part of the research contribution

### Presentation Notes

Best figure types:

- grouped bar chart for final `food_delivered`
- grouped bar chart for `delivery_conversion`
- short curriculum diagram showing what the full system teaches

### Expected Interpretation

If the current curriculum wins clearly, the conclusion is:

> The model does not merely need more training. It needs structured learning stages. The curriculum is part of the algorithmic contribution.

---

## Experiment 2: Pheromone Ablation

### Purpose

Test the central scientific claim directly: does stigmergic trail information actually improve swarm performance?

### Hypothesis

Under the same task difficulty, a pheromone-enabled trained swarm should outperform a no-pheromone or pheromone-disabled condition in delivery performance and search efficiency.

### Independent Variable

Pheromone condition during evaluation, and ideally during training as well.

Main conditions:

1. trained with pheromone, evaluated with pheromone
2. trained with pheromone, evaluated without pheromone
3. trained without pheromone, evaluated without pheromone

Optional fourth condition:

4. random or rule-based baseline

### Dependent Variables

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`
- `time_to_first_discovery` if extracted from evaluation output

### Controls

Keep fixed:

- same swarm size
- same environment size and obstacles
- same evaluation steps

### Why This Matters

This experiment tests the central idea of stigmergy instead of assuming it.

### Presentation Notes

Use:

- line or bar plots comparing delivery across conditions
- one or two representative heatmaps or screenshots as qualitative support

The graphs need to carry the main argument. The images are only supporting evidence.

### Expected Interpretation

A strong result would look like:

- pheromone-on achieves similar or better exploration
- pheromone-on achieves clearly better delivery efficiency
- pheromone-off requires more rediscovery and shows weaker route reuse

---

## Experiment 3: Swarm-Size Scaling

### Purpose

Show whether the collective behavior scales as the swarm gets larger.

### Hypothesis

As swarm size increases, delivery performance should improve up to a point, but not linearly forever. The pheromone-enabled system should scale more effectively than weaker baselines.

### Independent Variable

Number of agents.

Core sizes:

- `1`
- `2`
- `3`
- `5`
- `6`

If compute allows:

- `8`
- `10`

### Dependent Variables

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`
- `efficiency per robot`

### Controls

Keep fixed:

- environment difficulty
- evaluation horizon
- checkpoint family

### Why This Matters

This is the experiment that turns the project from a swarm demo into a collective-intelligence study.

### Presentation Notes

Use:

- line plot of `food_delivered` vs `n_agents`
- line plot of `delivery_conversion` vs `n_agents`
- line plot of `efficiency per robot` vs `n_agents`

This should make it possible to show:

- improvement from cooperation
- diminishing returns at higher swarm sizes

That tradeoff makes the result more believable and more scientific.

### Expected Interpretation

The strongest argument is not "more agents always better." It is:

> The system shows a measurable scaling law, with cooperative benefit up to a point and tradeoffs beyond that point.

---

## Experiment 4: Robustness Under Harder Environments

### Purpose

Show that the trained system is not overfitting to one easy layout family.

### Hypothesis

The current MAPPO curriculum-trained system should retain meaningful delivery behavior under increased obstacle density, increased map size, or both.

### Independent Variable

Environment difficulty.

Main conditions:

1. moderate environment
2. large environment
3. final hard environment
4. final hard environment with extra clutter or modified obstacle layouts

### Dependent Variables

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `carrying_stall_fraction`
- `non_carrying_nest_loiter_fraction`

### Controls

Keep fixed:

- policy checkpoint
- swarm size
- evaluation horizon, unless there is a justified scaled horizon

### Why This Matters

This is the clearest way to demonstrate generalization and robustness rather than one curated easy demo.

### Presentation Notes

Use:

- grouped bar chart by difficulty level
- one or two diagnostic plots for `carrying_stall_fraction`

That makes it easier to explain why harder environments are harder.

### Expected Interpretation

The ideal result is:

- the system degrades gracefully rather than collapsing
- delivery drops with difficulty, but not to zero

That is a realistic and credible result.

---

## Experiment 5: Baseline Comparison

### Purpose

Show that the learned curriculum-trained MAPPO system beats simpler alternatives.

### Hypothesis

The current MAPPO system should outperform:

- random behavior
- rule-based behavior
- and probably older DQN baselines on the full delivery task

### Conditions

Main baselines:

1. MAPPO current best checkpoint
2. rule-based baseline
3. random baseline
4. optional shared-policy DQN baseline

### Dependent Variables

- `food_delivered`
- `delivery_conversion`
- `exploration_coverage`
- `pheromone_usage`

### Why This Matters

This is the clearest comparative claim in the project:

> Here is the method I designed, and here is how it compares to simpler alternatives.

### Presentation Notes

Bar charts are probably the best choice here.

The strongest single figure is likely:

- `food_delivered` by algorithm

with:

- `delivery_conversion`

as the secondary comparison.

## Final Set for the Science Fair Board

If time and compute are limited, the first four to prioritize are:

1. curriculum vs weaker training
2. pheromone ablation
3. swarm-size scaling
4. baseline comparison

If time permits, add:

5. robustness under harder environments

That set covers:

- method validity
- mechanism validity
- scaling
- comparative advantage
- robustness

## Suggested Experimental Matrix

This is the practical high-quality matrix I want to follow.

### Tier A: Must Run

#### A1. Curriculum vs simplified curriculum

- `n = 3` trained runs per condition if feasible
- `20` evaluation episodes per trained checkpoint

Conditions:

- current curriculum
- simplified curriculum

#### A2. Pheromone ablation

- `20` to `50` evaluation episodes per condition

Conditions:

- trained with pheromone / eval with pheromone
- trained with pheromone / eval without pheromone
- trained without pheromone / eval without pheromone

#### A3. Swarm scaling

- `20` evaluation episodes per swarm size

Sizes:

- `1, 2, 3, 5, 6`

### Tier B: Strong Additions

#### B1. Baseline comparison

Conditions:

- MAPPO
- rule-based
- random
- optional DQN

#### B2. Robustness stress test

Conditions:

- moderate
- large
- hard
- hard-plus-extra-clutter

## Statistical Framing

For each major chart, report:

- mean
- standard deviation
- `n`

For the two most important claims, also report:

- confidence intervals
- significance testing

The best candidates for formal significance tests are:

1. pheromone-on vs pheromone-off on `food_delivered`
2. curriculum vs simplified curriculum on `delivery_conversion`

Those are the two most publication-like claims in the whole project.

## Threats to Validity

To keep the write-up rigorous, explicitly discuss the limitations.

### Internal validity threats

- evaluation can be sensitive to environment seed
- training variance may be high across runs
- checkpoint selection can bias results if only the best-looking run is shown

### Construct validity threats

- `pheromone_usage` is only a proxy for trail exploitation
- high pickup counts do not necessarily mean successful coordination

### External validity threats

- the simulator is still simplified
- the physical robot path still has only partial observation fidelity

Stating these limitations honestly strengthens the project.

## What To Avoid

These are the weak patterns I do not want in the experiment package:

- showing only screenshots or demos
- reporting only reward
- comparing conditions with different step horizons or different checkpoints unfairly
- using one lucky seed and calling it a result
- making claims about stigmergy without a no-pheromone ablation
- making claims about collective intelligence without a swarm-size scaling study

Those are the mistakes that make the work look like a demo instead of research.

## Figure Set for a Poster or Board

If there is room for only six strong figures, the best set is:

1. **Curriculum vs simplified curriculum**
   - y-axis: `food_delivered`
2. **Curriculum vs simplified curriculum**
   - y-axis: `delivery_conversion`
3. **Pheromone ablation**
   - y-axis: `food_delivered`
4. **Swarm scaling**
   - y-axis: `food_delivered`
5. **Swarm scaling**
   - y-axis: `efficiency per robot`
6. **Baseline comparison**
   - y-axis: `food_delivered`

Then add one small qualitative panel:

- representative pheromone heatmap or delivery-route screenshot

The qualitative image should support the quantitative story, not replace it.

## Best Final Claim

If the experiments work as expected, the strongest final claim is:

> A curriculum-trained decentralized multi-agent reinforcement learning system with stigmergic trail formation achieves measurably better search-and-return performance than simpler baselines, scales across swarm sizes, and remains effective under harder environments.

That is a strong computer-science science-fair claim because it is:

- algorithmic
- quantitative
- testable
- comparative
- relevant

## Run Order

If the schedule is tight, the best order to run the experiments is:

1. pheromone ablation
2. swarm-size scaling
3. baseline comparison
4. curriculum vs simplified curriculum
5. robustness stress test

Why this order:

- the first three give the clearest board-ready figures fastest
- the fourth validates the training methodology
- the fifth adds depth

## Closing Note

The strongest version of the project is not:

> "I trained a swarm and it looked intelligent."

It is:

> "I designed, measured, and validated a decentralized RL system whose collective behavior emerges from curriculum learning and environmental communication, and I can show when and why it works."

That is the standard I want the experiments to support.
