# Statistical Analysis for Experiment Bundle `20260406_102744`

This document summarizes the main statistical findings from:

- [20260406_102744](/Users/christopherlin/dev/cwsf2026/sim/docs/gvrsf2026/experiments/20260406_102744)

Primary sources used:

- [report_20260406_102744.md](/Users/christopherlin/dev/cwsf2026/sim/docs/gvrsf2026/experiments/report_20260406_102744.md)
- [family_summary.csv](/Users/christopherlin/dev/cwsf2026/sim/docs/gvrsf2026/experiments/20260406_102744/tables/family_summary.csv)
- [statistical_tests.csv](/Users/christopherlin/dev/cwsf2026/sim/docs/gvrsf2026/experiments/20260406_102744/tables/statistical_tests.csv)
- [killer_scaling/condition_summary.csv](/Users/christopherlin/dev/cwsf2026/sim/docs/gvrsf2026/experiments/20260406_102744/killer_scaling/tables/condition_summary.csv)
- [killer_scaling/statistical_tests.csv](/Users/christopherlin/dev/cwsf2026/sim/docs/gvrsf2026/experiments/20260406_102744/killer_scaling/tables/statistical_tests.csv)

## Scope

This bundle contains two campaign families:

- `broad`
  - curriculum vs weaker training
  - baseline comparison
  - generic pheromone ablation
  - robustness
  - swarm-size scaling
- `killer_scaling`
  - repeated-source stigmergy tests, especially at `6` and `30` agents

Broad conditions mostly use `n = 20` episodes per condition.

The killer campaign uses paired seeds:

- `n = 50` at `6` agents
- `n = 50` at `30` agents
- `n = 20` for some supporting sizes

## 1. Strongest Supported Claims

The most statistically solid findings in this bundle are:

- Curriculum improvement is real.
  - `food_delivered`: `1.25` vs `0.00`
  - Welch t-test `p = 0.000828`
  - Cohen's `d = 1.254`
- Learned MAPPO beats both baselines.
  - vs rule-based: `1.25` vs `0.30`, `p = 0.01225`, `d = 0.846`
  - vs random: `1.25` vs `0.05`, `p = 0.001235`, `d = 1.189`
- The strongest stigmergy result is in the killer repeated-source task.
  - `6` agents: `1.32` vs `0.00`, paired t `p = 0.00653`, Wilcoxon `p = 0.00364`
  - `30` agents: `2.66` vs `0.00`, paired t `p = 0.000847`, Wilcoxon `p = 9.96e-06`

Those are the claims I would treat as the bundle's core evidence.

## 2. Curriculum vs Older Weaker Training

Comparison:

- `current_curriculum_final`
- vs `older_weaker_curriculum_final`

Main metric:

- `food_delivered`

Results:

- current curriculum:
  - `food_picked_up = 1.30`
  - `food_delivered = 1.25`
  - `delivery_conversion = 0.583`
  - `exploration_coverage = 0.212`
- older weaker model:
  - `food_picked_up = 1.90`
  - `food_delivered = 0.00`
  - `delivery_conversion = 0.000`
  - `exploration_coverage = 0.146`

Statistical test:

- Welch t-test `p = 0.000828`
- Cohen's `d = 1.254`

Interpretation:

- the weaker model still found and picked up food
- but it largely failed the return-and-deliver stage
- this is strong evidence that the curriculum improved completion of the full foraging loop, not just wandering or pickup frequency

This is a strong result because it is not just “better reward”; it is “better completion of the task loop.”

## 3. Baseline Comparison

Comparison family:

- `current_mappo`
- `rule_based`
- `random`

Results:

- MAPPO:
  - `food_delivered = 1.25`
  - `delivery_conversion = 0.583`
  - `exploration_coverage = 0.212`
- rule-based:
  - `food_delivered = 0.30`
  - `delivery_conversion = 0.125`
  - `exploration_coverage = 0.099`
- random:
  - `food_delivered = 0.05`
  - `delivery_conversion = 0.050`
  - `exploration_coverage = 0.273`

Statistical tests:

- MAPPO vs rule-based:
  - `p = 0.012250`
  - `d = 0.846`
- MAPPO vs random:
  - `p = 0.001235`
  - `d = 1.189`

Effect-size interpretation:

- MAPPO delivers about `4.17x` as much food as the rule-based baseline
- MAPPO delivers about `25x` as much food as the random baseline

Interpretation:

- the learned policy outperforms both simpler baselines
- the random baseline explores widely but converts that exploration into almost no successful delivery
- that matters because it shows exploration alone is not enough; coordinated return behavior is the real bottleneck

## 4. Generic Pheromone Ablation

Comparison:

- `trained_with_pheromone_eval_with_pheromone`
- vs `trained_with_pheromone_eval_without_pheromone`

Results:

- with pheromone:
  - `food_delivered = 1.25`
- without pheromone at eval:
  - `food_delivered = 1.05`

Change:

- absolute drop: `0.20`
- relative drop: about `16%`

Statistical test:

- `p = 0.625411`
- `d = 0.156`

Interpretation:

- this is a weak effect statistically
- on its own, this comparison is not strong enough to support the main stigmergy claim
- this justifies the need for the stronger repeated-source `killer_scaling` experiment

This is one of the most important nuance points in the whole bundle:

- broad generic ablation is not enough
- mechanism-specific repeated-source evaluation is what exposes the stigmergy effect

## 5. Swarm Size Scaling

Family:

- `swarm_size_scaling`

`food_delivered_mean` by swarm size:

- `1` agent: `0.05`
- `2` agents: `0.50`
- `3` agents: `0.35`
- `4` agents: `1.00`
- `6` agents: `1.25`
- `10` agents: `1.90`
- `15` agents: `2.75`
- `20` agents: `3.85`
- `30` agents: `4.65`

Other scaling metrics:

- `delivery_conversion`
  - `0.05` at `1` agent
  - `0.859` at `30` agents
- `exploration_coverage`
  - `0.0487` at `1` agent
  - `0.4402` at `30` agents

Computed descriptive correlations across swarm size:

- swarm size vs `food_delivered`: `r = 0.986`
- swarm size vs `delivery_conversion`: `r = 0.842`
- swarm size vs `exploration_coverage`: `r = 0.959`

Effect-size interpretation:

- `food_delivered` increases by about `93x` from `1` to `30` agents
- `exploration_coverage` increases by about `9.0x`

Interpretation:

- scaling is very strong descriptively
- the pattern is close to monotonic after the small `2` to `3` agent wobble
- this is some of the cleanest descriptive evidence in the bundle

Caution:

- the exported broad test table does not appear to include formal inferential tests for each scaling step
- so this should be framed as strong descriptive scaling evidence

## 6. Robustness Under Harder Conditions

Control condition:

- `control_final_stage`
  - `food_delivered = 1.25`

Harder conditions:

- `more_obstacles`
  - `food_delivered = 0.65`
- `sensor_noise`
  - `food_delivered = 1.05`
- `failed_agents_2`
  - `food_delivered = 0.70`

Relative to control:

- more obstacles:
  - absolute drop `0.60`
  - about `48%` lower
- sensor noise:
  - absolute drop `0.20`
  - about `16%` lower
- two failed agents:
  - absolute drop `0.55`
  - about `44%` lower

Interpretation:

- sensor noise is comparatively well tolerated
- more obstacles and failed agents are much more damaging
- performance degrades but does not collapse completely

Caution:

- I did not find broad inferential test rows for these robustness comparisons in the exported test table
- these should be presented as descriptive robustness degradation unless separate tests are added

## 7. Killer Repeated-Source Stigmergy Test

This is the strongest mechanism-specific evidence in the bundle.

### At 6 agents

Pheromone-trained, evaluated with pheromone:

- `food_delivered = 1.32`
- `late_deliveries = 0.90`
- `post_discovery_deliveries = 1.32`
- `pickup_to_delivery_latency = 35.48`

No-pheromone-trained control, evaluated without pheromone:

- `food_delivered = 0.00`
- `late_deliveries = 0.00`
- `post_discovery_deliveries = 0.00`

Statistical tests:

- `food_delivered`
  - paired t-test `p = 0.006534`
  - Wilcoxon `p = 0.003639`
  - Cohen's `d = 0.568`
- `late_deliveries`
  - paired t-test `p = 0.015349`
  - Wilcoxon `p = 0.031788`
  - Cohen's `d = 0.502`

### At 30 agents

Pheromone-trained, evaluated with pheromone:

- `food_delivered = 2.66`
- `late_deliveries = 1.12`
- `post_discovery_deliveries = 2.66`
- `pickup_to_delivery_latency = 100.2`

No-pheromone-trained control, evaluated without pheromone:

- `food_delivered = 0.00`
- `late_deliveries = 0.00`
- `post_discovery_deliveries = 0.00`

Statistical tests:

- `food_delivered`
  - paired t-test `p = 0.000847`
  - Wilcoxon `p = 9.96e-06`
  - Cohen's `d = 0.711`
- `late_deliveries`
  - paired t-test `p = 0.018681`
  - Wilcoxon `p = 0.011089`
  - Cohen's `d = 0.487`

Interpretation:

- this is the strongest scientific result in the package
- the no-pheromone-trained control essentially fails the repeated-source route-reuse task
- the pheromone-trained swarm retains a significant advantage even at `30` agents
- agreement between paired t-tests and Wilcoxon strengthens the result

## 8. Eval-Time Pheromone Removal Within the Pheromone-Trained Policy

This is a weaker but still informative comparison.

### At 6 agents

Pheromone-trained policy:

- with pheromone:
  - `food_delivered = 1.32`
  - `late_deliveries = 0.90`
- without pheromone:
  - `food_delivered = 0.96`
  - `late_deliveries = 0.42`

Change:

- food delivered drop:
  - `0.36`
  - about `27.3%`
- late deliveries drop:
  - `0.48`
  - about `53.3%`

Late-delivery test:

- paired t-test `p = 0.047414`
- Wilcoxon `p = 0.084207`

### At 30 agents

Pheromone-trained policy:

- with pheromone:
  - `food_delivered = 2.66`
  - `late_deliveries = 1.12`
- without pheromone:
  - `food_delivered = 2.12`
  - `late_deliveries = 0.84`

Change:

- food delivered drop:
  - `0.54`
  - about `20.3%`
- late deliveries drop:
  - `0.28`
  - about `25%`

Late-delivery test:

- paired t-test `p = 0.226574`
- Wilcoxon `p = 0.108243`

Interpretation:

- removing pheromone at evaluation hurts the trained policy
- the effect is visible, especially for later deliveries
- but this comparison is weaker and less stable than the matched-training stigmergy comparison

So the cleaner claim is still:

- training with stigmergy produces a better policy for repeated-source coordination
- not necessarily that removing pheromone at eval always creates a huge collapse

That is the right scientific restraint here.

## 9. Statistical Quality and Limitations

What is strong:

- current curriculum is better than weaker older training
- clear effect sizes for curriculum and baseline tests
- paired tests in the killer family
- agreement between t-tests and Wilcoxon in the strongest stigmergy results
- descriptive scaling pattern is very consistent

What is weaker:

- broad robustness comparisons appear descriptive rather than fully inferential in the exported test table
- generic pheromone ablation is weak and should not be oversold
- some means have large variance, especially around discovery timing and sparse-delivery conditions
- `latest` and `best_greedy_eval` checkpoint distinctions matter elsewhere in the repo, but this bundle already fixes the evaluated checkpoints inside the campaign definitions

## 10. Bottom-Line Interpretation

If I compress the whole bundle into one technically defensible conclusion:

- the bundle strongly supports that the current curriculum-trained recurrent MAPPO swarm is better than weaker training and better than simple baselines
- it also shows strong descriptive scaling up to `30` agents
- the broad generic pheromone ablation is weak by itself
- the strongest evidence for stigmergy comes from the focused repeated-source killer experiment, where pheromone-trained swarms significantly outperform no-pheromone-trained controls at both `6` and `30` agents
