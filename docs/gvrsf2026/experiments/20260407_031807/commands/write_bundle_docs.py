from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / "20260407_031807"
SUMMARY_CSV = BASE_DIR / "tables" / "condition_summary.csv"
STATS_CSV = BASE_DIR / "tables" / "statistical_tests.csv"


def load_csv(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_for(rows, condition, n_agents):
    for row in rows:
        if row["condition"] == condition and int(row["n_agents"]) == n_agents:
            return row
    raise KeyError((condition, n_agents))


def stat_for(rows, n_agents, condition_b, metric):
    for row in rows:
        if (
            int(row["n_agents"]) == n_agents
            and row["condition_a"] == "trained_with_pheromone__eval_with_pheromone"
            and row["condition_b"] == condition_b
            and row["metric"] == metric
        ):
            return row
    raise KeyError((n_agents, condition_b, metric))


def promising(summary_rows, stat_rows) -> bool:
    with_on = row_for(summary_rows, "trained_with_pheromone__eval_with_pheromone", 10)
    eval_off = row_for(summary_rows, "trained_with_pheromone__eval_without_pheromone", 10)
    no_pher = row_for(summary_rows, "trained_without_pheromone__eval_without_pheromone", 10)
    fd_adv = float(with_on["food_delivered_mean"]) > float(eval_off["food_delivered_mean"]) and float(with_on["food_delivered_mean"]) > float(no_pher["food_delivered_mean"])
    pd_adv = float(with_on["post_discovery_deliveries_mean"]) > float(no_pher["post_discovery_deliveries_mean"])
    food_stat = stat_for(stat_rows, 10, "trained_without_pheromone__eval_without_pheromone", "food_delivered")
    post_stat = stat_for(stat_rows, 10, "trained_without_pheromone__eval_without_pheromone", "post_discovery_deliveries")

    def promising_stat(row):
        try:
            p = float(row["paired_t_p_value"])
        except Exception:
            p = 1.0
        try:
            d = float(row["cohens_d"])
        except Exception:
            d = 0.0
        return p < 0.10 or d >= 0.3

    return fd_adv and pd_adv and (promising_stat(food_stat) or promising_stat(post_stat))


def main() -> None:
    summary_rows = load_csv(SUMMARY_CSV)
    stat_rows = load_csv(STATS_CSV)
    is_promising = promising(summary_rows, stat_rows)

    with_on_10 = row_for(summary_rows, "trained_with_pheromone__eval_with_pheromone", 10)
    eval_off_10 = row_for(summary_rows, "trained_with_pheromone__eval_without_pheromone", 10)
    no_pher_10 = row_for(summary_rows, "trained_without_pheromone__eval_without_pheromone", 10)
    food_stat_10 = stat_for(stat_rows, 10, "trained_without_pheromone__eval_without_pheromone", "food_delivered")
    post_stat_10 = stat_for(stat_rows, 10, "trained_without_pheromone__eval_without_pheromone", "post_discovery_deliveries")

    verdict = "promising" if is_promising else "not promising"
    recommendation = "extend to max_20" if is_promising else "stop at max_10 and do not extend"

    report = f"""# Chokepoint Route-Reuse Report: Stage max_10

## Bundle

**Bundle folder:** `docs/gvrsf2026/experiments/20260407_031807/`

## Abstract

This bundle tests the hypothesis that pheromone-based stigmergy becomes more valuable when the swarm must repeatedly traverse a narrow rediscovery-expensive route. The study uses same-curriculum matched checkpoints, one repeated source, a fixed two-wall chokepoint layout with offset gaps, a longer horizon, and swarm sizes `1`, `3`, `6`, and `10` with `20` paired seeds per condition.

At `10` agents, the pheromone-enabled condition reached mean `food_delivered = {float(with_on_10['food_delivered_mean']):.2f}` versus `{float(eval_off_10['food_delivered_mean']):.2f}` for trained-with-pheromone but eval-without-pheromone and `{float(no_pher_10['food_delivered_mean']):.2f}` for the true no-pheromone matched control. The paired test against the true no-pheromone control gave `p = {float(food_stat_10['paired_t_p_value']):.3f}` for `food_delivered` and `p = {float(post_stat_10['paired_t_p_value']):.3f}` for `post_discovery_deliveries`. By the predeclared extension rule, this stage is **{verdict}**, so the recommendation is: **{recommendation}**.

## Design

- same-curriculum matched checkpoints
- one repeated source
- fixed two-wall chokepoint world with offset gaps
- `food_source_capacity = 18`
- `max_steps = 1600`
- swarm sizes: `1`, `3`, `6`, `10`
- conditions:
  - trained with pheromone, eval with pheromone
  - trained with pheromone, eval without pheromone
  - trained without pheromone, eval without pheromone

## Figures

![Food Delivered by Swarm Size](./figures/food_delivered_by_swarm_size.png)

![Late Deliveries by Swarm Size](./figures/late_deliveries_by_swarm_size.png)

![Post-Discovery Deliveries by Swarm Size](./figures/post_discovery_deliveries_by_swarm_size.png)

![Pickup-to-Delivery Latency by Swarm Size](./figures/pickup_to_delivery_latency_by_swarm_size.png)

## `10`-Agent Checkpoint For Extension Decision

- with pheromone: `food_delivered = {float(with_on_10['food_delivered_mean']):.2f}`
- trained with pheromone, eval without pheromone: `food_delivered = {float(eval_off_10['food_delivered_mean']):.2f}`
- true no-pheromone control: `food_delivered = {float(no_pher_10['food_delivered_mean']):.2f}`
- paired `food_delivered` vs true no-pheromone:
  - `p = {float(food_stat_10['paired_t_p_value']):.3f}`
  - `cohens_d = {float(food_stat_10['cohens_d']):.3f}`
- paired `post_discovery_deliveries` vs true no-pheromone:
  - `p = {float(post_stat_10['paired_t_p_value']):.3f}`
  - `cohens_d = {float(post_stat_10['cohens_d']):.3f}`

## Conclusion

This stage is **{verdict}** under the predeclared extension rule. The next action is to **{recommendation}**.
"""

    paper = f"""# Chokepoint Route-Reuse Stigmergy Study

## Stage max_10

**Experiment bundle folder:** `docs/gvrsf2026/experiments/20260407_031807/`
**Primary report:** [report.md](./report.md)

## Abstract

This paper evaluates whether pheromone-based coordination becomes more useful when the swarm must repeatedly traverse a fixed chokepoint route. Using same-curriculum matched checkpoints and `20` paired seeds per condition, the study compares three conditions across swarm sizes `1`, `3`, `6`, and `10`.

At `10` agents, the pheromone-enabled condition reached mean `food_delivered = {float(with_on_10['food_delivered_mean']):.2f}` compared with `{float(eval_off_10['food_delivered_mean']):.2f}` for trained-with-pheromone but evaluated without pheromone and `{float(no_pher_10['food_delivered_mean']):.2f}` for the true no-pheromone matched control. The paired comparison against the true no-pheromone control gave `p = {float(food_stat_10['paired_t_p_value']):.3f}` for `food_delivered` and `p = {float(post_stat_10['paired_t_p_value']):.3f}` for `post_discovery_deliveries`. By the predeclared rule, the stage is **{verdict}**.

## Main Comparison

![Food Delivered by Swarm Size](./figures/food_delivered_by_swarm_size.png)

![Post-Discovery Deliveries by Swarm Size](./figures/post_discovery_deliveries_by_swarm_size.png)

![Late Deliveries by Swarm Size](./figures/late_deliveries_by_swarm_size.png)

## Extension Decision

The predeclared recommendation from this stage is to **{recommendation}**.
"""

    (BASE_DIR / "report.md").write_text(report, encoding="utf-8")
    (BASE_DIR / "paper.md").write_text(paper, encoding="utf-8")


if __name__ == "__main__":
    main()
