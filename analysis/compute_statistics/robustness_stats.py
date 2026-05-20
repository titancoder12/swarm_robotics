from __future__ import annotations

import argparse
from pathlib import Path

from scipy import stats

from common import (
    DEFAULT_RAW_EXPORT,
    cohens_d,
    eta_squared,
    filter_family,
    format_stat,
    load_rows,
    metric_values,
    print_group_summary,
)

FAMILY = "robustness_harder_environments"
ORDER = ["control_final_stage", "sensor_noise", "failed_agents_2", "more_obstacles"]
BASELINE = "control_final_stage"
COMPARE = ["sensor_noise", "failed_agents_2", "more_obstacles"]
METRICS = ["food_delivered", "exploration_coverage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute robustness omnibus ANOVA results and baseline-vs-stressor Welch t-tests."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_EXPORT, help="Path to all_episode_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    rows_by_condition = filter_family(rows, FAMILY, ORDER)

    for metric in METRICS:
        groups = metric_values(rows_by_condition, metric)
        print(f"Metric: {metric}")
        print_group_summary(groups)
        print()

        f_stat, p_value = stats.f_oneway(*(groups[label] for label in ORDER))
        print(
            "One-way ANOVA "
            f"F({len(ORDER) - 1}, {sum(len(v) for v in groups.values()) - len(groups)})="
            f"{format_stat(float(f_stat))} "
            f"p={format_stat(float(p_value))} "
            f"eta_sq={format_stat(eta_squared(groups))}"
        )
        print("Welch t-tests vs baseline")
        for other in COMPARE:
            result = stats.ttest_ind(groups[BASELINE], groups[other], equal_var=False)
            d_value = cohens_d(groups[BASELINE], groups[other])
            print(
                f"  {BASELINE} vs {other}: "
                f"t={format_stat(float(result.statistic))} "
                f"p={format_stat(float(result.pvalue))} "
                f"d={format_stat(d_value)}"
            )
        print()


if __name__ == "__main__":
    main()
