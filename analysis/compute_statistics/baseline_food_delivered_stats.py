from __future__ import annotations

import argparse
from pathlib import Path

from scipy import stats

from common import (
    DEFAULT_RAW_EXPORT,
    cohens_d,
    filter_family,
    format_stat,
    load_rows,
    metric_values,
    print_group_summary,
)

FAMILY = "baseline_comparison"
ORDER = ["current_mappo", "rule_based", "random"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute omnibus and pairwise food_delivered statistics for the baseline comparison family."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_EXPORT, help="Path to all_episode_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    groups = metric_values(filter_family(rows, FAMILY, ORDER), "food_delivered")

    print("Baseline food_delivered group summary")
    print_group_summary(groups)
    print()

    f_stat, p_value = stats.f_oneway(*(groups[label] for label in ORDER))
    h_stat, h_p = stats.kruskal(*(groups[label] for label in ORDER))
    print(
        "One-way ANOVA "
        f"F(2, {sum(len(v) for v in groups.values()) - len(groups)})={format_stat(float(f_stat))} "
        f"p={format_stat(float(p_value))}"
    )
    print(f"Kruskal-Wallis H(2)={format_stat(float(h_stat))} p={format_stat(float(h_p))}")
    print()

    pairwise = [
        ("current_mappo", "rule_based"),
        ("current_mappo", "random"),
        ("rule_based", "random"),
    ]
    print("Welch t-tests")
    for a, b in pairwise:
        result = stats.ttest_ind(groups[a], groups[b], equal_var=False)
        d_value = cohens_d(groups[a], groups[b])
        print(
            f"  {a} vs {b}: "
            f"t={format_stat(float(result.statistic))} "
            f"p={format_stat(float(result.pvalue))} "
            f"d={format_stat(d_value)}"
        )


if __name__ == "__main__":
    main()
