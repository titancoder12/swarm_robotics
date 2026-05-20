from __future__ import annotations

import argparse
from pathlib import Path

from scipy import stats

from common import DEFAULT_RAW_EXPORT, filter_family, format_stat, load_rows, mean

FAMILY = "swarm_size_scaling"
ORDER = ["1_agents", "2_agents", "3_agents", "4_agents", "6_agents", "10_agents", "15_agents", "20_agents", "30_agents"]
SIZE_LABELS = [1, 2, 3, 4, 6, 10, 15, 20, 30]
METRICS = ["food_delivered", "delivery_conversion", "exploration_coverage"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute raw-data and condition-mean Pearson/Spearman correlations for swarm-size scaling metrics."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_EXPORT, help="Path to all_episode_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    rows_by_condition = filter_family(rows, FAMILY, ORDER)

    raw_x = [int(row["n_agents"]) for row in rows if row.get("family") == FAMILY]

    for metric in METRICS:
        raw_y = [float(row[metric]) for row in rows if row.get("family") == FAMILY]
        mean_y = [mean([float(row[metric]) for row in rows_by_condition[condition]]) for condition in ORDER]

        raw_spearman = stats.spearmanr(raw_x, raw_y)
        raw_pearson = stats.pearsonr(raw_x, raw_y)
        mean_spearman = stats.spearmanr(SIZE_LABELS, mean_y)
        mean_pearson = stats.pearsonr(SIZE_LABELS, mean_y)

        print(f"Metric: {metric}")
        print("  Raw per-episode correlations")
        print(
            f"    Spearman rho={format_stat(float(raw_spearman.statistic))} "
            f"p={format_stat(float(raw_spearman.pvalue))}"
        )
        print(
            f"    Pearson r={format_stat(float(raw_pearson.statistic))} "
            f"p={format_stat(float(raw_pearson.pvalue))}"
        )
        print("  Condition-mean correlations")
        print(
            f"    Spearman rho={format_stat(float(mean_spearman.statistic))} "
            f"p={format_stat(float(mean_spearman.pvalue))}"
        )
        print(
            f"    Pearson r={format_stat(float(mean_pearson.statistic))} "
            f"p={format_stat(float(mean_pearson.pvalue))}"
        )
        print()


if __name__ == "__main__":
    main()
