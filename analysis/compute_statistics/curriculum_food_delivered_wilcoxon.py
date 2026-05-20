from __future__ import annotations

import argparse
from pathlib import Path

from scipy.stats import wilcoxon

from common import DEFAULT_RAW_EXPORT, format_stat, load_rows, mean

FAMILY = "curriculum_vs_weaker_training"
CURRENT = "current_curriculum_final"
OLDER = "older_weaker_curriculum_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the matched-seed Wilcoxon signed-rank test for the curriculum "
            "food_delivered comparison used by curriculum_food_delivered_delta_by_trial_final_4.png."
        )
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_RAW_EXPORT, help="Path to all_episode_results.csv")
    return parser.parse_args()


def build_matched_pairs(rows: list[dict[str, str]]) -> tuple[list[float], list[float], list[float]]:
    by_seed: dict[int, dict[str, float]] = {}
    for row in rows:
        if row.get("family") != FAMILY or row.get("condition") not in {CURRENT, OLDER}:
            continue
        by_seed.setdefault(int(row["seed"]), {})[row["condition"]] = float(row["food_delivered"])

    newer: list[float] = []
    older: list[float] = []
    deltas: list[float] = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if CURRENT in pair and OLDER in pair:
            newer.append(pair[CURRENT])
            older.append(pair[OLDER])
            deltas.append(pair[CURRENT] - pair[OLDER])
    return newer, older, deltas


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    newer, older, deltas = build_matched_pairs(rows)
    result = wilcoxon(newer, older)

    positive = sum(1 for delta in deltas if delta > 0)
    zero = sum(1 for delta in deltas if delta == 0)
    negative = sum(1 for delta in deltas if delta < 0)

    print("Wilcoxon signed-rank test for curriculum_food_delivered_delta_by_trial_final_4.png")
    print(f"Input CSV: {args.csv}")
    print("Matched pairs: current_curriculum_final vs older_weaker_curriculum_final")
    print(f"  n_pairs={len(deltas)}")
    print(f"  mean_current={format_stat(mean(newer))}")
    print(f"  mean_older={format_stat(mean(older))}")
    print(f"  mean_delta={format_stat(mean(deltas))}")
    print(f"  positive_deltas={positive} zero_deltas={zero} negative_deltas={negative}")
    print(f"  statistic={format_stat(float(result.statistic))} p={format_stat(float(result.pvalue))}")


if __name__ == "__main__":
    main()
