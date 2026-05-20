from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from scipy.stats import f, wilcoxon


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_EXPORT = ROOT / "docs/gvrsf2026/experiments/20260406_102744/killer_scaling/raw_exports/all_episode_results.csv"

WITH_PHEROMONE = "trained_with_pheromone__eval_with_pheromone"
WITHOUT_PHEROMONE = "trained_without_pheromone__eval_without_pheromone"
PLOT_CONDITIONS = [WITH_PHEROMONE, WITHOUT_PHEROMONE]
PLOT_LABELS = {
    WITH_PHEROMONE: "with_pheromone",
    WITHOUT_PHEROMONE: "without_pheromone",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the full-data two-way ANOVA for the stigmergy scaling plot "
            "(food_delivered by swarm size and pheromone condition)."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_RAW_EXPORT,
        help="Path to killer_scaling all_episode_results.csv",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def two_way_anova_food_delivered(rows: list[dict[str, str]]) -> None:
    subset = [row for row in rows if row["condition"] in PLOT_CONDITIONS]
    sizes = sorted({int(row["n_agents"]) for row in subset})

    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    by_size: dict[int, list[float]] = defaultdict(list)
    by_condition: dict[str, list[float]] = defaultdict(list)
    all_values: list[float] = []

    for row in subset:
        size = int(row["n_agents"])
        condition = row["condition"]
        value = float(row["food_delivered"])
        cells[(size, condition)].append(value)
        by_size[size].append(value)
        by_condition[condition].append(value)
        all_values.append(value)

    grand_mean = mean(all_values)

    ss_size = sum(len(by_size[size]) * (mean(by_size[size]) - grand_mean) ** 2 for size in sizes)
    ss_condition = sum(
        len(by_condition[condition]) * (mean(by_condition[condition]) - grand_mean) ** 2
        for condition in PLOT_CONDITIONS
    )
    ss_interaction = 0.0
    for size in sizes:
        size_mean = mean(by_size[size])
        for condition in PLOT_CONDITIONS:
            condition_mean = mean(by_condition[condition])
            cell_values = cells[(size, condition)]
            cell_mean = mean(cell_values)
            ss_interaction += len(cell_values) * (
                cell_mean - size_mean - condition_mean + grand_mean
            ) ** 2

    ss_error = 0.0
    for (size, condition), values in cells.items():
        cell_mean = mean(values)
        ss_error += sum((value - cell_mean) ** 2 for value in values)

    ss_total = sum((value - grand_mean) ** 2 for value in all_values)

    a_levels = len(sizes)
    b_levels = len(PLOT_CONDITIONS)
    df_size = a_levels - 1
    df_condition = b_levels - 1
    df_interaction = (a_levels - 1) * (b_levels - 1)
    df_error = len(all_values) - a_levels * b_levels

    ms_size = ss_size / df_size
    ms_condition = ss_condition / df_condition
    ms_interaction = ss_interaction / df_interaction
    ms_error = ss_error / df_error

    effects = [
        ("swarm_size", ss_size, df_size, ms_size),
        ("pheromone_condition", ss_condition, df_condition, ms_condition),
        ("interaction", ss_interaction, df_interaction, ms_interaction),
    ]

    print("Two-way ANOVA for stigmergy_scaling_final_2.png")
    print(f"Input CSV: {DEFAULT_RAW_EXPORT}")
    print("Dependent variable: food_delivered")
    print("Factors: swarm_size x pheromone_condition")
    print()
    print("Cell means")
    for size in sizes:
        for condition in PLOT_CONDITIONS:
            values = cells[(size, condition)]
            print(
                f"  size={size:>2} "
                f"condition={PLOT_LABELS[condition]:<18} "
                f"n={len(values):>2} mean={mean(values):.6f}"
            )
    print()

    print("ANOVA table")
    print(f"  total_ss={ss_total:.6f}")
    print(f"  error_ss={ss_error:.6f} df={df_error} ms={ms_error:.6f}")
    for name, ss_value, df_value, ms_value in effects:
        f_value = ms_value / ms_error
        p_value = float(1.0 - f.cdf(f_value, df_value, df_error))
        eta_sq = ss_value / ss_total if ss_total else float("nan")
        partial_eta_sq = ss_value / (ss_value + ss_error) if (ss_value + ss_error) else float("nan")
        print(
            f"  {name}: SS={ss_value:.6f} df={df_value} MS={ms_value:.6f} "
            f"F={f_value:.6f} p={p_value:.6g} eta_sq={eta_sq:.6f} partial_eta_sq={partial_eta_sq:.6f}"
        )


def wilcoxon_consistency_30_agents(rows: list[dict[str, str]], n_agents: int = 30) -> None:
    subset = [row for row in rows if row["condition"] in PLOT_CONDITIONS and int(row["n_agents"]) == n_agents]
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for row in subset:
        by_seed[int(row["seed"])][row["condition"]] = float(row["food_delivered"])

    paired_with: list[float] = []
    paired_without: list[float] = []
    deltas: list[float] = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if WITH_PHEROMONE in pair and WITHOUT_PHEROMONE in pair:
            paired_with.append(pair[WITH_PHEROMONE])
            paired_without.append(pair[WITHOUT_PHEROMONE])
            deltas.append(pair[WITH_PHEROMONE] - pair[WITHOUT_PHEROMONE])

    result = wilcoxon(paired_with, paired_without)
    positive_count = sum(1 for delta in deltas if delta > 0)
    zero_count = sum(1 for delta in deltas if delta == 0)
    negative_count = sum(1 for delta in deltas if delta < 0)

    print()
    print(f"Wilcoxon signed-rank test for stigmergy_consistency_final.png ({n_agents} agents)")
    print("Matched pairs: with_pheromone vs without_pheromone")
    print(f"  n_pairs={len(deltas)}")
    print(f"  mean_delta={mean(deltas):.6f}")
    print(f"  positive_deltas={positive_count} zero_deltas={zero_count} negative_deltas={negative_count}")
    print(f"  statistic={float(result.statistic):.6f} p={float(result.pvalue):.6g}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv)
    two_way_anova_food_delivered(rows)
    wilcoxon_consistency_30_agents(rows, 30)


if __name__ == "__main__":
    main()
