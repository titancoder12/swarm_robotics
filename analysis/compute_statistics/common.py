from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

DEFAULT_RAW_EXPORT = (
    Path(__file__).resolve().parents[2]
    / "docs/gvrsf2026/experiments/20260406_102744/raw_exports/all_episode_results.csv"
)


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_family(
    rows: list[dict[str, str]],
    family: str,
    conditions: list[str] | tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    filtered = {condition: [] for condition in conditions}
    for row in rows:
        if row.get("family") == family and row.get("condition") in filtered:
            filtered[row["condition"]].append(row)
    return filtered


def metric_values(
    rows_by_condition: dict[str, list[dict[str, str]]],
    metric: str,
) -> dict[str, list[float]]:
    return {
        condition: [float(row[metric]) for row in condition_rows]
        for condition, condition_rows in rows_by_condition.items()
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def sem(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return sample_sd(values) / math.sqrt(len(values))


def cohens_d(a: list[float], b: list[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    var_a = statistics.variance(a)
    var_b = statistics.variance(b)
    pooled = math.sqrt((((len(a) - 1) * var_a) + ((len(b) - 1) * var_b)) / (len(a) + len(b) - 2))
    if pooled == 0.0:
        return float("nan")
    return (mean(a) - mean(b)) / pooled


def eta_squared(groups: dict[str, list[float]]) -> float:
    all_values = [value for values in groups.values() for value in values]
    grand_mean = mean(all_values)
    ss_between = sum(len(values) * (mean(values) - grand_mean) ** 2 for values in groups.values())
    ss_total = sum((value - grand_mean) ** 2 for value in all_values)
    if ss_total == 0.0:
        return float("nan")
    return ss_between / ss_total


def print_group_summary(groups: dict[str, list[float]]) -> None:
    for label, values in groups.items():
        print(
            f"  {label}: "
            f"n={len(values)} mean={mean(values):.6f} sd={sample_sd(values):.6f}"
        )


def format_stat(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    return str(value)
