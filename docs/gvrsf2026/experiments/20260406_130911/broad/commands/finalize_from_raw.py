from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TIMESTAMP = "20260406_130911"
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / TIMESTAMP
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
RAW_CSV = BASE_DIR / "raw_exports" / "all_episode_results.csv"
META_DIR = BASE_DIR / "metadata"


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def std_or_zero(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * std_or_zero(values) / math.sqrt(len(values))


def cohens_d(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_num = (len(a) - 1) * var_a + (len(b) - 1) * var_b
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return None
    pooled = math.sqrt(max(pooled_num / pooled_den, 1e-12))
    return (mean_a - mean_b) / pooled


def welch_ttest(a: list[float], b: list[float]) -> dict[str, float | None]:
    try:
        from scipy.stats import ttest_ind
    except Exception:
        return {"t_stat": None, "p_value": None}
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": None, "p_value": None}
    stat = ttest_ind(a, b, equal_var=False)
    return {"t_stat": float(stat.statistic), "p_value": float(stat.pvalue)}


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    metrics = [
        "mean_episode_reward",
        "food_picked_up",
        "food_delivered",
        "delivery_conversion",
        "exploration_coverage",
        "pheromone_usage",
        "episode_length",
        "time_to_first_discovery",
        "carrying_stall_fraction",
        "carrying_low_progress_fraction",
        "non_carrying_nest_loiter_fraction",
        "non_carrying_nest_crowding_fraction",
        "non_carrying_explore_active_fraction",
        "non_carrying_force_explore_fraction",
        "non_carrying_random_explore_fraction",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    summaries = []
    for key, group_rows in grouped.items():
        base = {k: v for k, v in zip(group_keys, key)}
        base["n"] = len(group_rows)
        for metric in metrics:
            values = [float(r[metric]) for r in group_rows]
            base[f"{metric}_mean"] = mean_or_zero(values)
            base[f"{metric}_std"] = std_or_zero(values)
            base[f"{metric}_ci95"] = ci95(values)
        summaries.append(base)
    return summaries


def plot_grouped_bars(summary_rows: list[dict[str, Any]], family: str, metric: str, out_name: str) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in summary_rows if r["family"] == family]
    labels = [r["condition"] for r in rows]
    means = [r[f"{metric}_mean"] for r in rows]
    cis = [r[f"{metric}_ci95"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, capsize=4, color="#4472C4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{family.replace('_', ' ').title()}: {metric.replace('_', ' ').title()}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name)
    plt.close(fig)


def plot_scaling(summary_rows: list[dict[str, Any]], metric: str, out_name: str) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in summary_rows if r["family"] == "swarm_size_scaling"]
    rows.sort(key=lambda r: int(r["n_agents"]))
    x = [int(r["n_agents"]) for r in rows]
    y = [r[f"{metric}_mean"] for r in rows]
    ci = [r[f"{metric}_ci95"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(x, y, yerr=ci, marker="o", linewidth=2, color="#2F5597", capsize=4)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Swarm-Size Scaling: {metric.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / out_name)
    plt.close(fig)


def load_rows() -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    int_keys = {"episode", "seed", "n_agents", "width", "height", "n_obstacles", "max_steps"}
    bool_keys = {"pheromone_enabled"}
    str_keys = {"family", "condition", "policy_kind", "checkpoint_dir"}
    with RAW_CSV.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in str_keys:
                    parsed[key] = value
                elif key in bool_keys:
                    parsed[key] = value == "True"
                elif key in int_keys:
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            dedupe_key = (
                parsed["family"],
                parsed["condition"],
                parsed["episode"],
                parsed["seed"],
            )
            rows_by_key[dedupe_key] = parsed
    return list(rows_by_key.values())


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("No raw rows found")

    summary_rows = summarize_rows(rows, ["family", "condition", "n_agents", "width", "height", "n_obstacles", "pheromone_enabled"])
    save_csv(TABLES_DIR / "condition_summary.csv", summary_rows)

    by_family = summarize_rows(rows, ["family", "condition"])
    save_csv(TABLES_DIR / "family_summary.csv", by_family)

    stats_rows = []
    pairs = [
        ("curriculum_vs_weaker_training", "current_curriculum_final", "older_weaker_curriculum_final"),
        ("pheromone_ablation", "trained_with_pheromone_eval_with_pheromone", "trained_with_pheromone_eval_without_pheromone"),
        ("baseline_comparison", "current_mappo", "rule_based"),
        ("baseline_comparison", "current_mappo", "random"),
    ]
    for family_name, a_label, b_label in pairs:
        a_vals = [r["food_delivered"] for r in rows if r["family"] == family_name and r["condition"] == a_label]
        b_vals = [r["food_delivered"] for r in rows if r["family"] == family_name and r["condition"] == b_label]
        stats_rows.append(
            {
                "family": family_name,
                "condition_a": a_label,
                "condition_b": b_label,
                "metric": "food_delivered",
                "cohens_d": cohens_d(a_vals, b_vals),
                **welch_ttest(a_vals, b_vals),
            }
        )
    save_csv(TABLES_DIR / "statistical_tests.csv", stats_rows)

    plot_grouped_bars(by_family, "curriculum_vs_weaker_training", "food_delivered", "curriculum_food_delivered.png")
    plot_grouped_bars(by_family, "curriculum_vs_weaker_training", "delivery_conversion", "curriculum_delivery_conversion.png")
    plot_grouped_bars(by_family, "pheromone_ablation", "food_delivered", "pheromone_food_delivered.png")
    plot_grouped_bars(by_family, "baseline_comparison", "food_delivered", "baseline_food_delivered.png")
    plot_grouped_bars(by_family, "robustness_harder_environments", "food_delivered", "robustness_food_delivered.png")
    plot_grouped_bars(by_family, "robustness_harder_environments", "exploration_coverage", "robustness_exploration_coverage.png")
    plot_scaling(summary_rows, "food_delivered", "scaling_food_delivered.png")
    plot_scaling(summary_rows, "delivery_conversion", "scaling_delivery_conversion.png")
    plot_scaling(summary_rows, "exploration_coverage", "scaling_exploration_coverage.png")

    (META_DIR / "run_complete.json").write_text(
        json.dumps(
            {
                "timestamp": TIMESTAMP,
                "status": "completed_from_raw",
                "raw_episode_rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
