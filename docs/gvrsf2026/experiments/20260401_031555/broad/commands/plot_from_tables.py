from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[6]
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / "20260401_031555" / "broad"
FIGURES_DIR = BASE_DIR / "figures"
SUMMARY_CSV = BASE_DIR / "tables" / "family_summary.csv"


def load_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def bar_plot(rows, family, metric, filename):
    subset = [r for r in rows if r["family"] == family]
    labels = [r["condition"] for r in subset]
    means = [float(r[f"{metric}_mean"]) for r in subset]
    cis = [float(r[f"{metric}_ci95"]) for r in subset]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=cis, color="#4472C4", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"{family.replace('_', ' ').title()}: {metric.replace('_', ' ').title()}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def line_plot(rows, metric, filename):
    subset = [r for r in rows if r["family"] == "swarm_size_scaling"]
    subset.sort(key=lambda r: int(r["condition"].split("_")[0]))
    xs = [int(r["condition"].split("_")[0]) for r in subset]
    ys = [float(r[f"{metric}_mean"]) for r in subset]
    cis = [float(r[f"{metric}_ci95"]) for r in subset]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(xs, ys, yerr=cis, marker="o", linewidth=2, color="#2F5597", capsize=4)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Swarm-Size Scaling: {metric.replace('_', ' ').title()}")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    bar_plot(rows, "curriculum_vs_weaker_training", "food_delivered", "curriculum_food_delivered.png")
    bar_plot(rows, "curriculum_vs_weaker_training", "delivery_conversion", "curriculum_delivery_conversion.png")
    bar_plot(rows, "pheromone_ablation", "food_delivered", "pheromone_food_delivered.png")
    bar_plot(rows, "baseline_comparison", "food_delivered", "baseline_food_delivered.png")
    bar_plot(rows, "robustness_harder_environments", "food_delivered", "robustness_food_delivered.png")
    bar_plot(rows, "robustness_harder_environments", "exploration_coverage", "robustness_exploration_coverage.png")
    line_plot(rows, "food_delivered", "scaling_food_delivered.png")
    line_plot(rows, "delivery_conversion", "scaling_delivery_conversion.png")
    line_plot(rows, "exploration_coverage", "scaling_exploration_coverage.png")


if __name__ == "__main__":
    main()
