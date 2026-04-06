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
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / "20260406_132534" / "killer_scaling"
FIGURES_DIR = BASE_DIR / "figures"
SUMMARY_CSV = BASE_DIR / "tables" / "condition_summary.csv"


def load_rows():
    with SUMMARY_CSV.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def grouped_metric_plot(rows, metric, filename, title):
    order = [
        "trained_with_pheromone__eval_with_pheromone",
        "trained_with_pheromone__eval_without_pheromone",
        "trained_without_pheromone__eval_without_pheromone",
    ]
    labels = {
        "trained_with_pheromone__eval_with_pheromone": "Trained with pheromone, eval with pheromone",
        "trained_with_pheromone__eval_without_pheromone": "Trained with pheromone, eval without pheromone",
        "trained_without_pheromone__eval_without_pheromone": "Trained without pheromone",
    }
    swarm_sizes = sorted({int(r["n_agents"]) for r in rows})
    fig, ax = plt.subplots(figsize=(10, 5.4))
    colors = ["#2F5597", "#70AD47", "#C0504D"]
    markers = ["o", "s", "^"]
    for idx, condition in enumerate(order):
        subset = [r for r in rows if r["condition"] == condition]
        subset.sort(key=lambda r: int(r["n_agents"]))
        means = [float(r[f"{metric}_mean"]) for r in subset]
        cis = [float(r[f"{metric}_ci95"]) for r in subset]
        ax.errorbar(
            swarm_sizes,
            means,
            yerr=cis,
            marker=markers[idx],
            linewidth=2.2,
            markersize=6,
            capsize=4,
            label=labels[condition],
            color=colors[idx],
        )
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xticks(swarm_sizes)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    grouped_metric_plot(rows, "food_delivered", "food_delivered_by_swarm_size.png", "Stronger Pheromone Study: Deliveries by Swarm Size")
    grouped_metric_plot(rows, "delivery_conversion", "delivery_conversion_by_swarm_size.png", "Stronger Pheromone Study: Delivery Conversion by Swarm Size")
    grouped_metric_plot(rows, "exploration_coverage", "exploration_coverage_by_swarm_size.png", "Stronger Pheromone Study: Exploration Coverage by Swarm Size")
    grouped_metric_plot(rows, "late_deliveries", "late_deliveries_by_swarm_size.png", "Stronger Pheromone Study: Late Deliveries by Swarm Size")
    grouped_metric_plot(rows, "post_discovery_deliveries", "post_discovery_deliveries_by_swarm_size.png", "Stronger Pheromone Study: Post-Discovery Deliveries by Swarm Size")
    grouped_metric_plot(rows, "pickup_to_delivery_latency", "pickup_to_delivery_latency_by_swarm_size.png", "Stronger Pheromone Study: Pickup-to-Delivery Latency")


if __name__ == "__main__":
    main()
