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


ROOT = Path(__file__).resolve().parents[5]
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / "20260401_021226"
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
    swarm_sizes = sorted({int(r["n_agents"]) for r in rows})
    x = np.arange(len(swarm_sizes))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2F5597", "#70AD47", "#C0504D"]
    for idx, condition in enumerate(order):
        subset = [r for r in rows if r["condition"] == condition]
        subset.sort(key=lambda r: int(r["n_agents"]))
        means = [float(r[f"{metric}_mean"]) for r in subset]
        cis = [float(r[f"{metric}_ci95"]) for r in subset]
        ax.bar(x + (idx - 1) * width, means, width, yerr=cis, capsize=4, label=condition, color=colors[idx])
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in swarm_sizes])
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    grouped_metric_plot(rows, "food_delivered", "food_delivered_by_swarm_size.png", "Pheromone Ablation: Deliveries by Swarm Size")
    grouped_metric_plot(rows, "late_deliveries", "late_deliveries_by_swarm_size.png", "Pheromone Ablation: Late Deliveries by Swarm Size")
    grouped_metric_plot(rows, "post_discovery_deliveries", "post_discovery_deliveries_by_swarm_size.png", "Pheromone Ablation: Post-Discovery Deliveries by Swarm Size")
    grouped_metric_plot(rows, "pickup_to_delivery_latency", "pickup_to_delivery_latency_by_swarm_size.png", "Pheromone Ablation: Pickup-to-Delivery Latency")


if __name__ == "__main__":
    main()
