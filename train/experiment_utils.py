from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable


def make_run_dir(output_dir: str, experiment_name: str) -> str:
    """Create a timestamped run directory for logs and plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


class CSVLogger:
    """Simple append-only CSV logger with fixed fieldnames."""

    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def plot_training_metrics(csv_path: str, out_dir: str) -> None:
    """Generate standard training plots from an episode metrics CSV."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate training plots.") from exc

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["mean_episode_reward"]) for row in rows]
    food_retrieval = [float(row["food_retrieved"]) for row in rows]
    efficiency = [float(row["swarm_efficiency"]) for row in rows]

    plots = [
        ("reward_vs_episode.png", rewards, "Mean Episode Reward", "Reward vs Episode"),
        ("food_retrieval_vs_episode.png", food_retrieval, "Food Retrieved", "Food Retrieval vs Episode"),
        ("swarm_efficiency_vs_episode.png", efficiency, "Swarm Efficiency", "Swarm Efficiency vs Episode"),
    ]

    for filename, series, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(episodes, series, linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)

