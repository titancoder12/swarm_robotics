from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List

from env.config import SwarmConfig


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


def add_env_config_args(parser) -> None:
    """Add shared environment override flags to a CLI parser."""
    parser.add_argument("--n-targets", type=int, default=4)
    parser.add_argument("--n-obstacles", type=int, default=6)
    parser.add_argument("--max-steps-per-episode", type=int, default=600)
    parser.add_argument("--dynamics-mode", choices=["tank", "hover", "mixed"], default="tank")
    parser.add_argument("--pheromone-disabled", action="store_true")
    parser.add_argument("--failed-agent-count", type=int, default=0)
    parser.add_argument("--observation-noise-std", type=float, default=0.0)


def make_swarm_config(args) -> SwarmConfig:
    """Build a SwarmConfig from parsed CLI args without changing defaults elsewhere."""
    pheromone_enabled = not getattr(args, "pheromone_disabled", False)
    return SwarmConfig(
        n_agents=getattr(args, "n_agents", 6),
        n_targets=getattr(args, "n_targets", 4),
        n_obstacles=getattr(args, "n_obstacles", 6),
        max_steps=getattr(args, "max_steps_per_episode", 600),
        dynamics_mode=getattr(args, "dynamics_mode", "tank"),
        pheromone_enabled=pheromone_enabled,
        render_pheromone=pheromone_enabled,
        obs_include_pheromone=pheromone_enabled,
        failed_agent_count=getattr(args, "failed_agent_count", 0),
        observation_noise_std=getattr(args, "observation_noise_std", 0.0),
    )


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    """Load a CSV file into a list of dict rows."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_rows(rows: List[Dict[str, float]], metric_keys: List[str]) -> Dict[str, float]:
    """Compute mean/std aggregates for a list of metric dicts."""
    import numpy as np

    summary: Dict[str, float] = {}
    for key in metric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=np.float32) if rows else np.array([], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean()) if values.size else 0.0
        summary[f"{key}_std"] = float(values.std()) if values.size else 0.0
    return summary
