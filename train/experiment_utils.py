from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List

from env.config import SwarmConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sanitize_filename(value: str | None, default: str = "default_run") -> str:
    """Return a filesystem-safe experiment label."""
    raw = (value or "").strip()
    if not raw:
        return default
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return safe or default


def resolve_filename(args, fallback: str = "default_run") -> str:
    """Resolve the preferred user-visible filename for outputs."""
    folder_name = getattr(args, "folder_name", "")
    if folder_name:
        return sanitize_filename(folder_name, default=fallback)
    explicit = getattr(args, "filename", "")
    if explicit:
        return sanitize_filename(explicit, default=fallback)
    return sanitize_filename(getattr(args, "experiment_name", ""), default=fallback)


def resolve_repo_path(path: str | None) -> str:
    """Resolve repo-relative paths against the repository root."""
    raw = (path or "").strip()
    if not raw:
        return ROOT
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(ROOT, raw))


def make_run_dir(output_dir: str, experiment_name: str) -> str:
    """Create a timestamped run directory for logs and plots."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(resolve_repo_path(output_dir), f"{sanitize_filename(experiment_name)}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON payload with stable formatting."""
    resolved_path = resolve_repo_path(path)
    parent = os.path.dirname(resolved_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


class CSVLogger:
    """Simple append-only CSV logger with fixed fieldnames."""

    def __init__(self, path: str, fieldnames: Iterable[str]):
        self.path = resolve_repo_path(path)
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        self._writer.writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _prepare_plot_series(x_values: List[int], y_values: List[float]) -> tuple[list[int], list[float]]:
    """Sort, filter, and deduplicate a series before plotting."""
    points = []
    for x, y in zip(x_values, y_values):
        try:
            x_val = int(x)
            y_val = float(y)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(y_val):
            continue
        points.append((x_val, y_val))

    if not points:
        return [], []

    points.sort(key=lambda pair: pair[0])
    collapsed: list[tuple[int, float]] = []
    for x_val, y_val in points:
        if collapsed and collapsed[-1][0] == x_val:
            prev_x, prev_y = collapsed[-1]
            collapsed[-1] = (prev_x, 0.5 * (prev_y + y_val))
        else:
            collapsed.append((x_val, y_val))
    return [x for x, _ in collapsed], [y for _, y in collapsed]


def plot_training_metrics(csv_path: str, out_dir: str) -> None:
    """Generate standard training plots from an episode metrics CSV."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate training plots.") from exc

    csv_path = resolve_repo_path(csv_path)
    out_dir = resolve_repo_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["mean_episode_reward"]) for row in rows]
    food_discovery = [float(row.get("food_picked_up", row.get("food_discovered", 0.0))) for row in rows]
    food_retrieval = [float(row.get("food_delivered", row.get("food_retrieved", 0.0))) for row in rows]
    efficiency = [float(row["swarm_efficiency"]) for row in rows]

    plots = [
        ("reward_vs_episode.png", rewards, "Mean Episode Reward", "Reward vs Episode"),
        ("food_discovery_vs_episode.png", food_discovery, "Food Picked Up", "Food Pickup vs Episode"),
        ("food_retrieval_vs_episode.png", food_retrieval, "Food Delivered", "Food Delivery vs Episode"),
        ("swarm_efficiency_vs_episode.png", efficiency, "Swarm Efficiency", "Swarm Efficiency vs Episode"),
    ]

    for filename, series, ylabel, title in plots:
        x_plot, y_plot = _prepare_plot_series(episodes, series)
        if not x_plot:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_plot, y_plot, linewidth=2, linestyle="-")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)


def plot_eval_metrics(csv_path: str, out_dir: str) -> None:
    """Generate evaluation plots from an evaluation metrics CSV."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to generate evaluation plots.") from exc

    csv_path = resolve_repo_path(csv_path)
    out_dir = resolve_repo_path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    steps = [int(row["global_step"]) for row in rows]
    rewards = [float(row["mean_episode_reward"]) for row in rows]
    food_discovery = [float(row.get("food_discovered", 0.0)) for row in rows]
    food_retrieval = [float(row.get("food_delivered", row.get("food_retrieved", 0.0))) for row in rows]
    coverage = [float(row["exploration_coverage"]) for row in rows]

    plots = [
        ("eval_reward_vs_step.png", rewards, "Mean Episode Reward", "Evaluation Reward vs Step"),
        ("eval_food_discovery_vs_step.png", food_discovery, "Food Picked Up", "Evaluation Food Pickup vs Step"),
        ("eval_food_retrieval_vs_step.png", food_retrieval, "Food Delivered", "Evaluation Food Delivery vs Step"),
        ("eval_coverage_vs_step.png", coverage, "Exploration Coverage", "Evaluation Coverage vs Step"),
    ]

    for filename, series, ylabel, title in plots:
        x_plot, y_plot = _prepare_plot_series(steps, series)
        if not x_plot:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x_plot, y_plot, linewidth=2, linestyle="-")
        ax.set_xlabel("Global Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, filename))
        plt.close(fig)


def add_env_config_args(parser) -> None:
    """Add shared environment override flags to a CLI parser."""
    parser.add_argument("--folder-name", type=str, default="", help="Preferred enclosing output/checkpoint folder name.")
    parser.add_argument("--filename", type=str, default="")
    parser.add_argument("--n-targets", type=int, default=3)
    parser.add_argument("--n-obstacles", type=int, default=6)
    parser.add_argument("--max-steps-per-episode", type=int, default=600)
    parser.add_argument("--dynamics-mode", choices=["tank", "hover", "mixed"], default="tank")
    parser.add_argument("--action-repeat-steps", type=int, default=2)
    parser.add_argument("--use-pheromone", dest="use_pheromone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pheromone-disabled", action="store_true")
    parser.add_argument("--failed-agent-count", type=int, default=0)
    parser.add_argument("--observation-noise-std", type=float, default=0.0)
    parser.add_argument("--observation-history-steps", type=int, default=3)
    parser.add_argument("--food-detection-radius", type=float, default=150.0)
    parser.add_argument("--target-nest-distance-min", type=float, default=0.0)
    parser.add_argument("--target-nest-distance-max", type=float, default=0.0)
    parser.add_argument("--agent-spawn-near-target-radius", type=float, default=0.0)
    parser.add_argument("--target-nest-corridor-clearance", type=float, default=0.0)
    parser.add_argument("--reward-food-approach", type=float, default=0.03)
    parser.add_argument("--reward-food-detected", type=float, default=0.15)
    parser.add_argument("--reward-pheromone-follow", type=float, default=0.02)
    parser.add_argument("--reward-target", type=float, default=0.0)
    parser.add_argument("--reward-pickup", type=float, default=6.0)
    parser.add_argument("--reward-nest-delivery", type=float, default=30.0)
    parser.add_argument("--reward-nest-approach", type=float, default=0.08)
    parser.add_argument("--reward-nest-approach-sustained", type=float, default=0.04)
    parser.add_argument("--reward-undelivered-food", type=float, default=-10.0)
    parser.add_argument("--reward-pheromone-deposit-cost", type=float, default=-0.001)
    parser.add_argument("--reward-action-switch", type=float, default=-0.01)
    parser.add_argument("--pheromone-follow-min-gradient", type=float, default=0.05)
    parser.add_argument("--reward-new-cell", type=float, default=0.01)
    parser.add_argument("--carrying-reward-new-cell-scale", type=float, default=0.1)
    parser.add_argument("--carrying-no-progress-penalty", type=float, default=-0.03)
    parser.add_argument("--carrying-low-displacement-penalty", type=float, default=-0.02)
    parser.add_argument("--carrying-progress-epsilon", type=float, default=2.0)
    parser.add_argument("--carrying-low-displacement-threshold", type=float, default=3.0)
    parser.add_argument("--carrying-stall-trigger-steps", type=int, default=6)
    parser.add_argument("--carrying-progress-streak-threshold", type=int, default=3)
    parser.add_argument("--pheromone-requires-food", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pheromone-deposit-requires-nest-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--post-delivery-cooldown-steps", type=int, default=0)
    parser.add_argument("--post-delivery-exit-radius", type=float, default=0.0)
    parser.add_argument("--post-delivery-outward-reward", type=float, default=0.0)
    parser.add_argument("--post-delivery-loiter-penalty", type=float, default=0.0)
    parser.add_argument("--post-delivery-pheromone-suppression-radius", type=float, default=0.0)
    parser.add_argument("--post-delivery-require-exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--post-delivery-crowding-penalty", type=float, default=0.0)
    parser.add_argument("--post-delivery-crowding-threshold", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, default=600)
    parser.add_argument("--active-targets", type=int, default=3)
    parser.add_argument("--target-respawn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--food-source-capacity", type=int, default=4)


def make_swarm_config(args) -> SwarmConfig:
    """Build a SwarmConfig from parsed CLI args without changing defaults elsewhere."""
    pheromone_enabled = bool(getattr(args, "use_pheromone", True)) and not getattr(args, "pheromone_disabled", False)
    target_respawn = bool(getattr(args, "target_respawn", True))
    active_targets = max(1, int(getattr(args, "active_targets", getattr(args, "n_targets", 3))))
    return SwarmConfig(
        n_agents=getattr(args, "n_agents", 6),
        n_targets=getattr(args, "n_targets", 3),
        n_obstacles=getattr(args, "n_obstacles", 6),
        max_steps=getattr(args, "max_steps_per_episode", 600),
        dynamics_mode=getattr(args, "dynamics_mode", "tank"),
        action_repeat_steps=max(1, int(getattr(args, "action_repeat_steps", 2))),
        pheromone_enabled=pheromone_enabled,
        render_pheromone=pheromone_enabled,
        obs_include_pheromone=True,
        food_detection_radius=getattr(args, "food_detection_radius", 150.0),
        target_nest_distance_min=getattr(args, "target_nest_distance_min", 0.0),
        target_nest_distance_max=getattr(args, "target_nest_distance_max", 0.0),
        agent_spawn_near_target_radius=getattr(args, "agent_spawn_near_target_radius", 0.0),
        target_nest_corridor_clearance=getattr(args, "target_nest_corridor_clearance", 0.0),
        pheromone_requires_food=bool(getattr(args, "pheromone_requires_food", True)),
        pheromone_deposit_requires_nest_progress=bool(getattr(args, "pheromone_deposit_requires_nest_progress", True)),
        non_carrying_nest_pheromone_suppression_radius=getattr(args, "non_carrying_nest_pheromone_suppression_radius", 0.0),
        non_carrying_nest_loiter_radius=getattr(args, "non_carrying_nest_loiter_radius", 0.0),
        non_carrying_nest_loiter_penalty=getattr(args, "non_carrying_nest_loiter_penalty", 0.0),
        non_carrying_nest_crowding_radius=getattr(args, "non_carrying_nest_crowding_radius", 0.0),
        non_carrying_nest_crowding_penalty=getattr(args, "non_carrying_nest_crowding_penalty", 0.0),
        non_carrying_nest_crowding_threshold=max(1, int(getattr(args, "non_carrying_nest_crowding_threshold", 2))),
        post_delivery_cooldown_steps=max(0, int(getattr(args, "post_delivery_cooldown_steps", 0))),
        post_delivery_exit_radius=getattr(args, "post_delivery_exit_radius", 0.0),
        post_delivery_outward_reward=getattr(args, "post_delivery_outward_reward", 0.0),
        post_delivery_loiter_penalty=getattr(args, "post_delivery_loiter_penalty", 0.0),
        post_delivery_pheromone_suppression_radius=getattr(args, "post_delivery_pheromone_suppression_radius", 0.0),
        post_delivery_require_exit=bool(getattr(args, "post_delivery_require_exit", False)),
        post_delivery_crowding_penalty=getattr(args, "post_delivery_crowding_penalty", 0.0),
        post_delivery_crowding_threshold=max(1, int(getattr(args, "post_delivery_crowding_threshold", 2))),
        failed_agent_count=getattr(args, "failed_agent_count", 0),
        observation_noise_std=getattr(args, "observation_noise_std", 0.0),
        observation_history_steps=max(1, int(getattr(args, "observation_history_steps", 3))),
        reward_new_cell=getattr(args, "reward_new_cell", 0.01),
        carrying_reward_new_cell_scale=getattr(args, "carrying_reward_new_cell_scale", 0.1),
        carrying_no_progress_penalty=getattr(args, "carrying_no_progress_penalty", -0.03),
        carrying_low_displacement_penalty=getattr(args, "carrying_low_displacement_penalty", -0.02),
        carrying_progress_epsilon=getattr(args, "carrying_progress_epsilon", 2.0),
        carrying_low_displacement_threshold=getattr(args, "carrying_low_displacement_threshold", 3.0),
        carrying_stall_trigger_steps=max(1, int(getattr(args, "carrying_stall_trigger_steps", 6))),
        carrying_progress_streak_threshold=max(1, int(getattr(args, "carrying_progress_streak_threshold", 3))),
        reward_target=getattr(args, "reward_target", 0.0),
        reward_pickup=getattr(args, "reward_pickup", 6.0),
        reward_nest_delivery=getattr(args, "reward_nest_delivery", 30.0),
        reward_nest_approach=getattr(args, "reward_nest_approach", 0.08),
        reward_nest_approach_sustained=getattr(args, "reward_nest_approach_sustained", 0.04),
        reward_undelivered_food=getattr(args, "reward_undelivered_food", -10.0),
        reward_food_approach=getattr(args, "reward_food_approach", 0.03),
        reward_food_detected=getattr(args, "reward_food_detected", 0.15),
        reward_pheromone_follow=getattr(args, "reward_pheromone_follow", 0.02),
        reward_pheromone_deposit_cost=getattr(args, "reward_pheromone_deposit_cost", -0.001),
        reward_action_switch=getattr(args, "reward_action_switch", -0.01),
        pheromone_follow_min_gradient=getattr(args, "pheromone_follow_min_gradient", 0.05),
        active_targets=active_targets,
        target_respawn=target_respawn,
        food_source_capacity=max(1, int(getattr(args, "food_source_capacity", 4))),
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
