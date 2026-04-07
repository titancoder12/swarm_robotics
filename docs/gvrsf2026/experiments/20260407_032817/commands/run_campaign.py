from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pygame
import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.inference import load_actor
from env.config import SwarmConfig
from env.swarm_env import AgentState, SwarmEnv


TIMESTAMP = "20260407_032817"
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / TIMESTAMP
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
RAW_DIR = BASE_DIR / "raw_exports"
META_DIR = BASE_DIR / "metadata"
NOTES_DIR = BASE_DIR / "analysis_notes"

SEEDS = list(range(100, 120))
SWARM_SIZES = [1, 3, 6, 10]


def ensure_dirs() -> None:
    for path in [BASE_DIR, FIGURES_DIR, TABLES_DIR, RAW_DIR, META_DIR, NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def cfg_from_metadata(metadata_path: str | Path, **overrides: Any) -> SwarmConfig:
    meta = json.loads(Path(metadata_path).read_text())
    cfg = SwarmConfig()
    field_names = {f.name for f in fields(SwarmConfig)}
    for key, value in meta.items():
        if key in field_names:
            setattr(cfg, key, value)
    for key, value in overrides.items():
        if key in field_names:
            setattr(cfg, key, value)
    return cfg


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


def paired_stats(a: list[float], b: list[float]) -> dict[str, float | None]:
    result = {
        "paired_t_stat": None,
        "paired_t_p_value": None,
        "wilcoxon_stat": None,
        "wilcoxon_p_value": None,
    }
    if len(a) != len(b) or len(a) < 2:
        return result
    try:
        from scipy.stats import ttest_rel, wilcoxon

        t_stat = ttest_rel(a, b)
        result["paired_t_stat"] = float(t_stat.statistic)
        result["paired_t_p_value"] = float(t_stat.pvalue)
        w_stat = wilcoxon(a, b, zero_method="zsplit")
        result["wilcoxon_stat"] = float(w_stat.statistic)
        result["wilcoxon_p_value"] = float(w_stat.pvalue)
    except Exception:
        pass
    return result


def build_policy(checkpoint_dir: Path, obs_dim: int, action_dim: int, n_agents: int) -> dict[str, Any]:
    actor, device = load_actor(str(checkpoint_dir), obs_dim, action_dim, device="cpu")
    return {"actor": actor, "device": device, "n_agents": n_agents}


def apply_seeded_single_gap_world(env: SwarmEnv, seed: int) -> np.ndarray:
    width = int(env.width)
    height = int(env.height)
    rng = np.random.default_rng(seed)

    gap_height = 210.0
    wall_width = 80
    wall_x = int(width * 0.50)
    gap_center = float(height * 0.50 + rng.integers(-60, 61))
    gap_top = int(max(100, gap_center - gap_height * 0.5))
    gap_bottom = int(min(height - 100, gap_center + gap_height * 0.5))

    env.obstacles = [
        pygame.Rect(wall_x, 0, wall_width, max(0, gap_top)),
        pygame.Rect(wall_x, gap_bottom, wall_width, max(0, height - gap_bottom)),
    ]

    nest_y = float(height * 0.50 + rng.integers(-40, 41))
    target_y = float(gap_center + rng.integers(-45, 46))
    env.nest_position = (180.0, nest_y)
    env.targets = [(float(width - 180), target_y)]
    env.target_remaining_uses = [int(env.cfg.food_source_capacity)]

    cluster_center_x = 245.0
    cluster_center_y = nest_y
    spacing = 24.0
    agent_positions: list[tuple[float, float]] = []
    cols = 3
    for idx in range(env.cfg.n_agents):
        row = idx // cols
        col = idx % cols
        dx = (col - 1) * spacing + float(rng.integers(-4, 5))
        dy = (row - 1) * spacing + float(rng.integers(-4, 5))
        agent_positions.append((cluster_center_x + dx, cluster_center_y + dy))

    env.agent_states = []
    for x, y in agent_positions:
        theta = float(rng.uniform(-0.20, 0.20))
        env.agent_states.append(AgentState(x, y, theta, carrying_food=False))

    env.food_delivered = 0
    env.episode_food_source_respawns = 0
    env.episode_targets_collected = 0
    env.episode_pheromone_deposit_events = 0
    env.first_pickup_step = None
    env.first_delivery_step = None
    env._init_coverage_grid()
    env._update_coverage()
    env._reset_food_shaping_state()
    env._prev_nest_distances = env._compute_nest_distance_state()
    env._prev_agent_positions = np.array([[agent.x, agent.y] for agent in env.agent_states], dtype=np.float32)
    env._carrying_no_progress_counts.fill(0)
    env._carrying_progress_streaks.fill(0)
    if env.pheromone_grid is not None:
        env.pheromone_grid.fill(0.0)
    return env._get_obs(reset_history=True)


def run_episode(env: SwarmEnv, policy: dict[str, Any], seed: int) -> dict[str, Any]:
    env.reset(seed=seed)
    obs = apply_seeded_single_gap_world(env, seed)
    cfg = env.cfg
    hidden = policy["actor"].initial_hidden(policy["n_agents"], policy["device"])
    prev_done = np.zeros((cfg.n_agents,), dtype=np.float32)

    total_pickups = 0
    total_deliveries = 0
    early_deliveries = 0
    late_deliveries = 0
    first_pickup_step = -1
    first_delivery_step = -1
    post_discovery_deliveries = 0
    pheromone_usage_values: list[float] = []

    while True:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy["device"])
        done_mask = torch.as_tensor(1.0 - prev_done, dtype=torch.float32, device=policy["device"])
        with torch.no_grad():
            logits, hidden = policy["actor"](obs_tensor, hidden, done_mask)
            actions = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)

        next_obs, rewards, terminations, truncations, infos = env.step(
            {agent_id: int(actions[i]) for i, agent_id in enumerate(env.possible_agents)}
        )
        obs = np.stack([next_obs[a] for a in env.possible_agents], axis=0)
        prev_done = np.array([float(terminations[a] or truncations[a]) for a in env.possible_agents], dtype=np.float32)
        info = infos[env.possible_agents[0]]
        step = int(info.get("episode_length", 0))
        pickups = int(info.get("targets_collected", 0))
        deliveries = int(info.get("food_delivered", 0))
        total_pickups += pickups
        total_deliveries += deliveries
        if step <= 500:
            early_deliveries += deliveries
        else:
            late_deliveries += deliveries
        if first_pickup_step < 0 and pickups > 0:
            first_pickup_step = step
        if first_delivery_step < 0 and deliveries > 0:
            first_delivery_step = step
        if first_pickup_step >= 0 and step > first_pickup_step:
            post_discovery_deliveries += deliveries
        pheromone_usage_values.append(float(info.get("pheromone_usage", 0.0)))
        if any(terminations.values()) or any(truncations.values()):
            final_info = info
            break

    return {
        "food_picked_up": total_pickups,
        "food_delivered": total_deliveries,
        "delivery_conversion": float(total_deliveries / total_pickups) if total_pickups > 0 else 0.0,
        "early_deliveries": early_deliveries,
        "late_deliveries": late_deliveries,
        "late_delivery_fraction": float(late_deliveries / total_deliveries) if total_deliveries > 0 else 0.0,
        "post_discovery_deliveries": post_discovery_deliveries,
        "time_to_first_discovery": first_pickup_step,
        "time_to_first_delivery": first_delivery_step,
        "pickup_to_delivery_latency": float(first_delivery_step - first_pickup_step)
        if first_pickup_step >= 0 and first_delivery_step >= 0
        else -1.0,
        "exploration_coverage": float(final_info.get("exploration_coverage", 0.0)),
        "pheromone_usage": mean_or_zero(pheromone_usage_values),
        "episode_length": int(final_info.get("episode_length", 0)),
    }


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    metrics = [
        "food_picked_up",
        "food_delivered",
        "delivery_conversion",
        "early_deliveries",
        "late_deliveries",
        "late_delivery_fraction",
        "post_discovery_deliveries",
        "time_to_first_discovery",
        "time_to_first_delivery",
        "pickup_to_delivery_latency",
        "exploration_coverage",
        "pheromone_usage",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[k] for k in group_keys), []).append(row)
    summaries = []
    for key, group in grouped.items():
        base = {k: v for k, v in zip(group_keys, key)}
        base["n"] = len(group)
        for metric in metrics:
            values = [float(r[metric]) for r in group]
            base[f"{metric}_mean"] = mean_or_zero(values)
            base[f"{metric}_std"] = std_or_zero(values)
            base[f"{metric}_ci95"] = ci95(values)
        summaries.append(base)
    return summaries


def write_notes() -> None:
    text = """# Single-Gap Chokepoint Hypothesis Study: Stage max_10

Hypothesis:

- pheromone should help more when a repeated source sits behind a single narrow passage, but the task still remains solvable enough to avoid floor effects

Design:

- same-curriculum matched checkpoints
- one repeated source with high capacity
- fixed single-wall chokepoint world with one gap
- nest and initial swarm cluster on the left side
- target on the right side, roughly aligned to the gap
- longer horizon
- swarm sizes: `1`, `3`, `6`, `10`
- `20` paired seeds per size and condition

Conditions:

- trained with pheromone, eval with pheromone
- trained with pheromone, eval without pheromone
- trained without pheromone, eval without pheromone

Predeclared extension rule:

- extend to `max_20` only if the `10`-agent result is promising:
  - `food_delivered_mean` is higher for pheromone-on than for both comparison conditions
  - and `post_discovery_deliveries_mean` is higher than the true no-pheromone condition
  - and at least one `10`-agent matched test against the true no-pheromone condition has
    `p < 0.10` or `cohens_d >= 0.3`
"""
    (NOTES_DIR / "execution_plan.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_notes()
    base_meta = ROOT / "checkpoints" / "mappo_g" / "latest" / "metadata.json"

    conditions = [
        {
            "condition": "trained_with_pheromone__eval_with_pheromone",
            "checkpoint_dir": ROOT / "checkpoints" / "mappo_g" / "latest",
            "pheromone_enabled": True,
        },
        {
            "condition": "trained_with_pheromone__eval_without_pheromone",
            "checkpoint_dir": ROOT / "checkpoints" / "mappo_g" / "latest",
            "pheromone_enabled": False,
        },
        {
            "condition": "trained_without_pheromone__eval_without_pheromone",
            "checkpoint_dir": ROOT / "checkpoints" / "mappo_g_no_pher_true" / "latest",
            "pheromone_enabled": False,
        },
    ]

    manifest = {
        "timestamp": TIMESTAMP,
        "stage": "max_10",
        "swarm_sizes": SWARM_SIZES,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "conditions": [c["condition"] for c in conditions],
        "design": "calibrated single-gap chokepoint route-reuse hypothesis study",
        "task_overrides": {
            "active_targets": 1,
            "target_respawn": False,
            "food_source_capacity": 15,
            "n_obstacles": 2,
            "max_steps": 1400,
            "layout": "single vertical wall with one gap",
        },
        "extension_rule": {
            "highest_size": 10,
            "requires_food_delivered_advantage": True,
            "requires_post_discovery_advantage_vs_true_no_pher": True,
            "requires_p_lt": 0.10,
            "or_cohens_d_gte": 0.3,
        },
    }
    (META_DIR / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    raw_rows: list[dict[str, Any]] = []
    for n_agents in SWARM_SIZES:
        print(f"[campaign] running n_agents={n_agents}", flush=True)
        for condition in conditions:
            print(f"[campaign]   condition={condition['condition']} episodes={len(SEEDS)}", flush=True)
            cfg = cfg_from_metadata(
                base_meta,
                n_agents=n_agents,
                pheromone_enabled=bool(condition["pheromone_enabled"]),
                n_targets=1,
                active_targets=1,
                target_respawn=False,
                food_source_capacity=15,
                n_obstacles=2,
                max_steps=1400,
            )
            env = SwarmEnv(cfg, headless=True)
            env.reset(seed=SEEDS[0])
            obs = apply_seeded_single_gap_world(env, SEEDS[0])
            policy = build_policy(condition["checkpoint_dir"], obs.shape[1], cfg.num_actions, cfg.n_agents)
            for seed in SEEDS:
                metrics = run_episode(env, policy, seed)
                raw_rows.append(
                    {
                        "family": "single_gap_chokepoint_hypothesis_max_10",
                        "condition": condition["condition"],
                        "seed": seed,
                        "n_agents": n_agents,
                        "max_steps": cfg.max_steps,
                        "active_targets": cfg.active_targets,
                        "food_source_capacity": cfg.food_source_capacity,
                        "n_obstacles": cfg.n_obstacles,
                        "pheromone_enabled": cfg.pheromone_enabled,
                        "checkpoint_dir": str(condition["checkpoint_dir"].relative_to(ROOT)),
                        **metrics,
                    }
                )
            env.close()
            save_csv(RAW_DIR / "all_episode_results.csv", raw_rows)
            save_csv(TABLES_DIR / "condition_summary.csv", summarize_rows(raw_rows, ["condition", "n_agents"]))

    save_csv(RAW_DIR / "all_episode_results.csv", raw_rows)
    summary_rows = summarize_rows(raw_rows, ["condition", "n_agents"])
    save_csv(TABLES_DIR / "condition_summary.csv", summary_rows)

    stats_rows = []
    for n_agents in SWARM_SIZES:
        for a_cond, b_cond, metric in [
            (
                "trained_with_pheromone__eval_with_pheromone",
                "trained_without_pheromone__eval_without_pheromone",
                "food_delivered",
            ),
            (
                "trained_with_pheromone__eval_with_pheromone",
                "trained_without_pheromone__eval_without_pheromone",
                "late_deliveries",
            ),
            (
                "trained_with_pheromone__eval_with_pheromone",
                "trained_without_pheromone__eval_without_pheromone",
                "post_discovery_deliveries",
            ),
            (
                "trained_with_pheromone__eval_with_pheromone",
                "trained_with_pheromone__eval_without_pheromone",
                "food_delivered",
            ),
            (
                "trained_with_pheromone__eval_with_pheromone",
                "trained_with_pheromone__eval_without_pheromone",
                "post_discovery_deliveries",
            ),
        ]:
            a_vals = [float(r[metric]) for r in raw_rows if r["n_agents"] == n_agents and r["condition"] == a_cond]
            b_vals = [float(r[metric]) for r in raw_rows if r["n_agents"] == n_agents and r["condition"] == b_cond]
            stats_rows.append(
                {
                    "n_agents": n_agents,
                    "condition_a": a_cond,
                    "condition_b": b_cond,
                    "metric": metric,
                    "cohens_d": cohens_d(a_vals, b_vals),
                    **paired_stats(a_vals, b_vals),
                }
            )
    save_csv(TABLES_DIR / "statistical_tests.csv", stats_rows)

if __name__ == "__main__":
    main()
