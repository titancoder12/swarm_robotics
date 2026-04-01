from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.inference import load_actor
from env.config import SwarmConfig
from env.swarm_env import SwarmEnv


TIMESTAMP = "20260401_021226"
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / TIMESTAMP
RAW_DIR = BASE_DIR / "raw_exports"
TABLES_DIR = BASE_DIR / "tables"
META_DIR = BASE_DIR / "metadata"
NOTES_DIR = BASE_DIR / "analysis_notes"

SEEDS = list(range(100, 120))
SWARM_SIZES = [1, 3, 6]


def ensure_dirs() -> None:
    for path in [BASE_DIR, RAW_DIR, TABLES_DIR, META_DIR, NOTES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def cfg_from_default(**overrides: Any) -> SwarmConfig:
    cfg = SwarmConfig()
    field_names = {f.name for f in fields(SwarmConfig)}
    for key, value in overrides.items():
        if key in field_names:
            setattr(cfg, key, value)
    return cfg


def run_episode(seed: int, checkpoint_dir: Path, cfg: SwarmConfig) -> dict[str, Any]:
    env = SwarmEnv(cfg, headless=True)
    obs_dict, _ = env.reset(seed=seed)
    obs = np.stack([obs_dict[a] for a in env.possible_agents], axis=0)
    actor, device = load_actor(str(checkpoint_dir), obs.shape[1], cfg.num_actions, device="cpu")
    hidden = actor.initial_hidden(cfg.n_agents, device)
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
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        done_mask = torch.as_tensor(1.0 - prev_done, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, hidden = actor(obs_tensor, hidden, done_mask)
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
        if step <= 400:
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

    env.close()
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
    text = """# Focused Pheromone Experiment Plan

This campaign is a focused rerun of the GVRSF experiment prompt with a stronger pheromone design.

Why the design changed:

- The previous generic final-stage pheromone toggle test produced only a modest effect.
- That design mainly tested inference-time dependence, not whether pheromone training and trail reuse improve swarm coordination.
- This campaign instead uses a repeated-source foraging environment where trail reuse should matter.

Key design choices:

- matched training conditions:
  - `mappo_gru_pheromone/stage3_full_marl`
  - `mappo_gru_no_pheromone/stage3_full_marl`
- paired evaluation layouts:
  - seeds `100..119`
- multiple swarm sizes:
  - `1`, `3`, `6`
- trail-reuse-friendly task:
  - one active food source
  - source capacity `12`
  - horizon `1200`

Primary claim being tested:

> Pheromone-trained swarms should outperform no-pheromone-trained swarms, especially in post-discovery and late-episode delivery behavior, and the effect should be strongest at larger swarm sizes.
"""
    (NOTES_DIR / "execution_plan.md").write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    write_notes()

    conditions = [
        {
            "condition": "trained_with_pheromone__eval_with_pheromone",
            "checkpoint_dir": ROOT / "checkpoints" / "mappo_gru_pheromone" / "stage3_full_marl",
            "pheromone_enabled": True,
        },
        {
            "condition": "trained_with_pheromone__eval_without_pheromone",
            "checkpoint_dir": ROOT / "checkpoints" / "mappo_gru_pheromone" / "stage3_full_marl",
            "pheromone_enabled": False,
        },
        {
            "condition": "trained_without_pheromone__eval_without_pheromone",
            "checkpoint_dir": ROOT / "checkpoints" / "mappo_gru_no_pheromone" / "stage3_full_marl",
            "pheromone_enabled": False,
        },
    ]

    raw_rows: list[dict[str, Any]] = []
    for n_agents in SWARM_SIZES:
        for seed in SEEDS:
            for condition in conditions:
                cfg = cfg_from_default(
                    n_agents=n_agents,
                    n_targets=1,
                    active_targets=1,
                    food_source_capacity=12,
                    target_respawn=True,
                    max_steps=1200,
                    pheromone_enabled=bool(condition["pheromone_enabled"]),
                )
                metrics = run_episode(seed, condition["checkpoint_dir"], cfg)
                raw_rows.append(
                    {
                        "family": "focused_pheromone_ablation",
                        "condition": condition["condition"],
                        "seed": seed,
                        "n_agents": n_agents,
                        "max_steps": cfg.max_steps,
                        "active_targets": cfg.active_targets,
                        "food_source_capacity": cfg.food_source_capacity,
                        "pheromone_enabled": cfg.pheromone_enabled,
                        "checkpoint_dir": str(condition["checkpoint_dir"].relative_to(ROOT)),
                        **metrics,
                    }
                )

    save_csv(RAW_DIR / "all_episode_results.csv", raw_rows)
    summary_rows = summarize_rows(raw_rows, ["condition", "n_agents"])
    save_csv(TABLES_DIR / "condition_summary.csv", summary_rows)

    stats_rows = []
    stat_pairs = [
        (6, "trained_with_pheromone__eval_with_pheromone", "trained_without_pheromone__eval_without_pheromone", "food_delivered"),
        (6, "trained_with_pheromone__eval_with_pheromone", "trained_without_pheromone__eval_without_pheromone", "late_deliveries"),
        (6, "trained_with_pheromone__eval_with_pheromone", "trained_with_pheromone__eval_without_pheromone", "late_deliveries"),
        (3, "trained_with_pheromone__eval_with_pheromone", "trained_without_pheromone__eval_without_pheromone", "food_delivered"),
    ]
    for n_agents, a_cond, b_cond, metric in stat_pairs:
        a_vals = [float(r[metric]) for r in raw_rows if r["n_agents"] == n_agents and r["condition"] == a_cond]
        b_vals = [float(r[metric]) for r in raw_rows if r["n_agents"] == n_agents and r["condition"] == b_cond]
        stats_rows.append(
            {
                "n_agents": n_agents,
                "condition_a": a_cond,
                "condition_b": b_cond,
                "metric": metric,
                "cohens_d": cohens_d(a_vals, b_vals),
                **welch_ttest(a_vals, b_vals),
            }
        )
    save_csv(TABLES_DIR / "statistical_tests.csv", stats_rows)

    manifest = {
        "timestamp": TIMESTAMP,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "swarm_sizes": SWARM_SIZES,
        "conditions": [c["condition"] for c in conditions],
        "design": "matched-training repeated-source pheromone ablation",
    }
    (META_DIR / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
