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

ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.mappo.inference import load_actor
from env.config import SwarmConfig
from env.swarm_env import SwarmEnv


TIMESTAMP = "20260407_020816"
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / TIMESTAMP / "killer_scaling"
RAW_DIR = BASE_DIR / "raw_exports"
TABLES_DIR = BASE_DIR / "tables"
META_DIR = BASE_DIR / "metadata"
NOTES_DIR = BASE_DIR / "analysis_notes"

PRIMARY_SEEDS = list(range(100, 120))
SUPPORTING_SEEDS = list(range(100, 120))
SWARM_SIZES = [1, 3, 6, 10, 15, 20, 30]
PRIMARY_SWARM_SIZES = list(SWARM_SIZES)


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


def cfg_from_default(**overrides: Any) -> SwarmConfig:
    cfg = SwarmConfig()
    field_names = {f.name for f in fields(SwarmConfig)}
    for key, value in overrides.items():
        if key in field_names:
            setattr(cfg, key, value)
    return cfg


def build_policy(checkpoint_dir: Path, obs_dim: int, action_dim: int, n_agents: int) -> dict[str, Any]:
    actor, device = load_actor(str(checkpoint_dir), obs_dim, action_dim, device="cpu")
    return {
        "actor": actor,
        "device": device,
        "n_agents": n_agents,
    }


def run_episode(env: SwarmEnv, policy: dict[str, Any], seed: int) -> dict[str, Any]:
    cfg = env.cfg
    obs_dict, _ = env.reset(seed=seed)
    obs = np.stack([obs_dict[a] for a in env.possible_agents], axis=0)
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
    text = """# 30-Agent Stigmergy Scaling Plan

This campaign redesigns the decisive pheromone study to be more sensitive to
trail reuse while keeping the condition comparison fair.

Primary goal:

- test whether pheromone-enabled training and evaluation produce stronger
  repeated-source exploitation than matched no-pheromone conditions as swarm
  size increases up to `30`

Design:

- conditions:
  - trained with pheromone, evaluated with pheromone
  - trained with pheromone, evaluated without pheromone
  - trained without pheromone, evaluated without pheromone
- task:
  - full final-stage arena geometry from `checkpoints/mappo_g/latest`
  - `1` active target
  - `1` total target source
  - `food_source_capacity = 12`
  - `target_respawn = false`
  - `18` obstacles
  - target constrained away from the nest to reduce trivial finds
  - horizon `1200`
- swarm sizes:
  - `1`, `3`, `6`, `10`, `15`, `20`, `30`

Repetitions:

- `20` paired seeds for every swarm size and condition
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

    raw_rows: list[dict[str, Any]] = []
    for n_agents in SWARM_SIZES:
        print(f"[campaign] running n_agents={n_agents}", flush=True)
        seeds = PRIMARY_SEEDS if n_agents in PRIMARY_SWARM_SIZES else SUPPORTING_SEEDS
        for condition in conditions:
            print(
                f"[campaign]   condition={condition['condition']} episodes={len(seeds)}",
                flush=True,
            )
            cfg = cfg_from_metadata(
                base_meta,
                n_agents=n_agents,
                pheromone_enabled=bool(condition["pheromone_enabled"]),
                n_targets=1,
                active_targets=1,
                target_respawn=False,
                food_source_capacity=12,
                max_steps=1200,
                target_nest_distance_min=260.0,
                target_nest_distance_max=420.0,
                target_nest_corridor_clearance=100.0,
            )
            env = SwarmEnv(cfg, headless=True)
            obs_dict, _ = env.reset(seed=seeds[0])
            obs = np.stack([obs_dict[a] for a in env.possible_agents], axis=0)
            policy = build_policy(condition["checkpoint_dir"], obs.shape[1], cfg.num_actions, cfg.n_agents)
            for seed in seeds:
                metrics = run_episode(env, policy, seed)
                raw_rows.append(
                    {
                        "family": "focused_pheromone_ablation_extended_30_agents",
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
            env.close()

    save_csv(RAW_DIR / "all_episode_results.csv", raw_rows)
    summary_rows = summarize_rows(raw_rows, ["condition", "n_agents"])
    save_csv(TABLES_DIR / "condition_summary.csv", summary_rows)

    stats_rows = []
    stat_pairs = []
    for primary_n in PRIMARY_SWARM_SIZES:
        stat_pairs.extend(
            [
                (
                    primary_n,
                    "trained_with_pheromone__eval_with_pheromone",
                    "trained_without_pheromone__eval_without_pheromone",
                    "food_delivered",
                ),
                (
                    primary_n,
                    "trained_with_pheromone__eval_with_pheromone",
                    "trained_without_pheromone__eval_without_pheromone",
                    "late_deliveries",
                ),
                (
                    primary_n,
                    "trained_with_pheromone__eval_with_pheromone",
                    "trained_without_pheromone__eval_without_pheromone",
                    "post_discovery_deliveries",
                ),
                (
                    primary_n,
                    "trained_with_pheromone__eval_with_pheromone",
                    "trained_with_pheromone__eval_without_pheromone",
                    "late_deliveries",
                ),
                (
                    primary_n,
                    "trained_with_pheromone__eval_with_pheromone",
                    "trained_with_pheromone__eval_without_pheromone",
                    "food_delivered",
                ),
                (
                    primary_n,
                    "trained_with_pheromone__eval_with_pheromone",
                    "trained_with_pheromone__eval_without_pheromone",
                    "post_discovery_deliveries",
                ),
            ]
        )

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
                **paired_stats(a_vals, b_vals),
            }
        )
    save_csv(TABLES_DIR / "statistical_tests.csv", stats_rows)

    manifest = {
        "timestamp": TIMESTAMP,
        "primary_seed_range": [PRIMARY_SEEDS[0], PRIMARY_SEEDS[-1]],
        "supporting_seed_range": [SUPPORTING_SEEDS[0], SUPPORTING_SEEDS[-1]],
        "primary_swarm_sizes": PRIMARY_SWARM_SIZES,
        "swarm_sizes": SWARM_SIZES,
        "conditions": [c["condition"] for c in conditions],
        "design": "single-source long-horizon repeated-use pheromone scaling up to 30 agents",
        "task_overrides": {
            "n_targets": 1,
            "active_targets": 1,
            "target_respawn": False,
            "food_source_capacity": 12,
            "max_steps": 1200,
            "target_nest_distance_min": 260.0,
            "target_nest_distance_max": 420.0,
            "target_nest_corridor_clearance": 100.0,
        },
    }
    (META_DIR / "campaign_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (META_DIR / "run_complete.json").write_text(
        json.dumps(
            {
                "timestamp": TIMESTAMP,
                "status": "completed",
                "raw_episode_rows": len(raw_rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
