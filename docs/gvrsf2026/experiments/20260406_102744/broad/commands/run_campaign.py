from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import asdict, fields
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
from models.rule_based_policy import RuleBasedSwarmPolicy


TIMESTAMP = "20260406_102744"
BASE_DIR = ROOT / "docs" / "gvrsf2026" / "experiments" / TIMESTAMP
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
RAW_DIR = BASE_DIR / "raw_exports"
META_DIR = BASE_DIR / "metadata"
NOTES_DIR = BASE_DIR / "analysis_notes"

SEED_START = 100
EPISODES_PER_CONDITION = 20
LEAKED_ENVS: list[SwarmEnv] = []


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


def welch_ttest(a: list[float], b: list[float]) -> dict[str, float | None]:
    try:
        from scipy.stats import ttest_ind
    except Exception:
        return {"t_stat": None, "p_value": None}
    if len(a) < 2 or len(b) < 2:
        return {"t_stat": None, "p_value": None}
    stat = ttest_ind(a, b, equal_var=False)
    return {"t_stat": float(stat.statistic), "p_value": float(stat.pvalue)}


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_policy(
    policy_kind: str,
    checkpoint_dir: Path | None,
    obs_dim: int,
    action_dim: int,
    n_agents: int,
    cfg: SwarmConfig,
):
    device = torch.device("cpu")
    if policy_kind == "mappo_gru":
        if checkpoint_dir is None:
            raise ValueError("MAPPO policy requires a checkpoint_dir")
        actor, device = load_actor(str(checkpoint_dir), obs_dim, action_dim, device="cpu")
        return {"kind": "mappo_gru", "actor": actor, "device": device}
    if policy_kind == "rule_based":
        return {
            "kind": "rule_based",
            "policies": [RuleBasedSwarmPolicy(cfg, seed=i) for i in range(n_agents)],
            "device": device,
        }
    if policy_kind == "random":
        rngs = [np.random.default_rng(10_000 + i) for i in range(n_agents)]
        return {"kind": "random", "rngs": rngs, "device": device}
    raise ValueError(f"Unsupported policy_kind: {policy_kind}")


def run_episode(env: SwarmEnv, policy: dict[str, Any], seed: int) -> dict[str, Any]:
    obs_dict, _ = env.reset(seed=seed)
    agent_ids = env.possible_agents
    obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
    episode_rewards = np.zeros(env.cfg.n_agents, dtype=np.float32)
    episode_length = 0
    mappo_hidden = None
    prev_done = np.zeros((env.cfg.n_agents,), dtype=np.float32)
    if policy["kind"] == "mappo_gru":
        mappo_hidden = policy["actor"].initial_hidden(env.cfg.n_agents, policy["device"])

    pickups = 0
    deliveries = 0
    discovery_step = -1
    pheromone_usage_values: list[float] = []
    latest_info: dict[str, Any] = {}

    while True:
        actions = np.zeros(env.cfg.n_agents, dtype=np.int64)
        if policy["kind"] == "mappo_gru":
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=policy["device"])
            done_mask = torch.as_tensor(1.0 - prev_done, dtype=torch.float32, device=policy["device"])
            with torch.no_grad():
                logits, mappo_hidden = policy["actor"](obs_tensor, mappo_hidden, done_mask)
                actions = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
        elif policy["kind"] == "rule_based":
            for i in range(env.cfg.n_agents):
                actions[i] = int(policy["policies"][i].act(obs[i]))
        else:
            for i in range(env.cfg.n_agents):
                actions[i] = int(policy["rngs"][i].integers(0, env.cfg.num_actions))

        action_dict = {agent_id: int(actions[i]) for i, agent_id in enumerate(agent_ids)}
        next_obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(action_dict)
        obs = np.stack([next_obs_dict[agent] for agent in agent_ids], axis=0)
        rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
        prev_done = np.array([float(terminations[a] or truncations[a]) for a in agent_ids], dtype=np.float32)
        latest_info = info_dict[agent_ids[0]]

        episode_rewards += rewards
        step_pickups = int(latest_info.get("targets_collected", 0))
        step_deliveries = int(latest_info.get("food_delivered", 0))
        pickups += step_pickups
        deliveries += step_deliveries
        if discovery_step < 0 and step_pickups > 0:
            discovery_step = int(latest_info.get("episode_length", episode_length + 1))
        pheromone_usage_values.append(float(latest_info.get("pheromone_usage", 0.0)))
        episode_length = int(latest_info.get("episode_length", episode_length + 1))

        if any(terminations.values()) or any(truncations.values()):
            break

    metrics = {
        "mean_episode_reward": float(np.mean(episode_rewards)),
        "food_picked_up": pickups,
        "food_delivered": deliveries,
        "delivery_conversion": float(deliveries / pickups) if pickups > 0 else 0.0,
        "exploration_coverage": float(latest_info.get("exploration_coverage", 0.0)),
        "pheromone_usage": mean_or_zero(pheromone_usage_values),
        "episode_length": episode_length,
        "time_to_first_discovery": discovery_step,
        "carrying_stall_fraction": float(latest_info.get("carrying_stall_fraction", 0.0)),
        "carrying_low_progress_fraction": float(latest_info.get("carrying_low_progress_fraction", 0.0)),
        "non_carrying_nest_loiter_fraction": float(latest_info.get("non_carrying_nest_loiter_fraction", 0.0)),
        "non_carrying_nest_crowding_fraction": float(latest_info.get("non_carrying_nest_crowding_fraction", 0.0)),
        "non_carrying_explore_active_fraction": float(latest_info.get("non_carrying_explore_active_fraction", 0.0)),
        "non_carrying_force_explore_fraction": float(latest_info.get("non_carrying_force_explore_fraction", 0.0)),
        "non_carrying_random_explore_fraction": float(latest_info.get("non_carrying_random_explore_fraction", 0.0)),
    }
    return metrics


def evaluate_condition(
    family: str,
    label: str,
    policy_kind: str,
    cfg: SwarmConfig,
    checkpoint_dir: Path | None,
    seeds: list[int],
    raw_csv_path: Path | None = None,
) -> list[dict[str, Any]]:
    env = SwarmEnv(cfg, headless=True)
    obs_dict, _ = env.reset(seed=seeds[0])
    obs = np.stack([obs_dict[a] for a in env.possible_agents], axis=0)
    policy = build_policy(policy_kind, checkpoint_dir, obs.shape[1], cfg.num_actions, cfg.n_agents, cfg)
    rows = []
    for episode_index, seed in enumerate(seeds, start=1):
        print(f"[episode] family={family} condition={label} episode={episode_index} seed={seed}", flush=True)
        metrics = run_episode(env, policy, seed)
        row = {
            "family": family,
            "condition": label,
            "episode": episode_index,
            "seed": seed,
            "policy_kind": policy_kind,
            "checkpoint_dir": "" if checkpoint_dir is None else str(checkpoint_dir.relative_to(ROOT)),
            "n_agents": cfg.n_agents,
            "width": cfg.width,
            "height": cfg.height,
            "n_obstacles": cfg.n_obstacles,
            "max_steps": cfg.max_steps,
            "pheromone_enabled": bool(cfg.pheromone_enabled),
        }
        row.update(metrics)
        rows.append(row)
        if raw_csv_path is not None:
            append_csv_row(raw_csv_path, row)
    LEAKED_ENVS.append(env)
    return rows


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
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)
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


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def write_analysis_notes(campaign: dict[str, Any]) -> None:
    text = f"""# Experiment Execution Notes

Timestamped campaign folder: `{TIMESTAMP}`

This campaign reruns the strongest feasible evaluation-heavy subset of the GVRSF plan as part of the 30-agent full-bundle extension:

- curriculum learning vs weaker/simplified training
- pheromone ablation
- swarm-size scaling
- robustness under harder environments
- baseline comparison

Why this design:

- The repository already contains multiple strong and weaker MAPPO checkpoints.
- Full multi-condition retraining would be compute-heavy and would reduce scientific rigor if done underpowered.
- The chosen design reuses existing checkpoints where possible and evaluates each condition over {EPISODES_PER_CONDITION} deterministic episodes with matched controls.

Primary current checkpoint:

- `checkpoints/mappo_g/stage3b_full_swarm_final`

Weaker curriculum checkpoint:

- `checkpoints/mappo_full_600k_33/stage3b_full_swarm_final`

Environment matching rule:

- The runner reconstructs `SwarmConfig` from each checkpoint's `metadata.json`.
- This avoids the mismatch risk in the generic CLI evaluators, which do not fully restore stage geometry from metadata.
"""
    (NOTES_DIR / "execution_plan.md").write_text(text, encoding="utf-8")
    (META_DIR / "campaign_manifest.json").write_text(json.dumps(campaign, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    ensure_dirs()

    strong_ckpt = ROOT / "checkpoints" / "mappo_g" / "stage3b_full_swarm_final"
    weak_ckpt = ROOT / "checkpoints" / "mappo_full_600k_33" / "stage3b_full_swarm_final"
    base_meta = strong_ckpt / "metadata.json"
    weak_meta = weak_ckpt / "metadata.json"

    seeds = list(range(SEED_START, SEED_START + EPISODES_PER_CONDITION))
    campaign = {
        "timestamp": TIMESTAMP,
        "episodes_per_condition": EPISODES_PER_CONDITION,
        "seed_range": [seeds[0], seeds[-1]],
        "primary_checkpoint": str(strong_ckpt.relative_to(ROOT)),
        "weaker_checkpoint": str(weak_ckpt.relative_to(ROOT)),
        "families": [],
    }
    write_analysis_notes(campaign)

    all_rows: list[dict[str, Any]] = []
    raw_csv_path = RAW_DIR / "all_episode_results.csv"
    raw_csv_path.write_text("", encoding="utf-8")

    families = []

    # 1. Curriculum vs weaker training.
    families.append(
        (
            "curriculum_vs_weaker_training",
            [
                {
                    "label": "current_curriculum_final",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta),
                    "checkpoint_dir": strong_ckpt,
                },
                {
                    "label": "older_weaker_curriculum_final",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta),
                    "checkpoint_dir": weak_ckpt,
                },
            ],
        )
    )

    # 2. Pheromone ablation.
    families.append(
        (
            "pheromone_ablation",
            [
                {
                    "label": "trained_with_pheromone_eval_with_pheromone",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta, pheromone_enabled=True),
                    "checkpoint_dir": strong_ckpt,
                },
                {
                    "label": "trained_with_pheromone_eval_without_pheromone",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta, pheromone_enabled=False),
                    "checkpoint_dir": strong_ckpt,
                },
            ],
        )
    )

    # 3. Swarm-size scaling.
    scaling_conditions = []
    for n_agents in [1, 2, 3, 4, 6, 10, 15, 20, 30]:
        scaling_conditions.append(
            {
                "label": f"{n_agents}_agents",
                "policy_kind": "mappo_gru",
                "cfg": cfg_from_metadata(base_meta, n_agents=n_agents),
                "checkpoint_dir": strong_ckpt,
            }
        )
    families.append(("swarm_size_scaling", scaling_conditions))

    # 4. Robustness under harder environments.
    families.append(
        (
            "robustness_harder_environments",
            [
                {
                    "label": "control_final_stage",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta),
                    "checkpoint_dir": strong_ckpt,
                },
                {
                    "label": "more_obstacles",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta, n_obstacles=24),
                    "checkpoint_dir": strong_ckpt,
                },
                {
                    "label": "sensor_noise",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta, observation_noise_std=0.05),
                    "checkpoint_dir": strong_ckpt,
                },
                {
                    "label": "failed_agents_2",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta, failed_agent_count=2),
                    "checkpoint_dir": strong_ckpt,
                },
            ],
        )
    )

    # 5. Baseline comparison.
    families.append(
        (
            "baseline_comparison",
            [
                {
                    "label": "current_mappo",
                    "policy_kind": "mappo_gru",
                    "cfg": cfg_from_metadata(base_meta),
                    "checkpoint_dir": strong_ckpt,
                },
                {
                    "label": "rule_based",
                    "policy_kind": "rule_based",
                    "cfg": cfg_from_metadata(base_meta),
                    "checkpoint_dir": None,
                },
                {
                    "label": "random",
                    "policy_kind": "random",
                    "cfg": cfg_from_metadata(base_meta),
                    "checkpoint_dir": None,
                },
            ],
        )
    )

    campaign["families"] = [
        {"family": family_name, "conditions": [cond["label"] for cond in conditions]}
        for family_name, conditions in families
    ]
    write_analysis_notes(campaign)

    for family_name, conditions in families:
        for condition in conditions:
            print(f"[campaign] running family={family_name} condition={condition['label']}")
            rows = evaluate_condition(
                family=family_name,
                label=condition["label"],
                policy_kind=condition["policy_kind"],
                cfg=condition["cfg"],
                checkpoint_dir=condition["checkpoint_dir"],
                seeds=seeds,
                raw_csv_path=raw_csv_path,
            )
            all_rows.extend(rows)

    summary_rows = summarize_rows(all_rows, ["family", "condition", "n_agents", "width", "height", "n_obstacles", "pheromone_enabled"])
    save_csv(TABLES_DIR / "condition_summary.csv", summary_rows)

    by_family = summarize_rows(all_rows, ["family", "condition"])
    save_csv(TABLES_DIR / "family_summary.csv", by_family)

    # Statistical comparisons where they are most relevant.
    stats_rows = []
    pairs = [
        ("curriculum_vs_weaker_training", "current_curriculum_final", "older_weaker_curriculum_final"),
        ("pheromone_ablation", "trained_with_pheromone_eval_with_pheromone", "trained_with_pheromone_eval_without_pheromone"),
        ("baseline_comparison", "current_mappo", "rule_based"),
        ("baseline_comparison", "current_mappo", "random"),
    ]
    for family_name, a_label, b_label in pairs:
        a_vals = [r["food_delivered"] for r in all_rows if r["family"] == family_name and r["condition"] == a_label]
        b_vals = [r["food_delivered"] for r in all_rows if r["family"] == family_name and r["condition"] == b_label]
        stats = welch_ttest(a_vals, b_vals)
        stats_rows.append(
            {
                "family": family_name,
                "condition_a": a_label,
                "condition_b": b_label,
                "metric": "food_delivered",
                "cohens_d": cohens_d(a_vals, b_vals),
                **stats,
            }
        )
    save_csv(TABLES_DIR / "statistical_tests.csv", stats_rows)

    plot_grouped_bars(by_family, "curriculum_vs_weaker_training", "food_delivered", "curriculum_food_delivered.png")
    plot_grouped_bars(by_family, "curriculum_vs_weaker_training", "delivery_conversion", "curriculum_delivery_conversion.png")
    plot_grouped_bars(by_family, "pheromone_ablation", "food_delivered", "pheromone_food_delivered.png")
    plot_grouped_bars(by_family, "baseline_comparison", "food_delivered", "baseline_food_delivered.png")
    plot_grouped_bars(by_family, "robustness_harder_environments", "food_delivered", "robustness_food_delivered.png")
    plot_grouped_bars(by_family, "robustness_harder_environments", "exploration_coverage", "robustness_exploration_coverage.png")

    (META_DIR / "run_complete.json").write_text(
        json.dumps(
            {
                "timestamp": TIMESTAMP,
                "status": "completed",
                "raw_episode_rows": len(all_rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    plot_scaling(summary_rows, "food_delivered", "scaling_food_delivered.png")
    plot_scaling(summary_rows, "delivery_conversion", "scaling_delivery_conversion.png")
    plot_scaling(summary_rows, "exploration_coverage", "scaling_exploration_coverage.png")

    with (META_DIR / "checkpoint_metadata_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "strong_checkpoint_metadata": json.loads(base_meta.read_text()),
                "weaker_checkpoint_metadata": json.loads(weak_meta.read_text()),
            },
            f,
            indent=2,
            sort_keys=True,
        )


if __name__ == "__main__":
    main()
