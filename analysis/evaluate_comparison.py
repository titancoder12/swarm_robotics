from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.swarm_env import SwarmEnv
from models.q_network import QNetwork
from train.experiment_utils import add_env_config_args, make_swarm_config, resolve_filename, resolve_repo_path, sanitize_filename, write_json


CONDITIONS = [
    {
        "comparison_label": "trained_with_pheromone__eval_with_pheromone",
        "checkpoint_attr": "checkpoint_with_pheromone",
        "trained_with_pheromone": True,
        "eval_with_pheromone": True,
    },
    {
        "comparison_label": "trained_without_pheromone__eval_without_pheromone",
        "checkpoint_attr": "checkpoint_without_pheromone",
        "trained_with_pheromone": False,
        "eval_with_pheromone": False,
    },
    {
        "comparison_label": "trained_with_pheromone__eval_without_pheromone",
        "checkpoint_attr": "checkpoint_with_pheromone",
        "trained_with_pheromone": True,
        "eval_with_pheromone": False,
    },
    {
        "comparison_label": "random_walk",
        "checkpoint_attr": None,
        "trained_with_pheromone": False,
        "eval_with_pheromone": False,
        "is_random_policy": True,
    },
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-with-pheromone", type=str, required=True)
    parser.add_argument("--checkpoint-without-pheromone", type=str, required=True)
    parser.add_argument("--shared-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--agent-min", type=int, default=1)
    parser.add_argument("--agent-max", type=int, default=30)
    parser.add_argument("--agent-step", type=int, default=1)
    parser.add_argument("--episodes-per-agent", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="experiments/experiment_data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Alias for eval-steps/fixed evaluation horizon.")
    add_env_config_args(parser)
    return parser.parse_args(argv)


class RandomPolicy:
    """Uniform random policy over the existing discrete action space."""

    def __init__(self, action_dim: int, seed: int):
        self.action_dim = int(action_dim)
        self.rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray) -> int:
        del obs
        return int(self.rng.integers(0, self.action_dim))

def _resolve_checkpoint_file(path: str, shared_policy: bool, agent_index: int = 0) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "shared.pt" if shared_policy else f"agent_{agent_index}.pt")
        if not os.path.exists(candidate):
            raise FileNotFoundError(f"Checkpoint file not found: {candidate}")
        return candidate
    if shared_policy:
        return path
    raise FileNotFoundError(f"Independent-policy evaluation requires a checkpoint directory, got: {path}")


def _load_models(checkpoint_path: str, obs_dim: int, action_dim: int, n_agents: int, shared_policy: bool):
    device = torch.device("cpu")
    if shared_policy:
        model_path = _resolve_checkpoint_file(checkpoint_path, shared_policy=True)
        net = QNetwork(obs_dim, action_dim).to(device)
        net.load_state_dict(torch.load(model_path, map_location=device))
        nets = [net for _ in range(n_agents)]
    else:
        nets = []
        for agent_index in range(n_agents):
            model_path = _resolve_checkpoint_file(checkpoint_path, shared_policy=False, agent_index=agent_index)
            net = QNetwork(obs_dim, action_dim).to(device)
            net.load_state_dict(torch.load(model_path, map_location=device))
            nets.append(net)
    for net in nets:
        net.eval()
    return nets, device


def _build_cfg(args, n_agents: int, eval_with_pheromone: bool):
    cfg_args = argparse.Namespace(**vars(args))
    cfg_args.n_agents = int(n_agents)
    cfg_args.use_pheromone = bool(eval_with_pheromone)
    cfg_args.pheromone_disabled = not bool(eval_with_pheromone)
    eval_steps = int(args.max_steps) if int(getattr(args, "max_steps", 0)) > 0 else int(args.eval_steps)
    cfg_args.max_steps_per_episode = eval_steps
    cfg_args.n_targets = int(args.active_targets)
    cfg_args.target_respawn = True
    cfg = make_swarm_config(cfg_args)
    cfg.max_steps = eval_steps
    cfg.n_targets = int(args.active_targets)
    cfg.active_targets = int(args.active_targets)
    cfg.target_respawn = True
    return cfg


def _save_exploration_visual(path_png: str, path_pdf: str, env: SwarmEnv, title: str) -> None:
    import matplotlib.pyplot as plt

    grid = env.coverage_grid.astype(np.float32) if env.coverage_grid is not None else np.zeros((1, 1), dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.imshow(grid, cmap="magma", origin="lower", interpolation="nearest")
    if env.cfg.nest_enabled and env.coverage_grid is not None:
        cell = max(env.cfg.coverage_cell_size, 1)
        ax.scatter(
            [env.nest_position[0] / cell],
            [env.nest_position[1] / cell],
            c="cyan",
            s=70,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            label="Nest",
        )
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_xlabel("Coverage Grid X")
    ax.set_ylabel("Coverage Grid Y")
    fig.tight_layout()
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def _progress_interval(max_steps: int) -> int:
    if max_steps <= 0:
        return 100
    return max(1, min(500, max_steps // 5 if max_steps >= 5 else 1))


def _run_episode(
    env: SwarmEnv,
    nets,
    device,
    seed: int,
    random_policy: RandomPolicy | None = None,
    progress_label: str = "",
):
    obs_dict, _ = env.reset(seed=seed)
    agent_ids = env.possible_agents
    obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)

    episode_rewards = np.zeros(env.cfg.n_agents, dtype=np.float32)
    targets_collected = 0
    food_delivered = 0
    coverage = 0.0
    pheromone_usage_values = []
    collisions = 0
    new_cells_visited = 0
    episode_length = 0
    time_to_first_discovery = -1
    done_reason = ""
    progress_interval = _progress_interval(int(env.cfg.max_steps))

    while True:
        actions = np.zeros(env.cfg.n_agents, dtype=np.int64)
        for index in range(env.cfg.n_agents):
            if random_policy is not None:
                actions[index] = random_policy.act(obs[index])
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs[index], dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = nets[index](obs_tensor)
                    actions[index] = int(torch.argmax(q_vals, dim=1).item())

        action_dict = {agent: int(actions[index]) for index, agent in enumerate(agent_ids)}
        next_obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(action_dict)
        obs = np.stack([next_obs_dict[agent] for agent in agent_ids], axis=0)
        rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
        info = info_dict[agent_ids[0]]

        episode_rewards += rewards
        picked_up = int(info.get("targets_collected", 0))
        delivered = int(info.get("food_delivered", 0))
        # targets_collected refers to pickup events, not delivery events.
        targets_collected += picked_up
        food_delivered += delivered
        coverage = max(coverage, float(info.get("exploration_coverage", 0.0)))
        pheromone_usage_values.append(float(info.get("pheromone_usage", 0.0)))
        collisions += int(info.get("collisions", 0))
        new_cells_visited += int(info.get("new_cells_visited", 0))
        episode_length = int(info.get("episode_length", episode_length + 1))
        if time_to_first_discovery < 0 and picked_up > 0:
            time_to_first_discovery = episode_length
        done_reason = str(info.get("episode_done_reason", done_reason))
        if progress_label and episode_length % progress_interval == 0 and episode_length < int(env.cfg.max_steps):
            print(
                f"[compare] {progress_label} "
                f"step {episode_length}/{env.cfg.max_steps} "
                f"targets={targets_collected} coverage={coverage:.3f}"
            )

        if any(terminations.values()) or any(truncations.values()):
            break

    return {
        "mean_episode_reward": float(episode_rewards.mean()),
        "total_reward": float(episode_rewards.sum()),
        "targets_collected": int(targets_collected),
        "targets_picked_up": int(targets_collected),
        "food_discovered": int(targets_collected),
        "food_picked_up": int(targets_collected),
        "food_retrieved": int(food_delivered),
        "food_delivered": int(food_delivered),
        "coverage": float(coverage),
        "exploration_coverage": float(coverage),
        "coverage_efficiency": float(coverage / max(episode_length, 1)),
        "efficiency": float(targets_collected / max(env.cfg.n_agents, 1)),
        "time_to_first_discovery": int(time_to_first_discovery),
        "pheromone_usage": float(np.mean(pheromone_usage_values)) if pheromone_usage_values else 0.0,
        "collisions": int(collisions),
        "new_cells_visited": int(new_cells_visited),
        "episode_length": int(episode_length),
        "total_steps_taken": int(episode_length),
        "episode_done_reason": done_reason,
        "swarm_efficiency": float(food_delivered / max(episode_length, 1)),
    }


def _write_csv(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    normalized_fieldnames = list(fieldnames)
    known_fields = set(normalized_fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in known_fields:
                normalized_fieldnames.append(key)
                known_fields.add(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=normalized_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["comparison_label"]), int(row["number_of_agents"]))].append(row)

    summary_rows = []
    metric_keys = [
        "targets_collected",
        "coverage_efficiency",
        "efficiency",
        "time_to_first_discovery",
        "total_reward",
        "pheromone_usage",
        "collisions",
        "coverage",
        "food_delivered",
        "swarm_efficiency",
    ]
    for (label, n_agents), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary = {
            "comparison_label": label,
            "number_of_agents": n_agents,
            "episodes": len(bucket),
        }
        for key in metric_keys:
            values = np.array([float(row[key]) for row in bucket if key != "time_to_first_discovery" or float(row[key]) >= 0], dtype=np.float32)
            if values.size == 0:
                summary[f"mean_{key}"] = float("nan")
                summary[f"std_{key}"] = float("nan")
            else:
                summary[f"mean_{key}"] = float(values.mean())
                summary[f"std_{key}"] = float(values.std())
        summary_rows.append(summary)
    return summary_rows


def _save_comparison_plot(
    summary_rows: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    filename_root: str,
    out_png_dir: str,
    out_pdf_dir: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = [condition["comparison_label"] for condition in CONDITIONS]
    for label in labels:
        rows = [row for row in summary_rows if row["comparison_label"] == label]
        if not rows:
            continue
        xs = [int(row["number_of_agents"]) for row in rows]
        ys = [float(row[f"mean_{metric_key}"]) for row in rows]
        ax.plot(xs, ys, marker="o", linewidth=2, label=label)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    os.makedirs(out_png_dir, exist_ok=True)
    os.makedirs(out_pdf_dir, exist_ok=True)
    fig.savefig(os.path.join(out_png_dir, f"{filename_root}.png"))
    fig.savefig(os.path.join(out_pdf_dir, f"{filename_root}.pdf"))
    plt.close(fig)


def run(args):
    args.output_dir = resolve_repo_path(args.output_dir)
    args.checkpoint_with_pheromone = resolve_repo_path(args.checkpoint_with_pheromone)
    args.checkpoint_without_pheromone = resolve_repo_path(args.checkpoint_without_pheromone)
    filename = resolve_filename(args, fallback="pheromone_comparison")
    raw_dir = os.path.join(args.output_dir, "raw")
    exploration_png_dir = os.path.join(args.output_dir, "exploration_graphs", "PNG")
    exploration_pdf_dir = os.path.join(args.output_dir, "exploration_graphs", "PDF")
    graph_png_dir = os.path.join(args.output_dir, "graphs", "PNG")
    graph_pdf_dir = os.path.join(args.output_dir, "graphs", "PDF")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(exploration_png_dir, exist_ok=True)
    os.makedirs(exploration_pdf_dir, exist_ok=True)
    os.makedirs(graph_png_dir, exist_ok=True)
    os.makedirs(graph_pdf_dir, exist_ok=True)

    representative_sizes = {size for size in (1, 5, 10, 20, 30) if args.agent_min <= size <= args.agent_max}
    raw_rows: list[dict] = []
    failures: list[dict] = []
    saved_exploration: set[tuple[str, int]] = set()
    total_conditions = len(CONDITIONS)
    total_agent_sizes = len(range(args.agent_min, args.agent_max + 1, args.agent_step))
    total_episodes = total_conditions * total_agent_sizes * args.episodes_per_agent
    completed_episodes = 0

    print(
        f"[compare] start filename={filename} conditions={total_conditions} "
        f"agent_sizes={total_agent_sizes} episodes_per_agent={args.episodes_per_agent} "
        f"total_episodes={total_episodes} output_dir={args.output_dir}"
    )

    for condition_index, condition in enumerate(CONDITIONS):
        checkpoint_path = getattr(args, condition["checkpoint_attr"]) if condition.get("checkpoint_attr") else ""
        print(
            f"[compare] condition {condition_index + 1}/{total_conditions} "
            f"label={condition['comparison_label']} "
            f"eval_pheromone={'on' if condition['eval_with_pheromone'] else 'off'} "
            f"policy={'random_walk' if condition.get('is_random_policy', False) else 'checkpoint'}"
        )
        for n_agents in range(args.agent_min, args.agent_max + 1, args.agent_step):
            cfg = _build_cfg(args, n_agents=n_agents, eval_with_pheromone=condition["eval_with_pheromone"])
            env = SwarmEnv(cfg, headless=True)
            try:
                print(
                    f"[compare] running label={condition['comparison_label']} "
                    f"agents={n_agents} episodes={args.episodes_per_agent} max_steps={cfg.max_steps}"
                )
                obs_dict, _ = env.reset(seed=args.seed + condition_index * 100_000 + n_agents * 1_000)
                obs_dim = np.stack([obs_dict[agent] for agent in env.possible_agents], axis=0).shape[1]
                nets = []
                device = torch.device("cpu")
                random_policy = None
                if condition.get("is_random_policy", False):
                    random_policy = RandomPolicy(cfg.num_actions, seed=args.seed + condition_index * 100_000 + n_agents)
                else:
                    nets, device = _load_models(checkpoint_path, obs_dim, cfg.num_actions, cfg.n_agents, args.shared_policy)
                for episode_index in range(args.episodes_per_agent):
                    episode_seed = args.seed + condition_index * 100_000 + n_agents * 1_000 + episode_index
                    print(
                        f"[compare] episode {episode_index + 1}/{args.episodes_per_agent} "
                        f"label={condition['comparison_label']} agents={n_agents} seed={episode_seed}"
                    )
                    metrics = _run_episode(
                        env,
                        nets,
                        device,
                        episode_seed,
                        random_policy=random_policy,
                        progress_label=(
                            f"label={condition['comparison_label']} "
                            f"agents={n_agents} episode={episode_index + 1}/{args.episodes_per_agent}"
                        ),
                    )
                    row = {
                        "filename": filename,
                        "comparison_label": condition["comparison_label"],
                        "checkpoint_path": checkpoint_path,
                        "trained_with_pheromone": bool(condition["trained_with_pheromone"]),
                        "eval_with_pheromone": bool(condition["eval_with_pheromone"]),
                        "is_random_policy": bool(condition.get("is_random_policy", False)),
                        "number_of_agents": int(n_agents),
                        "episode_index": int(episode_index),
                        "seed": int(episode_seed),
                        "active_targets": int(cfg.active_targets),
                        "eval_steps_configured": int(cfg.max_steps),
                        **metrics,
                    }
                    raw_rows.append(row)
                    completed_episodes += 1
                    print(
                        f"[compare] done {completed_episodes}/{total_episodes} "
                        f"label={condition['comparison_label']} agents={n_agents} "
                        f"episode={episode_index + 1}/{args.episodes_per_agent} "
                        f"targets={metrics['targets_collected']} reward={metrics['total_reward']:.3f} "
                        f"done_reason={metrics['episode_done_reason'] or 'unknown'}"
                    )

                    marker = (condition["comparison_label"], n_agents)
                    if n_agents in representative_sizes and episode_index == 0 and marker not in saved_exploration:
                        slug = sanitize_filename(condition["comparison_label"])
                        png_path = os.path.join(exploration_png_dir, f"{filename}_{slug}_agents_{n_agents}.png")
                        pdf_path = os.path.join(exploration_pdf_dir, f"{filename}_{slug}_agents_{n_agents}.pdf")
                        _save_exploration_visual(
                            png_path,
                            pdf_path,
                            env,
                            title=f"{condition['comparison_label']} | agents={n_agents}",
                        )
                        saved_exploration.add(marker)
            except Exception as exc:
                failures.append(
                    {
                        "comparison_label": condition["comparison_label"],
                        "checkpoint_path": checkpoint_path,
                        "number_of_agents": int(n_agents),
                        "error": str(exc),
                    }
                )
                print(
                    f"[compare] failure label={condition['comparison_label']} "
                    f"agents={n_agents} error={exc}"
                )
            finally:
                env.close()

    fieldnames = [
        "filename",
        "comparison_label",
        "checkpoint_path",
        "trained_with_pheromone",
        "eval_with_pheromone",
        "is_random_policy",
        "number_of_agents",
        "episode_index",
        "seed",
        "active_targets",
        "eval_steps_configured",
        "total_steps_taken",
        "targets_collected",
        "targets_picked_up",
        "coverage",
        "exploration_coverage",
        "coverage_efficiency",
        "efficiency",
        "time_to_first_discovery",
        "total_reward",
        "mean_episode_reward",
        "pheromone_usage",
        "collisions",
        "new_cells_visited",
        "food_discovered",
        "food_picked_up",
        "food_retrieved",
        "food_delivered",
        "swarm_efficiency",
        "episode_length",
        "episode_done_reason",
    ]
    master_raw_path = os.path.join(raw_dir, f"{filename}_all_conditions_raw.csv")
    _write_csv(master_raw_path, fieldnames, raw_rows)

    for condition in CONDITIONS:
        slug = sanitize_filename(condition["comparison_label"])
        rows = [row for row in raw_rows if row["comparison_label"] == condition["comparison_label"]]
        _write_csv(os.path.join(raw_dir, f"{filename}_{slug}_raw.csv"), fieldnames, rows)

    summary_rows = _aggregate_rows(raw_rows)
    summary_path = os.path.join(raw_dir, f"{filename}_summary.csv")
    if summary_rows:
        _write_csv(summary_path, list(summary_rows[0].keys()), summary_rows)

    _save_comparison_plot(summary_rows, "targets_collected", "Mean Targets Collected", "Agents vs Targets Collected in Time", f"{filename}_agents_vs_targets_collected", graph_png_dir, graph_pdf_dir)
    _save_comparison_plot(summary_rows, "coverage_efficiency", "Mean Coverage Efficiency", "Coverage Efficiency vs Agents", f"{filename}_coverage_efficiency_vs_agents", graph_png_dir, graph_pdf_dir)
    _save_comparison_plot(summary_rows, "efficiency", "Mean Efficiency", "Efficiency vs Agents", f"{filename}_efficiency_vs_agents", graph_png_dir, graph_pdf_dir)
    _save_comparison_plot(summary_rows, "time_to_first_discovery", "Mean Time to First Discovery", "Time to First Discovery vs Agents", f"{filename}_time_to_first_discovery_vs_agents", graph_png_dir, graph_pdf_dir)

    write_json(
        os.path.join(args.output_dir, f"{filename}_metadata.json"),
        {
            "filename": filename,
            "checkpoints": {
                "checkpoint_with_pheromone": args.checkpoint_with_pheromone,
                "checkpoint_without_pheromone": args.checkpoint_without_pheromone,
            },
            "targets_collected_definition": "pickup_events_not_delivery",
            "agent_range": {
                "min": int(args.agent_min),
                "max": int(args.agent_max),
                "step": int(args.agent_step),
            },
            "episodes_per_agent": int(args.episodes_per_agent),
            "eval_steps": int(args.eval_steps),
            "active_targets": int(args.active_targets),
            "failures": failures,
            "raw_csv": os.path.relpath(master_raw_path, args.output_dir),
            "summary_csv": os.path.relpath(summary_path, args.output_dir) if summary_rows else "",
        },
    )
    print(
        f"[compare] complete episodes={completed_episodes}/{total_episodes} "
        f"raw_csv={master_raw_path} summary_csv={summary_path if summary_rows else 'none'} "
        f"failures={len(failures)}"
    )
    return args.output_dir


if __name__ == "__main__":
    run(parse_args())
