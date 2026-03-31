from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithms.mappo.inference import load_actor
from env.config import SwarmConfig
from env.swarm_env import SwarmEnv
from models.q_network import QNetwork
from models.rule_based_policy import RuleBasedSwarmPolicy
from policy_debug import make_policy_debug_config, print_policy_debug, should_debug_policy
from train.experiment_utils import CSVLogger, add_env_config_args, make_swarm_config, resolve_filename, resolve_repo_path, write_json


def _progress_interval(max_steps: int) -> int:
    if max_steps <= 0:
        return 100
    return max(1, min(500, max_steps // 5 if max_steps >= 5 else 1))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--policy-kind", choices=["dqn", "mappo_gru", "rule_based"], default="dqn")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs/eval")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--debug-policy", action="store_true")
    parser.add_argument("--debug-policy-agents", type=str, default="")
    parser.add_argument("--debug-policy-max-steps", type=int, default=0)
    add_env_config_args(parser)
    return parser.parse_args(argv)


def _load_models(checkpoint_dir: str, obs_dim: int, action_dim: int, n_agents: int, shared: bool):
    device = torch.device("cpu")
    if shared:
        net = QNetwork(obs_dim, action_dim).to(device)
        net.load_state_dict(torch.load(os.path.join(checkpoint_dir, "shared.pt"), map_location=device))
        nets = [net for _ in range(n_agents)]
    else:
        nets = []
        for i in range(n_agents):
            net = QNetwork(obs_dim, action_dim).to(device)
            net.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"agent_{i}.pt"), map_location=device))
            nets.append(net)
    for net in nets:
        net.eval()
    return nets, device


def _build_rule_based_policies(cfg: SwarmConfig, seed: int):
    return [RuleBasedSwarmPolicy(cfg, seed=seed + i) for i in range(cfg.n_agents)]


def run(args):
    args.output_dir = resolve_repo_path(args.output_dir)
    args.checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    args.max_steps_per_episode = int(getattr(args, "eval_steps", getattr(args, "max_steps_per_episode", 600)))
    args.n_targets = int(getattr(args, "active_targets", getattr(args, "n_targets", 3)))
    args.target_respawn = True
    cfg = make_swarm_config(args)
    env = SwarmEnv(cfg, headless=bool(args.headless))
    obs_dict, _ = env.reset(seed=args.seed)
    agent_ids = env.possible_agents
    obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
    obs_dim = obs.shape[1]
    filename = resolve_filename(args, fallback="eval")
    nets = []
    policies = []
    mappo_actor = None
    device = torch.device("cpu")
    if args.policy_kind == "dqn":
        nets, device = _load_models(args.checkpoint_dir, obs_dim, cfg.num_actions, cfg.n_agents, args.shared_policy)
    elif args.policy_kind == "mappo_gru":
        mappo_actor, device = load_actor(args.checkpoint_dir, obs_dim, cfg.num_actions, device="cpu")
    else:
        policies = _build_rule_based_policies(cfg, args.seed)
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)

    metrics_path = os.path.join(args.output_dir, f"{filename}_eval_metrics.csv")
    logger = CSVLogger(
        metrics_path,
        [
            "filename",
            "episode",
            "seed",
            "checkpoint_path",
            "use_pheromone",
            "active_targets",
            "eval_steps_configured",
            "mean_episode_reward",
            "food_discovered",
            "food_picked_up",
            "food_retrieved",
            "food_delivered",
            "targets_collected",
            "targets_picked_up",
            "exploration_coverage",
            "coverage_efficiency",
            "efficiency",
            "time_to_first_discovery",
            "pheromone_usage",
            "episode_length",
            "total_steps_taken",
            "collisions",
            "new_cells_visited",
            "episode_done_reason",
            "swarm_efficiency",
            "food_source_respawns",
            "food_units_remaining",
        ],
    )
    summaries = []
    progress_interval = _progress_interval(int(cfg.max_steps))

    print(
        f"[evaluate] start filename={filename} policy={args.policy_kind} "
        f"episodes={args.episodes} agents={cfg.n_agents} max_steps={cfg.max_steps} "
        f"pheromone={'on' if cfg.pheromone_enabled else 'off'} output_dir={args.output_dir}"
    )

    try:
        for episode in range(1, args.episodes + 1):
            print(f"[evaluate] episode {episode}/{args.episodes} starting seed={args.seed + episode - 1}")
            obs_dict, _ = env.reset(seed=args.seed + episode - 1)
            obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
            episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)
            food_discovered = 0
            food_retrieved = 0
            exploration_coverage = 0.0
            pheromone_usage_values = []
            episode_length = 0
            collisions = 0
            new_cells_visited = 0
            food_source_respawns = 0
            food_units_remaining = 0
            time_to_first_discovery = -1
            done_reason = ""
            prev_rewards = None
            prev_done_debug = None
            mappo_prev_done = np.zeros((cfg.n_agents,), dtype=np.float32)
            mappo_hidden = None
            if args.policy_kind == "mappo_gru":
                mappo_hidden = mappo_actor.initial_hidden(cfg.n_agents, device)

            while True:
                actions = np.zeros(cfg.n_agents, dtype=np.int64)
                if args.policy_kind == "dqn":
                    for i in range(cfg.n_agents):
                        with torch.no_grad():
                            obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                            q_vals = nets[i](obs_tensor)
                            actions[i] = int(torch.argmax(q_vals, dim=1).item())
                        if should_debug_policy(debug_cfg, episode_length, i, agent_ids[i]):
                            print_policy_debug(
                                step=episode_length,
                                agent_index=i,
                                agent_id=agent_ids[i],
                                policy_label="shared" if args.shared_policy else f"agent_{i}",
                                mode="greedy",
                                output_name="q_values",
                                output_values=q_vals.detach().cpu().numpy(),
                                action=int(actions[i]),
                                num_actions=cfg.num_actions,
                                prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                                prev_done=None if prev_done_debug is None else bool(prev_done_debug[i]),
                            )
                elif args.policy_kind == "mappo_gru":
                    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    done_mask = torch.as_tensor(1.0 - mappo_prev_done, dtype=torch.float32, device=device)
                    with torch.no_grad():
                        logits, mappo_hidden = mappo_actor(obs_tensor, mappo_hidden, done_mask)
                        greedy_actions = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
                    actions[:] = greedy_actions
                    for i in range(cfg.n_agents):
                        if should_debug_policy(debug_cfg, episode_length, i, agent_ids[i]):
                            print_policy_debug(
                                step=episode_length,
                                agent_index=i,
                                agent_id=agent_ids[i],
                                policy_label="mappo_gru",
                                mode="greedy",
                                output_name="policy_logits",
                                output_values=logits[i].detach().cpu().numpy(),
                                action=int(actions[i]),
                                num_actions=cfg.num_actions,
                                prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                                prev_done=bool(mappo_prev_done[i]),
                            )
                else:
                    for i in range(cfg.n_agents):
                        actions[i] = int(policies[i].act(obs[i]))
                        if should_debug_policy(debug_cfg, episode_length, i, agent_ids[i]):
                            print_policy_debug(
                                step=episode_length,
                                agent_index=i,
                                agent_id=agent_ids[i],
                                policy_label="rule_based",
                                mode="rule_based",
                                output_name=None,
                                output_values=None,
                                action=int(actions[i]),
                                num_actions=cfg.num_actions,
                                prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                                prev_done=None if prev_done_debug is None else bool(prev_done_debug[i]),
                            )

                action_dict = {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
                next_obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(action_dict)
                obs = np.stack([next_obs_dict[agent] for agent in agent_ids], axis=0)
                rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
                prev_rewards = rewards
                prev_done_debug = np.array(
                    [bool(terminations[agent] or truncations[agent]) for agent in agent_ids],
                    dtype=np.bool_,
                )
                if args.policy_kind == "mappo_gru":
                    mappo_prev_done = prev_done_debug.astype(np.float32)
                info = info_dict[agent_ids[0]]

                episode_rewards += rewards
                # targets_collected refers to pickup events, not delivery events.
                food_discovered += int(info.get("targets_collected", 0))
                food_retrieved += int(info.get("food_delivered", 0))
                exploration_coverage = max(exploration_coverage, float(info.get("exploration_coverage", 0.0)))
                pheromone_usage_values.append(float(info.get("pheromone_usage", 0.0)))
                episode_length = int(info.get("episode_length", episode_length + 1))
                collisions += int(info.get("collisions", 0))
                new_cells_visited += int(info.get("new_cells_visited", 0))
                food_source_respawns = int(info.get("food_source_respawns", food_source_respawns))
                food_units_remaining = int(info.get("food_units_remaining", food_units_remaining))
                if time_to_first_discovery < 0 and int(info.get("targets_collected", 0)) > 0:
                    time_to_first_discovery = episode_length
                done_reason = str(info.get("episode_done_reason", done_reason))
                if episode_length % progress_interval == 0 and episode_length < int(cfg.max_steps):
                    print(
                        f"[evaluate] episode {episode}/{args.episodes} "
                        f"step {episode_length}/{cfg.max_steps} "
                        f"targets={food_discovered} coverage={exploration_coverage:.3f}"
                    )

                if any(terminations.values()) or any(truncations.values()):
                    break

            mean_reward = float(episode_rewards.mean())
            mean_pheromone = float(np.mean(pheromone_usage_values)) if pheromone_usage_values else 0.0
            efficiency = float(food_retrieved / max(episode_length, 1))
            coverage_efficiency = float(exploration_coverage / max(episode_length, 1))
            row = {
                "filename": filename,
                "episode": episode,
                "seed": args.seed + episode - 1,
                "checkpoint_path": args.checkpoint_dir,
                "use_pheromone": bool(cfg.pheromone_enabled),
                "active_targets": int(cfg.active_targets),
                "eval_steps_configured": int(cfg.max_steps),
                "mean_episode_reward": mean_reward,
                "food_discovered": food_discovered,
                "food_picked_up": food_discovered,
                "food_retrieved": food_retrieved,
                "food_delivered": food_retrieved,
                "targets_collected": food_discovered,
                "targets_picked_up": food_discovered,
                "exploration_coverage": exploration_coverage,
                "coverage_efficiency": coverage_efficiency,
                "efficiency": float(food_discovered / max(cfg.n_agents, 1)),
                "time_to_first_discovery": time_to_first_discovery,
                "pheromone_usage": mean_pheromone,
                "episode_length": episode_length,
                "total_steps_taken": episode_length,
                "collisions": collisions,
                "new_cells_visited": new_cells_visited,
                "episode_done_reason": done_reason,
                "swarm_efficiency": efficiency,
                "food_source_respawns": food_source_respawns,
                "food_units_remaining": food_units_remaining,
            }
            logger.log(row)
            summaries.append(row)
            print(
                f"[evaluate] episode {episode}/{args.episodes} done "
                f"steps={episode_length} targets={food_discovered} delivered={food_retrieved} "
                f"reward={mean_reward:.3f} done_reason={done_reason or 'unknown'}"
            )
    finally:
        logger.close()
        env.close()

    valid_discovery_times = [row["time_to_first_discovery"] for row in summaries if float(row["time_to_first_discovery"]) >= 0]
    write_json(
        os.path.join(args.output_dir, f"{filename}_eval_summary.json"),
        {
            "filename": filename,
            "episodes": args.episodes,
            "obs_dim": obs_dim,
            "policy_kind": args.policy_kind,
            "use_pheromone": bool(cfg.pheromone_enabled),
            "active_targets": int(cfg.active_targets),
            "eval_steps": int(cfg.max_steps),
            "checkpoint_dir": args.checkpoint_dir,
            "metrics_csv": os.path.basename(metrics_path),
            "metrics": {
                "targets_collected_definition": "pickup_events_not_delivery",
                "mean_reward": float(np.mean([row["mean_episode_reward"] for row in summaries])) if summaries else 0.0,
                "mean_food_discovered": float(np.mean([row["food_discovered"] for row in summaries])) if summaries else 0.0,
                "mean_food_retrieved": float(np.mean([row["food_retrieved"] for row in summaries])) if summaries else 0.0,
                "mean_exploration_coverage": float(np.mean([row["exploration_coverage"] for row in summaries])) if summaries else 0.0,
                "mean_coverage_efficiency": float(np.mean([row["coverage_efficiency"] for row in summaries])) if summaries else 0.0,
                "mean_efficiency": float(np.mean([row["efficiency"] for row in summaries])) if summaries else 0.0,
                "mean_time_to_first_discovery": float(np.mean(valid_discovery_times)) if valid_discovery_times else -1.0,
                "mean_pheromone_usage": float(np.mean([row["pheromone_usage"] for row in summaries])) if summaries else 0.0,
                "mean_episode_length": float(np.mean([row["episode_length"] for row in summaries])) if summaries else 0.0,
                "mean_food_source_respawns": float(np.mean([row["food_source_respawns"] for row in summaries])) if summaries else 0.0,
                "mean_food_units_remaining": float(np.mean([row["food_units_remaining"] for row in summaries])) if summaries else 0.0,
            },
        },
    )
    print(
        f"[evaluate] complete episodes={len(summaries)} "
        f"metrics_csv={metrics_path} summary_json={os.path.join(args.output_dir, f'{filename}_eval_summary.json')}"
    )
    return args.output_dir


if __name__ == "__main__":
    run(parse_args())
