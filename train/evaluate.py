from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv
from models.q_network import QNetwork
from train.experiment_utils import CSVLogger, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="runs/eval")
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


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = SwarmConfig(n_agents=args.n_agents)
    env = SwarmEnv(cfg, headless=True)
    obs_dict, _ = env.reset(seed=args.seed)
    agent_ids = env.possible_agents
    obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
    obs_dim = obs.shape[1]
    nets, device = _load_models(args.checkpoint_dir, obs_dim, cfg.num_actions, cfg.n_agents, args.shared_policy)

    logger = CSVLogger(
        os.path.join(args.output_dir, "eval_metrics.csv"),
        [
            "episode",
            "seed",
            "mean_episode_reward",
            "food_retrieved",
            "exploration_coverage",
            "pheromone_usage",
            "episode_length",
            "swarm_efficiency",
        ],
    )
    summaries = []

    try:
        for episode in range(1, args.episodes + 1):
            obs_dict, _ = env.reset(seed=args.seed + episode - 1)
            obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
            episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)
            food_retrieved = 0
            exploration_coverage = 0.0
            pheromone_usage_values = []
            episode_length = 0

            while True:
                actions = np.zeros(cfg.n_agents, dtype=np.int64)
                for i in range(cfg.n_agents):
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                        q_vals = nets[i](obs_tensor)
                        actions[i] = int(torch.argmax(q_vals, dim=1).item())

                action_dict = {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
                next_obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(action_dict)
                obs = np.stack([next_obs_dict[agent] for agent in agent_ids], axis=0)
                rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
                info = info_dict[agent_ids[0]]

                episode_rewards += rewards
                food_retrieved += int(info.get("food_delivered", 0))
                exploration_coverage = max(exploration_coverage, float(info.get("exploration_coverage", 0.0)))
                pheromone_usage_values.append(float(info.get("pheromone_usage", 0.0)))
                episode_length = int(info.get("episode_length", episode_length + 1))

                if any(terminations.values()) or any(truncations.values()):
                    break

            mean_reward = float(episode_rewards.mean())
            mean_pheromone = float(np.mean(pheromone_usage_values)) if pheromone_usage_values else 0.0
            efficiency = float(food_retrieved / max(episode_length, 1))
            row = {
                "episode": episode,
                "seed": args.seed + episode - 1,
                "mean_episode_reward": mean_reward,
                "food_retrieved": food_retrieved,
                "exploration_coverage": exploration_coverage,
                "pheromone_usage": mean_pheromone,
                "episode_length": episode_length,
                "swarm_efficiency": efficiency,
            }
            logger.log(row)
            summaries.append(row)
    finally:
        logger.close()
        env.close()

    write_json(
        os.path.join(args.output_dir, "eval_summary.json"),
        {
            "episodes": args.episodes,
            "obs_dim": obs_dim,
            "metrics": {
                "mean_reward": float(np.mean([row["mean_episode_reward"] for row in summaries])) if summaries else 0.0,
                "mean_food_retrieved": float(np.mean([row["food_retrieved"] for row in summaries])) if summaries else 0.0,
                "mean_exploration_coverage": float(np.mean([row["exploration_coverage"] for row in summaries])) if summaries else 0.0,
                "mean_pheromone_usage": float(np.mean([row["pheromone_usage"] for row in summaries])) if summaries else 0.0,
                "mean_episode_length": float(np.mean([row["episode_length"] for row in summaries])) if summaries else 0.0,
            },
        },
    )


if __name__ == "__main__":
    run(parse_args())

