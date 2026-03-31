from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithms.common.env_api import extract_env_spaces
from algorithms.mappo.curriculum import default_curriculum, select_curriculum
from algorithms.mappo.networks import CentralizedGRUCritic, SharedGRUActor
from env.swarm_env import SwarmEnv
from train.experiment_utils import (
    CSVLogger,
    add_env_config_args,
    make_run_dir,
    make_swarm_config,
    plot_eval_metrics,
    plot_training_metrics,
    resolve_filename,
    resolve_repo_path,
    write_json,
)


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 3e-4
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    eval_episodes: int = 3


@dataclass(frozen=True)
class StagePromotionTarget:
    min_pickups: float = 0.0
    min_deliveries: float = 0.0


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--total-steps", type=int, default=30_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--curriculum", choices=["stage1", "stage1_to_2", "full"], default="full")
    parser.add_argument("--resume-checkpoint", type=str, default="")
    parser.add_argument("--stage-repeat-limit", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    add_env_config_args(parser)
    return parser.parse_args(argv)


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_env(args, stage):
    args_copy = argparse.Namespace(**vars(args))
    args_copy.n_agents = int(stage.n_agents)
    args_copy.n_targets = int(stage.n_targets)
    args_copy.n_obstacles = int(stage.n_obstacles)
    args_copy.max_steps_per_episode = int(stage.max_steps)
    args_copy.active_targets = int(stage.active_targets)
    args_copy.target_respawn = bool(stage.target_respawn)
    args_copy.action_repeat_steps = int(stage.action_repeat_steps)
    args_copy.reward_new_cell = float(stage.reward_new_cell)
    cfg = make_swarm_config(args_copy)
    cfg.width = int(stage.width)
    cfg.height = int(stage.height)
    if stage.reward_step is not None:
        cfg.reward_step = float(stage.reward_step)
    if stage.reward_collision is not None:
        cfg.reward_collision = float(stage.reward_collision)
    if stage.reward_pickup is not None:
        cfg.reward_pickup = float(stage.reward_pickup)
    if stage.reward_nest_approach is not None:
        cfg.reward_nest_approach = float(stage.reward_nest_approach)
    if stage.reward_nest_delivery is not None:
        cfg.reward_nest_delivery = float(stage.reward_nest_delivery)
    if stage.reward_undelivered_food is not None:
        cfg.reward_undelivered_food = float(stage.reward_undelivered_food)
    if stage.reward_food_approach is not None:
        cfg.reward_food_approach = float(stage.reward_food_approach)
    if stage.reward_food_detected is not None:
        cfg.reward_food_detected = float(stage.reward_food_detected)
    if stage.reward_pheromone_follow is not None:
        cfg.reward_pheromone_follow = float(stage.reward_pheromone_follow)
    if stage.carrying_reward_new_cell_scale is not None:
        cfg.carrying_reward_new_cell_scale = float(stage.carrying_reward_new_cell_scale)
    if stage.pheromone_enabled is not None:
        cfg.pheromone_enabled = bool(stage.pheromone_enabled)
        cfg.render_pheromone = bool(stage.pheromone_enabled)
    env = SwarmEnv(cfg, headless=bool(args.headless))
    return cfg, env


def _estimate_stage_state_dim(stage) -> int:
    target_slots = max(int(stage.n_targets), int(stage.active_targets))
    return int(stage.n_agents) * 7 + target_slots * 4 + int(stage.n_obstacles) * 4 + 7


def _critic_state_dim(curriculum) -> int:
    return max(_estimate_stage_state_dim(stage) for stage in curriculum)


def _pad_state(state: np.ndarray, target_dim: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] == target_dim:
        return state.astype(np.float32, copy=False)
    if state.shape[0] > target_dim:
        return state[:target_dim].astype(np.float32, copy=False)
    padded = np.full((target_dim,), -1.0, dtype=np.float32)
    padded[: state.shape[0]] = state
    return padded


def _stage_promotion_target(stage) -> StagePromotionTarget:
    if stage.name == "stage1a_single_agent_miniscule":
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=1.0)
    if stage.name in {"stage1b_single_agent_tiny", "stage1c_single_agent_small"}:
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=1.0)
    if stage.name == "stage1d_single_agent_return_medium":
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=1.5)
    if stage.name == "stage1e_single_agent_delivery_bridge":
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=1.0)
    if stage.name == "stage1f_single_agent_delivery_obstacles":
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=0.8)
    if int(stage.n_agents) == 1:
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=0.5)
    if not bool(stage.target_respawn):
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=0.5)
    if int(stage.n_agents) < 5:
        return StagePromotionTarget(min_pickups=1.0, min_deliveries=0.5)
    return StagePromotionTarget(min_pickups=1.0, min_deliveries=0.2)


def _meets_stage_promotion(stage, eval_metrics) -> bool:
    target = _stage_promotion_target(stage)
    return (
        float(eval_metrics.get("food_picked_up", 0.0)) >= target.min_pickups
        and float(eval_metrics.get("food_retrieved", 0.0)) >= target.min_deliveries
    )


def _greedy_eval_score(eval_metrics) -> float:
    delivered = float(eval_metrics.get("food_retrieved", 0.0))
    picked_up = float(eval_metrics.get("food_picked_up", 0.0))
    reward = float(eval_metrics.get("mean_episode_reward", 0.0))
    coverage = float(eval_metrics.get("exploration_coverage", 0.0))
    return delivered * 1000.0 + picked_up * 100.0 + reward + coverage


def _stage_entropy_coef(stage, stage_steps: int) -> float:
    progress = min(1.0, max(0.0, float(stage_steps) / max(1.0, float(stage.total_steps))))
    start = float(stage.entropy_start)
    end = float(stage.entropy_end)
    return start + (end - start) * progress


def _save_best_checkpoint(
    path,
    actor,
    critic,
    actor_opt,
    critic_opt,
    cfg,
    stage_name,
    global_step,
    hidden_size,
    metadata,
):
    _save_checkpoint(path, actor, critic, actor_opt, critic_opt, cfg, stage_name, global_step, hidden_size)
    write_json(os.path.join(path, "metadata.json"), metadata)


def _evaluate(actor, critic, cfg, critic_state_dim: int, device, episodes: int, seed: int):
    env = SwarmEnv(cfg, headless=True)
    actor.eval()
    critic.eval()
    results = []
    try:
        for ep in range(episodes):
            obs_dict, _ = env.reset(seed=seed + ep)
            agent_ids = env.possible_agents
            obs = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
            actor_hidden = actor.initial_hidden(cfg.n_agents, device)
            critic_hidden = critic.initial_hidden(1, device)
            prev_done = np.zeros((cfg.n_agents,), dtype=np.float32)
            ep_rewards = np.zeros((cfg.n_agents,), dtype=np.float32)
            coverage = 0.0
            pheromone = []
            length = 0
            picked_up = 0
            delivered = 0
            pheromone_deposits = 0
            source_respawns = 0
            food_units_remaining = 0
            first_pickup_step = -1
            first_delivery_step = -1
            while True:
                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                mask_t = torch.as_tensor(1.0 - prev_done, dtype=torch.float32, device=device)
                logits, actor_hidden = actor(obs_t, actor_hidden, mask_t)
                actions = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64)
                action_dict = {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
                next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(action_dict)
                obs = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                padded_state = _pad_state(env.state(), critic_state_dim)
                state_t = torch.as_tensor(padded_state, dtype=torch.float32, device=device).unsqueeze(0)
                critic_mask_t = torch.as_tensor([1.0 - float(prev_done.any())], dtype=torch.float32, device=device)
                _, critic_hidden = critic(state_t, critic_hidden, critic_mask_t)
                rewards = np.array([rewards_dict[a] for a in agent_ids], dtype=np.float32)
                info = infos[agent_ids[0]]
                ep_rewards += rewards
                coverage = max(coverage, float(info.get("exploration_coverage", 0.0)))
                pheromone.append(float(info.get("pheromone_usage", 0.0)))
                length = int(info.get("episode_length", length + 1))
                picked_up += int(info.get("targets_collected", 0))
                delivered += int(info.get("food_delivered", 0))
                pheromone_deposits += int(info.get("pheromone_deposit_events", 0))
                source_respawns = int(info.get("food_source_respawns", source_respawns))
                food_units_remaining = int(info.get("food_units_remaining", food_units_remaining))
                if first_pickup_step < 0 and int(info.get("first_pickup_step", -1)) >= 0:
                    first_pickup_step = int(info.get("first_pickup_step", -1))
                if first_delivery_step < 0 and int(info.get("first_delivery_step", -1)) >= 0:
                    first_delivery_step = int(info.get("first_delivery_step", -1))
                prev_done = np.array([float(terminations[a] or truncations[a]) for a in agent_ids], dtype=np.float32)
                if prev_done.any():
                    break
            results.append(
                {
                    "mean_episode_reward": float(ep_rewards.mean()),
                    "food_discovered": float(picked_up),
                    "food_picked_up": float(picked_up),
                    "food_retrieved": float(delivered),
                    "exploration_coverage": float(coverage),
                    "pheromone_usage": float(np.mean(pheromone)) if pheromone else 0.0,
                    "pheromone_deposit_events": float(pheromone_deposits),
                    "food_source_respawns": float(source_respawns),
                    "food_units_remaining": float(food_units_remaining),
                    "episode_length": float(length),
                    "first_pickup_step": float(first_pickup_step),
                    "first_delivery_step": float(first_delivery_step),
                    "pickup_to_delivery_latency": float(
                        first_delivery_step - first_pickup_step
                        if first_pickup_step >= 0 and first_delivery_step >= 0
                        else -1
                    ),
                    "delivery_conversion": float(delivered / max(picked_up, 1)),
                    "swarm_efficiency": float(delivered / max(length, 1)),
                }
            )
    finally:
        env.close()
        actor.train()
        critic.train()
    return {k: float(np.mean([row[k] for row in results])) for k in results[0]} if results else {}


def _save_checkpoint(path, actor, critic, actor_opt, critic_opt, cfg, stage_name, global_step, hidden_size):
    os.makedirs(path, exist_ok=True)
    torch.save({"state_dict": actor.state_dict(), "hidden_size": hidden_size}, os.path.join(path, "actor.pt"))
    torch.save({"state_dict": critic.state_dict(), "hidden_size": hidden_size}, os.path.join(path, "critic.pt"))
    torch.save(
        {
            "actor_optimizer": actor_opt.state_dict(),
            "critic_optimizer": critic_opt.state_dict(),
            "stage_name": stage_name,
            "global_step": global_step,
            "n_agents": cfg.n_agents,
            "width": cfg.width,
            "height": cfg.height,
            "n_targets": cfg.n_targets,
            "n_obstacles": cfg.n_obstacles,
            "obs_dim": actor.obs_encoder[0].in_features,
            "action_dim": actor.policy_head[-1].out_features,
            "state_dim": critic.state_encoder[0].in_features,
        },
        os.path.join(path, "trainer.pt"),
    )


def _load_checkpoint(path, actor, critic, actor_opt, critic_opt, device):
    actor_payload = torch.load(os.path.join(path, "actor.pt"), map_location=device)
    trainer_payload = torch.load(os.path.join(path, "trainer.pt"), map_location=device)
    actor.load_state_dict(actor_payload["state_dict"])
    actor_opt.load_state_dict(trainer_payload["actor_optimizer"])
    critic_state_loaded = False
    if int(trainer_payload.get("state_dim", -1)) == critic.state_encoder[0].in_features:
        critic_payload = torch.load(os.path.join(path, "critic.pt"), map_location=device)
        critic.load_state_dict(critic_payload["state_dict"])
        critic_opt.load_state_dict(trainer_payload["critic_optimizer"])
        critic_state_loaded = True
    trainer_payload["critic_state_loaded"] = critic_state_loaded
    return trainer_payload


def _compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_adv = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def _flatten_rollout(rollout):
    flat = {}
    for key, value in rollout.items():
        flat[key] = value.reshape(-1, *value.shape[2:]) if value.ndim > 2 else value.reshape(-1)
    return flat


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def train(args):
    _set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    filename = resolve_filename(args, fallback="mappo_gru")
    run_dir = make_run_dir(args.output_dir, filename)
    checkpoint_root = os.path.join(resolve_repo_path(args.save_dir), filename)
    os.makedirs(checkpoint_root, exist_ok=True)

    mappo_cfg = MAPPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        lr=args.lr,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        hidden_size=args.hidden_size,
        eval_episodes=args.eval_episodes,
    )

    curriculum = select_curriculum(default_curriculum(args.n_agents), args.curriculum, args.total_steps)
    critic_state_dim = _critic_state_dim(curriculum)
    episode_logger = CSVLogger(
        os.path.join(run_dir, "episode_metrics.csv"),
        [
            "stage",
            "stage_index",
            "stage_attempt",
            "global_step",
            "episode",
            "n_agents",
            "width",
            "height",
            "n_targets",
            "n_obstacles",
            "mean_episode_reward",
            "food_discovered",
            "food_picked_up",
            "food_retrieved",
            "exploration_coverage",
            "pheromone_usage",
            "pheromone_deposit_events",
            "food_source_respawns",
            "food_units_remaining",
            "episode_length",
            "first_pickup_step",
            "first_delivery_step",
            "pickup_to_delivery_latency",
            "delivery_conversion",
            "swarm_efficiency",
        ],
    )
    eval_logger = CSVLogger(
        os.path.join(run_dir, "eval_metrics.csv"),
        [
            "stage",
            "stage_attempt",
            "global_step",
            "width",
            "height",
            "n_targets",
            "n_obstacles",
            "mean_episode_reward",
            "food_discovered",
            "food_picked_up",
            "food_retrieved",
            "exploration_coverage",
            "pheromone_usage",
            "pheromone_deposit_events",
            "food_source_respawns",
            "food_units_remaining",
            "episode_length",
            "first_pickup_step",
            "first_delivery_step",
            "pickup_to_delivery_latency",
            "delivery_conversion",
            "swarm_efficiency",
        ],
    )

    global_step = 0
    completed_episodes = 0
    resume_path = resolve_repo_path(args.resume_checkpoint) if args.resume_checkpoint else ""
    train_start_time = time.time()
    best_greedy_eval_score = float("-inf")
    best_greedy_eval_metadata = None

    write_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "algorithm": "recurrent_mappo_gru",
            "curriculum_mode": args.curriculum,
            "stages": [stage.__dict__ for stage in curriculum],
            "seed": args.seed,
            "hidden_size": args.hidden_size,
            "rollout_steps": args.rollout_steps,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "food_source_capacity": args.food_source_capacity,
            "stage_repeat_limit": args.stage_repeat_limit,
            "critic_state_dim": critic_state_dim,
            "domain_randomization_note": "Environment resets already randomize layout/configuration.",
        },
    )

    bootstrap_cfg, bootstrap_env = _build_env(args, curriculum[0])
    bootstrap_spaces = extract_env_spaces(bootstrap_env)
    bootstrap_env.close()
    actor = SharedGRUActor(bootstrap_spaces.obs_dim, bootstrap_spaces.action_dim, hidden_size=mappo_cfg.hidden_size).to(device)
    critic = CentralizedGRUCritic(critic_state_dim, hidden_size=mappo_cfg.hidden_size).to(device)
    actor_opt = optim.Adam(actor.parameters(), lr=mappo_cfg.lr)
    critic_opt = optim.Adam(critic.parameters(), lr=mappo_cfg.lr)
    if resume_path:
        _load_checkpoint(resume_path, actor, critic, actor_opt, critic_opt, device)
        resume_path = ""

    for stage_index, stage in enumerate(curriculum, start=1):
        stage_attempt = 0
        while True:
            stage_attempt += 1
            stage_start_step = global_step
            stage_start_time = time.time()
            cfg, env = _build_env(args, stage)
            obs_dict, _ = env.reset(seed=args.seed + stage_index - 1 + (stage_attempt - 1) * 10_000)
            spaces = extract_env_spaces(env)

            agent_ids = env.possible_agents
            obs = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
            state = _pad_state(env.state(), critic_state_dim)
            actor_hidden = actor.initial_hidden(cfg.n_agents, device)
            critic_hidden = critic.initial_hidden(1, device)
            prev_done = np.zeros((cfg.n_agents,), dtype=np.float32)
            stage_steps = 0
            stage_episode = 0
            episode_reward = np.zeros((cfg.n_agents,), dtype=np.float32)
            episode_pheromone = []
            episode_coverage = 0.0
            episode_food_picked_up = 0
            episode_food_delivered = 0
            episode_pheromone_deposit_events = 0
            episode_food_source_respawns = 0
            episode_food_units_remaining = 0
            episode_length = 0
            first_pickup_step = -1
            first_delivery_step = -1
            stage_best_actor_state = copy.deepcopy(actor.state_dict())
            stage_best_critic_state = copy.deepcopy(critic.state_dict())
            stage_best_actor_opt_state = copy.deepcopy(actor_opt.state_dict())
            stage_best_critic_opt_state = copy.deepcopy(critic_opt.state_dict())
            stage_best_eval_score = float("-inf")
            sampled_reward_sum = 0.0
            sampled_pickups_sum = 0.0
            sampled_deliveries_sum = 0.0
            sampled_episode_count = 0

            print(
                f"[MAPPO] Stage {stage_index}/{len(curriculum)} {stage.name} attempt={stage_attempt} | "
                f"agents={cfg.n_agents} | target_steps={stage.total_steps} | "
                f"size={cfg.width}x{cfg.height} | targets={cfg.n_targets} | obstacles={cfg.n_obstacles} | "
                f"action_repeat={cfg.action_repeat_steps} | reward_new_cell={cfg.reward_new_cell:.4f} | "
                f"carrying_new_cell_scale={cfg.carrying_reward_new_cell_scale:.2f} | "
                f"reward_step={cfg.reward_step:.4f} | reward_collision={cfg.reward_collision:.2f} | "
                f"reward_pickup={cfg.reward_pickup:.2f} | reward_nest_approach={cfg.reward_nest_approach:.2f} | "
                f"reward_delivery={cfg.reward_nest_delivery:.2f} | "
                f"reward_undelivered={cfg.reward_undelivered_food:.2f} | "
                f"pheromone={'on' if cfg.pheromone_enabled else 'off'} | "
                f"entropy={stage.entropy_start:.4f}->{stage.entropy_end:.4f} | "
                f"obs_dim={spaces.obs_dim} | action_dim={spaces.action_dim} | "
                f"env_state_dim={spaces.state_dim} | critic_state_dim={critic_state_dim}"
            )

            while stage_steps < stage.total_steps:
                rollout = {
                    "obs": np.zeros((mappo_cfg.rollout_steps, cfg.n_agents, spaces.obs_dim), dtype=np.float32),
                    "state": np.zeros((mappo_cfg.rollout_steps, critic_state_dim), dtype=np.float32),
                    "actions": np.zeros((mappo_cfg.rollout_steps, cfg.n_agents), dtype=np.int64),
                    "log_probs": np.zeros((mappo_cfg.rollout_steps, cfg.n_agents), dtype=np.float32),
                    "values": np.zeros((mappo_cfg.rollout_steps,), dtype=np.float32),
                    "rewards": np.zeros((mappo_cfg.rollout_steps,), dtype=np.float32),
                    "dones": np.zeros((mappo_cfg.rollout_steps,), dtype=np.float32),
                    "actor_hidden": np.zeros((mappo_cfg.rollout_steps, cfg.n_agents, mappo_cfg.hidden_size), dtype=np.float32),
                    "critic_hidden": np.zeros((mappo_cfg.rollout_steps, 1, mappo_cfg.hidden_size), dtype=np.float32),
                    "masks": np.zeros((mappo_cfg.rollout_steps, cfg.n_agents), dtype=np.float32),
                    "critic_masks": np.zeros((mappo_cfg.rollout_steps, 1), dtype=np.float32),
                }

                collected = 0
                for t in range(mappo_cfg.rollout_steps):
                    rollout["obs"][t] = obs
                    rollout["state"][t] = state
                    rollout["actor_hidden"][t] = actor_hidden.squeeze(0).detach().cpu().numpy()
                    rollout["critic_hidden"][t] = critic_hidden.squeeze(0).detach().cpu().numpy()[None, :]
                    rollout["masks"][t] = 1.0 - prev_done
                    rollout["critic_masks"][t, 0] = 1.0 - float(prev_done.any())

                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    actor_mask_t = torch.as_tensor(1.0 - prev_done, dtype=torch.float32, device=device)
                    critic_mask_t = torch.as_tensor([1.0 - float(prev_done.any())], dtype=torch.float32, device=device)

                    logits, actor_hidden = actor(obs_t, actor_hidden, actor_mask_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    actions = dist.sample()
                    log_probs = dist.log_prob(actions)
                    value, critic_hidden = critic(state_t, critic_hidden, critic_mask_t)

                    action_dict = {agent: int(actions[i].item()) for i, agent in enumerate(agent_ids)}
                    next_obs_dict, rewards_dict, terminations, truncations, infos = env.step(action_dict)
                    next_obs = np.stack([next_obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                    next_state = _pad_state(env.state(), critic_state_dim)
                    rewards = np.array([rewards_dict[a] for a in agent_ids], dtype=np.float32)
                    done_flags = np.array([float(terminations[a] or truncations[a]) for a in agent_ids], dtype=np.float32)
                    done = float(done_flags.any())
                    info = infos[agent_ids[0]]

                    rollout["actions"][t] = actions.detach().cpu().numpy()
                    rollout["log_probs"][t] = log_probs.detach().cpu().numpy()
                    rollout["values"][t] = float(value.item())
                    rollout["rewards"][t] = float(rewards.mean())
                    rollout["dones"][t] = done

                    obs = next_obs
                    state = next_state
                    prev_done = done_flags
                    global_step += cfg.n_agents
                    stage_steps += cfg.n_agents
                    collected += 1
                    episode_reward += rewards
                    episode_pheromone.append(float(info.get("pheromone_usage", 0.0)))
                    episode_coverage = max(episode_coverage, float(info.get("exploration_coverage", 0.0)))
                    episode_food_picked_up += int(info.get("targets_collected", 0))
                    episode_food_delivered += int(info.get("food_delivered", 0))
                    episode_pheromone_deposit_events += int(info.get("pheromone_deposit_events", 0))
                    episode_food_source_respawns = int(info.get("food_source_respawns", episode_food_source_respawns))
                    episode_food_units_remaining = int(info.get("food_units_remaining", episode_food_units_remaining))
                    episode_length = int(info.get("episode_length", episode_length + 1))
                    if first_pickup_step < 0 and int(info.get("first_pickup_step", -1)) >= 0:
                        first_pickup_step = int(info.get("first_pickup_step", -1))
                    if first_delivery_step < 0 and int(info.get("first_delivery_step", -1)) >= 0:
                        first_delivery_step = int(info.get("first_delivery_step", -1))

                    if done:
                        stage_episode += 1
                        completed_episodes += 1
                        mean_episode_reward = float(episode_reward.mean())
                        pheromone_usage = float(np.mean(episode_pheromone)) if episode_pheromone else 0.0
                        swarm_efficiency = float(episode_food_delivered / max(episode_length, 1))
                        episode_logger.log(
                            {
                                "stage": stage.name,
                                "stage_index": stage_index,
                                "stage_attempt": stage_attempt,
                                "global_step": global_step,
                                "episode": completed_episodes,
                                "n_agents": cfg.n_agents,
                                "width": cfg.width,
                                "height": cfg.height,
                                "n_targets": cfg.n_targets,
                                "n_obstacles": cfg.n_obstacles,
                                "mean_episode_reward": mean_episode_reward,
                                "food_discovered": float(episode_food_picked_up),
                                "food_picked_up": float(episode_food_picked_up),
                                "food_retrieved": float(episode_food_delivered),
                                "exploration_coverage": float(episode_coverage),
                                "pheromone_usage": pheromone_usage,
                                "pheromone_deposit_events": int(episode_pheromone_deposit_events),
                                "food_source_respawns": int(episode_food_source_respawns),
                                "food_units_remaining": int(episode_food_units_remaining),
                                "episode_length": int(episode_length),
                                "first_pickup_step": int(first_pickup_step),
                                "first_delivery_step": int(first_delivery_step),
                                "pickup_to_delivery_latency": int(
                                    first_delivery_step - first_pickup_step
                                    if first_pickup_step >= 0 and first_delivery_step >= 0
                                    else -1
                                ),
                                "delivery_conversion": float(episode_food_delivered / max(episode_food_picked_up, 1)),
                                "swarm_efficiency": swarm_efficiency,
                            }
                        )
                        sampled_reward_sum += mean_episode_reward
                        sampled_pickups_sum += float(episode_food_picked_up)
                        sampled_deliveries_sum += float(episode_food_delivered)
                        sampled_episode_count += 1
                        elapsed = time.time() - train_start_time
                        steps_per_sec = global_step / max(elapsed, 1e-6)
                        remaining_steps = max(args.total_steps - global_step, 0)
                        eta_seconds = remaining_steps / max(steps_per_sec, 1e-6)
                        print(
                            f"[MAPPO] {stage.name} attempt={stage_attempt} ep={completed_episodes} stage_ep={stage_episode} "
                            f"step={global_step}/{args.total_steps} reward={mean_episode_reward:.2f} "
                            f"picked_up={episode_food_picked_up} delivered={episode_food_delivered} "
                            f"deposits={episode_pheromone_deposit_events} respawns={episode_food_source_respawns} "
                            f"first_pickup={first_pickup_step} first_delivery={first_delivery_step} "
                            f"coverage={episode_coverage:.3f} "
                            f"pheromone={pheromone_usage:.3f} len={episode_length} "
                            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)}"
                        )
                        obs_dict, _ = env.reset(seed=args.seed + completed_episodes + stage_index + (stage_attempt - 1) * 10_000)
                        obs = np.stack([obs_dict[a] for a in agent_ids], axis=0).astype(np.float32)
                        state = _pad_state(env.state(), critic_state_dim)
                        actor_hidden = actor.initial_hidden(cfg.n_agents, device)
                        critic_hidden = critic.initial_hidden(1, device)
                        prev_done = np.zeros((cfg.n_agents,), dtype=np.float32)
                        episode_reward.fill(0.0)
                        episode_pheromone = []
                        episode_coverage = 0.0
                        episode_food_picked_up = 0
                        episode_food_delivered = 0
                        episode_pheromone_deposit_events = 0
                        episode_food_source_respawns = 0
                        episode_food_units_remaining = 0
                        episode_length = 0
                        first_pickup_step = -1
                        first_delivery_step = -1

                    if stage_steps >= stage.total_steps:
                        break

                with torch.no_grad():
                    critic_mask_t = torch.as_tensor([1.0 - float(prev_done.any())], dtype=torch.float32, device=device)
                    next_value, _ = critic(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0), critic_hidden, critic_mask_t)
                    next_value = float(next_value.item())

                advantages, returns = _compute_gae(
                    rollout["rewards"][:collected],
                    rollout["values"][:collected],
                    rollout["dones"][:collected],
                    next_value,
                    mappo_cfg.gamma,
                    mappo_cfg.gae_lambda,
                )
                rollout["advantages"] = advantages[:, None].repeat(cfg.n_agents, axis=1)
                rollout["returns"] = returns

                obs_batch = torch.as_tensor(rollout["obs"][:collected], dtype=torch.float32, device=device)
                actions_batch = torch.as_tensor(rollout["actions"][:collected], dtype=torch.int64, device=device)
                old_log_probs_batch = torch.as_tensor(rollout["log_probs"][:collected], dtype=torch.float32, device=device)
                values_batch = torch.as_tensor(rollout["values"][:collected], dtype=torch.float32, device=device)
                state_batch = torch.as_tensor(rollout["state"][:collected], dtype=torch.float32, device=device)
                returns_batch = torch.as_tensor(rollout["returns"][:collected], dtype=torch.float32, device=device)
                advantages_batch = torch.as_tensor(rollout["advantages"][:collected], dtype=torch.float32, device=device)
                masks_batch = torch.as_tensor(rollout["masks"][:collected], dtype=torch.float32, device=device)
                critic_masks_batch = torch.as_tensor(rollout["critic_masks"][:collected], dtype=torch.float32, device=device).squeeze(-1)
                actor_hidden_batch = torch.as_tensor(rollout["actor_hidden"][:collected], dtype=torch.float32, device=device)
                critic_hidden_batch = torch.as_tensor(rollout["critic_hidden"][:collected], dtype=torch.float32, device=device)

                adv_mean = advantages_batch.mean()
                adv_std = advantages_batch.std(unbiased=False) + 1e-8
                advantages_batch = (advantages_batch - adv_mean) / adv_std

                flat_indices = np.arange(collected)
                for _ in range(mappo_cfg.update_epochs):
                    np.random.shuffle(flat_indices)
                    for start in range(0, collected, max(1, min(mappo_cfg.minibatch_size, collected))):
                        idx = flat_indices[start : start + max(1, min(mappo_cfg.minibatch_size, collected))]
                        mb_obs = obs_batch[idx].reshape(-1, spaces.obs_dim)
                        mb_actions = actions_batch[idx].reshape(-1)
                        mb_old_log_probs = old_log_probs_batch[idx].reshape(-1)
                        mb_advantages = advantages_batch[idx].reshape(-1)
                        mb_masks = masks_batch[idx].reshape(-1)
                        mb_actor_hidden = actor_hidden_batch[idx].reshape(-1, mappo_cfg.hidden_size).unsqueeze(0)
                        entropy_coef = _stage_entropy_coef(stage, stage_steps)

                        logits, _ = actor(mb_obs, mb_actor_hidden, mb_masks)
                        dist = torch.distributions.Categorical(logits=logits)
                        new_log_probs = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()
                        ratio = torch.exp(new_log_probs - mb_old_log_probs)
                        surrogate1 = ratio * mb_advantages
                        surrogate2 = torch.clamp(ratio, 1.0 - mappo_cfg.clip_ratio, 1.0 + mappo_cfg.clip_ratio) * mb_advantages
                        actor_loss = -torch.min(surrogate1, surrogate2).mean() - entropy_coef * entropy

                        mb_states = state_batch[idx]
                        mb_returns = returns_batch[idx]
                        mb_values = values_batch[idx]
                        mb_critic_masks = critic_masks_batch[idx]
                        mb_critic_hidden = critic_hidden_batch[idx].reshape(-1, mappo_cfg.hidden_size).unsqueeze(0)
                        critic_values, _ = critic(mb_states, mb_critic_hidden, mb_critic_masks)
                        value_clipped = mb_values + (critic_values - mb_values).clamp(-mappo_cfg.clip_ratio, mappo_cfg.clip_ratio)
                        value_loss = 0.5 * torch.max(
                            F.mse_loss(critic_values, mb_returns, reduction="none"),
                            F.mse_loss(value_clipped, mb_returns, reduction="none"),
                        ).mean()

                        actor_opt.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), mappo_cfg.max_grad_norm)
                        actor_opt.step()

                        critic_opt.zero_grad()
                        (mappo_cfg.value_coef * value_loss).backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), mappo_cfg.max_grad_norm)
                        critic_opt.step()

                if args.eval_every > 0 and global_step > 0 and global_step % args.eval_every < cfg.n_agents:
                    eval_metrics = _evaluate(actor, critic, cfg, critic_state_dim, device, args.eval_episodes, args.seed + 10_000 + stage_index)
                    if eval_metrics:
                        eval_logger.log(
                            {
                                "stage": stage.name,
                                "stage_attempt": stage_attempt,
                                "global_step": global_step,
                                "width": cfg.width,
                                "height": cfg.height,
                                "n_targets": cfg.n_targets,
                                "n_obstacles": cfg.n_obstacles,
                                "food_discovered": eval_metrics["food_discovered"],
                                **eval_metrics,
                            }
                        )
                        eval_score = _greedy_eval_score(eval_metrics)
                        if eval_score > stage_best_eval_score:
                            stage_best_eval_score = eval_score
                            stage_best_actor_state = copy.deepcopy(actor.state_dict())
                            stage_best_critic_state = copy.deepcopy(critic.state_dict())
                            stage_best_actor_opt_state = copy.deepcopy(actor_opt.state_dict())
                            stage_best_critic_opt_state = copy.deepcopy(critic_opt.state_dict())
                        print(
                            f"[MAPPO][Eval] {stage.name} attempt={stage_attempt} step={global_step} "
                            f"reward={eval_metrics['mean_episode_reward']:.2f} "
                            f"picked_up={eval_metrics['food_picked_up']:.2f} "
                            f"delivered={eval_metrics['food_retrieved']:.2f} "
                            f"conversion={eval_metrics['delivery_conversion']:.2f} "
                            f"gap_pickup={max(0.0, (sampled_pickups_sum / max(sampled_episode_count, 1)) - eval_metrics['food_picked_up']):.2f} "
                            f"gap_delivery={max(0.0, (sampled_deliveries_sum / max(sampled_episode_count, 1)) - eval_metrics['food_retrieved']):.2f} "
                            f"deposits={eval_metrics['pheromone_deposit_events']:.2f} "
                            f"respawns={eval_metrics['food_source_respawns']:.2f} "
                            f"first_pickup={eval_metrics['first_pickup_step']:.1f} "
                            f"first_delivery={eval_metrics['first_delivery_step']:.1f} "
                            f"coverage={eval_metrics['exploration_coverage']:.3f} "
                            f"pheromone={eval_metrics['pheromone_usage']:.3f} "
                            f"len={eval_metrics['episode_length']:.1f}"
                        )

            stage_end_eval = _evaluate(
                actor,
                critic,
                cfg,
                critic_state_dim,
                device,
                max(1, args.eval_episodes),
                args.seed + 20_000 + stage_index + stage_attempt,
            )
            stage_end_score = _greedy_eval_score(stage_end_eval) if stage_end_eval else float("-inf")
            if stage_end_eval:
                eval_logger.log(
                    {
                        "stage": stage.name,
                        "stage_attempt": stage_attempt,
                        "global_step": global_step,
                        "width": cfg.width,
                        "height": cfg.height,
                        "n_targets": cfg.n_targets,
                        "n_obstacles": cfg.n_obstacles,
                        "food_discovered": stage_end_eval["food_discovered"],
                        **stage_end_eval,
                    }
                )
                if stage_end_score > stage_best_eval_score:
                    stage_best_eval_score = stage_end_score
                    stage_best_actor_state = copy.deepcopy(actor.state_dict())
                    stage_best_critic_state = copy.deepcopy(critic.state_dict())
                    stage_best_actor_opt_state = copy.deepcopy(actor_opt.state_dict())
                    stage_best_critic_opt_state = copy.deepcopy(critic_opt.state_dict())

            actor.load_state_dict(stage_best_actor_state)
            critic.load_state_dict(stage_best_critic_state)
            actor_opt.load_state_dict(stage_best_actor_opt_state)
            critic_opt.load_state_dict(stage_best_critic_opt_state)

            stage_promoted = bool(stage_end_eval) and _meets_stage_promotion(stage, stage_end_eval)
            promotion_target = _stage_promotion_target(stage)
            sampled_mean_reward = sampled_reward_sum / max(sampled_episode_count, 1)
            sampled_mean_pickups = sampled_pickups_sum / max(sampled_episode_count, 1)
            sampled_mean_deliveries = sampled_deliveries_sum / max(sampled_episode_count, 1)
            sampled_delivery_conversion = sampled_mean_deliveries / max(sampled_mean_pickups, 1.0)
            current_entropy_coef = _stage_entropy_coef(stage, stage_steps)

            metadata = {
                "algorithm": "recurrent_mappo_gru",
                "stage": stage.name,
                "stage_index": stage_index,
                "stage_attempt": stage_attempt,
                "global_step": global_step,
                "n_agents": cfg.n_agents,
                "width": cfg.width,
                "height": cfg.height,
                "n_targets": cfg.n_targets,
                "n_obstacles": cfg.n_obstacles,
                "max_steps": cfg.max_steps,
                "active_targets": cfg.active_targets,
                "target_respawn": bool(cfg.target_respawn),
                "food_source_capacity": int(cfg.food_source_capacity),
                "action_repeat_steps": int(cfg.action_repeat_steps),
                "reward_new_cell": float(cfg.reward_new_cell),
                "carrying_reward_new_cell_scale": float(cfg.carrying_reward_new_cell_scale),
                "reward_step": float(cfg.reward_step),
                "reward_collision": float(cfg.reward_collision),
                "reward_pickup": float(cfg.reward_pickup),
                "reward_nest_approach": float(cfg.reward_nest_approach),
                "reward_nest_delivery": float(cfg.reward_nest_delivery),
                "reward_undelivered_food": float(cfg.reward_undelivered_food),
                "reward_food_approach": float(cfg.reward_food_approach),
                "reward_food_detected": float(cfg.reward_food_detected),
                "reward_pheromone_follow": float(cfg.reward_pheromone_follow),
                "pheromone_enabled": bool(cfg.pheromone_enabled),
                "obs_dim": spaces.obs_dim,
                "action_dim": spaces.action_dim,
                "env_state_dim": spaces.state_dim,
                "critic_state_dim": critic_state_dim,
                "hidden_size": mappo_cfg.hidden_size,
                "entropy_start": float(stage.entropy_start),
                "entropy_end": float(stage.entropy_end),
                "final_entropy_coef": float(current_entropy_coef),
                "decentralized_execution": True,
                "curriculum_actor_transfer": True,
                "curriculum_critic_transfer": True,
                "promotion_min_pickups": promotion_target.min_pickups,
                "promotion_min_deliveries": promotion_target.min_deliveries,
                "stage_promoted": stage_promoted,
                "sampled_mean_episode_reward": float(sampled_mean_reward),
                "sampled_mean_food_picked_up": float(sampled_mean_pickups),
                "sampled_mean_food_retrieved": float(sampled_mean_deliveries),
                "sampled_delivery_conversion": float(sampled_delivery_conversion),
                "stage_end_eval": stage_end_eval,
                "recommended_demo_checkpoint": "best_greedy_eval",
            }

            stage_ckpt_dir = os.path.join(checkpoint_root, stage.name)
            _save_best_checkpoint(
                stage_ckpt_dir,
                actor,
                critic,
                actor_opt,
                critic_opt,
                cfg,
                stage.name,
                global_step,
                mappo_cfg.hidden_size,
                metadata,
            )
            latest_ckpt_dir = os.path.join(checkpoint_root, "latest")
            _save_best_checkpoint(
                latest_ckpt_dir,
                actor,
                critic,
                actor_opt,
                critic_opt,
                cfg,
                stage.name,
                global_step,
                mappo_cfg.hidden_size,
                metadata,
            )

            if stage_end_eval and stage_end_score > best_greedy_eval_score:
                best_greedy_eval_score = stage_end_score
                best_greedy_eval_metadata = {
                    **metadata,
                    "best_greedy_eval_score": best_greedy_eval_score,
                    "best_greedy_eval_stage": stage.name,
                }
                best_ckpt_dir = os.path.join(checkpoint_root, "best_greedy_eval")
                _save_best_checkpoint(
                    best_ckpt_dir,
                    actor,
                    critic,
                    actor_opt,
                    critic_opt,
                    cfg,
                    stage.name,
                    global_step,
                    mappo_cfg.hidden_size,
                    best_greedy_eval_metadata,
                )

            stage_elapsed = time.time() - stage_start_time
            print(
                f"[MAPPO] Completed {stage.name} attempt={stage_attempt} | "
                f"stage_steps={global_step - stage_start_step} | episodes={stage_episode} | "
                f"sampled_reward={sampled_mean_reward:.2f} | sampled_pickup={sampled_mean_pickups:.2f} | "
                f"sampled_delivery={sampled_mean_deliveries:.2f} | sampled_conversion={sampled_delivery_conversion:.2f} | "
                f"greedy_reward={float(stage_end_eval.get('mean_episode_reward', 0.0)) if stage_end_eval else 0.0:.2f} | "
                f"greedy_pickup={float(stage_end_eval.get('food_picked_up', 0.0)) if stage_end_eval else 0.0:.2f} | "
                f"greedy_delivery={float(stage_end_eval.get('food_retrieved', 0.0)) if stage_end_eval else 0.0:.2f} | "
                f"greedy_conversion={float(stage_end_eval.get('delivery_conversion', 0.0)) if stage_end_eval else 0.0:.2f} | "
                f"promoted={stage_promoted} | best_eval_score={stage_best_eval_score:.2f} | "
                f"elapsed={_format_duration(stage_elapsed)}"
            )
            env.close()
            if stage_promoted or stage_attempt > args.stage_repeat_limit:
                if not stage_promoted:
                    print(
                        f"[MAPPO] Advancing despite unmet stage target after {stage_attempt} attempt(s): "
                        f"required pick_up>={promotion_target.min_pickups:.2f}, "
                        f"delivery>={promotion_target.min_deliveries:.2f}, "
                        f"got pick_up={float(stage_end_eval.get('food_picked_up', 0.0)) if stage_end_eval else 0.0:.2f}, "
                        f"delivery={float(stage_end_eval.get('food_retrieved', 0.0)) if stage_end_eval else 0.0:.2f}"
                    )
                break
            print(
                f"[MAPPO] Repeating stage {stage.name} because greedy eval did not meet promotion target: "
                f"required pick_up>={promotion_target.min_pickups:.2f}, "
                f"delivery>={promotion_target.min_deliveries:.2f}, "
                f"got pick_up={float(stage_end_eval.get('food_picked_up', 0.0)) if stage_end_eval else 0.0:.2f}, "
                f"delivery={float(stage_end_eval.get('food_retrieved', 0.0)) if stage_end_eval else 0.0:.2f}"
            )

    episode_logger.close()
    eval_logger.close()
    if not args.no_plots:
        plot_training_metrics(os.path.join(run_dir, "episode_metrics.csv"), run_dir)
        if os.path.exists(os.path.join(run_dir, "eval_metrics.csv")):
            plot_eval_metrics(os.path.join(run_dir, "eval_metrics.csv"), run_dir)
    if best_greedy_eval_metadata is not None:
        write_json(os.path.join(run_dir, "best_greedy_eval.json"), best_greedy_eval_metadata)


if __name__ == "__main__":
    train(parse_args())
