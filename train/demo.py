from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pygame
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv
from algorithms.mappo.inference import load_actor
from models.q_network import QNetwork
from policy_debug import make_policy_debug_config, print_policy_debug, should_debug_policy
from train.experiment_utils import add_env_config_args, make_swarm_config


def parse_args(argv=None):
    # 1) Parse CLI args (checkpoint location, backend, shared policy flag, agent count, seed).
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["custom", "sb3", "rllib", "mappo"], default="custom")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--sb3-model", type=str, default="checkpoints/sb3_dqn.zip")
    parser.add_argument("--rllib-checkpoint", type=str, default="checkpoints/rllib_dqn")
    parser.add_argument(
        "--ray-tmpdir",
        type=str,
        default="",
        help="Override Ray temp dir (useful to avoid /tmp space or socket path length issues).",
    )
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--n-agents", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="Exit after N steps (0 = run until window closed)")
    parser.add_argument(
        "--demo-epsilon",
        type=float,
        default=0.0,
        help="Probability of taking a random action during demo to mimic exploratory behavior.",
    )
    parser.add_argument("--debug-policy", action="store_true")
    parser.add_argument("--debug-policy-agents", type=str, default="")
    parser.add_argument("--debug-policy-max-steps", type=int, default=0)
    add_env_config_args(parser)
    return parser.parse_args(argv)


def load_models(checkpoint_dir: str, obs_dim: int, action_dim: int, n_agents: int, shared: bool, device):
    if shared:
        net = QNetwork(obs_dim, action_dim).to(device)
        path = os.path.join(checkpoint_dir, "shared.pt")
        net.load_state_dict(torch.load(path, map_location=device))
        nets = [net for _ in range(n_agents)]
    else:
        nets = []
        for i in range(n_agents):
            net = QNetwork(obs_dim, action_dim).to(device)
            path = os.path.join(checkpoint_dir, f"agent_{i}.pt")
            net.load_state_dict(torch.load(path, map_location=device))
            nets.append(net)
    for net in nets:
        net.eval()
    return nets


def _custom_demo(env, obs, agent_ids, args):
    # Infer model dimensions and device.
    obs_dim = obs.shape[1]
    action_dim = env.cfg.num_actions
    device = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    next_reset_seed = args.seed + 1
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = None

    nets = load_models(args.checkpoint_dir, obs_dim, action_dim, env.cfg.n_agents, args.shared_policy, device)

    running = True
    steps = 0
    while running:
        if not args.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        actions = np.zeros(env.cfg.n_agents, dtype=np.int64)
        for i in range(env.cfg.n_agents):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = nets[i](obs_tensor)
                greedy_action = int(torch.argmax(q_vals, dim=1).item())
            if random.random() < args.demo_epsilon:
                actions[i] = np.random.randint(0, action_dim)
                mode = "explore"
            else:
                actions[i] = greedy_action
                mode = "greedy"
            if should_debug_policy(debug_cfg, steps, i, agent_ids[i]):
                print_policy_debug(
                    step=steps,
                    agent_index=i,
                    agent_id=agent_ids[i],
                    policy_label="shared" if args.shared_policy else f"agent_{i}",
                    epsilon=args.demo_epsilon,
                    mode=mode,
                    output_name="q_values",
                    output_values=q_vals.detach().cpu().numpy(),
                    action=int(actions[i]),
                    num_actions=action_dim,
                    prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                    prev_done=None if prev_done is None else bool(prev_done[i]),
                )

        action_dict = {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
        prev_rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
        prev_done = np.array(
            [bool(terminations[agent] or truncations[agent]) for agent in agent_ids],
            dtype=np.bool_,
        )
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        if not args.headless:
            env.render(fps=60)
        steps += 1
        if args.max_steps and steps >= args.max_steps:
            running = False
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=next_reset_seed)
            obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
            next_reset_seed += 1

    return obs


def _sb3_demo(env, obs_dict, agent_ids, args):
    from stable_baselines3 import DQN

    model = DQN.load(args.sb3_model, device="cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    next_reset_seed = args.seed + 1
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = None

    running = True
    steps = 0
    while running:
        if not args.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        action_dict = {}
        for i, agent in enumerate(agent_ids):
            q_values = None
            if args.debug_policy:
                obs_tensor = torch.as_tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    q_values = model.q_net(obs_tensor).detach().cpu().numpy()
            action, _ = model.predict(obs_dict[agent], deterministic=True)
            greedy_action = int(action)
            if random.random() < args.demo_epsilon:
                action_dict[agent] = int(np.random.randint(0, env.cfg.num_actions))
                mode = "explore"
            else:
                action_dict[agent] = greedy_action
                mode = "greedy"
            if should_debug_policy(debug_cfg, steps, i, agent):
                print_policy_debug(
                    step=steps,
                    agent_index=i,
                    agent_id=agent,
                    policy_label="shared",
                    epsilon=args.demo_epsilon,
                    mode=mode,
                    output_name="q_values",
                    output_values=q_values,
                    action=int(action_dict[agent]),
                    num_actions=env.cfg.num_actions,
                    prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                    prev_done=None if prev_done is None else bool(prev_done[i]),
                )

        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        prev_rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
        prev_done = np.array(
            [bool(terminations[agent] or truncations[agent]) for agent in agent_ids],
            dtype=np.bool_,
        )
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        if not args.headless:
            env.render(fps=60)
        steps += 1
        if args.max_steps and steps >= args.max_steps:
            running = False
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=next_reset_seed)
            next_reset_seed += 1

    return obs_dict


def _rllib_demo(env, obs_dict, agent_ids, args):
    import os

    os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
    if args.ray_tmpdir:
        os.environ["RAY_TMPDIR"] = args.ray_tmpdir

    import ray
    from ray.rllib.algorithms.algorithm import Algorithm
    import ray.rllib.algorithms.algorithm as alg_module
    import ray.rllib.algorithms.dqn.dqn as dqn_module
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
    from ray.tune.registry import register_env

    try:
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False, _skip_env_hook=True)
    except TypeError:
        ray.init(address="local", ignore_reinit_error=True, include_dashboard=False)

    # Patch RLlib old-stack replay buffer type handling (coerce class -> string).
    orig_create = alg_module.Algorithm._create_local_replay_buffer_if_necessary

    def _patched_create(self, cfg):
        rb_cfg = cfg.get("replay_buffer_config", {})
        rb_type = rb_cfg.get("type")
        if isinstance(rb_type, type):
            rb_cfg["type"] = rb_type.__name__
        return orig_create(self, cfg)

    alg_module.Algorithm._create_local_replay_buffer_if_necessary = _patched_create

    # Bypass strict validation for replay buffer type on this Ray version.
    dqn_module.DQNConfig.validate = lambda self: None

    def env_creator(_):
        cfg = make_swarm_config(args)
        return ParallelPettingZooEnv(SwarmEnv(cfg, headless=args.headless))

    register_env("swarm_pz", env_creator)
    checkpoint_path = args.rllib_checkpoint
    if not checkpoint_path.startswith(("file://", "s3://", "gs://")):
        checkpoint_path = os.path.abspath(checkpoint_path)
        checkpoint_path = f"file://{checkpoint_path}"
    algo = Algorithm.from_checkpoint(checkpoint_path)

    running = True
    steps = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    next_reset_seed = args.seed + 1
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = None
    while running:
        if not args.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        action_dict = {}
        for i, agent in enumerate(agent_ids):
            action = int(algo.compute_single_action(obs_dict[agent], policy_id="shared_policy"))
            if random.random() < args.demo_epsilon:
                action_dict[agent] = int(np.random.randint(0, env.cfg.num_actions))
                mode = "explore"
            else:
                action_dict[agent] = action
                mode = "greedy"
            if should_debug_policy(debug_cfg, steps, i, agent):
                print_policy_debug(
                    step=steps,
                    agent_index=i,
                    agent_id=agent,
                    policy_label="shared_policy",
                    epsilon=args.demo_epsilon,
                    mode=mode,
                    output_name=None,
                    output_values=None,
                    action=int(action_dict[agent]),
                    num_actions=env.cfg.num_actions,
                    prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                    prev_done=None if prev_done is None else bool(prev_done[i]),
                )

        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        prev_rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
        prev_done = np.array(
            [bool(terminations[agent] or truncations[agent]) for agent in agent_ids],
            dtype=np.bool_,
        )
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        if not args.headless:
            env.render(fps=60)
        steps += 1
        if args.max_steps and steps >= args.max_steps:
            running = False
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=next_reset_seed)
            next_reset_seed += 1

    algo.stop()
    ray.shutdown()
    return obs_dict


def _mappo_demo(env, obs_dict, agent_ids, args):
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.cfg.num_actions
    actor, device = load_actor(args.checkpoint_dir, obs_dim, action_dim, device="cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    next_reset_seed = args.seed + 1
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = np.zeros((env.cfg.n_agents,), dtype=np.float32)
    hidden_state = actor.initial_hidden(env.cfg.n_agents, device)

    running = True
    steps = 0
    while running:
        if not args.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0).astype(np.float32)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        done_mask = torch.as_tensor(1.0 - prev_done, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, hidden_state = actor(obs_tensor, hidden_state, done_mask)
            greedy_actions = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)

        action_dict = {}
        for i, agent in enumerate(agent_ids):
            if random.random() < args.demo_epsilon:
                action_dict[agent] = int(np.random.randint(0, env.cfg.num_actions))
                mode = "explore"
            else:
                action_dict[agent] = int(greedy_actions[i])
                mode = "greedy"
            if should_debug_policy(debug_cfg, steps, i, agent):
                print_policy_debug(
                    step=steps,
                    agent_index=i,
                    agent_id=agent,
                    policy_label="mappo_gru",
                    epsilon=args.demo_epsilon,
                    mode=mode,
                    output_name="policy_logits",
                    output_values=logits[i].detach().cpu().numpy(),
                    action=int(action_dict[agent]),
                    num_actions=env.cfg.num_actions,
                    prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                    prev_done=bool(prev_done[i]),
                )

        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        prev_rewards = np.array([rewards_dict[agent] for agent in agent_ids], dtype=np.float32)
        prev_done = np.array(
            [float(terminations[agent] or truncations[agent]) for agent in agent_ids],
            dtype=np.float32,
        )
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        if not args.headless:
            env.render(fps=60)
        steps += 1
        if args.max_steps and steps >= args.max_steps:
            running = False
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=next_reset_seed)
            next_reset_seed += 1
            hidden_state = actor.initial_hidden(env.cfg.n_agents, device)
            prev_done = np.zeros((env.cfg.n_agents,), dtype=np.float32)

    return obs_dict


def main():
    args = parse_args()

    # 2) Build config + environment, then reset to get initial observations.
    cfg = make_swarm_config(args)
    env = SwarmEnv(cfg, headless=args.headless)
    obs_dict, _ = env.reset(seed=args.seed)
    agent_ids = env.possible_agents
    obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
    if args.headless and args.max_steps <= 0:
        args.max_steps = int(cfg.max_steps)
    if not args.headless:
        env.render(fps=60) # render first frame

    if args.backend == "custom":
        _custom_demo(env, obs, agent_ids, args)
    elif args.backend == "sb3":
        _sb3_demo(env, obs_dict, agent_ids, args)
    elif args.backend == "rllib":
        _rllib_demo(env, obs_dict, agent_ids, args)
    elif args.backend == "mappo":
        _mappo_demo(env, obs_dict, agent_ids, args)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    env.close()


if __name__ == "__main__":
    main()
