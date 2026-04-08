from __future__ import annotations

import argparse
import json
import os
import random
import sys
import math

import numpy as np
import pygame
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig
from env.swarm_env import SwarmEnv
from algorithms.mappo.inference import load_actor
from firmware.command_center_client import CommandCenterTCPClient
from firmware.relay_client import CommandCenterRelayClient
from models.q_network import QNetwork
from policy_debug import make_policy_debug_config, print_policy_debug, should_debug_policy
from train.experiment_utils import add_env_config_args, make_swarm_config


def parse_args(argv=None):
    # 1) Parse CLI args (checkpoint location, backend, shared policy flag, agent count, seed).
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["custom", "sb3", "rllib", "mappo", "random"], default="custom")
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
    parser.add_argument(
        "--seed-list",
        type=str,
        default="",
        help="Comma-separated reset seeds to cycle through during demo (for example: 3,17,45).",
    )
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
    parser.add_argument("--mc-relay-url", type=str, default="")
    parser.add_argument("--mc-relay-session", type=str, default="sim_demo")
    parser.add_argument("--mc-relay-timeout", type=float, default=1.0)
    parser.add_argument("--mc-relay-debug", action="store_true")
    parser.add_argument("--mc-tcp-host", type=str, default="")
    parser.add_argument("--mc-tcp-port", type=int, default=8765)
    parser.add_argument("--mc-tcp-timeout", type=float, default=1.0)
    parser.add_argument("--mc-tcp-debug", action="store_true")
    add_env_config_args(parser)
    parser.set_defaults(n_obstacles=18)
    return parser.parse_args(argv)


class SimulatorMissionControlRelayPublisher:
    """Write-only simulator telemetry publisher for Mission Control relay mode."""

    def __init__(self, relay_url: str, session: str, timeout_s: float = 1.0, debug: bool = False) -> None:
        self.client = CommandCenterRelayClient(
            relay_url=relay_url,
            session=session,
            timeout_s=timeout_s,
            debug=debug,
        )

    @staticmethod
    def _to_command_center_pose(env: SwarmEnv, agent_state) -> tuple[float, float, float]:
        nest_x, nest_y = env.nest_position
        x_cm = float(agent_state.x) - float(nest_x)
        y_cm = float(nest_y) - float(agent_state.y)
        heading_deg = -math.degrees(float(agent_state.theta))
        return x_cm, y_cm, heading_deg

    def publish_positions(self, env: SwarmEnv, agent_ids: list[str]) -> None:
        for i, agent_id in enumerate(agent_ids):
            if i >= len(env.agent_states) or i in env.failed_agent_indices:
                continue
            x_cm, y_cm, heading_deg = self._to_command_center_pose(env, env.agent_states[i])
            self.client.send_position(agent_id, x_cm, y_cm, heading_deg)

    def publish_step(self, env: SwarmEnv, agent_ids: list[str], chosen_actions: np.ndarray, prev_nest_distances: np.ndarray) -> None:
        self.publish_positions(env, agent_ids)
        if not bool(getattr(env.cfg, "pheromone_enabled", False)):
            return

        current_nest_distances = env._compute_nest_distance_state()
        for i, agent_id in enumerate(agent_ids):
            if i >= len(env.agent_states) or i in env.failed_agent_indices:
                continue
            _, _, deposit_requested = env.action_table[int(chosen_actions[i])]
            if not deposit_requested:
                continue
            agent = env.agent_states[i]
            if env.cfg.pheromone_requires_food and not agent.carrying_food:
                continue
            if env.cfg.pheromone_deposit_requires_nest_progress:
                prev_dist = float(prev_nest_distances[i])
                curr_dist = float(current_nest_distances[i])
                if not (np.isfinite(prev_dist) and np.isfinite(curr_dist)) or curr_dist >= prev_dist:
                    continue
            amount = float(env.cfg.pheromone_deposit)
            if agent.carrying_food:
                amount *= float(env.cfg.pheromone_deposit_carrying_scale)
            x_cm, y_cm, _heading_deg = self._to_command_center_pose(env, agent)
            self.client.deposit_pheromone(agent_id, x_cm, y_cm, amount)

    def close(self) -> None:
        self.client.close()


class SimulatorMissionControlTCPPublisher:
    """Write-only simulator telemetry publisher for direct Mission Control TCP mode."""

    def __init__(self, host: str, port: int, timeout_s: float = 1.0, debug: bool = False) -> None:
        self.client = CommandCenterTCPClient(
            host=host,
            port=port,
            timeout_s=timeout_s,
            debug=debug,
        )

    @staticmethod
    def _to_command_center_pose(env: SwarmEnv, agent_state) -> tuple[float, float, float]:
        nest_x, nest_y = env.nest_position
        x_cm = float(agent_state.x) - float(nest_x)
        y_cm = float(nest_y) - float(agent_state.y)
        heading_deg = -math.degrees(float(agent_state.theta))
        return x_cm, y_cm, heading_deg

    def publish_positions(self, env: SwarmEnv, agent_ids: list[str]) -> None:
        for i, agent_id in enumerate(agent_ids):
            if i >= len(env.agent_states) or i in env.failed_agent_indices:
                continue
            x_cm, y_cm, heading_deg = self._to_command_center_pose(env, env.agent_states[i])
            self.client.send_position(agent_id, x_cm, y_cm, heading_deg)

    def publish_step(self, env: SwarmEnv, agent_ids: list[str], chosen_actions: np.ndarray, prev_nest_distances: np.ndarray) -> None:
        self.publish_positions(env, agent_ids)
        if not bool(getattr(env.cfg, "pheromone_enabled", False)):
            return

        current_nest_distances = env._compute_nest_distance_state()
        for i, agent_id in enumerate(agent_ids):
            if i >= len(env.agent_states) or i in env.failed_agent_indices:
                continue
            _, _, deposit_requested = env.action_table[int(chosen_actions[i])]
            if not deposit_requested:
                continue
            agent = env.agent_states[i]
            if env.cfg.pheromone_requires_food and not agent.carrying_food:
                continue
            if env.cfg.pheromone_deposit_requires_nest_progress:
                prev_dist = float(prev_nest_distances[i])
                curr_dist = float(current_nest_distances[i])
                if not (np.isfinite(prev_dist) and np.isfinite(curr_dist)) or curr_dist >= prev_dist:
                    continue
            amount = float(env.cfg.pheromone_deposit)
            if agent.carrying_food:
                amount *= float(env.cfg.pheromone_deposit_carrying_scale)
            x_cm, y_cm, _heading_deg = self._to_command_center_pose(env, agent)
            self.client.deposit_pheromone(agent_id, x_cm, y_cm, amount)

    def close(self) -> None:
        self.client.close()


def _parse_seed_list(seed_list_raw: str) -> list[int]:
    values = []
    for part in (seed_list_raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _make_reset_seed_provider(args):
    seeds = _parse_seed_list(getattr(args, "seed_list", ""))
    if seeds:
        state = {"idx": 1}

        def next_seed():
            seed = seeds[state["idx"] % len(seeds)]
            state["idx"] += 1
            return int(seed)

        return int(seeds[0]), next_seed

    state = {"next_seed": int(args.seed) + 1}

    def next_seed():
        seed = state["next_seed"]
        state["next_seed"] += 1
        return int(seed)

    return int(args.seed), next_seed


def _make_demo_ui_state():
    return {"paused": False, "selected_agent_index": None}


def _handle_demo_events(env, ui_state):
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            ui_state["paused"] = not bool(ui_state.get("paused", False))
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            ui_state["selected_agent_index"] = env.pick_agent_at_screen_pos(event.pos)
    return running


def _action_label(env, action_id: int) -> str:
    throttle, turn, deposit = env.action_table[int(action_id)]
    throttle_label = {1.0: "FWD", 0.0: "STOP", -1.0: "REV"}.get(float(throttle), f"thr={throttle:.1f}")
    turn_label = {1.0: "RIGHT", 0.0: "STRAIGHT", -1.0: "LEFT"}.get(float(turn), f"turn={turn:.1f}")
    deposit_label = "DROP" if int(deposit) else "NO-DROP"
    return f"{throttle_label} | {turn_label} | {deposit_label}"


def _format_vector_block(label: str, values, *, chunk_size: int = 6) -> list[str]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    lines = [f"{label}:"]
    for start in range(0, arr.size, chunk_size):
        chunk = arr[start : start + chunk_size]
        parts = [f"{start + idx:02d}:{value:+.2f}" for idx, value in enumerate(chunk)]
        lines.append("  " + "  ".join(parts))
    return lines


def _format_top_actions(action_scores, top_k: int = 5) -> list[str]:
    arr = np.asarray(action_scores, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return ["Top actions:", "  unavailable"]
    order = np.argsort(arr)[::-1][: max(1, min(top_k, arr.size))]
    lines = ["Top actions:"]
    for idx in order:
        lines.append(f"  a{int(idx):02d}: {float(arr[idx]):+.3f}")
    return lines


def _build_agent_panel(
    env,
    agent_ids,
    selected_agent_index,
    obs_matrix,
    output_name,
    output_matrix,
    chosen_actions,
    modes,
):
    if selected_agent_index is None:
        return "", []
    if selected_agent_index < 0 or selected_agent_index >= len(agent_ids):
        return "", []
    agent = env.agent_states[selected_agent_index]
    agent_id = agent_ids[selected_agent_index]
    obs_vec = np.asarray(obs_matrix[selected_agent_index], dtype=np.float32)
    action_id = int(chosen_actions[selected_agent_index])
    mode = str(modes[selected_agent_index])
    output_vec = None if output_matrix is None else np.asarray(output_matrix[selected_agent_index], dtype=np.float32)

    lines = [
        "Status:",
        f"  id: {agent_id}",
        f"  pos: ({agent.x:.1f}, {agent.y:.1f})",
        f"  theta: {math.degrees(agent.theta):+.1f} deg",
        f"  carrying: {'yes' if agent.carrying_food else 'no'}",
        f"  post_delivery: {int(getattr(agent, 'post_delivery_steps', 0))}",
        f"  mode: {mode}",
        f"  action: a{action_id:02d} ({_action_label(env, action_id)})",
    ]
    if output_vec is not None:
        lines.extend(_format_top_actions(output_vec))
    else:
        lines.extend([f"{output_name}:", "  unavailable"])
    lines.extend(_format_vector_block("Input", obs_vec))
    if output_vec is not None:
        lines.extend(_format_vector_block(output_name, output_vec))
    title = f"Agent Inspector: {agent_id}"
    return title, lines


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


def _load_checkpoint_metadata(checkpoint_dir: str) -> dict | None:
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_demo_env(args):
    metadata = None
    args_copy = argparse.Namespace(**vars(args))
    if args.backend == "mappo":
        metadata = _load_checkpoint_metadata(args.checkpoint_dir)
        if metadata:
            args_copy.n_agents = int(metadata.get("n_agents", args_copy.n_agents))
            args_copy.n_targets = int(metadata.get("n_targets", getattr(args_copy, "n_targets", 3)))
            args_copy.n_obstacles = int(metadata.get("n_obstacles", getattr(args_copy, "n_obstacles", 6)))
            args_copy.max_steps_per_episode = int(metadata.get("max_steps", getattr(args_copy, "max_steps_per_episode", 600)))
            args_copy.active_targets = int(metadata.get("active_targets", getattr(args_copy, "active_targets", 3)))
            args_copy.target_respawn = bool(metadata.get("target_respawn", getattr(args_copy, "target_respawn", True)))
            args_copy.action_repeat_steps = int(
                metadata.get("action_repeat_steps", getattr(args_copy, "action_repeat_steps", 2))
            )
            args_copy.food_source_capacity = int(
                metadata.get("food_source_capacity", getattr(args_copy, "food_source_capacity", 4))
            )
            args_copy.non_carrying_nest_pheromone_suppression_radius = float(
                metadata.get(
                    "non_carrying_nest_pheromone_suppression_radius",
                    getattr(args_copy, "non_carrying_nest_pheromone_suppression_radius", 0.0),
                )
            )
            args_copy.non_carrying_nest_loiter_radius = float(
                metadata.get(
                    "non_carrying_nest_loiter_radius",
                    getattr(args_copy, "non_carrying_nest_loiter_radius", 0.0),
                )
            )
            args_copy.non_carrying_nest_loiter_penalty = float(
                metadata.get(
                    "non_carrying_nest_loiter_penalty",
                    getattr(args_copy, "non_carrying_nest_loiter_penalty", 0.0),
                )
            )
            args_copy.non_carrying_nest_crowding_radius = float(
                metadata.get(
                    "non_carrying_nest_crowding_radius",
                    getattr(args_copy, "non_carrying_nest_crowding_radius", 0.0),
                )
            )
            args_copy.non_carrying_nest_crowding_penalty = float(
                metadata.get(
                    "non_carrying_nest_crowding_penalty",
                    getattr(args_copy, "non_carrying_nest_crowding_penalty", 0.0),
                )
            )
            args_copy.non_carrying_nest_crowding_threshold = int(
                metadata.get(
                    "non_carrying_nest_crowding_threshold",
                    getattr(args_copy, "non_carrying_nest_crowding_threshold", 2),
                )
            )
            args_copy.post_delivery_cooldown_steps = int(
                metadata.get(
                    "post_delivery_cooldown_steps",
                    getattr(args_copy, "post_delivery_cooldown_steps", 0),
                )
            )
            args_copy.post_delivery_exit_radius = float(
                metadata.get(
                    "post_delivery_exit_radius",
                    getattr(args_copy, "post_delivery_exit_radius", 0.0),
                )
            )
            args_copy.post_delivery_outward_reward = float(
                metadata.get(
                    "post_delivery_outward_reward",
                    getattr(args_copy, "post_delivery_outward_reward", 0.0),
                )
            )
            args_copy.post_delivery_loiter_penalty = float(
                metadata.get(
                    "post_delivery_loiter_penalty",
                    getattr(args_copy, "post_delivery_loiter_penalty", 0.0),
                )
            )
            args_copy.post_delivery_pheromone_suppression_radius = float(
                metadata.get(
                    "post_delivery_pheromone_suppression_radius",
                    getattr(args_copy, "post_delivery_pheromone_suppression_radius", 0.0),
                )
            )
            args_copy.post_delivery_require_exit = bool(
                metadata.get(
                    "post_delivery_require_exit",
                    getattr(args_copy, "post_delivery_require_exit", False),
                )
            )
            args_copy.post_delivery_crowding_penalty = float(
                metadata.get(
                    "post_delivery_crowding_penalty",
                    getattr(args_copy, "post_delivery_crowding_penalty", 0.0),
                )
            )
            args_copy.post_delivery_crowding_threshold = int(
                metadata.get(
                    "post_delivery_crowding_threshold",
                    getattr(args_copy, "post_delivery_crowding_threshold", 2),
                )
            )
            args_copy.non_carrying_explore_radius = float(
                metadata.get(
                    "non_carrying_explore_radius",
                    getattr(args_copy, "non_carrying_explore_radius", 0.0),
                )
            )
            args_copy.non_carrying_outward_reward = float(
                metadata.get(
                    "non_carrying_outward_reward",
                    getattr(args_copy, "non_carrying_outward_reward", 0.0),
                )
            )
            args_copy.non_carrying_no_outward_progress_penalty = float(
                metadata.get(
                    "non_carrying_no_outward_progress_penalty",
                    getattr(args_copy, "non_carrying_no_outward_progress_penalty", 0.0),
                )
            )
            args_copy.non_carrying_idle_near_nest_penalty = float(
                metadata.get(
                    "non_carrying_idle_near_nest_penalty",
                    getattr(args_copy, "non_carrying_idle_near_nest_penalty", 0.0),
                )
            )
            args_copy.non_carrying_low_displacement_threshold = float(
                metadata.get(
                    "non_carrying_low_displacement_threshold",
                    getattr(args_copy, "non_carrying_low_displacement_threshold", 0.0),
                )
            )
            args_copy.non_carrying_explore_ignore_pheromone = bool(
                metadata.get(
                    "non_carrying_explore_ignore_pheromone",
                    getattr(args_copy, "non_carrying_explore_ignore_pheromone", False),
                )
            )
            args_copy.non_carrying_explore_random_action_prob = float(
                metadata.get(
                    "non_carrying_explore_random_action_prob",
                    getattr(args_copy, "non_carrying_explore_random_action_prob", 0.0),
                )
            )
            args_copy.non_carrying_force_explore_mode = bool(
                metadata.get(
                    "non_carrying_force_explore_mode",
                    getattr(args_copy, "non_carrying_force_explore_mode", False),
                )
            )
            args_copy.non_carrying_force_explore_radius = float(
                metadata.get(
                    "non_carrying_force_explore_radius",
                    getattr(args_copy, "non_carrying_force_explore_radius", 0.0),
                )
            )

    cfg = make_swarm_config(args_copy)
    if metadata:
        cfg.width = int(metadata.get("width", cfg.width))
        cfg.height = int(metadata.get("height", cfg.height))
    env = SwarmEnv(cfg, headless=args.headless)
    return args_copy, cfg, env, metadata


def _custom_demo(env, obs, agent_ids, args, relay_publisher: SimulatorMissionControlRelayPublisher | SimulatorMissionControlTCPPublisher | None = None):
    # Infer model dimensions and device.
    obs_dim = obs.shape[1]
    action_dim = env.cfg.num_actions
    device = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    _, next_reset_seed = _make_reset_seed_provider(args)
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = None
    ui_state = _make_demo_ui_state()

    nets = load_models(args.checkpoint_dir, obs_dim, action_dim, env.cfg.n_agents, args.shared_policy, device)

    running = True
    steps = 0
    while running:
        if not args.headless:
            running = _handle_demo_events(env, ui_state)

        actions = np.zeros(env.cfg.n_agents, dtype=np.int64)
        q_matrix = np.zeros((env.cfg.n_agents, action_dim), dtype=np.float32)
        modes = ["greedy"] * env.cfg.n_agents
        for i in range(env.cfg.n_agents):
            with torch.no_grad():
                obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = nets[i](obs_tensor)
                q_matrix[i] = q_vals.detach().cpu().numpy().reshape(-1)
                greedy_action = int(torch.argmax(q_vals, dim=1).item())
            if random.random() < args.demo_epsilon:
                actions[i] = np.random.randint(0, action_dim)
                mode = "explore"
            else:
                actions[i] = greedy_action
                mode = "greedy"
            modes[i] = mode
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

        panel_title, panel_lines = _build_agent_panel(
            env,
            agent_ids,
            ui_state.get("selected_agent_index"),
            obs,
            "Q-values",
            q_matrix,
            actions,
            modes,
        )
        env.set_demo_overlay(
            paused=bool(ui_state.get("paused", False)),
            selected_agent_index=ui_state.get("selected_agent_index"),
            panel_title=panel_title,
            panel_lines=panel_lines,
        )
        if ui_state.get("paused", False):
            if not args.headless:
                env.render(fps=30)
            continue

        prev_nest_distances = env._compute_nest_distance_state()
        action_dict = {agent: int(actions[i]) for i, agent in enumerate(agent_ids)}
        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        if relay_publisher is not None:
            relay_publisher.publish_step(env, agent_ids, actions, prev_nest_distances)
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
            obs_dict, _ = env.reset(seed=next_reset_seed())
            obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)

    return obs


def _sb3_demo(env, obs_dict, agent_ids, args, relay_publisher: SimulatorMissionControlRelayPublisher | SimulatorMissionControlTCPPublisher | None = None):
    from stable_baselines3 import DQN

    model = DQN.load(args.sb3_model, device="cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    _, next_reset_seed = _make_reset_seed_provider(args)
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = None
    ui_state = _make_demo_ui_state()

    running = True
    steps = 0
    while running:
        if not args.headless:
            running = _handle_demo_events(env, ui_state)

        action_dict = {}
        obs_matrix = np.stack([obs_dict[agent] for agent in agent_ids], axis=0).astype(np.float32)
        q_matrix = np.zeros((env.cfg.n_agents, env.cfg.num_actions), dtype=np.float32)
        chosen_actions = np.zeros((env.cfg.n_agents,), dtype=np.int64)
        modes = ["greedy"] * env.cfg.n_agents
        for i, agent in enumerate(agent_ids):
            obs_tensor = torch.as_tensor(obs_dict[agent], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = model.q_net(obs_tensor).detach().cpu().numpy()
            q_matrix[i] = q_values.reshape(-1)
            action, _ = model.predict(obs_dict[agent], deterministic=True)
            greedy_action = int(action)
            if random.random() < args.demo_epsilon:
                action_dict[agent] = int(np.random.randint(0, env.cfg.num_actions))
                mode = "explore"
            else:
                action_dict[agent] = greedy_action
                mode = "greedy"
            chosen_actions[i] = int(action_dict[agent])
            modes[i] = mode
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

        panel_title, panel_lines = _build_agent_panel(
            env,
            agent_ids,
            ui_state.get("selected_agent_index"),
            obs_matrix,
            "Q-values",
            q_matrix,
            chosen_actions,
            modes,
        )
        env.set_demo_overlay(
            paused=bool(ui_state.get("paused", False)),
            selected_agent_index=ui_state.get("selected_agent_index"),
            panel_title=panel_title,
            panel_lines=panel_lines,
        )
        if ui_state.get("paused", False):
            if not args.headless:
                env.render(fps=30)
            continue

        prev_nest_distances = env._compute_nest_distance_state()
        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        if relay_publisher is not None:
            relay_publisher.publish_step(env, agent_ids, chosen_actions, prev_nest_distances)
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
            obs_dict, _ = env.reset(seed=next_reset_seed())

    return obs_dict


def _rllib_demo(env, obs_dict, agent_ids, args, relay_publisher: SimulatorMissionControlRelayPublisher | SimulatorMissionControlTCPPublisher | None = None):
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
    _, next_reset_seed = _make_reset_seed_provider(args)
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = None
    ui_state = _make_demo_ui_state()
    while running:
        if not args.headless:
            running = _handle_demo_events(env, ui_state)

        action_dict = {}
        obs_matrix = np.stack([obs_dict[agent] for agent in agent_ids], axis=0).astype(np.float32)
        chosen_actions = np.zeros((env.cfg.n_agents,), dtype=np.int64)
        modes = ["greedy"] * env.cfg.n_agents
        for i, agent in enumerate(agent_ids):
            action = int(algo.compute_single_action(obs_dict[agent], policy_id="shared_policy"))
            if random.random() < args.demo_epsilon:
                action_dict[agent] = int(np.random.randint(0, env.cfg.num_actions))
                mode = "explore"
            else:
                action_dict[agent] = action
                mode = "greedy"
            chosen_actions[i] = int(action_dict[agent])
            modes[i] = mode
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

        panel_title, panel_lines = _build_agent_panel(
            env,
            agent_ids,
            ui_state.get("selected_agent_index"),
            obs_matrix,
            "Policy output",
            None,
            chosen_actions,
            modes,
        )
        env.set_demo_overlay(
            paused=bool(ui_state.get("paused", False)),
            selected_agent_index=ui_state.get("selected_agent_index"),
            panel_title=panel_title,
            panel_lines=panel_lines,
        )
        if ui_state.get("paused", False):
            if not args.headless:
                env.render(fps=30)
            continue

        prev_nest_distances = env._compute_nest_distance_state()
        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        if relay_publisher is not None:
            relay_publisher.publish_step(env, agent_ids, chosen_actions, prev_nest_distances)
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
            obs_dict, _ = env.reset(seed=next_reset_seed())

    algo.stop()
    ray.shutdown()
    return obs_dict


def _mappo_demo(env, obs_dict, agent_ids, args, relay_publisher: SimulatorMissionControlRelayPublisher | SimulatorMissionControlTCPPublisher | None = None):
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.cfg.num_actions
    actor, device = load_actor(args.checkpoint_dir, obs_dim, action_dim, device="cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    _, next_reset_seed = _make_reset_seed_provider(args)
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = np.zeros((env.cfg.n_agents,), dtype=np.float32)
    hidden_state = actor.initial_hidden(env.cfg.n_agents, device)
    ui_state = _make_demo_ui_state()

    running = True
    steps = 0
    while running:
        if not args.headless:
            running = _handle_demo_events(env, ui_state)

        obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0).astype(np.float32)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        done_mask = torch.as_tensor(1.0 - prev_done, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits, hidden_state = actor(obs_tensor, hidden_state, done_mask)
            greedy_actions = torch.argmax(logits, dim=-1).detach().cpu().numpy().astype(np.int64, copy=False)
        logits_matrix = logits.detach().cpu().numpy()

        action_dict = {}
        chosen_actions = np.zeros((env.cfg.n_agents,), dtype=np.int64)
        modes = ["greedy"] * env.cfg.n_agents
        for i, agent in enumerate(agent_ids):
            if random.random() < args.demo_epsilon:
                action_dict[agent] = int(np.random.randint(0, env.cfg.num_actions))
                mode = "explore"
            else:
                action_dict[agent] = int(greedy_actions[i])
                mode = "greedy"
            chosen_actions[i] = int(action_dict[agent])
            modes[i] = mode
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

        panel_title, panel_lines = _build_agent_panel(
            env,
            agent_ids,
            ui_state.get("selected_agent_index"),
            obs,
            "Policy logits",
            logits_matrix,
            chosen_actions,
            modes,
        )
        env.set_demo_overlay(
            paused=bool(ui_state.get("paused", False)),
            selected_agent_index=ui_state.get("selected_agent_index"),
            panel_title=panel_title,
            panel_lines=panel_lines,
        )
        if ui_state.get("paused", False):
            if not args.headless:
                env.render(fps=30)
            continue

        prev_nest_distances = env._compute_nest_distance_state()
        obs_dict, rewards_dict, terminations, truncations, _ = env.step(action_dict)
        if relay_publisher is not None:
            relay_publisher.publish_step(env, agent_ids, chosen_actions, prev_nest_distances)
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
            obs_dict, _ = env.reset(seed=next_reset_seed())
            hidden_state = actor.initial_hidden(env.cfg.n_agents, device)
            prev_done = np.zeros((env.cfg.n_agents,), dtype=np.float32)

    return obs_dict


def _random_demo(env, obs_dict, agent_ids, args, relay_publisher: SimulatorMissionControlRelayPublisher | SimulatorMissionControlTCPPublisher | None = None):
    random.seed(args.seed)
    np.random.seed(args.seed)
    _, next_reset_seed = _make_reset_seed_provider(args)
    ui_state = _make_demo_ui_state()

    running = True
    steps = 0
    while running:
        if not args.headless:
            running = _handle_demo_events(env, ui_state)

        obs_matrix = np.stack([obs_dict[agent] for agent in agent_ids], axis=0).astype(np.float32)
        chosen_actions = np.array([int(np.random.randint(0, env.cfg.num_actions)) for _ in agent_ids], dtype=np.int64)
        action_dict = {agent: int(chosen_actions[i]) for i, agent in enumerate(agent_ids)}
        panel_title, panel_lines = _build_agent_panel(
            env,
            agent_ids,
            ui_state.get("selected_agent_index"),
            obs_matrix,
            "Policy output",
            None,
            chosen_actions,
            ["random"] * env.cfg.n_agents,
        )
        env.set_demo_overlay(
            paused=bool(ui_state.get("paused", False)),
            selected_agent_index=ui_state.get("selected_agent_index"),
            panel_title=panel_title,
            panel_lines=panel_lines,
        )
        if ui_state.get("paused", False):
            if not args.headless:
                env.render(fps=30)
            continue

        prev_nest_distances = env._compute_nest_distance_state()
        obs_dict, _, terminations, truncations, _ = env.step(action_dict)
        if relay_publisher is not None:
            relay_publisher.publish_step(env, agent_ids, chosen_actions, prev_nest_distances)
        terminated = any(terminations.values())
        truncated = any(truncations.values())

        if not args.headless:
            env.render(fps=60)
        steps += 1
        if args.max_steps and steps >= args.max_steps:
            running = False
        if terminated or truncated:
            obs_dict, _ = env.reset(seed=next_reset_seed())

    return obs_dict


def main():
    args = parse_args()
    initial_seed, _ = _make_reset_seed_provider(args)

    # 2) Build config + environment, then reset to get initial observations.
    demo_args, cfg, env, metadata = _build_demo_env(args)
    if metadata:
        print(
            f"[demo] Loaded MAPPO stage metadata from {args.checkpoint_dir}: "
            f"agents={cfg.n_agents} size={cfg.width}x{cfg.height} "
            f"targets={cfg.n_targets} obstacles={cfg.n_obstacles} "
            f"max_steps={cfg.max_steps}"
        )
    obs_dict, _ = env.reset(seed=initial_seed)
    agent_ids = env.possible_agents
    obs = np.stack([obs_dict[agent] for agent in agent_ids], axis=0)
    relay_publisher = None
    if demo_args.mc_tcp_host:
        relay_publisher = SimulatorMissionControlTCPPublisher(
            host=demo_args.mc_tcp_host,
            port=demo_args.mc_tcp_port,
            timeout_s=demo_args.mc_tcp_timeout,
            debug=demo_args.mc_tcp_debug,
        )
        relay_publisher.publish_positions(env, agent_ids)
    elif demo_args.mc_relay_url:
        relay_publisher = SimulatorMissionControlRelayPublisher(
            relay_url=demo_args.mc_relay_url,
            session=demo_args.mc_relay_session,
            timeout_s=demo_args.mc_relay_timeout,
            debug=demo_args.mc_relay_debug,
        )
        relay_publisher.publish_positions(env, agent_ids)
    if demo_args.headless and demo_args.max_steps <= 0:
        demo_args.max_steps = int(cfg.max_steps)
    if not demo_args.headless:
        env.render(fps=60) # render first frame

    try:
        if demo_args.backend == "custom":
            _custom_demo(env, obs, agent_ids, demo_args, relay_publisher=relay_publisher)
        elif demo_args.backend == "sb3":
            _sb3_demo(env, obs_dict, agent_ids, demo_args, relay_publisher=relay_publisher)
        elif demo_args.backend == "rllib":
            _rllib_demo(env, obs_dict, agent_ids, demo_args, relay_publisher=relay_publisher)
        elif demo_args.backend == "mappo":
            _mappo_demo(env, obs_dict, agent_ids, demo_args, relay_publisher=relay_publisher)
        elif demo_args.backend == "random":
            _random_demo(env, obs_dict, agent_ids, demo_args, relay_publisher=relay_publisher)
        else:
            raise ValueError(f"Unsupported backend: {demo_args.backend}")
    finally:
        if relay_publisher is not None:
            relay_publisher.close()
        env.close()


if __name__ == "__main__":
    main()
