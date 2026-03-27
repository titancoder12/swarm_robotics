"""
Classic DQN / Q-learning setup (with replay buffer + target network)
and an ε-greedy behavior policy.
"""
from __future__ import annotations

import argparse  # CLI argument parsing.
import copy
import os  # Filesystem paths.
import random  # Epsilon-greedy randomness.
import sys  # Path tweaks for local imports.
from dataclasses import dataclass  # Simple config container.
from typing import List  # Type hints.

import numpy as np  # Numeric arrays for buffers/obs.
import torch  # Neural nets + tensors.
import torch.nn as nn  # NN modules.
import torch.optim as optim  # Optimizers (Adam).

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Repo root.
if ROOT not in sys.path:  # Ensure local imports work.
    sys.path.insert(0, ROOT)

from env.config import SwarmConfig  # Environment config.
from env.swarm_env import SwarmEnv  # PettingZoo env.
from models.q_network import QNetwork  # Shared Q-network definition.
from policy_debug import make_policy_debug_config, print_policy_debug, should_debug_policy
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


@dataclass  # Declarative hyperparameter container.
class DQNConfig:
    """Hyperparameters for DQN training."""
    gamma: float = 0.98 # Discount factor for future rewards.
    batch_size: int = 64 # Minibatch size for sampling from replay buffer.
    buffer_size: int = 50_000 # Experience memory size.
    lr: float = 3e-4 # Learning rate for Adam optimizer.
    target_update: int = 200
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_steps: int = 8000
    warmup_steps: int = 500


class ReplayBuffer:
    """Simple FIFO replay buffer for experience tuples."""
    def __init__(self, capacity: int, obs_dim: int):
        """Allocate storage for fixed-size experience arrays."""
        self.capacity = capacity  # Max number of stored transitions.
        self.obs_dim = obs_dim  # Observation vector length.
        self.ptr = 0  # Write pointer (circular buffer).
        self.size = 0  # Current number of stored transitions.
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s_t
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)  # s_{t+1}
        self.actions = np.zeros((capacity,), dtype=np.int64)  # a_t
        self.rewards = np.zeros((capacity,), dtype=np.float32)  # r_t
        self.dones = np.zeros((capacity,), dtype=np.float32)  # done flag

    def add(self, obs, action, reward, next_obs, done):
        """Insert a transition, overwriting oldest when capacity is exceeded."""
        self.obs[self.ptr] = obs  # Store state.
        self.actions[self.ptr] = action  # Store action.
        self.rewards[self.ptr] = reward  # Store reward.
        self.next_obs[self.ptr] = next_obs  # Store next state.
        self.dones[self.ptr] = done  # Store done.
        self.ptr = (self.ptr + 1) % self.capacity  # Advance pointer (wrap).
        self.size = min(self.size + 1, self.capacity)  # Track current size.

    def sample(self, batch_size: int):
        """Sample a random minibatch and return as torch tensors."""
        idx = np.random.randint(0, self.size, size=batch_size)  # Random indices.
        return (
            torch.tensor(self.obs[idx]),  # Batch of states.
            torch.tensor(self.actions[idx]),  # Batch of actions.
            torch.tensor(self.rewards[idx]),  # Batch of rewards.
            torch.tensor(self.next_obs[idx]),  # Batch of next states.
            torch.tensor(self.dones[idx]),  # Batch of done flags.
        )


def _dict_to_array(data, agent_ids, dtype=None):
    arr = np.array([data[agent] for agent in agent_ids])
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _array_to_dict(arr, agent_ids):
    return {agent: arr[i] for i, agent in enumerate(agent_ids)}


def linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    """Linearly anneal a value from start to end over decay_steps."""
    if step >= decay_steps:  # Stop decaying after schedule ends.
        return end
    frac = step / decay_steps  # Progress ratio.
    return start + frac * (end - start)  # Linear interpolation.


def _evaluate_policy(cfg: SwarmConfig, q_nets: List[QNetwork], shared: bool, device, episodes: int, base_seed: int):
    """Run deterministic evaluation episodes with the current policy."""
    eval_cfg = copy.deepcopy(cfg)
    env = SwarmEnv(eval_cfg, headless=True)
    try:
        metrics = []
        for episode in range(episodes):
            obs_dict, _ = env.reset(seed=base_seed + episode)
            agent_ids = env.possible_agents
            obs = _dict_to_array(obs_dict, agent_ids, dtype=np.float32)
            episode_rewards = np.zeros(eval_cfg.n_agents, dtype=np.float32)
            food_discovered = 0
            food_retrieved = 0
            exploration_coverage = 0.0
            pheromone_usage_values = []
            episode_length = 0

            while True:
                actions = np.zeros(eval_cfg.n_agents, dtype=np.int64)
                for i in range(eval_cfg.n_agents):
                    with torch.no_grad():
                        obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                        q_vals = q_nets[i](obs_tensor)
                        actions[i] = int(torch.argmax(q_vals, dim=1).item())

                action_dict = _array_to_dict(actions, agent_ids)
                next_obs_dict, rewards_dict, terminations, truncations, info_dict = env.step(action_dict)
                obs = _dict_to_array(next_obs_dict, agent_ids, dtype=np.float32)
                rewards = _dict_to_array(rewards_dict, agent_ids, dtype=np.float32)
                info = info_dict[agent_ids[0]]

                episode_rewards += rewards
                food_discovered += int(info.get("targets_collected", 0))
                food_retrieved += int(info.get("food_delivered", 0))
                exploration_coverage = max(exploration_coverage, float(info.get("exploration_coverage", 0.0)))
                pheromone_usage_values.append(float(info.get("pheromone_usage", 0.0)))
                episode_length = int(info.get("episode_length", episode_length + 1))
                if any(terminations.values()) or any(truncations.values()):
                    break

            metrics.append(
                {
                    "mean_episode_reward": float(episode_rewards.mean()),
                    "food_discovered": food_discovered,
                    "food_retrieved": food_retrieved,
                    "exploration_coverage": exploration_coverage,
                    "pheromone_usage": float(np.mean(pheromone_usage_values)) if pheromone_usage_values else 0.0,
                    "episode_length": episode_length,
                    "swarm_efficiency": float(food_retrieved / max(episode_length, 1)),
                }
            )
        return {
            "mean_episode_reward": float(np.mean([m["mean_episode_reward"] for m in metrics])) if metrics else 0.0,
            "food_discovered": float(np.mean([m["food_discovered"] for m in metrics])) if metrics else 0.0,
            "food_retrieved": float(np.mean([m["food_retrieved"] for m in metrics])) if metrics else 0.0,
            "exploration_coverage": float(np.mean([m["exploration_coverage"] for m in metrics])) if metrics else 0.0,
            "pheromone_usage": float(np.mean([m["pheromone_usage"] for m in metrics])) if metrics else 0.0,
            "episode_length": float(np.mean([m["episode_length"] for m in metrics])) if metrics else 0.0,
            "swarm_efficiency": float(np.mean([m["swarm_efficiency"] for m in metrics])) if metrics else 0.0,
        }
    finally:
        env.close()


def _milestone_steps(total_steps: int) -> list[tuple[int, str]]:
    """Return one-shot checkpoint milestones for 25/50/75/100 percent progress."""
    total = max(1, int(total_steps))
    return [
        (max(1, int(np.ceil(total * 0.25))), "1_4_trained"),
        (max(1, int(np.ceil(total * 0.50))), "1_2_trained"),
        (max(1, int(np.ceil(total * 0.75))), "3_4_trained"),
        (total, "full_policy"),
    ]


def train(args):
    """Train independent (or shared) DQN policies for each agent."""
    args.output_dir = resolve_repo_path(args.output_dir)
    args.save_dir = resolve_repo_path(args.save_dir)
    # 1) Environment and config setup.
    cfg = make_swarm_config(args)  # Env config.
    filename = resolve_filename(args, fallback="dqn_foraging")
    experiment_name = filename
    dqn_cfg = DQNConfig(
        epsilon_start=args.epsilon_start,
        epsilon_final=args.epsilon_final,
        epsilon_decay_steps=args.epsilon_decay_steps,
        warmup_steps=args.warmup_steps,
    )  # Training config.

    env = SwarmEnv(cfg, headless=args.headless)  # Create env.
    obs_dict, _ = env.reset(seed=args.seed)  # Reset -> initial observations.
    agent_ids = env.possible_agents  # Ordered agent IDs.
    obs = _dict_to_array(obs_dict, agent_ids, dtype=np.float32)  # Dict -> array.
    obs_dim = obs.shape[1]  # Observation length.
    action_dim = cfg.num_actions  # Number of discrete actions.
    print(f"training obs_dim={obs_dim} action_dim={action_dim}")

    # Use GPU if requested and available.
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")  # Device.

    run_dir = make_run_dir(args.output_dir, experiment_name)
    graph_dir = resolve_repo_path(os.path.join("training_graphs", os.path.basename(run_dir)))
    os.makedirs(graph_dir, exist_ok=True)
    episode_logger = CSVLogger(
        os.path.join(run_dir, "episode_metrics.csv"),
        [
            "episode",
            "global_step",
            "epsilon",
            "mean_episode_reward",
            "food_discovered",
            "food_picked_up",
            "food_retrieved",
            "food_delivered",
            "exploration_coverage",
            "coverage_reward_total",
            "pheromone_usage",
            "pheromone_deposit_events",
            "episode_length",
            "collisions",
            "reward_step_component",
            "reward_pickup_component",
            "reward_delivery_component",
            "reward_collision_component",
            "reward_exploration_component",
            "reward_new_cell_component",
            "reward_food_approach_component",
            "reward_food_detected_component",
            "reward_pheromone_follow_component",
            "reward_pheromone_usage_component",
            "reward_pheromone_deposit_component",
            "reward_pheromone_component",
            "swarm_efficiency",
        ],
    )
    eval_logger = CSVLogger(
        os.path.join(run_dir, "eval_metrics.csv"),
        [
            "global_step",
            "mean_episode_reward",
            "food_discovered",
            "food_retrieved",
            "exploration_coverage",
            "pheromone_usage",
            "episode_length",
            "swarm_efficiency",
        ],
    )
    write_json(
        os.path.join(run_dir, "run_config.json"),
        {
            "experiment_name": experiment_name,
            "filename": filename,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "shared_policy": bool(args.shared_policy),
            "total_steps": args.total_steps,
            "seed": args.seed,
            "n_agents": args.n_agents,
            "use_pheromone": bool(cfg.pheromone_enabled),
        },
    )
    save_metadata = {
        "filename": filename,
        "use_pheromone": bool(cfg.pheromone_enabled),
        "experiment_name": experiment_name,
    }
    milestone_targets = _milestone_steps(args.total_steps)
    saved_milestones: set[str] = set()

    # 2) Initialize Q-networks, target networks, optimizers, and replay buffers.
    if args.shared_policy:
        # One shared policy across all agents.
        q_net = QNetwork(obs_dim, action_dim).to(device)  # Online Q-network.
        target_net = QNetwork(obs_dim, action_dim).to(device)  # Target Q-network.
        target_net.load_state_dict(q_net.state_dict())  # Sync target.
        optimizer = optim.Adam(q_net.parameters(), lr=dqn_cfg.lr)  # Optimizer.
        buffers = [ReplayBuffer(dqn_cfg.buffer_size, obs_dim) for _ in range(cfg.n_agents)]  # Per-agent replay.
        q_nets = [q_net for _ in range(cfg.n_agents)]  # Shared network reference.
        target_nets = [target_net for _ in range(cfg.n_agents)]  # Shared target reference.
        optimizers = [optimizer for _ in range(cfg.n_agents)]  # Shared optimizer reference.
    else:
        # Independent policies (one per agent).
        q_nets = []  # One Q-net per agent.
        target_nets = []  # One target net per agent.
        optimizers = []  # One optimizer per agent.
        buffers = []  # One replay buffer per agent.
        for _ in range(cfg.n_agents):
            net = QNetwork(obs_dim, action_dim).to(device)  # Online Q-network.
            target = QNetwork(obs_dim, action_dim).to(device)  # Target Q-network.
            target.load_state_dict(net.state_dict())  # Sync target.
            q_nets.append(net)  # Track Q-net.
            target_nets.append(target)  # Track target net.
            optimizers.append(optim.Adam(net.parameters(), lr=dqn_cfg.lr))  # Optimizer.
            buffers.append(ReplayBuffer(dqn_cfg.buffer_size, obs_dim))  # Replay buffer.

    # 3) Training loop state.
    global_step = 0  # Total steps across training.
    episode = 0  # Episode counter.
    episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)  # Accumulated rewards.
    episode_food_discovered = 0
    episode_food_retrieved = 0
    episode_collisions = 0
    episode_exploration_coverage = 0.0
    episode_coverage_reward_total = 0.0
    episode_pheromone_usage = []
    episode_pheromone_deposit_events = 0
    episode_length = 0
    episode_reward_breakdown = {
        "step": 0.0,
        "pickup": 0.0,
        "delivery": 0.0,
        "collision": 0.0,
        "new_cell": 0.0,
        "food_approach": 0.0,
        "food_detected": 0.0,
        "pheromone_follow": 0.0,
        "pheromone_usage": 0.0,
        "pheromone_deposit": 0.0,
    }
    next_eval_step = args.eval_every
    next_reset_seed = args.seed + 1
    debug_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    prev_rewards = None
    prev_done = None

    # 4) Main training loop.
    try:
        while global_step < args.total_steps:
            # Epsilon-greedy exploration schedule.
            epsilon = linear_schedule(dqn_cfg.epsilon_start, dqn_cfg.epsilon_final, global_step, dqn_cfg.epsilon_decay_steps)  # Exploration rate.
            actions = np.zeros(cfg.n_agents, dtype=np.int64)  # Action array.

            # Select actions for each agent (random with prob epsilon, else greedy).
            for i in range(cfg.n_agents):
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=device).unsqueeze(0)  # Batch-1 obs.
                    q_vals = q_nets[i](obs_tensor)  # Q-values.
                    greedy_action = int(torch.argmax(q_vals, dim=1).item())  # Greedy action.

                if random.random() < epsilon:
                    actions[i] = np.random.randint(0, action_dim)  # Explore.
                    mode = "explore"
                else:
                    actions[i] = greedy_action
                    mode = "greedy"

                if should_debug_policy(debug_cfg, global_step, i, agent_ids[i]):
                    print_policy_debug(
                        step=global_step,
                        agent_index=i,
                        agent_id=agent_ids[i],
                        policy_label="shared" if args.shared_policy else f"agent_{i}",
                        epsilon=epsilon,
                        mode=mode,
                        output_name="q_values",
                        output_values=q_vals.detach().cpu().numpy(),
                        action=int(actions[i]),
                        num_actions=action_dim,
                        prev_reward=None if prev_rewards is None else float(prev_rewards[i]),
                        prev_done=None if prev_done is None else bool(prev_done[i]),
                    )

            # Agent sees observation, picks action, gets reward, environment changes.
            action_dict = _array_to_dict(actions, agent_ids)  # Array -> dict (PettingZoo).
            next_obs_dict, rewards_dict, terminations, truncations, info = env.step(action_dict)  # Env step.
            next_obs = _dict_to_array(next_obs_dict, agent_ids, dtype=np.float32)  # Dict -> array.
            rewards = _dict_to_array(rewards_dict, agent_ids, dtype=np.float32)  # Dict -> array.
            step_info = info[agent_ids[0]]
            terminated = any(terminations.values())  # Episode ended (success).
            truncated = any(truncations.values())  # Episode ended (time limit).
            prev_rewards = rewards.copy()
            prev_done = np.full((cfg.n_agents,), bool(terminated or truncated), dtype=np.bool_)
            done_flag = float(terminated or truncated)  # Done flag for training target.

            # Store transitions in each agent's replay buffer.
            for i in range(cfg.n_agents):
                buffers[i].add(obs[i], actions[i], rewards[i], next_obs[i], done_flag)
                episode_rewards[i] += rewards[i]

            episode_food_discovered += int(step_info.get("targets_collected", 0))
            episode_food_retrieved += int(step_info.get("food_delivered", 0))
            episode_collisions += int(step_info.get("collisions", 0))
            episode_exploration_coverage = max(
                episode_exploration_coverage,
                float(step_info.get("exploration_coverage", 0.0)),
            )
            episode_coverage_reward_total += float(step_info.get("coverage_reward_total", 0.0))
            episode_pheromone_usage.append(float(step_info.get("pheromone_usage", 0.0)))
            episode_pheromone_deposit_events += int(step_info.get("pheromone_deposit_events", 0))
            episode_length = int(step_info.get("episode_length", episode_length + 1))
            reward_breakdown = step_info.get("reward_breakdown", {})
            for key in episode_reward_breakdown:
                episode_reward_breakdown[key] += float(reward_breakdown.get(key, 0.0))

            # Move to next observation for the next step.
            obs = next_obs
            global_step += 1

            # 5) Start learning after warmup (collecting initial experience).
            if global_step > dqn_cfg.warmup_steps:
                for i in range(cfg.n_agents):
                    if buffers[i].size < dqn_cfg.batch_size:
                        continue
                    batch = buffers[i].sample(dqn_cfg.batch_size)
                    batch = [b.to(device) for b in batch]
                    b_obs, b_actions, b_rewards, b_next_obs, b_dones = batch

                    q_vals = q_nets[i](b_obs).gather(1, b_actions.unsqueeze(1)).squeeze(1)  # Q(s,a).
                    with torch.no_grad():
                        max_next = target_nets[i](b_next_obs).max(dim=1)[0]  # max_a' Q_target(s',a')
                        target = b_rewards + dqn_cfg.gamma * (1.0 - b_dones) * max_next  # Bellman target.

                    loss = nn.functional.smooth_l1_loss(q_vals, target)  # Huber loss.
                    optimizers[i].zero_grad()  # Clear gradients.
                    loss.backward()  # Backprop.
                    optimizers[i].step()  # Update weights.

            # 6) Periodically sync target networks.
            if global_step % dqn_cfg.target_update == 0:
                for i in range(cfg.n_agents):
                    target_nets[i].load_state_dict(q_nets[i].state_dict())  # Sync target net.

            # 7) Periodically evaluate current policy.
            if args.eval_every > 0 and global_step >= next_eval_step:
                eval_metrics = _evaluate_policy(cfg, q_nets, args.shared_policy, device, args.eval_episodes, args.seed + 10_000)
                eval_logger.log({"global_step": global_step, **eval_metrics})
                next_eval_step += args.eval_every

            # 7b) One-shot milestone checkpoints.
            for threshold, milestone_name in milestone_targets:
                if global_step >= threshold and milestone_name not in saved_milestones:
                    _save_models(
                        os.path.join(args.save_dir, milestone_name),
                        q_nets,
                        args.shared_policy,
                        obs_dim,
                        global_step,
                        extra_metadata=save_metadata,
                    )
                    saved_milestones.add(milestone_name)

            # 8) Episode bookkeeping + logging.
            if terminated or truncated:
                episode += 1
                pheromone_usage_mean = float(np.mean(episode_pheromone_usage)) if episode_pheromone_usage else 0.0
                swarm_efficiency = float(episode_food_retrieved / max(episode_length, 1))
                episode_row = {
                    "episode": episode,
                    "global_step": global_step,
                    "epsilon": epsilon,
                    "mean_episode_reward": float(episode_rewards.mean()),
                    "food_discovered": episode_food_discovered,
                    "food_picked_up": episode_food_discovered,
                    "food_retrieved": episode_food_retrieved,
                    "food_delivered": episode_food_retrieved,
                    "exploration_coverage": episode_exploration_coverage,
                    "coverage_reward_total": episode_coverage_reward_total,
                    "pheromone_usage": pheromone_usage_mean,
                    "pheromone_deposit_events": episode_pheromone_deposit_events,
                    "episode_length": episode_length,
                    "collisions": episode_collisions,
                    "reward_step_component": episode_reward_breakdown["step"],
                    "reward_pickup_component": episode_reward_breakdown["pickup"],
                    "reward_delivery_component": episode_reward_breakdown["delivery"],
                    "reward_collision_component": episode_reward_breakdown["collision"],
                    "reward_exploration_component": episode_reward_breakdown["new_cell"],
                    "reward_new_cell_component": episode_reward_breakdown["new_cell"],
                    "reward_food_approach_component": episode_reward_breakdown["food_approach"],
                    "reward_food_detected_component": episode_reward_breakdown["food_detected"],
                    "reward_pheromone_follow_component": episode_reward_breakdown["pheromone_follow"],
                    "reward_pheromone_usage_component": episode_reward_breakdown["pheromone_usage"],
                    "reward_pheromone_deposit_component": episode_reward_breakdown["pheromone_deposit"],
                    "reward_pheromone_component": (
                        episode_reward_breakdown["pheromone_follow"]
                        + episode_reward_breakdown["pheromone_usage"]
                        + episode_reward_breakdown["pheromone_deposit"]
                    ),
                    "swarm_efficiency": swarm_efficiency,
                }
                episode_logger.log(episode_row)
                print(
                    f"episode {episode} step {global_step} reward {episode_row['mean_episode_reward']:.2f} "
                    f"food_found {episode_food_discovered} food_delivered {episode_food_retrieved} "
                    f"coverage {episode_exploration_coverage:.3f} "
                    f"pheromone {pheromone_usage_mean:.3f} epsilon {epsilon:.2f}"
                )
                obs_dict, _ = env.reset(seed=next_reset_seed)  # Reset environment.
                obs = _dict_to_array(obs_dict, agent_ids, dtype=np.float32)  # Dict -> array.
                next_reset_seed += 1
                episode_rewards = np.zeros(cfg.n_agents, dtype=np.float32)
                episode_food_discovered = 0
                episode_food_retrieved = 0
                episode_collisions = 0
                episode_exploration_coverage = 0.0
                episode_coverage_reward_total = 0.0
                episode_pheromone_usage = []
                episode_pheromone_deposit_events = 0
                episode_length = 0
                episode_reward_breakdown = {key: 0.0 for key in episode_reward_breakdown}

            # 9) Optional checkpointing.
            if args.save_every > 0 and global_step % args.save_every == 0:
                _save_models(
                    args.save_dir,
                    q_nets,
                    args.shared_policy,
                    obs_dim,
                    global_step,
                    extra_metadata=save_metadata,
                )

        _save_models(
            args.save_dir,
            q_nets,
            args.shared_policy,
            obs_dim,
            global_step,
            extra_metadata=save_metadata,
        )
        if "full_policy" not in saved_milestones:
            _save_models(
                os.path.join(args.save_dir, "full_policy"),
                q_nets,
                args.shared_policy,
                obs_dim,
                global_step,
                extra_metadata=save_metadata,
            )
        if not args.no_plots:
            plot_training_metrics(os.path.join(run_dir, "episode_metrics.csv"), run_dir)
            plot_training_metrics(os.path.join(run_dir, "episode_metrics.csv"), graph_dir)
            plot_eval_metrics(os.path.join(run_dir, "eval_metrics.csv"), graph_dir)
        write_json(
            os.path.join(run_dir, "summary.json"),
            {
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "total_steps": global_step,
                "episodes_completed": episode,
                "checkpoint_dir": args.save_dir,
                "graph_dir": graph_dir,
                "filename": filename,
                "use_pheromone": bool(cfg.pheromone_enabled),
            },
        )
        return run_dir
    finally:
        episode_logger.close()
        eval_logger.close()
        env.close()


def parse_args(argv=None):
    """Parse CLI args for training configuration."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=10000)  # Training steps.
    parser.add_argument("--n-agents", type=int, default=6)  # Number of agents.
    parser.add_argument("--shared-policy", action="store_true")  # Shared vs independent.
    parser.add_argument("--headless", action="store_true")  # No render window.
    parser.add_argument("--cuda", action="store_true")  # Use GPU if available.
    parser.add_argument("--seed", type=int, default=0)  # RNG seed.
    parser.add_argument("--save-dir", type=str, default="checkpoints")  # Checkpoint dir.
    parser.add_argument("--save-every", type=int, default=0, help="Save every N steps (0 = only at end)")  # Save cadence.
    parser.add_argument("--experiment-name", type=str, default="dqn_foraging")
    parser.add_argument("--output-dir", type=str, default="runs")
    parser.add_argument("--eval-every", type=int, default=0, help="Evaluate every N training steps (0 = disabled)")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon for epsilon-greedy exploration.")
    parser.add_argument("--epsilon-final", type=float, default=0.05, help="Final epsilon after annealing.")
    parser.add_argument("--epsilon-decay-steps", type=int, default=8000, help="Number of training steps over which epsilon decays.")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Number of steps to collect before gradient updates start.")
    parser.add_argument("--debug-policy", action="store_true")
    parser.add_argument("--debug-policy-agents", type=str, default="")
    parser.add_argument("--debug-policy-max-steps", type=int, default=0)
    add_env_config_args(parser)
    return parser.parse_args(argv)


def _save_models(
    save_dir: str,
    q_nets: List[QNetwork],
    shared: bool,
    obs_dim: int,
    step: int,
    extra_metadata: dict | None = None,
):
    """Save model weights to disk, shared or per-agent."""
    os.makedirs(save_dir, exist_ok=True)  # Ensure dir exists.
    metadata = {"obs_dim": obs_dim, "shared_policy": bool(shared), "step": int(step)}
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(os.path.join(save_dir, "metadata.json"), metadata)
    if shared:
        torch.save(q_nets[0].state_dict(), os.path.join(save_dir, "shared.pt"))  # One shared file.
        return
    for i, net in enumerate(q_nets):
        torch.save(net.state_dict(), os.path.join(save_dir, f"agent_{i}.pt"))  # Per-agent file.


if __name__ == "__main__":
    train(parse_args())  # Entry point.
