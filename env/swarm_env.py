from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pygame
from gymnasium import spaces
from pettingzoo import ParallelEnv

from env.config import SwarmConfig


@dataclass
class AgentState:
    """Lightweight container for an agent's kinematic state in the world."""
    x: float
    y: float
    theta: float
    v: float = 0.0
    omega: float = 0.0
    v_lat: float = 0.0
    carrying_food: bool = False


class DynamicsDriver:
    """Interface for mapping actions into kinematic updates."""
    def apply(self, state: AgentState, action: Tuple[float, float], dt: float, cfg: SwarmConfig, rng: np.random.Generator) -> AgentState:
        """Return the next AgentState given the current state and action."""
        raise NotImplementedError


class TankKinematicsDriver(DynamicsDriver):
    """Differential-drive style dynamics (forward speed + yaw rate)."""
    def apply(self, state: AgentState, action: Tuple[float, float], dt: float, cfg: SwarmConfig, rng: np.random.Generator) -> AgentState:
        """Apply tank kinematics with acceleration limits."""
        throttle, turn = action
        target_v = throttle * cfg.max_speed
        target_omega = turn * cfg.max_yaw_rate
        dv = np.clip(target_v - state.v, -cfg.accel * dt, cfg.accel * dt)
        domega = np.clip(target_omega - state.omega, -cfg.ang_accel * dt, cfg.ang_accel * dt)

        v = state.v + dv
        omega = state.omega + domega
        theta = state.theta + omega * dt

        nx = state.x + math.cos(theta) * v * dt
        ny = state.y + math.sin(theta) * v * dt
        return AgentState(nx, ny, theta, v=v, omega=omega, v_lat=0.0)


class HovercraftDriver(DynamicsDriver):
    """Hover-like dynamics with lateral drift and slip."""
    def apply(self, state: AgentState, action: Tuple[float, float], dt: float, cfg: SwarmConfig, rng: np.random.Generator) -> AgentState:
        """Apply hovercraft kinematics including lateral noise and slip."""
        throttle, turn = action
        target_v = throttle * cfg.max_speed
        target_omega = turn * cfg.max_yaw_rate
        dv = np.clip(target_v - state.v, -cfg.accel * dt, cfg.accel * dt)
        domega = np.clip(target_omega - state.omega, -cfg.ang_accel * dt, cfg.ang_accel * dt)

        v = state.v + dv
        omega = state.omega + domega
        theta = state.theta + omega * dt

        v_lat = state.v_lat * cfg.hover_lat_damping
        v_lat += rng.normal(0.0, cfg.hover_lat_noise)
        if rng.random() < cfg.hover_slip_chance:
            v *= cfg.hover_slip_scale

        forward = np.array([math.cos(theta), math.sin(theta)])
        right = np.array([math.cos(theta + math.pi / 2.0), math.sin(theta + math.pi / 2.0)])
        vel = forward * v + right * v_lat

        nx = state.x + vel[0] * dt
        ny = state.y + vel[1] * dt
        return AgentState(nx, ny, theta, v=v, omega=omega, v_lat=v_lat)


class SwarmEnv(ParallelEnv):
    """Multi-agent 2D swarm environment with optional stigmergy and PyGame rendering."""
    metadata = {"name": "swarm_env_v0", "render_modes": ["human"], "is_parallel": True}

    def __init__(self, cfg: SwarmConfig, headless: bool = False):
        """Create an environment instance with a given config."""
        self.cfg = cfg
        self.headless = headless
        self.render_mode = None if headless else "human"
        self.rng = np.random.default_rng(cfg.seed)

        self.width = cfg.width
        self.height = cfg.height

        self.action_table = self._build_action_table()
        self.driver_mode = cfg.dynamics_mode
        self.driver = self._select_driver(cfg.dynamics_mode)

        self.possible_agents = [f"agent_{i}" for i in range(self.cfg.n_agents)]
        self.agents = self.possible_agents[:]
        self.agent_states: List[AgentState] = []
        self.targets: List[Tuple[float, float]] = []
        self.target_remaining_uses: List[int] = []
        self.obstacles: List[pygame.Rect] = []
        self.nest_position: Tuple[float, float] = (self.width * 0.5, self.height * 0.5)
        self.food_delivered = 0
        self.episode_food_source_respawns = 0
        self.coverage_grid = None
        self.covered_cells = 0
        self.total_cover_cells = 1
        self.failed_agent_indices: set[int] = set()
        self._prev_detectable_food_distances = np.full((self.cfg.n_agents,), np.inf, dtype=np.float32)
        self._prev_food_detected = np.zeros((self.cfg.n_agents,), dtype=np.bool_)
        self._prev_nest_distances = np.full((self.cfg.n_agents,), np.inf, dtype=np.float32)
        self._held_actions = np.zeros((self.cfg.n_agents,), dtype=np.int64)
        self._hold_remaining = np.zeros((self.cfg.n_agents,), dtype=np.int32)
        self._prev_executed_actions = np.full((self.cfg.n_agents,), -1, dtype=np.int64)
        self.episode_targets_collected = 0
        self.episode_pheromone_deposit_events = 0
        self.first_pickup_step: int | None = None
        self.first_delivery_step: int | None = None

        self.step_count = 0
        self.terminated = False
        self.truncated = False

        self.pheromone_grid = None

        self._pygame_inited = False
        self._screen = None
        self._clock = None

        self._single_obs_dim = self._compute_single_obs_dim()
        self._obs_dim = self._compute_obs_dim()
        self._obs_history = np.zeros(
            (self.cfg.n_agents, self.cfg.observation_history_steps, self._single_obs_dim),
            dtype=np.float32,
        )
        self._observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(self.cfg.num_actions)
            for agent in self.possible_agents
        }
        self._state_dim = self._compute_state_dim()
        self._state_space = spaces.Box(low=-1.0, high=1.0, shape=(self._state_dim,), dtype=np.float32)

    def observation_space(self, agent: str):
        """Return the observation space for a given agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        """Return the action space for a given agent."""
        return self._action_spaces[agent]

    def state_space(self):
        """Return the centralized training-time state space for CTDE algorithms."""
        return self._state_space

    def state(self) -> np.ndarray:
        """Return a fixed-layout centralized training-time state vector."""
        return self._get_global_state()

    def _compute_single_obs_dim(self) -> int:
        """Return the length of one per-agent observation frame."""
        return (
            self.cfg.lidar_rays
            + 2  # detectable target distance/angle
            + (2 if self.cfg.obs_include_nest_direction else 0)
            + 2  # neighbor vector
            + 2  # heading (sin, cos)
            + 1  # speed
            + (1 if self.cfg.obs_include_food_presence else 0)
            + (1 if self.cfg.obs_include_carrying else 0)
            + self.cfg.pheromone_samples
        )

    def _compute_obs_dim(self) -> int:
        """Return the flattened length of the per-agent observation history."""
        return self._single_obs_dim * self.cfg.observation_history_steps

    def _compute_state_dim(self) -> int:
        """Return the centralized training-state dimension for CTDE algorithms."""
        target_slots = max(int(self.cfg.n_targets), int(self.cfg.active_targets))
        return (
            self.cfg.n_agents * 7
            + target_slots * 4
            + self.cfg.n_obstacles * 4
            + 2
            + 5
        )

    def _get_global_state(self) -> np.ndarray:
        """Encode the centralized training-time state."""
        parts: list[np.ndarray] = []
        width = max(float(self.width), 1.0)
        height = max(float(self.height), 1.0)
        max_speed = max(float(self.cfg.max_speed), 1e-6)
        target_slots = max(int(self.cfg.n_targets), int(self.cfg.active_targets))
        max_targets = max(float(target_slots), 1.0)

        for i in range(self.cfg.n_agents):
            if i < len(self.agent_states):
                agent = self.agent_states[i]
                parts.append(
                    np.array(
                        [
                            np.clip((float(agent.x) / width) * 2.0 - 1.0, -1.0, 1.0),
                            np.clip((float(agent.y) / height) * 2.0 - 1.0, -1.0, 1.0),
                            math.sin(float(agent.theta)),
                            math.cos(float(agent.theta)),
                            np.clip(float(agent.v) / max_speed, -1.0, 1.0),
                            1.0 if agent.carrying_food else -1.0,
                            1.0 if i in self.failed_agent_indices else -1.0,
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                parts.append(np.full((7,), -1.0, dtype=np.float32))

        source_capacity = max(float(self.cfg.food_source_capacity), 1.0)
        for idx in range(target_slots):
            if idx < len(self.targets):
                tx, ty = self.targets[idx]
                remaining_uses = float(self.target_remaining_uses[idx]) if idx < len(self.target_remaining_uses) else 0.0
                parts.append(
                    np.array(
                        [
                            np.clip((float(tx) / width) * 2.0 - 1.0, -1.0, 1.0),
                            np.clip((float(ty) / height) * 2.0 - 1.0, -1.0, 1.0),
                            1.0,
                            np.clip(remaining_uses / source_capacity, 0.0, 1.0),
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                parts.append(np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32))

        for idx in range(self.cfg.n_obstacles):
            if idx < len(self.obstacles):
                obstacle = self.obstacles[idx]
                parts.append(
                    np.array(
                        [
                            np.clip((float(obstacle.x) / width) * 2.0 - 1.0, -1.0, 1.0),
                            np.clip((float(obstacle.y) / height) * 2.0 - 1.0, -1.0, 1.0),
                            np.clip(float(obstacle.width) / width, 0.0, 1.0),
                            np.clip(float(obstacle.height) / height, 0.0, 1.0),
                        ],
                        dtype=np.float32,
                    )
                )
            else:
                parts.append(np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32))

        parts.append(
            np.array(
                [
                    np.clip((float(self.nest_position[0]) / width) * 2.0 - 1.0, -1.0, 1.0),
                    np.clip((float(self.nest_position[1]) / height) * 2.0 - 1.0, -1.0, 1.0),
                ],
                dtype=np.float32,
            )
        )
        parts.append(
            np.array(
                [
                    np.clip(float(self.step_count) / max(float(self.cfg.max_steps), 1.0), 0.0, 1.0),
                    np.clip(float(self.food_delivered) / max_targets, 0.0, 1.0),
                    np.clip(float(self._coverage_ratio()), 0.0, 1.0),
                    np.clip(float(self._mean_pheromone_usage()), 0.0, 1.0),
                    np.clip(float(len(self.targets)) / max_targets, 0.0, 1.0),
                ],
                dtype=np.float32,
            )
        )

        state = np.concatenate(parts).astype(np.float32)
        if state.shape != (self._state_dim,):
            raise RuntimeError(f"Centralized state shape mismatch: expected {(self._state_dim,)}, got {state.shape}")
        return state

    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment and return initial observations and info."""
        # (Re)initialize RNG and episode state, then spawn a fresh world.
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.agents = self.possible_agents[:]
        self._held_actions.fill(0)
        self._hold_remaining.fill(0)
        self._prev_executed_actions.fill(-1)

        # Randomize dynamics driver if mixed mode is enabled.
        if self.cfg.dynamics_mode == "mixed":
            self.driver = self._select_driver(self.rng.choice(["tank", "hover"]))
        else:
            self.driver = self._select_driver(self.cfg.dynamics_mode)

        # World state: obstacles, targets, agents.
        self._spawn_obstacles()
        self._spawn_nest()
        self._spawn_targets()
        self._spawn_agents()
        self.food_delivered = 0
        self.episode_food_source_respawns = 0
        self.episode_targets_collected = 0
        self.episode_pheromone_deposit_events = 0
        self.first_pickup_step = None
        self.first_delivery_step = None
        self._init_coverage_grid()
        self._assign_failed_agents()
        self._update_coverage()
        self._reset_food_shaping_state()
        self._prev_nest_distances = self._compute_nest_distance_state()

        # Optional pheromone grid for stigmergy.
        if self.cfg.pheromone_enabled:
            grid_w = self.width // self.cfg.pheromone_cell_size + 1
            grid_h = self.height // self.cfg.pheromone_cell_size + 1
            self.pheromone_grid = np.zeros((grid_h, grid_w), dtype=np.float32)
        else:
            self.pheromone_grid = None

        obs = self._get_obs(reset_history=True)
        obs_dict = {agent: obs[i] for i, agent in enumerate(self.possible_agents)}
        info = {
            "n_targets": len(self.targets),
            "food_delivered": self.food_delivered,
            "exploration_coverage": self._coverage_ratio(),
            "food_source_respawns": self.episode_food_source_respawns,
            "food_source_capacity": int(self.cfg.food_source_capacity),
            "food_units_remaining": int(sum(self.target_remaining_uses)),
        }
        info_dict = {agent: info for agent in self.possible_agents}
        return obs_dict, info_dict

    def step(self, actions: Dict[str, int]):
        """Advance the simulation by one step using agent actions."""
        if not self.agents:
            raise RuntimeError("step() called with no active agents. Call reset() to start a new episode.")
        if not isinstance(actions, dict):
            raise ValueError("actions must be a dict keyed by agent id (Parallel API).")
        for agent in self.agents:
            if agent not in actions:
                raise ValueError(f"Missing action for agent {agent}.")

        # One environment tick: apply actions, move agents, compute rewards/obs.
        requested_actions = np.array([actions[agent] for agent in self.agents], dtype=np.int64)
        actions = np.empty_like(requested_actions)
        for i in range(self.cfg.n_agents):
            if self._hold_remaining[i] > 0:
                actions[i] = self._held_actions[i]
                self._hold_remaining[i] -= 1
            else:
                actions[i] = requested_actions[i]
                self._held_actions[i] = actions[i]
                self._hold_remaining[i] = max(int(self.cfg.action_repeat_steps) - 1, 0)

        current_step = self.step_count + 1
        prev_nest_distances = self._prev_nest_distances.copy()

        # Start with per-step reward for all agents.
        rewards = np.full((self.cfg.n_agents,), self.cfg.reward_step, dtype=np.float32)
        collisions = 0
        reward_breakdown = {
            "step": float(self.cfg.reward_step * self.cfg.n_agents),
            "pickup": 0.0,
            "delivery": 0.0,
            "undelivered": 0.0,
            "nest_approach": 0.0,
            "collision": 0.0,
            "new_cell": 0.0,
            "food_approach": 0.0,
            "food_detected": 0.0,
            "action_switch": 0.0,
            "pheromone_follow": 0.0,
            "pheromone_usage": 0.0,
            "pheromone_deposit": 0.0,
        }

        for i, action_id in enumerate(actions):
            if i in self.failed_agent_indices:
                continue
            prev_action = int(self._prev_executed_actions[i])
            if prev_action >= 0 and int(action_id) != prev_action:
                rewards[i] += float(self.cfg.reward_action_switch)
                reward_breakdown["action_switch"] += float(self.cfg.reward_action_switch)
            self._prev_executed_actions[i] = int(action_id)

        for i, agent in enumerate(self.agent_states):
            if i in self.failed_agent_indices:
                continue
            action_id = int(actions[i])
            throttle, turn, deposit_requested = self.action_table[action_id]
            # Propose next state from dynamics, then check collisions.
            proposed = self.driver.apply(agent, (throttle, turn), self.cfg.dt, self.cfg, self.rng)

            collided = self._handle_collisions(proposed)
            if collided:
                rewards[i] += self.cfg.reward_collision
                collisions += 1
                reward_breakdown["collision"] += float(self.cfg.reward_collision)
                # Clear action hold on impact and apply a Newton-style contact
                # response so the agent can separate from the collider.
                self._hold_remaining[i] = 0
                self._held_actions[i] = 0
                self.agent_states[i] = self._resolve_collision_state(agent, proposed)
            else:
                self.agent_states[i] = proposed

        # Handle target collection and pheromone updates.
        picked_up, delivered, pickup_reward, delivery_reward = self._handle_targets(rewards)
        reward_breakdown["pickup"] += float(pickup_reward)
        reward_breakdown["delivery"] += float(delivery_reward)
        nest_approach_reward = self._apply_nest_return_shaping(rewards, prev_nest_distances)
        reward_breakdown["nest_approach"] += float(nest_approach_reward)

        food_approach_reward, food_detect_reward = self._apply_food_shaping(rewards)
        reward_breakdown["food_approach"] += float(food_approach_reward)
        reward_breakdown["food_detected"] += float(food_detect_reward)

        coverage_reward_total, new_cells = self._apply_exploration_reward(rewards)
        reward_breakdown["new_cell"] += float(coverage_reward_total)

        pheromone_usage_reward, pheromone_follow_reward, pheromone_usage = self._apply_pheromone_reward(rewards, actions)
        reward_breakdown["pheromone_usage"] += float(pheromone_usage_reward)
        reward_breakdown["pheromone_follow"] += float(pheromone_follow_reward)

        if self.cfg.pheromone_enabled:
            pheromone_deposit_events, pheromone_deposit_cost = self._update_pheromone(
                actions,
                rewards,
                prev_nest_distances,
            )
        else:
            pheromone_deposit_events = 0
            pheromone_deposit_cost = 0.0
        reward_breakdown["pheromone_deposit"] += float(pheromone_deposit_cost)
        self._prev_nest_distances = self._compute_nest_distance_state()
        self.episode_targets_collected += int(picked_up)
        self.episode_pheromone_deposit_events += int(pheromone_deposit_events)
        if picked_up > 0 and self.first_pickup_step is None:
            self.first_pickup_step = current_step
        if delivered > 0 and self.first_delivery_step is None:
            self.first_delivery_step = current_step

        # Episode end conditions.
        self.step_count += 1
        if (
            not self.cfg.target_respawn
            and len(self.targets) == 0
            and not any(agent.carrying_food for agent in self.agent_states)
        ):
            self.terminated = True
        if self.step_count >= self.cfg.max_steps:
            self.truncated = True

        undelivered_count = 0
        undelivered_penalty = 0.0
        if self.terminated or self.truncated:
            undelivered_count, undelivered_penalty = self._apply_undelivered_food_penalty(rewards)
            reward_breakdown["undelivered"] += float(undelivered_penalty)

        obs = self._get_obs()
        obs_dict = {agent: obs[i] for i, agent in enumerate(self.possible_agents)}
        rewards_dict = {agent: float(rewards[i]) for i, agent in enumerate(self.possible_agents)}
        terminations = {agent: self.terminated for agent in self.possible_agents}
        truncations = {agent: self.truncated for agent in self.possible_agents}
        info = {
            "targets_collected": picked_up,
            "episode_targets_collected": self.episode_targets_collected,
            "food_delivered": delivered,
            "episode_food_delivered": self.food_delivered,
            "collisions": collisions,
            "new_cells_visited": new_cells,
            "coverage_reward_total": coverage_reward_total,
            "exploration_coverage": self._coverage_ratio(),
            "pheromone_usage": pheromone_usage,
            "pheromone_deposit_events": pheromone_deposit_events,
            "episode_pheromone_deposit_events": self.episode_pheromone_deposit_events,
            "first_pickup_step": self.first_pickup_step if self.first_pickup_step is not None else -1,
            "first_delivery_step": self.first_delivery_step if self.first_delivery_step is not None else -1,
            "pickup_to_delivery_latency": (
                self.first_delivery_step - self.first_pickup_step
                if self.first_pickup_step is not None and self.first_delivery_step is not None
                else -1
            ),
            "carrying_agents": int(sum(1 for agent in self.agent_states if agent.carrying_food)),
            "episode_length": self.step_count,
            "failed_agents": len(self.failed_agent_indices),
            "active_targets": len(self.targets),
            "food_source_respawns": self.episode_food_source_respawns,
            "food_source_capacity": int(self.cfg.food_source_capacity),
            "food_units_remaining": int(sum(self.target_remaining_uses)),
            "food_source_uses_remaining": [int(value) for value in self.target_remaining_uses],
            "undelivered_carrying_agents": int(undelivered_count),
            "undelivered_food_penalty": float(undelivered_penalty),
            "episode_done_reason": (
                "max_steps"
                if self.truncated
                else "all_targets_cleared"
                if self.terminated
                else ""
            ),
            "reward_breakdown": reward_breakdown,
        }
        infos = {agent: info for agent in self.possible_agents}
        if self.terminated or self.truncated:
            self.agents = []
        return obs_dict, rewards_dict, terminations, truncations, infos

    def render(self, mode: str = "human", fps: int = 60):
        """Render the current state to a PyGame window."""
        # Draw the current world state to a PyGame window.
        if self.headless:
            return
        self._ensure_pygame()

        # Background.
        self._screen.fill((20, 20, 26))

        if self.cfg.pheromone_enabled and self.cfg.render_pheromone:
            self._draw_pheromone()

        # Obstacles.
        for rect in self.obstacles:
            pygame.draw.rect(self._screen, (70, 70, 80), rect)

        # Targets.
        source_capacity = max(int(self.cfg.food_source_capacity), 1)
        target_draw_data = []
        for idx, (tx, ty) in enumerate(self.targets):
            remaining_uses = self.target_remaining_uses[idx] if idx < len(self.target_remaining_uses) else source_capacity
            ratio = np.clip(float(remaining_uses) / float(source_capacity), 0.0, 1.0)
            target_color = (
                int(70 + 40 * (1.0 - ratio)),
                int(150 + 70 * ratio),
                int(70 + 20 * ratio),
            )
            target_draw_data.append((float(tx), float(ty), int(remaining_uses), target_color))

        # Nest.
        if self.cfg.nest_enabled:
            nx, ny = self.nest_position
            pygame.draw.circle(self._screen, (80, 140, 220), (int(nx), int(ny)), int(self.cfg.nest_radius), 3)
            pygame.draw.circle(self._screen, (50, 80, 140), (int(nx), int(ny)), int(self.cfg.nest_radius // 2))

        # Agents (body + heading line).
        for i, agent in enumerate(self.agent_states):
            x, y = int(agent.x), int(agent.y)
            if i in self.failed_agent_indices:
                body_color = (120, 120, 120)
            else:
                body_color = (70, 220, 140) if agent.carrying_food else (200, 160, 50)
            pygame.draw.circle(self._screen, body_color, (x, y), int(self.cfg.agent_radius))
            if agent.carrying_food:
                pygame.draw.circle(self._screen, (235, 255, 180), (x, y), int(self.cfg.agent_radius) + 3, 2)
                pygame.draw.circle(self._screen, (240, 255, 200), (x, y), max(2, int(self.cfg.agent_radius // 2)))
                carried_y = y - int(self.cfg.agent_radius) - 5
                pygame.draw.circle(self._screen, (255, 220, 90), (x, carried_y), max(3, int(self.cfg.target_radius)))
                pygame.draw.circle(self._screen, (60, 50, 20), (x, carried_y), max(3, int(self.cfg.target_radius)), 1)
            hx = x + int(math.cos(agent.theta) * self.cfg.agent_radius)
            hy = y + int(math.sin(agent.theta) * self.cfg.agent_radius)
            pygame.draw.line(self._screen, (255, 240, 180), (x, y), (hx, hy), 2)

        # Draw food sources after agents so they remain visible even when an
        # agent is standing on top of the source. Also add simple "remaining
        # uses" pips so repeated-use sources are obvious in demo.
        for tx, ty, remaining_uses, target_color in target_draw_data:
            center = (int(tx), int(ty))
            pygame.draw.circle(self._screen, target_color, center, int(self.cfg.target_radius))
            pygame.draw.circle(self._screen, (230, 245, 210), center, int(self.cfg.target_radius) + 2, 1)
            pygame.draw.circle(self._screen, (30, 30, 30), center, int(self.cfg.target_radius), 1)
            pip_radius = 2
            pip_spacing = 5
            start_x = int(tx) - ((remaining_uses - 1) * pip_spacing) // 2
            pip_y = int(ty) - int(self.cfg.target_radius) - 6
            for pip_idx in range(max(0, remaining_uses)):
                px = start_x + pip_idx * pip_spacing
                pygame.draw.circle(self._screen, (255, 240, 150), (px, pip_y), pip_radius)
                pygame.draw.circle(self._screen, (50, 40, 20), (px, pip_y), pip_radius, 1)

        # Present frame and limit FPS.
        pygame.display.flip()
        self._clock.tick(fps)

    def save_screenshot(self, path: str):
        """Save the current render buffer to an image file."""
        if self.headless:
            raise RuntimeError("Cannot save screenshot in headless mode.")
        self._ensure_pygame()
        pygame.image.save(self._screen, path)

    def close(self):
        """Shut down PyGame resources if they were initialized."""
        if self._pygame_inited:
            pygame.quit()
            self._pygame_inited = False

    def _ensure_pygame(self):
        """Initialize PyGame on first render, no-op if already initialized."""
        # Lazy-init PyGame so headless training doesn't open a window.
        if not self._pygame_inited:
            pygame.init()
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Swarm RL")
            self._clock = pygame.time.Clock()
            self._pygame_inited = True

    def _build_action_table(self):
        """Create the discrete action lookup table (throttle, turn, deposit)."""
        throttle_vals = [-1.0, 0.0, 1.0]
        turn_vals = [-1.0, 0.0, 1.0]
        deposit_vals = [0, 1]
        table = []
        for throttle in throttle_vals:
            for turn in turn_vals:
                for deposit in deposit_vals:
                    table.append((throttle, turn, deposit))
        return table

    def _select_driver(self, mode: str) -> DynamicsDriver:
        """Return a dynamics driver based on the configured mode."""
        if mode == "hover":
            return HovercraftDriver()
        return TankKinematicsDriver()

    def _spawn_agents(self):
        """Randomly place agents in non-colliding free space."""
        # Randomly place agents in free space.
        self.agent_states = []
        for _ in range(self.cfg.n_agents):
            pos = self._sample_free_position(self.cfg.agent_radius)
            theta = self.rng.uniform(-math.pi, math.pi)
            self.agent_states.append(AgentState(pos[0], pos[1], theta))

    def _spawn_targets(self):
        """Randomly place targets in non-colliding free space."""
        # Randomly place targets in free space.
        self.targets = []
        self.target_remaining_uses = []
        self._respawn_targets(target_count=self._target_spawn_count())

    def _spawn_nest(self):
        """Place a single nest location away from obstacles and borders."""
        if not self.cfg.nest_enabled:
            self.nest_position = (self.width * 0.5, self.height * 0.5)
            return
        self.nest_position = self._sample_free_position(self.cfg.nest_radius)

    def _assign_failed_agents(self):
        """Randomly choose a subset of agents that are inactive for the episode."""
        failed = min(max(self.cfg.failed_agent_count, 0), self.cfg.n_agents)
        if failed <= 0:
            self.failed_agent_indices = set()
            return
        indices = self.rng.choice(self.cfg.n_agents, size=failed, replace=False)
        self.failed_agent_indices = {int(i) for i in np.atleast_1d(indices)}

    def _spawn_obstacles(self):
        """Create random obstacle rectangles with simple overlap avoidance."""
        # Randomly generate rectangular obstacles without overlaps.
        self.obstacles = []
        attempts = 0
        while len(self.obstacles) < self.cfg.n_obstacles and attempts < 200:
            attempts += 1
            w = self.rng.integers(60, 120)
            h = self.rng.integers(40, 120)
            x = self.rng.integers(30, self.width - 30 - w)
            y = self.rng.integers(30, self.height - 30 - h)
            rect = pygame.Rect(int(x), int(y), int(w), int(h))
            if any(rect.colliderect(o) for o in self.obstacles):
                continue
            self.obstacles.append(rect)

    def _sample_free_position(self, radius: float):
        """Sample a random position that does not overlap obstacles or entities."""
        for _ in range(200):
            x = self.rng.uniform(radius, self.width - radius)
            y = self.rng.uniform(radius, self.height - radius)
            circle = pygame.Rect(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
            if any(circle.colliderect(o) for o in self.obstacles):
                continue
            if self.cfg.nest_enabled:
                nx, ny = self.nest_position
                nest_clearance = radius + self.cfg.nest_radius
                if (nx - x) ** 2 + (ny - y) ** 2 < nest_clearance ** 2:
                    continue
            if any((ax - x) ** 2 + (ay - y) ** 2 < (radius * 2) ** 2 for ax, ay in self.targets):
                continue
            if any((agent.x - x) ** 2 + (agent.y - y) ** 2 < (radius * 2) ** 2 for agent in self.agent_states):
                continue
            return x, y
        return radius, radius

    def _target_spawn_count(self) -> int:
        """Return the desired number of simultaneously active targets."""
        if self.cfg.target_respawn:
            return max(1, int(self.cfg.active_targets))
        return max(0, int(self.cfg.n_targets))

    def _respawn_targets(self, target_count: int | None = None) -> int:
        """Spawn replacement targets until the active target count is reached."""
        desired = self._target_spawn_count() if target_count is None else max(0, int(target_count))
        spawned = 0
        while len(self.targets) < desired:
            pos = self._sample_free_position(self.cfg.target_radius)
            self.targets.append(pos)
            self.target_remaining_uses.append(int(self.cfg.food_source_capacity))
            spawned += 1
        return spawned

    def _handle_collisions(self, proposed: AgentState) -> bool:
        """Return True if the proposed state collides with bounds/obstacles."""
        # Check arena bounds and obstacle collision for a proposed state.
        if proposed.x < self.cfg.agent_radius or proposed.x > self.width - self.cfg.agent_radius:
            return True
        if proposed.y < self.cfg.agent_radius or proposed.y > self.height - self.cfg.agent_radius:
            return True
        agent_rect = pygame.Rect(
            int(proposed.x - self.cfg.agent_radius),
            int(proposed.y - self.cfg.agent_radius),
            int(self.cfg.agent_radius * 2),
            int(self.cfg.agent_radius * 2),
        )
        return any(agent_rect.colliderect(o) for o in self.obstacles)

    def _collision_normal(self, proposed: AgentState) -> np.ndarray | None:
        """Estimate a contact normal for boundary or obstacle collisions."""
        normals: list[np.ndarray] = []
        radius = float(self.cfg.agent_radius)

        if proposed.x < radius:
            normals.append(np.array([1.0, 0.0], dtype=np.float32))
        elif proposed.x > float(self.width) - radius:
            normals.append(np.array([-1.0, 0.0], dtype=np.float32))
        if proposed.y < radius:
            normals.append(np.array([0.0, 1.0], dtype=np.float32))
        elif proposed.y > float(self.height) - radius:
            normals.append(np.array([0.0, -1.0], dtype=np.float32))

        cx = float(proposed.x)
        cy = float(proposed.y)
        for rect in self.obstacles:
            nearest_x = float(np.clip(cx, rect.left, rect.right))
            nearest_y = float(np.clip(cy, rect.top, rect.bottom))
            dx = cx - nearest_x
            dy = cy - nearest_y
            dist_sq = dx * dx + dy * dy
            if dist_sq > radius * radius:
                continue
            if dist_sq > 1e-8:
                inv_dist = 1.0 / math.sqrt(dist_sq)
                normals.append(np.array([dx * inv_dist, dy * inv_dist], dtype=np.float32))
            else:
                left_gap = abs(cx - rect.left)
                right_gap = abs(rect.right - cx)
                top_gap = abs(cy - rect.top)
                bottom_gap = abs(rect.bottom - cy)
                min_gap = min(left_gap, right_gap, top_gap, bottom_gap)
                if min_gap == left_gap:
                    normals.append(np.array([-1.0, 0.0], dtype=np.float32))
                elif min_gap == right_gap:
                    normals.append(np.array([1.0, 0.0], dtype=np.float32))
                elif min_gap == top_gap:
                    normals.append(np.array([0.0, -1.0], dtype=np.float32))
                else:
                    normals.append(np.array([0.0, 1.0], dtype=np.float32))

        if not normals:
            return None
        normal = np.sum(normals, axis=0)
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-8:
            return np.array([0.0, -1.0], dtype=np.float32)
        return (normal / norm).astype(np.float32)

    def _resolve_collision_state(self, agent: AgentState, proposed: AgentState) -> AgentState:
        """Apply a normal impulse and separation step after collision."""
        normal = self._collision_normal(proposed)
        if normal is None:
            normal = np.array([-math.cos(proposed.theta), -math.sin(proposed.theta)], dtype=np.float32)

        forward = np.array([math.cos(proposed.theta), math.sin(proposed.theta)], dtype=np.float32)
        right = np.array([math.cos(proposed.theta + math.pi / 2.0), math.sin(proposed.theta + math.pi / 2.0)], dtype=np.float32)

        vel_world = forward * float(proposed.v) + right * float(proposed.v_lat)
        vn = float(np.dot(vel_world, normal))
        restitution = 0.2
        if vn < 0.0:
            vel_world = vel_world - (1.0 + restitution) * vn * normal
        vel_world = vel_world + normal * (0.05 * float(self.cfg.max_speed))

        next_v = float(np.clip(np.dot(vel_world, forward), -self.cfg.max_speed, self.cfg.max_speed))
        next_v_lat = float(np.clip(np.dot(vel_world, right), -self.cfg.max_speed, self.cfg.max_speed))

        for step_scale in (0.6, 1.0, 1.4):
            sep = max(1.0, float(self.cfg.agent_radius) * step_scale)
            cand = AgentState(
                x=float(np.clip(agent.x + normal[0] * sep, self.cfg.agent_radius + 1.0, self.width - self.cfg.agent_radius - 1.0)),
                y=float(np.clip(agent.y + normal[1] * sep, self.cfg.agent_radius + 1.0, self.height - self.cfg.agent_radius - 1.0)),
                theta=proposed.theta,
                v=next_v * 0.35,
                omega=proposed.omega,
                v_lat=next_v_lat * 0.35,
                carrying_food=agent.carrying_food,
            )
            if not self._handle_collisions(cand):
                return cand

        return AgentState(
            x=float(np.clip(agent.x, self.cfg.agent_radius + 1.0, self.width - self.cfg.agent_radius - 1.0)),
            y=float(np.clip(agent.y, self.cfg.agent_radius + 1.0, self.height - self.cfg.agent_radius - 1.0)),
            theta=proposed.theta,
            v=0.0,
            omega=proposed.omega,
            v_lat=0.0,
            carrying_food=agent.carrying_food,
        )

    def _handle_targets(self, rewards: np.ndarray) -> tuple[int, int, float, float]:
        """Handle food pickup and optional nest delivery."""
        picked_up = 0
        pickup_reward = 0.0
        remaining_targets = []
        remaining_uses = []
        respawns = 0
        for source_index, (tx, ty) in enumerate(self.targets):
            uses_left = int(self.target_remaining_uses[source_index]) if source_index < len(self.target_remaining_uses) else int(self.cfg.food_source_capacity)
            collected_by = None
            for i, agent in enumerate(self.agent_states):
                if agent.carrying_food:
                    continue
                if (agent.x - tx) ** 2 + (agent.y - ty) ** 2 <= (self.cfg.target_radius + self.cfg.agent_radius) ** 2:
                    collected_by = i
                    break
            if collected_by is not None:
                if self.cfg.require_nest_delivery and self.cfg.nest_enabled:
                    self.agent_states[collected_by].carrying_food = True
                    rewards[collected_by] += self.cfg.reward_pickup
                    pickup_reward += float(self.cfg.reward_pickup)
                else:
                    rewards[collected_by] += self.cfg.reward_target
                    pickup_reward += float(self.cfg.reward_target)
                picked_up += 1
                uses_left -= 1
                if uses_left > 0:
                    remaining_targets.append((tx, ty))
                    remaining_uses.append(uses_left)
            else:
                remaining_targets.append((tx, ty))
                remaining_uses.append(uses_left)
        self.targets = remaining_targets
        self.target_remaining_uses = remaining_uses
        if self.cfg.target_respawn:
            respawns = self._respawn_targets()
            self.episode_food_source_respawns += int(respawns)
        delivered, delivery_reward = self._handle_nest_delivery(rewards)
        return picked_up, delivered, pickup_reward, delivery_reward

    def _handle_nest_delivery(self, rewards: np.ndarray) -> tuple[int, float]:
        """Reward agents that return carried food to the nest."""
        if not (self.cfg.nest_enabled and self.cfg.require_nest_delivery):
            return 0, 0.0
        delivered = 0
        reward_total = 0.0
        nx, ny = self.nest_position
        nest_reach = (self.cfg.nest_radius + self.cfg.agent_radius) ** 2
        for i, agent in enumerate(self.agent_states):
            if not agent.carrying_food:
                continue
            if (agent.x - nx) ** 2 + (agent.y - ny) ** 2 <= nest_reach:
                agent.carrying_food = False
                rewards[i] += self.cfg.reward_nest_delivery
                self.food_delivered += 1
                delivered += 1
                reward_total += float(self.cfg.reward_nest_delivery)
        return delivered, reward_total

    def _apply_undelivered_food_penalty(self, rewards: np.ndarray) -> tuple[int, float]:
        """Penalize agents that finish an episode while still carrying food."""
        penalty_value = float(self.cfg.reward_undelivered_food)
        if penalty_value == 0.0:
            return 0, 0.0
        penalized = 0
        penalty_total = 0.0
        for i, agent in enumerate(self.agent_states):
            if not agent.carrying_food:
                continue
            rewards[i] += penalty_value
            penalized += 1
            penalty_total += penalty_value
        return penalized, penalty_total

    def _reset_food_shaping_state(self) -> None:
        """Initialize detectable-food shaping state from the freshly reset world."""
        distances, detected = self._compute_detectable_food_state()
        self._prev_detectable_food_distances = distances
        self._prev_food_detected = detected

    def _init_coverage_grid(self):
        """Initialize per-episode exploration coverage tracking."""
        cell = max(self.cfg.coverage_cell_size, 1)
        grid_w = self.width // cell + 1
        grid_h = self.height // cell + 1
        self.coverage_grid = np.zeros((grid_h, grid_w), dtype=np.bool_)
        self.covered_cells = 0
        self.total_cover_cells = int(self.coverage_grid.size)

    def _update_coverage(self) -> tuple[int, np.ndarray]:
        """Mark currently occupied coverage cells and return total/per-agent new-cell counts."""
        if self.coverage_grid is None:
            return 0, np.zeros((self.cfg.n_agents,), dtype=np.int32)
        cell = max(self.cfg.coverage_cell_size, 1)
        new_cells = 0
        per_agent_new_cells = np.zeros((self.cfg.n_agents,), dtype=np.int32)
        for idx, agent in enumerate(self.agent_states):
            gx = int(np.clip(agent.x // cell, 0, self.coverage_grid.shape[1] - 1))
            gy = int(np.clip(agent.y // cell, 0, self.coverage_grid.shape[0] - 1))
            if not self.coverage_grid[gy, gx]:
                self.coverage_grid[gy, gx] = True
                self.covered_cells += 1
                new_cells += 1
                per_agent_new_cells[idx] += 1
        return new_cells, per_agent_new_cells

    def _coverage_ratio(self) -> float:
        """Return the fraction of visited coverage cells this episode."""
        if self.total_cover_cells <= 0:
            return 0.0
        return float(self.covered_cells / self.total_cover_cells)

    def _apply_exploration_reward(self, rewards: np.ndarray) -> tuple[float, int]:
        """Reward visiting previously unseen coverage cells."""
        new_cells, per_agent_new_cells = self._update_coverage()
        reward_total = 0.0
        if new_cells > 0:
            for i, count in enumerate(per_agent_new_cells):
                if count <= 0:
                    continue
                scale = float(self.cfg.carrying_reward_new_cell_scale) if self.agent_states[i].carrying_food else 1.0
                reward = float(count) * float(self.cfg.reward_new_cell) * scale
                rewards[i] += reward
                reward_total += reward
        return reward_total, new_cells

    def _apply_food_shaping(self, rewards: np.ndarray) -> tuple[float, float]:
        """Apply local food-detection shaping without changing the observation contract."""
        # This uses environment-side distance bookkeeping for training reward
        # shaping only. The policy still consumes the existing local observation.
        current_distances, current_detected = self._compute_detectable_food_state()
        approach_total = 0.0
        detected_total = 0.0

        for i, agent in enumerate(self.agent_states):
            if i in self.failed_agent_indices or agent.carrying_food:
                continue

            prev_dist = float(self._prev_detectable_food_distances[i])
            curr_dist = float(current_distances[i])
            if np.isfinite(prev_dist) and np.isfinite(curr_dist) and curr_dist != prev_dist:
                progress = np.clip(
                    (prev_dist - curr_dist) / max(self.cfg.lidar_max_range, 1e-6),
                    -1.0,
                    1.0,
                )
                reward = float(self.cfg.reward_food_approach) * progress
                rewards[i] += reward
                approach_total += reward

            if bool(current_detected[i]) and not bool(self._prev_food_detected[i]):
                reward = float(self.cfg.reward_food_detected)
                rewards[i] += reward
                detected_total += reward

        self._prev_detectable_food_distances = current_distances
        self._prev_food_detected = current_detected
        return approach_total, detected_total

    def _compute_nest_distance_state(self) -> np.ndarray:
        """Return current per-agent distances to the nest."""
        if not self.cfg.nest_enabled:
            return np.full((self.cfg.n_agents,), np.inf, dtype=np.float32)
        nx, ny = self.nest_position
        distances = np.full((self.cfg.n_agents,), np.inf, dtype=np.float32)
        for i, agent in enumerate(self.agent_states):
            distances[i] = float(math.hypot(agent.x - nx, agent.y - ny))
        return distances

    def _apply_nest_return_shaping(self, rewards: np.ndarray, prev_nest_distances: np.ndarray) -> float:
        """Reward carrying agents for making signed progress back toward the nest."""
        if not (self.cfg.nest_enabled and self.cfg.require_nest_delivery):
            return 0.0
        current_distances = self._compute_nest_distance_state()
        reward_total = 0.0
        for i, agent in enumerate(self.agent_states):
            if i in self.failed_agent_indices or not agent.carrying_food:
                continue
            prev_dist = float(prev_nest_distances[i])
            curr_dist = float(current_distances[i])
            if not (np.isfinite(prev_dist) and np.isfinite(curr_dist)) or curr_dist == prev_dist:
                continue
            progress = np.clip(
                (prev_dist - curr_dist) / max(float(self.cfg.lidar_max_range), 1e-6),
                -1.0,
                1.0,
            )
            reward = float(self.cfg.reward_nest_approach) * progress
            rewards[i] += reward
            reward_total += reward
        return reward_total

    def _apply_pheromone_reward(self, rewards: np.ndarray, actions: np.ndarray) -> tuple[float, float, float]:
        """Apply conservative pheromone shaping using existing sample geometry."""
        usage = self._mean_pheromone_usage()
        usage_reward_total = float(self.cfg.reward_pheromone_following * usage * self.cfg.n_agents)
        if usage_reward_total != 0.0 and self.cfg.n_agents > 0:
            rewards += usage_reward_total / self.cfg.n_agents

        follow_reward_total = 0.0
        for i, agent in enumerate(self.agent_states):
            if i in self.failed_agent_indices or agent.carrying_food:
                continue
            throttle, _, _ = self.action_table[int(actions[i])]
            if throttle <= 0.0:
                continue
            samples = self._pheromone_samples(agent)
            if samples.size == 0:
                continue
            split = max(1, len(samples) // 2)
            near = float(np.mean(samples[:split]))
            far = float(np.mean(samples[split:])) if split < len(samples) else near
            gradient = far - near
            if gradient <= self.cfg.pheromone_follow_min_gradient:
                continue
            reward = float(self.cfg.reward_pheromone_follow) * min(gradient, 1.0)
            rewards[i] += reward
            follow_reward_total += reward
        return usage_reward_total, follow_reward_total, usage

    def _update_pheromone(
        self,
        actions: np.ndarray,
        rewards: np.ndarray,
        prev_nest_distances: np.ndarray,
    ) -> tuple[int, float]:
        """Deposit, decay, and diffuse pheromone values."""
        # Deposit pheromone only for agents whose action explicitly requested it,
        # then decay/diffuse the grid globally. When pheromone_requires_food is
        # enabled, deposition is a physically motivated food-trail signal only.
        grid = self.pheromone_grid
        cell = self.cfg.pheromone_cell_size
        deposit_events = 0
        deposit_cost_total = 0.0
        current_nest_distances = self._compute_nest_distance_state()
        for i, agent in enumerate(self.agent_states):
            if i in self.failed_agent_indices:
                continue
            _, _, deposit_requested = self.action_table[int(actions[i])]
            if not deposit_requested:
                continue
            if self.cfg.pheromone_requires_food and not agent.carrying_food:
                continue
            if self.cfg.pheromone_deposit_requires_nest_progress:
                prev_dist = float(prev_nest_distances[i])
                curr_dist = float(current_nest_distances[i])
                if not (np.isfinite(prev_dist) and np.isfinite(curr_dist)) or curr_dist >= prev_dist:
                    continue
            gx = int(agent.x // cell)
            gy = int(agent.y // cell)
            if 0 <= gy < grid.shape[0] and 0 <= gx < grid.shape[1]:
                deposit_amount = self.cfg.pheromone_deposit
                if agent.carrying_food:
                    deposit_amount *= self.cfg.pheromone_deposit_carrying_scale
                grid[gy, gx] += deposit_amount
                rewards[i] += self.cfg.reward_pheromone_deposit_cost
                deposit_cost_total += float(self.cfg.reward_pheromone_deposit_cost)
                deposit_events += 1

        grid *= self.cfg.pheromone_decay
        diff = self.cfg.pheromone_diffuse_rate
        if diff > 0:
            # Use zero-padded neighbors so pheromone diffuses within the arena
            # instead of wrapping from one edge of the grid to the opposite side.
            padded = np.pad(grid, 1, mode="constant")
            up = padded[:-2, 1:-1]
            down = padded[2:, 1:-1]
            left = padded[1:-1, :-2]
            right = padded[1:-1, 2:]
            neighbor_avg = (up + down + left + right) * 0.25
            grid[:] = grid * (1.0 - diff) + neighbor_avg * diff
        min_value = float(self.cfg.pheromone_min_value)
        if min_value > 0:
            grid[grid < min_value] = 0.0
        return deposit_events, deposit_cost_total

    def _get_obs(self, reset_history: bool = False):
        """Assemble per-agent observations and flatten the recent history window."""
        frame_obs = self._get_obs_frame()
        if reset_history:
            self._obs_history[:] = frame_obs[:, None, :]
        else:
            self._obs_history[:, :-1, :] = self._obs_history[:, 1:, :]
            self._obs_history[:, -1, :] = frame_obs
        return self._obs_history.reshape(self.cfg.n_agents, self._obs_dim)

    def _get_obs_frame(self):
        """Assemble a single per-agent observation frame without history stacking."""
        obs_list = []
        for idx, agent in enumerate(self.agent_states):
            lidar = self._lidar_scan(agent)
            target_features = self._nearest_target_features(agent)
            nest_vec = self._nest_direction(agent)
            neighbor_vec = self._nearest_agent_vector(agent, idx)
            heading = np.array([math.sin(agent.theta), math.cos(agent.theta)], dtype=np.float32)
            speed = np.array([np.clip(agent.v / self.cfg.max_speed, -1.0, 1.0)], dtype=np.float32)
            food_presence = self._food_presence(agent)
            carrying = self._carrying_food(agent)
            pheromone = self._pheromone_samples(agent)

            parts = [lidar, target_features]
            if self.cfg.obs_include_nest_direction:
                parts.append(nest_vec)
            parts.extend([neighbor_vec, heading, speed])
            if self.cfg.obs_include_food_presence:
                parts.append(food_presence)
            if self.cfg.obs_include_carrying:
                parts.append(carrying)
            parts.append(pheromone)
            obs = np.concatenate(parts).astype(np.float32)
            if idx in self.failed_agent_indices:
                obs[:] = 0.0
            elif self.cfg.observation_noise_std > 0:
                noise = self.rng.normal(0.0, self.cfg.observation_noise_std, size=obs.shape)
                obs = np.clip(obs + noise.astype(np.float32), -1.0, 1.0)
            obs_list.append(obs)
        return np.stack(obs_list, axis=0)

    def _lidar_scan(self, agent: AgentState) -> np.ndarray:
        """Return normalized lidar ray distances for a single agent."""
        # Cast lidar rays and return normalized distances.
        rays = []
        half = self.cfg.lidar_rays // 2
        for i in range(self.cfg.lidar_rays):
            angle = agent.theta + (i - half) * (math.pi / (self.cfg.lidar_rays - 1))
            dist = self._ray_distance(agent.x, agent.y, angle)
            rays.append(dist / self.cfg.lidar_max_range)
        return np.array(rays, dtype=np.float32)

    def _ray_distance(self, x: float, y: float, angle: float) -> float:
        """Return distance from (x, y) to nearest obstacle/boundary along a ray."""
        # March a ray forward until it hits an obstacle or max range.
        max_range = self.cfg.lidar_max_range
        step = self.cfg.lidar_step
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dist = 0.0
        while dist < max_range:
            px = x + cos_a * dist
            py = y + sin_a * dist
            if px < 0 or px > self.width or py < 0 or py > self.height:
                return dist
            point_rect = pygame.Rect(int(px), int(py), 2, 2)
            if any(point_rect.colliderect(o) for o in self.obstacles):
                return dist
            dist += step
        return max_range

    def _nearest_target_vector(self, agent: AgentState) -> np.ndarray:
        """Return nearest target vector in agent-local coordinates when detectable."""
        target_state = self._nearest_detectable_target(agent)
        if target_state is None:
            return np.zeros(2, dtype=np.float32)
        rel_body, _, _ = target_state
        return np.clip(rel_body / self.cfg.lidar_max_range, -1.0, 1.0)

    def _nearest_target_features(self, agent: AgentState) -> np.ndarray:
        """Return detectable target distance and relative angle."""
        target_state = self._nearest_detectable_target(agent)
        if target_state is None:
            return np.zeros(2, dtype=np.float32)
        _, dist, angle = target_state
        dist_norm = np.array([np.clip(dist / self.cfg.lidar_max_range, 0.0, 1.0)], dtype=np.float32)
        angle_norm = np.array([np.clip(angle / math.pi, -1.0, 1.0)], dtype=np.float32)
        return np.concatenate([dist_norm, angle_norm]).astype(np.float32)

    def _nearest_detectable_target(self, agent: AgentState) -> tuple[np.ndarray, float, float] | None:
        """Return the nearest target body-frame vector, distance, and angle if detectable."""
        if not self.targets:
            return None

        targets = np.array(self.targets, dtype=np.float32)
        dx = targets[:, 0] - float(agent.x)
        dy = targets[:, 1] - float(agent.y)
        dists = np.hypot(dx, dy)
        visible: list[tuple[float, np.ndarray, float]] = []
        max_range = float(self.cfg.lidar_max_range)

        for idx, dist in enumerate(dists):
            dist = float(dist)
            if dist > max_range:
                continue
            world_angle = math.atan2(float(dy[idx]), float(dx[idx]))
            if not self._target_visible(agent, dist, world_angle):
                continue
            rel_world = np.array([dx[idx], dy[idx]], dtype=np.float32)
            rel_body = self._to_agent_frame(rel_world, agent.theta)
            angle = float(math.atan2(float(rel_body[1]), float(rel_body[0])))
            visible.append((dist, rel_body, angle))

        if not visible:
            return None

        dist, rel_body, angle = min(visible, key=lambda item: item[0])
        return rel_body, dist, angle

    def _target_visible(self, agent: AgentState, target_dist: float, world_angle: float) -> bool:
        """Return whether a target is within the front 180 degrees and unobstructed."""
        relative_angle = math.atan2(
            math.sin(world_angle - float(agent.theta)),
            math.cos(world_angle - float(agent.theta)),
        )
        if abs(relative_angle) > (math.pi / 2.0):
            return False
        obstacle_dist = self._ray_distance(float(agent.x), float(agent.y), world_angle)
        return obstacle_dist + float(self.cfg.target_radius) >= target_dist

    def _compute_detectable_food_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-agent nearest detectable-food distances and detection flags."""
        distances = np.full((self.cfg.n_agents,), np.inf, dtype=np.float32)
        detected = np.zeros((self.cfg.n_agents,), dtype=np.bool_)
        for i, agent in enumerate(self.agent_states):
            target_state = self._nearest_detectable_target(agent)
            if target_state is not None:
                _, nearest, _ = target_state
                distances[i] = float(nearest)
                detected[i] = True
        return distances, detected

    def _nest_direction(self, agent: AgentState) -> np.ndarray:
        """Return nest direction in agent-local coordinates."""
        if not self.cfg.nest_enabled:
            return np.zeros(2, dtype=np.float32)
        rel = np.array(
            [self.nest_position[0] - agent.x, self.nest_position[1] - agent.y],
            dtype=np.float32,
        )
        rel = self._to_agent_frame(rel, agent.theta)
        return np.clip(rel / self.cfg.lidar_max_range, -1.0, 1.0)

    def _nearest_agent_vector(self, agent: AgentState, idx: int) -> np.ndarray:
        """Return nearest neighbor vector in agent-local coordinates."""
        # Vector from agent to nearest neighbor, in agent-local coordinates.
        if self.cfg.n_agents <= 1:
            return np.zeros(2, dtype=np.float32)
        best = None
        best_dist = float("inf")
        for j, other in enumerate(self.agent_states):
            if j == idx:
                continue
            dx = other.x - agent.x
            dy = other.y - agent.y
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best = np.array([dx, dy], dtype=np.float32)
        if best is None:
            return np.zeros(2, dtype=np.float32)
        rel = self._to_agent_frame(best, agent.theta)
        return np.clip(rel / self.cfg.lidar_max_range, -1.0, 1.0)

    def _pheromone_samples(self, agent: AgentState) -> np.ndarray:
        """Sample pheromone values along the agent's forward direction."""
        # Sample pheromone intensity in front of the agent.
        if not (self.cfg.pheromone_enabled and self.cfg.obs_include_pheromone):
            return np.zeros(self.cfg.pheromone_samples, dtype=np.float32)
        grid = self.pheromone_grid
        cell = self.cfg.pheromone_cell_size
        awareness_radius = self._pheromone_awareness_radius()
        samples = []
        for dist in self._pheromone_sample_distances():
            if dist > awareness_radius:
                samples.append(0.0)
                continue
            sx = agent.x + math.cos(agent.theta) * dist
            sy = agent.y + math.sin(agent.theta) * dist
            gx = int(np.clip(sx // cell, 0, grid.shape[1] - 1))
            gy = int(np.clip(sy // cell, 0, grid.shape[0] - 1))
            samples.append(grid[gy, gx])
        samples = np.array(samples, dtype=np.float32)
        if samples.max() > 0:
            samples = samples / (samples.max() + 1e-6)
        return samples

    def _pheromone_sample_distances(self) -> list[float]:
        """Return the forward pheromone sampling distances for one observation."""
        spacing = float(self.cfg.agent_radius) * float(self.cfg.pheromone_sample_spacing_scale)
        return [float(i + 1) * spacing for i in range(self.cfg.pheromone_samples)]

    def _pheromone_awareness_radius(self) -> float:
        """Return the farthest distance at which pheromone can affect the observation."""
        distances = self._pheromone_sample_distances()
        return max(distances) if distances else 0.0

    def _food_presence(self, agent: AgentState) -> np.ndarray:
        """Return a binary local food-presence cue."""
        if not self.cfg.obs_include_food_presence:
            return np.zeros(1, dtype=np.float32)
        value = 1.0 if self._nearest_detectable_target(agent) is not None else 0.0
        return np.array([value], dtype=np.float32)

    def _carrying_food(self, agent: AgentState) -> np.ndarray:
        """Return whether the agent is currently carrying food."""
        if not self.cfg.obs_include_carrying:
            return np.zeros(1, dtype=np.float32)
        return np.array([1.0 if agent.carrying_food else 0.0], dtype=np.float32)

    def _to_agent_frame(self, vec: np.ndarray, theta: float) -> np.ndarray:
        """Rotate a world-space vector into the agent's local frame."""
        # Rotate a world-space vector into the agent's local frame.
        c = math.cos(-theta)
        s = math.sin(-theta)
        x, y = vec
        return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)

    def _draw_pheromone(self):
        """Render a heatmap-style visualization of the pheromone grid."""
        # Render pheromone heatmap as colored grid cells.
        grid = self.pheromone_grid
        if grid is None:
            return
        cell = self.cfg.pheromone_cell_size
        max_val = grid.max()
        if max_val <= 0:
            return
        for gy in range(grid.shape[0]):
            for gx in range(grid.shape[1]):
                val = grid[gy, gx] / max_val
                if val <= 0.01:
                    continue
                color = (int(40 + 160 * val), int(40 + 40 * val), int(80 + 120 * val))
                rect = pygame.Rect(gx * cell, gy * cell, cell, cell)
                self._screen.fill(color, rect)

    def _mean_pheromone_usage(self) -> float:
        """Return the mean local pheromone intensity under the agents."""
        if self.pheromone_grid is None or self.pheromone_grid.size == 0:
            return 0.0
        max_val = float(self.pheromone_grid.max())
        if max_val <= 1e-6:
            return 0.0
        cell = self.cfg.pheromone_cell_size
        values = []
        for agent in self.agent_states:
            gx = int(np.clip(agent.x // cell, 0, self.pheromone_grid.shape[1] - 1))
            gy = int(np.clip(agent.y // cell, 0, self.pheromone_grid.shape[0] - 1))
            values.append(float(self.pheromone_grid[gy, gx] / max_val))
        return float(np.mean(values)) if values else 0.0


if __name__ == "__main__":
    cfg = SwarmConfig()
    env = SwarmEnv(cfg)
    obs, info = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        actions = {agent: int(np.random.randint(0, cfg.num_actions)) for agent in env.agents}
        env.step(actions)
        env.render()
    env.close()
