from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwarmConfig:
    # World
    width: int = 900
    height: int = 600
    n_agents: int = 6
    n_targets: int = 4
    n_obstacles: int = 6
    agent_radius: float = 7.0
    target_radius: float = 5.0
    nest_radius: float = 22.0
    max_steps: int = 600

    # Action space (discrete)
    action_dim: int = 1
    num_actions: int = 18

    # Dynamics
    dt: float = 0.1
    max_speed: float = 120.0
    max_yaw_rate: float = 2.5
    accel: float = 300.0
    ang_accel: float = 8.0
    dynamics_mode: str = "tank"  # "tank", "hover", or "mixed"

    # Hovercraft dynamics
    hover_lat_damping: float = 0.85
    hover_lat_noise: float = 5.0
    hover_slip_chance: float = 0.08
    hover_slip_scale: float = 0.6

    # Sensors
    lidar_rays: int = 9
    lidar_max_range: float = 160.0
    lidar_step: float = 6.0
    obs_include_pheromone: bool = True
    pheromone_samples: int = 3
    pheromone_sample_spacing_scale: float = 1.5
    obs_include_food_presence: bool = True
    obs_include_nest_direction: bool = True
    obs_include_carrying: bool = True
    food_detection_radius: float = 150.0
    food_presence_radius: float = 120.0
    observation_history_steps: int = 3

    # Pheromone grid
    pheromone_enabled: bool = True
    pheromone_cell_size: int = 6
    pheromone_deposit: float = 1.0
    pheromone_decay: float = 0.985
    pheromone_diffuse_rate: float = 0.25
    pheromone_min_value: float = 1e-3
    pheromone_deposit_carrying_scale: float = 1.5
    pheromone_requires_food: bool = False

    # Rewards
    reward_target: float = 8.0
    reward_step: float = -0.01
    reward_collision: float = -0.2
    reward_pickup: float = 8.0
    reward_nest_delivery: float = 0.0 #10.0
    reward_exploration: float = 0.02
    reward_new_cell: float = 0.02
    reward_pheromone_following: float = 0.2
    reward_food_approach: float = 1 #reward_food_approach: float = 0.2
    reward_food_detected: float = 0.2 #reward_food_detected: float = 0.05
    reward_pheromone_follow: float = 0.03
    reward_pheromone_deposit_cost: float = -0.02
    pheromone_follow_min_gradient: float = 0.05

    # Task
    nest_enabled: bool = True
    require_nest_delivery: bool = True
    active_targets: int = 4
    target_respawn: bool = False
    coverage_cell_size: int = 24
    failed_agent_count: int = 0
    observation_noise_std: float = 0.0

    # Rendering
    render_pheromone: bool = True
    render_scale: float = 1.0

    # Seeding
    seed: int | None = None
