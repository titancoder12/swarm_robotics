from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SwarmConfig:
    # World
    width: int = 900
    height: int = 600
    n_agents: int = 6
    n_targets: int = 3
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
    action_repeat_steps: int = 2

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
    target_nest_distance_min: float = 0.0
    target_nest_distance_max: float = 0.0
    agent_spawn_near_target_radius: float = 0.0
    target_nest_corridor_clearance: float = 0.0
    start_carrying_food: bool = False
    obstacle_min_size: int = 40
    obstacle_max_size: int = 120

    # Pheromone grid
    pheromone_enabled: bool = True
    pheromone_cell_size: int = 6
    pheromone_deposit: float = 1.0
    pheromone_decay: float = 0.985
    pheromone_diffuse_rate: float = 0.25
    pheromone_min_value: float = 1e-3
    pheromone_deposit_carrying_scale: float = 1.5
    pheromone_requires_food: bool = True
    pheromone_deposit_requires_nest_progress: bool = True
    non_carrying_nest_pheromone_suppression_radius: float = 0.0
    non_carrying_nest_loiter_radius: float = 0.0
    non_carrying_nest_loiter_penalty: float = 0.0
    non_carrying_nest_crowding_radius: float = 0.0
    non_carrying_nest_crowding_penalty: float = 0.0
    non_carrying_nest_crowding_threshold: int = 2
    non_carrying_explore_radius: float = 0.0
    non_carrying_outward_reward: float = 0.0
    non_carrying_no_outward_progress_penalty: float = 0.0
    non_carrying_idle_near_nest_penalty: float = 0.0
    non_carrying_low_displacement_threshold: float = 0.0
    non_carrying_explore_ignore_pheromone: bool = False
    post_delivery_cooldown_steps: int = 0
    post_delivery_exit_radius: float = 0.0
    post_delivery_outward_reward: float = 0.0
    post_delivery_loiter_penalty: float = 0.0
    post_delivery_pheromone_suppression_radius: float = 0.0
    post_delivery_require_exit: bool = False
    post_delivery_crowding_penalty: float = 0.0
    post_delivery_crowding_threshold: int = 2

    # Rewards
    #reward_target: float = 30.0
    #reward_step: float = -0.01 #-0.01 
    #reward_collision: float = -2.0 #-0.2
    #reward_pickup: float = 30.0
    #reward_nest_delivery: float = 0.0 #10.0
    #reward_exploration: float = 0.0 #0.2 #0.05
    #reward_new_cell: float = 0.00
    #reward_pheromone_following: float = 0.0 #0.2
    #reward_food_approach: float = 0.0 #0.5 #reward_food_approach: float = 0.2
    #reward_food_detected: float = 0.0 #reward_food_detected: float = 0.05
    #reward_pheromone_follow: float = 0.0 # 0.03
   # reward_pheromone_deposit_cost: float = -0.2 #-0.02
    #pheromone_follow_min_gradient: float = 0.05

    reward_target: float = 0.0#2.0               # fallback only when pickup/delivery is disabled
    reward_pickup: float = 6.0               # meaningful event, but clearly smaller than final completion
    reward_nest_delivery: float = 30.0       # main task completion reward
    reward_nest_approach: float = 0.08       # carrying-food progress toward nest
    reward_nest_approach_sustained: float = 0.04
    reward_undelivered_food: float = -10.0   # penalty if episode ends while still carrying food

    reward_step: float = -0.01              # gentle time pressure
    reward_collision: float = -1.5          # collisions should clearly hurt, but not dominate

    reward_exploration: float = 0.0         # currently unused in env step path
    reward_new_cell: float = 0.01           # small exploration bonus
    carrying_reward_new_cell_scale: float = 0.1  # heavily suppress exploration bonus while carrying
    carrying_no_progress_penalty: float = -0.03
    carrying_low_displacement_penalty: float = -0.02
    carrying_progress_epsilon: float = 2.0
    carrying_low_displacement_threshold: float = 3.0
    carrying_stall_trigger_steps: int = 6
    carrying_progress_streak_threshold: int = 3

    reward_food_approach: float = 0.03      # small signed shaping only
    reward_food_detected: float = 0.15      # one-time local cue, kept modest

    reward_pheromone_following: float = 0.0 # keep global usage reward off by default
    reward_pheromone_follow: float = 0.02   # light local gradient preference
    reward_pheromone_deposit_cost: float = -0.001  # discourage spam without suppressing use
    reward_action_switch: float = -0.01
    pheromone_follow_min_gradient: float = 0.05

    # Task
    nest_enabled: bool = True
    require_nest_delivery: bool = True
    active_targets: int = 3
    target_respawn: bool = True
    food_source_capacity: int = 4
    coverage_cell_size: int = 24
    failed_agent_count: int = 0
    observation_noise_std: float = 0.0

    # Rendering
    render_pheromone: bool = True
    render_scale: float = 1.0

    # Seeding
    seed: int | None = None
