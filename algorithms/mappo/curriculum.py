from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    n_agents: int
    total_steps: int
    budget_weight: int
    width: int
    height: int
    n_targets: int
    n_obstacles: int
    max_steps: int
    active_targets: int
    target_respawn: bool
    obstacle_min_size: int | None = None
    obstacle_max_size: int | None = None
    action_repeat_steps: int = 2
    reward_new_cell: float = 0.01
    reward_step: float | None = None
    reward_collision: float | None = None
    reward_pickup: float | None = None
    reward_nest_approach: float | None = None
    reward_nest_approach_sustained: float | None = None
    reward_nest_delivery: float | None = None
    reward_undelivered_food: float | None = None
    reward_food_approach: float | None = None
    reward_food_detected: float | None = None
    reward_pheromone_follow: float | None = None
    carrying_reward_new_cell_scale: float | None = None
    carrying_no_progress_penalty: float | None = None
    carrying_low_displacement_penalty: float | None = None
    carrying_progress_epsilon: float | None = None
    carrying_low_displacement_threshold: float | None = None
    carrying_stall_trigger_steps: int | None = None
    pheromone_enabled: bool | None = None
    non_carrying_nest_pheromone_suppression_radius: float | None = None
    non_carrying_nest_loiter_radius: float | None = None
    non_carrying_nest_loiter_penalty: float | None = None
    non_carrying_nest_crowding_radius: float | None = None
    non_carrying_nest_crowding_penalty: float | None = None
    non_carrying_nest_crowding_threshold: int | None = None
    non_carrying_explore_radius: float | None = None
    non_carrying_outward_reward: float | None = None
    non_carrying_no_outward_progress_penalty: float | None = None
    non_carrying_idle_near_nest_penalty: float | None = None
    non_carrying_low_displacement_threshold: float | None = None
    non_carrying_explore_ignore_pheromone: bool | None = None
    non_carrying_explore_random_action_prob: float | None = None
    non_carrying_force_explore_mode: bool | None = None
    non_carrying_force_explore_radius: float | None = None
    post_delivery_cooldown_steps: int | None = None
    post_delivery_exit_radius: float | None = None
    post_delivery_outward_reward: float | None = None
    post_delivery_loiter_penalty: float | None = None
    post_delivery_pheromone_suppression_radius: float | None = None
    post_delivery_require_exit: bool | None = None
    post_delivery_crowding_penalty: float | None = None
    post_delivery_crowding_threshold: int | None = None
    target_nest_distance_min: float | None = None
    target_nest_distance_max: float | None = None
    agent_spawn_near_target_radius: float | None = None
    target_nest_corridor_clearance: float | None = None
    start_carrying_food: bool = False
    repeat_limit_override: int | None = None
    entropy_start: float = 0.01
    entropy_end: float = 0.001


def _split_stage_steps(total_steps: int, weights: list[int]) -> list[int]:
    total_steps = max(len(weights), int(total_steps))
    weight_sum = max(1, sum(weights))
    raw = [max(1, int(total_steps * weight / weight_sum)) for weight in weights]
    diff = total_steps - sum(raw)
    raw[-1] += diff
    return raw


def _assign_stage_steps(stages: list[CurriculumStage], total_steps: int) -> list[CurriculumStage]:
    total_steps = max(len(stages), int(total_steps))
    stage_steps = _split_stage_steps(total_steps, [stage.budget_weight for stage in stages])
    return [replace(stage, total_steps=steps) for stage, steps in zip(stages, stage_steps)]


def default_curriculum(full_agents: int) -> list[CurriculumStage]:
    """Return curriculum stage templates from tiny single-agent to hard full-swarm."""
    full_agents = max(5, int(full_agents))
    small_agents = max(2, min(5, max(2, full_agents // 2)))
    stages: list[CurriculumStage] = []
    stages.append(
        CurriculumStage(
            name="stage1a_single_agent_miniscule",
            n_agents=1,
            total_steps=0,
            budget_weight=1,
            width=50,
            height=50,
            n_targets=1,
            n_obstacles=0,
            obstacle_min_size=40,
            obstacle_max_size=80,
            max_steps=120,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.02,
            reward_step=-0.002,
            reward_collision=-0.25,
            reward_pickup=8.0,
            reward_nest_approach=0.16,
            reward_nest_delivery=40.0,
            reward_undelivered_food=-12.0,
            reward_food_approach=0.05,
            reward_food_detected=0.30,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.04,
            carrying_low_displacement_penalty=-0.02,
            carrying_progress_epsilon=1.5,
            carrying_low_displacement_threshold=2.0,
            carrying_stall_trigger_steps=4,
            pheromone_enabled=False,
            target_nest_distance_min=22.0,
            target_nest_distance_max=30.0,
            agent_spawn_near_target_radius=18.0,
            entropy_start=0.030,
            entropy_end=0.004,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1b_single_agent_tiny",
            n_agents=1,
            total_steps=0,
            budget_weight=1,
            width=75,
            height=75,
            n_targets=1,
            n_obstacles=0,
            obstacle_min_size=40,
            obstacle_max_size=80,
            max_steps=120,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.02,
            reward_step=-0.003,
            reward_collision=-0.25,
            reward_pickup=8.0,
            reward_nest_approach=0.18,
            reward_nest_delivery=40.0,
            reward_undelivered_food=-12.0,
            reward_food_approach=0.05,
            reward_food_detected=0.25,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.04,
            carrying_low_displacement_penalty=-0.02,
            carrying_progress_epsilon=1.5,
            carrying_low_displacement_threshold=2.0,
            carrying_stall_trigger_steps=4,
            pheromone_enabled=False,
            target_nest_distance_min=26.0,
            target_nest_distance_max=36.0,
            agent_spawn_near_target_radius=22.0,
            entropy_start=0.025,
            entropy_end=0.004,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1c_single_agent_small",
            n_agents=1,
            total_steps=0,
            budget_weight=1,
            width=100,
            height=100,
            n_targets=1,
            n_obstacles=0,
            obstacle_min_size=40,
            obstacle_max_size=80,
            max_steps=140,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.02,
            reward_step=-0.004,
            reward_collision=-0.40,
            reward_pickup=8.0,
            reward_nest_approach=0.20,
            reward_nest_delivery=40.0,
            reward_undelivered_food=-12.0,
            reward_food_approach=0.05,
            reward_food_detected=0.20,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.045,
            carrying_low_displacement_penalty=-0.025,
            carrying_progress_epsilon=1.5,
            carrying_low_displacement_threshold=2.5,
            carrying_stall_trigger_steps=4,
            pheromone_enabled=False,
            target_nest_distance_min=32.0,
            target_nest_distance_max=46.0,
            agent_spawn_near_target_radius=28.0,
            entropy_start=0.020,
            entropy_end=0.003,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1d_single_agent_carry_bootstrap",
            n_agents=1,
            total_steps=0,
            budget_weight=1,
            width=120,
            height=100,
            n_targets=1,
            n_obstacles=0,
            obstacle_min_size=40,
            obstacle_max_size=80,
            max_steps=110,
            active_targets=0,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.0,
            reward_step=-0.004,
            reward_collision=-0.40,
            reward_pickup=0.0,
            reward_nest_approach=0.60,
            reward_nest_approach_sustained=0.18,
            reward_nest_delivery=60.0,
            reward_undelivered_food=-24.0,
            reward_food_approach=0.0,
            reward_food_detected=0.0,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.08,
            carrying_low_displacement_penalty=-0.04,
            carrying_progress_epsilon=1.0,
            carrying_low_displacement_threshold=2.0,
            carrying_stall_trigger_steps=2,
            pheromone_enabled=False,
            target_nest_distance_min=26.0,
            target_nest_distance_max=34.0,
            agent_spawn_near_target_radius=10.0,
            start_carrying_food=True,
            repeat_limit_override=1,
            entropy_start=0.0015,
            entropy_end=0.0001,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1e_single_agent_guaranteed_homing",
            n_agents=1,
            total_steps=0,
            budget_weight=1,
            width=140,
            height=110,
            n_targets=1,
            n_obstacles=0,
            obstacle_min_size=40,
            obstacle_max_size=80,
            max_steps=140,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.002,
            reward_step=-0.004,
            reward_collision=-0.40,
            reward_pickup=7.0,
            reward_nest_approach=0.50,
            reward_nest_approach_sustained=0.14,
            reward_nest_delivery=56.0,
            reward_undelivered_food=-20.0,
            reward_food_approach=0.02,
            reward_food_detected=0.14,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.06,
            carrying_low_displacement_penalty=-0.03,
            carrying_progress_epsilon=1.2,
            carrying_low_displacement_threshold=2.5,
            carrying_stall_trigger_steps=3,
            pheromone_enabled=False,
            target_nest_distance_min=34.0,
            target_nest_distance_max=46.0,
            agent_spawn_near_target_radius=18.0,
            target_nest_corridor_clearance=18.0,
            repeat_limit_override=1,
            entropy_start=0.003,
            entropy_end=0.0001,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1f_single_agent_delivery_bridge",
            n_agents=1,
            total_steps=0,
            budget_weight=2,
            width=190,
            height=160,
            n_targets=1,
            n_obstacles=1,
            obstacle_min_size=34,
            obstacle_max_size=56,
            max_steps=160,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.001,
            reward_step=-0.004,
            reward_collision=-0.30,
            reward_pickup=7.0,
            reward_nest_approach=0.54,
            reward_nest_approach_sustained=0.18,
            reward_nest_delivery=58.0,
            reward_undelivered_food=-22.0,
            reward_food_approach=0.025,
            reward_food_detected=0.16,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.07,
            carrying_low_displacement_penalty=-0.035,
            carrying_progress_epsilon=1.2,
            carrying_low_displacement_threshold=2.5,
            carrying_stall_trigger_steps=2,
            pheromone_enabled=False,
            target_nest_distance_min=34.0,
            target_nest_distance_max=48.0,
            agent_spawn_near_target_radius=18.0,
            target_nest_corridor_clearance=20.0,
            repeat_limit_override=2,
            entropy_start=0.002,
            entropy_end=0.0001,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1g_single_agent_delivery_obstacles",
            n_agents=1,
            total_steps=0,
            budget_weight=3,
            width=300,
            height=240,
            n_targets=1,
            n_obstacles=2,
            obstacle_min_size=48,
            obstacle_max_size=88,
            max_steps=200,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.002,
            reward_step=-0.004,
            reward_collision=-0.40,
            reward_pickup=7.0,
            reward_nest_approach=0.46,
            reward_nest_approach_sustained=0.14,
            reward_nest_delivery=52.0,
            reward_undelivered_food=-20.0,
            reward_food_approach=0.025,
            reward_food_detected=0.16,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.07,
            carrying_low_displacement_penalty=-0.035,
            carrying_progress_epsilon=1.5,
            carrying_low_displacement_threshold=3.0,
            carrying_stall_trigger_steps=3,
            pheromone_enabled=False,
            target_nest_distance_min=42.0,
            target_nest_distance_max=60.0,
            agent_spawn_near_target_radius=26.0,
            target_nest_corridor_clearance=8.0,
            repeat_limit_override=2,
            entropy_start=0.002,
            entropy_end=0.0001,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage2a_small_swarm_carry_bootstrap",
            n_agents=small_agents,
            total_steps=0,
            budget_weight=2,
            width=260,
            height=220,
            n_targets=1,
            n_obstacles=0,
            obstacle_min_size=48,
            obstacle_max_size=96,
            max_steps=180,
            active_targets=0,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.0,
            reward_step=-0.004,
            reward_collision=-0.60,
            reward_pickup=0.0,
            reward_nest_approach=0.65,
            reward_nest_approach_sustained=0.20,
            reward_nest_delivery=60.0,
            reward_undelivered_food=-24.0,
            reward_food_approach=0.0,
            reward_food_detected=0.0,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.08,
            carrying_low_displacement_penalty=-0.04,
            carrying_progress_epsilon=1.5,
            carrying_low_displacement_threshold=3.0,
            carrying_stall_trigger_steps=4,
            pheromone_enabled=False,
            start_carrying_food=True,
            repeat_limit_override=2,
            entropy_start=0.003,
            entropy_end=0.0002,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage2b_small_swarm_delivery_easy",
            n_agents=small_agents,
            total_steps=0,
            budget_weight=2,
            width=360,
            height=260,
            n_targets=1,
            n_obstacles=0,
            obstacle_min_size=48,
            obstacle_max_size=96,
            max_steps=240,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.004,
            reward_step=-0.005,
            reward_collision=-0.75,
            reward_pickup=7.0,
            reward_nest_approach=0.40,
            reward_nest_approach_sustained=0.14,
            reward_nest_delivery=48.0,
            reward_undelivered_food=-18.0,
            reward_food_approach=0.04,
            reward_food_detected=0.18,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.06,
            carrying_low_displacement_penalty=-0.03,
            carrying_progress_epsilon=1.8,
            carrying_low_displacement_threshold=3.0,
            carrying_stall_trigger_steps=4,
            pheromone_enabled=False,
            target_nest_distance_min=56.0,
            target_nest_distance_max=84.0,
            agent_spawn_near_target_radius=34.0,
            repeat_limit_override=2,
            entropy_start=0.006,
            entropy_end=0.0005,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage2c_small_swarm_medium",
            n_agents=small_agents,
            total_steps=0,
            budget_weight=2,
            width=520,
            height=380,
            n_targets=1,
            n_obstacles=1,
            obstacle_min_size=44,
            obstacle_max_size=88,
            max_steps=320,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.004,
            reward_step=-0.006,
            reward_collision=-0.85,
            reward_pickup=7.0,
            reward_nest_approach=0.30,
            reward_nest_approach_sustained=0.10,
            reward_nest_delivery=44.0,
            reward_undelivered_food=-16.0,
            reward_food_approach=0.035,
            reward_food_detected=0.16,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            carrying_no_progress_penalty=-0.055,
            carrying_low_displacement_penalty=-0.03,
            carrying_progress_epsilon=2.0,
            carrying_low_displacement_threshold=3.2,
            carrying_stall_trigger_steps=4,
            pheromone_enabled=False,
            target_nest_distance_min=72.0,
            target_nest_distance_max=110.0,
            target_nest_corridor_clearance=10.0,
            repeat_limit_override=2,
            entropy_start=0.008,
            entropy_end=0.001,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage2d_small_swarm_large",
            n_agents=small_agents,
            total_steps=0,
            budget_weight=3,
            width=950,
            height=700,
            n_targets=3,
            n_obstacles=8,
            obstacle_min_size=50,
            obstacle_max_size=110,
            max_steps=650,
            active_targets=3,
            target_respawn=True,
            action_repeat_steps=2,
            reward_new_cell=0.0075,
            reward_step=-0.010,
            reward_collision=-1.25,
            reward_pickup=6.0,
            reward_nest_approach=0.18,
            reward_nest_delivery=30.0,
            reward_undelivered_food=-10.0,
            reward_food_approach=0.03,
            reward_food_detected=0.15,
            reward_pheromone_follow=0.02,
            carrying_reward_new_cell_scale=0.05,
            carrying_no_progress_penalty=-0.04,
            carrying_low_displacement_penalty=-0.02,
            carrying_progress_epsilon=2.5,
            carrying_low_displacement_threshold=4.0,
            carrying_stall_trigger_steps=5,
            pheromone_enabled=True,
            non_carrying_nest_pheromone_suppression_radius=120.0,
            non_carrying_nest_loiter_radius=150.0,
            non_carrying_nest_loiter_penalty=-0.15,
            non_carrying_nest_crowding_radius=190.0,
            non_carrying_nest_crowding_penalty=-0.12,
            non_carrying_nest_crowding_threshold=2,
            non_carrying_explore_radius=260.0,
            non_carrying_outward_reward=0.12,
            non_carrying_no_outward_progress_penalty=-0.18,
            non_carrying_idle_near_nest_penalty=-0.14,
            non_carrying_low_displacement_threshold=4.0,
            non_carrying_explore_ignore_pheromone=True,
            non_carrying_explore_random_action_prob=0.10,
            non_carrying_force_explore_mode=True,
            non_carrying_force_explore_radius=220.0,
            post_delivery_cooldown_steps=18,
            post_delivery_exit_radius=105.0,
            post_delivery_outward_reward=0.05,
            post_delivery_loiter_penalty=-0.02,
            post_delivery_pheromone_suppression_radius=150.0,
            post_delivery_require_exit=True,
            post_delivery_crowding_penalty=-0.012,
            post_delivery_crowding_threshold=2,
            entropy_start=0.010,
            entropy_end=0.0015,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage3a_full_swarm_large",
            n_agents=full_agents,
            total_steps=0,
            budget_weight=4,
            width=950,
            height=700,
            n_targets=3,
            n_obstacles=8,
            obstacle_min_size=50,
            obstacle_max_size=110,
            max_steps=720,
            active_targets=3,
            target_respawn=True,
            action_repeat_steps=2,
            reward_new_cell=0.005,
            reward_step=-0.010,
            reward_collision=-1.50,
            reward_pickup=6.0,
            reward_nest_approach=0.14,
            reward_nest_delivery=30.0,
            reward_undelivered_food=-10.0,
            reward_food_approach=0.03,
            reward_food_detected=0.15,
            reward_pheromone_follow=0.02,
            carrying_reward_new_cell_scale=0.05,
            carrying_no_progress_penalty=-0.04,
            carrying_low_displacement_penalty=-0.02,
            carrying_progress_epsilon=2.5,
            carrying_low_displacement_threshold=4.0,
            carrying_stall_trigger_steps=5,
            pheromone_enabled=True,
            non_carrying_nest_pheromone_suppression_radius=120.0,
            non_carrying_nest_loiter_radius=170.0,
            non_carrying_nest_loiter_penalty=-0.18,
            non_carrying_nest_crowding_radius=220.0,
            non_carrying_nest_crowding_penalty=-0.15,
            non_carrying_nest_crowding_threshold=2,
            non_carrying_explore_radius=320.0,
            non_carrying_outward_reward=0.14,
            non_carrying_no_outward_progress_penalty=-0.22,
            non_carrying_idle_near_nest_penalty=-0.18,
            non_carrying_low_displacement_threshold=4.2,
            non_carrying_explore_ignore_pheromone=True,
            non_carrying_explore_random_action_prob=0.12,
            non_carrying_force_explore_mode=True,
            non_carrying_force_explore_radius=260.0,
            post_delivery_cooldown_steps=22,
            post_delivery_exit_radius=110.0,
            post_delivery_outward_reward=0.06,
            post_delivery_loiter_penalty=-0.022,
            post_delivery_pheromone_suppression_radius=160.0,
            post_delivery_require_exit=True,
            post_delivery_crowding_penalty=-0.014,
            post_delivery_crowding_threshold=2,
            entropy_start=0.008,
            entropy_end=0.0010,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage3b_full_swarm_final",
            n_agents=full_agents,
            total_steps=0,
            budget_weight=6,
            width=1400,
            height=950,
            n_targets=3,
            n_obstacles=18,
            obstacle_min_size=60,
            obstacle_max_size=120,
            max_steps=800,
            active_targets=3,
            target_respawn=True,
            action_repeat_steps=2,
            reward_new_cell=0.005,
            reward_step=-0.010,
            reward_collision=-1.50,
            reward_pickup=6.0,
            reward_nest_approach=0.12,
            reward_nest_delivery=30.0,
            reward_undelivered_food=-10.0,
            reward_food_approach=0.03,
            reward_food_detected=0.15,
            reward_pheromone_follow=0.02,
            carrying_reward_new_cell_scale=0.05,
            carrying_no_progress_penalty=-0.04,
            carrying_low_displacement_penalty=-0.02,
            carrying_progress_epsilon=3.0,
            carrying_low_displacement_threshold=4.5,
            carrying_stall_trigger_steps=6,
            pheromone_enabled=True,
            non_carrying_nest_pheromone_suppression_radius=140.0,
            non_carrying_nest_loiter_radius=210.0,
            non_carrying_nest_loiter_penalty=-0.25,
            non_carrying_nest_crowding_radius=280.0,
            non_carrying_nest_crowding_penalty=-0.20,
            non_carrying_nest_crowding_threshold=2,
            non_carrying_explore_radius=420.0,
            non_carrying_outward_reward=0.16,
            non_carrying_no_outward_progress_penalty=-0.30,
            non_carrying_idle_near_nest_penalty=-0.22,
            non_carrying_low_displacement_threshold=4.5,
            non_carrying_explore_ignore_pheromone=True,
            non_carrying_explore_random_action_prob=0.15,
            non_carrying_force_explore_mode=True,
            non_carrying_force_explore_radius=320.0,
            post_delivery_cooldown_steps=28,
            post_delivery_exit_radius=120.0,
            post_delivery_outward_reward=0.07,
            post_delivery_loiter_penalty=-0.024,
            post_delivery_pheromone_suppression_radius=180.0,
            post_delivery_require_exit=True,
            post_delivery_crowding_penalty=-0.016,
            post_delivery_crowding_threshold=2,
            entropy_start=0.006,
            entropy_end=0.0005,
        )
    )
    return stages


def select_curriculum(stages: list[CurriculumStage], mode: str, total_steps: int) -> list[CurriculumStage]:
    """Select a curriculum slice and assign the requested total budget across that slice."""
    if mode == "stage1":
        selected = stages[:7]
    elif mode == "stage1_to_2":
        selected = stages[:11]
    elif mode == "full":
        selected = stages
    else:
        raise ValueError(f"Unsupported curriculum mode: {mode}")
    return _assign_stage_steps(selected, total_steps)
