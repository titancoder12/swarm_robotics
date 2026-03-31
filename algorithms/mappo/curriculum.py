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
    action_repeat_steps: int = 2
    reward_new_cell: float = 0.01
    reward_step: float | None = None
    reward_collision: float | None = None
    reward_pickup: float | None = None
    reward_nest_approach: float | None = None
    reward_nest_delivery: float | None = None
    reward_undelivered_food: float | None = None
    reward_food_approach: float | None = None
    reward_food_detected: float | None = None
    reward_pheromone_follow: float | None = None
    carrying_reward_new_cell_scale: float | None = None
    pheromone_enabled: bool | None = None
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
            pheromone_enabled=False,
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
            pheromone_enabled=False,
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
            pheromone_enabled=False,
            entropy_start=0.020,
            entropy_end=0.003,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1d_single_agent_return_medium",
            n_agents=1,
            total_steps=0,
            budget_weight=2,
            width=160,
            height=120,
            n_targets=1,
            n_obstacles=0,
            max_steps=180,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.005,
            reward_step=-0.004,
            reward_collision=-0.40,
            reward_pickup=7.0,
            reward_nest_approach=0.34,
            reward_nest_delivery=48.0,
            reward_undelivered_food=-16.0,
            reward_food_approach=0.04,
            reward_food_detected=0.18,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            pheromone_enabled=False,
            entropy_start=0.014,
            entropy_end=0.002,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1e_single_agent_delivery_bridge",
            n_agents=1,
            total_steps=0,
            budget_weight=2,
            width=300,
            height=240,
            n_targets=1,
            n_obstacles=1,
            max_steps=190,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.004,
            reward_step=-0.003,
            reward_collision=-0.35,
            reward_pickup=7.0,
            reward_nest_approach=0.38,
            reward_nest_delivery=52.0,
            reward_undelivered_food=-16.0,
            reward_food_approach=0.025,
            reward_food_detected=0.16,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            pheromone_enabled=False,
            entropy_start=0.012,
            entropy_end=0.0015,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1f_single_agent_delivery_obstacles",
            n_agents=1,
            total_steps=0,
            budget_weight=2,
            width=360,
            height=280,
            n_targets=1,
            n_obstacles=2,
            max_steps=220,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.004,
            reward_step=-0.003,
            reward_collision=-0.40,
            reward_pickup=7.0,
            reward_nest_approach=0.36,
            reward_nest_delivery=50.0,
            reward_undelivered_food=-16.0,
            reward_food_approach=0.025,
            reward_food_detected=0.16,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            pheromone_enabled=False,
            entropy_start=0.010,
            entropy_end=0.0015,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage2a_small_swarm_medium",
            n_agents=small_agents,
            total_steps=0,
            budget_weight=2,
            width=600,
            height=420,
            n_targets=2,
            n_obstacles=2,
            max_steps=360,
            active_targets=2,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.005,
            reward_step=-0.006,
            reward_collision=-0.75,
            reward_pickup=6.0,
            reward_nest_approach=0.24,
            reward_nest_delivery=40.0,
            reward_undelivered_food=-14.0,
            reward_food_approach=0.03,
            reward_food_detected=0.16,
            reward_pheromone_follow=0.0,
            carrying_reward_new_cell_scale=0.0,
            pheromone_enabled=False,
            entropy_start=0.012,
            entropy_end=0.002,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage2b_small_swarm_large",
            n_agents=small_agents,
            total_steps=0,
            budget_weight=3,
            width=950,
            height=700,
            n_targets=3,
            n_obstacles=8,
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
            carrying_reward_new_cell_scale=0.1,
            pheromone_enabled=True,
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
            carrying_reward_new_cell_scale=0.15,
            pheromone_enabled=True,
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
            carrying_reward_new_cell_scale=0.20,
            pheromone_enabled=True,
            entropy_start=0.006,
            entropy_end=0.0005,
        )
    )
    return stages


def select_curriculum(stages: list[CurriculumStage], mode: str, total_steps: int) -> list[CurriculumStage]:
    """Select a curriculum slice and assign the requested total budget across that slice."""
    if mode == "stage1":
        selected = stages[:6]
    elif mode == "stage1_to_2":
        selected = stages[:8]
    elif mode == "full":
        selected = stages
    else:
        raise ValueError(f"Unsupported curriculum mode: {mode}")
    return _assign_stage_steps(selected, total_steps)
