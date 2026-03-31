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
        )
    )
    stages.append(
        CurriculumStage(
            name="stage1d_single_agent_delivery_obstacles",
            n_agents=1,
            total_steps=0,
            budget_weight=2,
            width=420,
            height=320,
            n_targets=1,
            n_obstacles=2,
            max_steps=320,
            active_targets=1,
            target_respawn=False,
            action_repeat_steps=1,
            reward_new_cell=0.01,
        )
    )
    stages.append(
        CurriculumStage(
            name="stage2a_small_swarm_medium",
            n_agents=small_agents,
            total_steps=0,
            budget_weight=2,
            width=700,
            height=500,
            n_targets=2,
            n_obstacles=4,
            max_steps=420,
            active_targets=2,
            target_respawn=False,
            action_repeat_steps=2,
            reward_new_cell=0.01,
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
        )
    )
    return stages


def select_curriculum(stages: list[CurriculumStage], mode: str, total_steps: int) -> list[CurriculumStage]:
    """Select a curriculum slice and assign the requested total budget across that slice."""
    if mode == "stage1":
        selected = stages[:4]
    elif mode == "stage1_to_2":
        selected = stages[:6]
    elif mode == "full":
        selected = stages
    else:
        raise ValueError(f"Unsupported curriculum mode: {mode}")
    return _assign_stage_steps(selected, total_steps)
