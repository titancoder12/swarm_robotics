from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    n_agents: int
    total_steps: int
    width: int
    height: int
    n_targets: int
    n_obstacles: int
    max_steps: int
    active_targets: int
    target_respawn: bool


def _split_stage_steps(total_steps: int, weights: list[int]) -> list[int]:
    total_steps = max(len(weights), int(total_steps))
    weight_sum = max(1, sum(weights))
    raw = [max(1, int(total_steps * weight / weight_sum)) for weight in weights]
    diff = total_steps - sum(raw)
    raw[-1] += diff
    return raw


def default_curriculum(total_steps: int, full_agents: int) -> list[CurriculumStage]:
    """Return a staged environment curriculum from tiny single-agent to hard full-swarm."""
    total_steps = max(6, int(total_steps))
    full_agents = max(5, int(full_agents))
    small_agents = max(2, min(5, max(2, full_agents // 2)))
    stage_steps = _split_stage_steps(total_steps, [1, 1, 1, 1, 1, 2])
    return [
        CurriculumStage(
            name="stage1a_single_agent_tiny",
            n_agents=1,
            total_steps=stage_steps[0],
            width=220,
            height=220,
            n_targets=1,
            n_obstacles=0,
            max_steps=120,
            active_targets=1,
            target_respawn=False,
        ),
        CurriculumStage(
            name="stage1b_single_agent_obstacles",
            n_agents=1,
            total_steps=stage_steps[1],
            width=420,
            height=320,
            n_targets=1,
            n_obstacles=2,
            max_steps=220,
            active_targets=1,
            target_respawn=False,
        ),
        CurriculumStage(
            name="stage2a_small_swarm_medium",
            n_agents=small_agents,
            total_steps=stage_steps[2],
            width=700,
            height=500,
            n_targets=2,
            n_obstacles=4,
            max_steps=360,
            active_targets=2,
            target_respawn=False,
        ),
        CurriculumStage(
            name="stage2b_small_swarm_large",
            n_agents=small_agents,
            total_steps=stage_steps[3],
            width=950,
            height=700,
            n_targets=3,
            n_obstacles=6,
            max_steps=500,
            active_targets=3,
            target_respawn=False,
        ),
        CurriculumStage(
            name="stage3a_full_swarm_large",
            n_agents=full_agents,
            total_steps=stage_steps[4],
            width=950,
            height=700,
            n_targets=3,
            n_obstacles=6,
            max_steps=500,
            active_targets=3,
            target_respawn=False,
        ),
        CurriculumStage(
            name="stage3b_full_swarm_final",
            n_agents=full_agents,
            total_steps=stage_steps[5],
            width=1400,
            height=950,
            n_targets=4,
            n_obstacles=12,
            max_steps=720,
            active_targets=4,
            target_respawn=False,
        ),
    ]


def select_curriculum(stages: list[CurriculumStage], mode: str) -> list[CurriculumStage]:
    """Select a curriculum slice from a list of stages."""
    if mode == "stage1":
        return stages[:2]
    if mode == "stage1_to_2":
        return stages[:4]
    if mode == "full":
        return stages
    raise ValueError(f"Unsupported curriculum mode: {mode}")
