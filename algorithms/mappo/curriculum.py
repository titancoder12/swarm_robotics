from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    n_agents: int
    total_steps: int


def default_curriculum(total_steps: int, full_agents: int) -> list[CurriculumStage]:
    """Return a simple 3-stage curriculum schedule."""
    total_steps = max(3, int(total_steps))
    stage_steps = max(1, total_steps // 3)
    full_agents = max(5, int(full_agents))
    small_agents = max(2, min(5, max(2, full_agents // 2)))
    return [
        CurriculumStage(name="stage1_single_agent", n_agents=1, total_steps=stage_steps),
        CurriculumStage(name="stage2_small_swarm", n_agents=small_agents, total_steps=stage_steps),
        CurriculumStage(name="stage3_full_marl", n_agents=full_agents, total_steps=total_steps - 2 * stage_steps),
    ]


def select_curriculum(stages: list[CurriculumStage], mode: str) -> list[CurriculumStage]:
    """Select a curriculum slice from a list of stages."""
    if mode == "stage1":
        return stages[:1]
    if mode == "stage1_to_2":
        return stages[:2]
    if mode == "full":
        return stages
    raise ValueError(f"Unsupported curriculum mode: {mode}")
