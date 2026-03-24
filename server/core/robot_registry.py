from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RobotState:
    robot_id: str
    x_cm: float
    y_cm: float
    heading_deg: float | None
    last_seen: float
    carrying_food: bool | None = None
    connection_label: str | None = None

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.last_seen)


class RobotRegistry:
    """Tracks the latest known state for each robot."""

    def __init__(self) -> None:
        self._robots: dict[str, RobotState] = {}

    def update_position(
        self,
        robot_id: str,
        x_cm: float,
        y_cm: float,
        heading_deg: float | None,
        connection_label: str | None = None,
        carrying_food: bool | None = None,
    ) -> RobotState:
        state = RobotState(
            robot_id=robot_id,
            x_cm=float(x_cm),
            y_cm=float(y_cm),
            heading_deg=None if heading_deg is None else float(heading_deg),
            last_seen=time.time(),
            carrying_food=carrying_food,
            connection_label=connection_label,
        )
        self._robots[robot_id] = state
        return state

    def get(self, robot_id: str) -> RobotState | None:
        return self._robots.get(robot_id)

    def snapshot(self) -> dict[str, RobotState]:
        return dict(self._robots)

    def stale_ids(self, stale_after_s: float) -> list[str]:
        now = time.time()
        return [
            robot_id
            for robot_id, state in self._robots.items()
            if now - state.last_seen > stale_after_s
        ]

    def telemetry(self) -> dict[str, Any]:
        return {
            "robot_count": len(self._robots),
            "robot_ids": sorted(self._robots.keys()),
        }

