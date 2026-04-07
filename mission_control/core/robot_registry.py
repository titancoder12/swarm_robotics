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
    lidar_ranges_mm: tuple[float, ...] = ()
    carrying_food: bool | None = None
    connection_label: str | None = None
    target_x_cm: float | None = None
    target_y_cm: float | None = None
    target_confidence: float = 0.0
    last_target_seen: float | None = None

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
        now = time.time()
        state = self._robots.get(robot_id)
        if state is None:
            state = RobotState(
                robot_id=robot_id,
                x_cm=float(x_cm),
                y_cm=float(y_cm),
                heading_deg=None if heading_deg is None else float(heading_deg),
                last_seen=now,
                carrying_food=carrying_food,
                connection_label=connection_label,
            )
            self._robots[robot_id] = state
            return state

        state.x_cm = float(x_cm)
        state.y_cm = float(y_cm)
        state.heading_deg = None if heading_deg is None else float(heading_deg)
        state.last_seen = now
        if carrying_food is not None:
            state.carrying_food = carrying_food
        if connection_label is not None:
            state.connection_label = connection_label
        return state

    def get(self, robot_id: str) -> RobotState | None:
        return self._robots.get(robot_id)

    def update_lidar(self, robot_id: str, ranges_mm: tuple[float, ...]) -> RobotState | None:
        state = self._robots.get(robot_id)
        if state is None:
            state = RobotState(
                robot_id=robot_id,
                x_cm=0.0,
                y_cm=0.0,
                heading_deg=None,
                last_seen=time.time(),
            )
            self._robots[robot_id] = state
        state.lidar_ranges_mm = tuple(float(value) for value in ranges_mm)
        state.last_seen = time.time()
        return state

    def update_target(self, robot_id: str, x_cm: float, y_cm: float, confidence: float) -> RobotState:
        state = self._robots.get(robot_id)
        if state is None:
            state = RobotState(
                robot_id=robot_id,
                x_cm=0.0,
                y_cm=0.0,
                heading_deg=None,
                last_seen=time.time(),
            )
            self._robots[robot_id] = state
        state.target_x_cm = float(x_cm)
        state.target_y_cm = float(y_cm)
        state.target_confidence = float(confidence)
        state.last_target_seen = time.time()
        return state

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
        robot_rows = []
        for robot_id, state in sorted(self._robots.items()):
            robot_rows.append(
                {
                    "robot_id": robot_id,
                    "x_cm": state.x_cm,
                    "y_cm": state.y_cm,
                    "heading_deg": state.heading_deg,
                    "age_s": state.age_s,
                    "lidar_ranges_mm": list(state.lidar_ranges_mm),
                    "target_x_cm": state.target_x_cm,
                    "target_y_cm": state.target_y_cm,
                    "target_confidence": state.target_confidence,
                    "target_age_s": None if state.last_target_seen is None else max(0.0, time.time() - state.last_target_seen),
                }
            )
        return {
            "robot_count": len(self._robots),
            "robot_ids": sorted(self._robots.keys()),
            "robot_rows": robot_rows,
        }
