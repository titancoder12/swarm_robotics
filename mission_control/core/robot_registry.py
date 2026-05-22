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
    control_mode: str = "unknown"
    action_id: int = -1
    throttle: float = 0.0
    turn: float = 0.0
    deposit: bool = False
    camera_found: bool = False
    camera_distance_m: float = 0.0
    camera_angle_deg: float = 0.0
    front_min_mm: float = 0.0
    left_min_mm: float = 0.0
    right_min_mm: float = 0.0
    serial_ok: bool = False
    serial_cmd: str = ""
    serial_reply: str = ""
    last_status_seen: float | None = None

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

    def update_status(
        self,
        robot_id: str,
        control_mode: str,
        action_id: int,
        throttle: float,
        turn: float,
        deposit: bool,
        camera_found: bool,
        camera_distance_m: float,
        camera_angle_deg: float,
        front_min_mm: float,
        left_min_mm: float,
        right_min_mm: float,
        serial_ok: bool,
        serial_cmd: str,
        serial_reply: str,
    ) -> RobotState:
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
        state.control_mode = str(control_mode)
        state.action_id = int(action_id)
        state.throttle = float(throttle)
        state.turn = float(turn)
        state.deposit = bool(deposit)
        state.camera_found = bool(camera_found)
        state.camera_distance_m = float(camera_distance_m)
        state.camera_angle_deg = float(camera_angle_deg)
        state.front_min_mm = float(front_min_mm)
        state.left_min_mm = float(left_min_mm)
        state.right_min_mm = float(right_min_mm)
        state.serial_ok = bool(serial_ok)
        state.serial_cmd = str(serial_cmd)
        state.serial_reply = str(serial_reply)
        state.last_status_seen = time.time()
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

    def telemetry(self, stale_after_s: float | None = None) -> dict[str, Any]:
        robot_rows = []
        stale_after_s = None if stale_after_s is None else max(0.0, float(stale_after_s))
        stale_ids: list[str] = []
        for robot_id, state in sorted(self._robots.items()):
            is_stale = False
            if stale_after_s is not None and state.age_s > stale_after_s:
                is_stale = True
                stale_ids.append(robot_id)
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
                    "control_mode": state.control_mode,
                    "action_id": state.action_id,
                    "throttle": state.throttle,
                    "turn": state.turn,
                    "deposit": state.deposit,
                    "camera_found": state.camera_found,
                    "camera_distance_m": state.camera_distance_m,
                    "camera_angle_deg": state.camera_angle_deg,
                    "front_min_mm": state.front_min_mm,
                    "left_min_mm": state.left_min_mm,
                    "right_min_mm": state.right_min_mm,
                    "serial_ok": state.serial_ok,
                    "serial_cmd": state.serial_cmd,
                    "serial_reply": state.serial_reply,
                    "status_age_s": None if state.last_status_seen is None else max(0.0, time.time() - state.last_status_seen),
                    "connected": not is_stale,
                }
            )
        return {
            "robot_count": max(0, len(self._robots) - len(stale_ids)),
            "tracked_count": len(self._robots),
            "robot_ids": sorted(self._robots.keys()),
            "stale_ids": stale_ids,
            "robot_rows": robot_rows,
        }
