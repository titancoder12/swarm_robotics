from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from mission_control.config import CommandCenterConfig
from mission_control.core.pheromone_field import PheromoneField
from mission_control.core.robot_registry import RobotRegistry
from mission_control.core.trail_store import TrailStore


@dataclass
class WorldSnapshot:
    robots: dict[str, Any]
    trails: dict[str, list[tuple[float, float]]]
    pheromone_grid: np.ndarray
    telemetry: dict[str, Any]


class WorldState:
    """Thread-safe state container for incoming robot data and rendered state."""

    def __init__(self, cfg: CommandCenterConfig) -> None:
        self.cfg = cfg
        self.robot_registry = RobotRegistry()
        self.trails = TrailStore(cfg.trail_max_points)
        self.pheromone = PheromoneField(cfg)
        self._lock = threading.RLock()
        self._paused = False
        self._show_trails = True
        self._show_pheromone = True
        self._show_targets = True
        self._last_decay = time.time()
        self._control_flash_until: dict[str, float] = {}

    def update_position(
        self,
        robot_id: str,
        x_cm: float,
        y_cm: float,
        heading_deg: float | None,
        connection_label: str | None = None,
    ) -> None:
        with self._lock:
            self.robot_registry.update_position(
                robot_id=robot_id,
                x_cm=x_cm,
                y_cm=y_cm,
                heading_deg=heading_deg,
                connection_label=connection_label,
            )
            self.trails.add_point(robot_id, x_cm, y_cm)

    def deposit_pheromone(self, x_cm: float, y_cm: float, amount: float) -> tuple[int, int]:
        with self._lock:
            return self.pheromone.deposit(x_cm, y_cm, amount)

    def update_target_detection(self, robot_id: str, x_cm: float, y_cm: float, confidence: float) -> None:
        with self._lock:
            self.robot_registry.update_target(robot_id, x_cm, y_cm, confidence)

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
    ) -> None:
        with self._lock:
            self.robot_registry.update_status(
                robot_id=robot_id,
                control_mode=control_mode,
                action_id=action_id,
                throttle=throttle,
                turn=turn,
                deposit=deposit,
                camera_found=camera_found,
                camera_distance_m=camera_distance_m,
                camera_angle_deg=camera_angle_deg,
                front_min_mm=front_min_mm,
                left_min_mm=left_min_mm,
                right_min_mm=right_min_mm,
                serial_ok=serial_ok,
                serial_cmd=serial_cmd,
                serial_reply=serial_reply,
            )

    def sample_pheromone(self, x_cm: float, y_cm: float, heading_deg: float) -> np.ndarray:
        heading_rad = math.radians(heading_deg)
        with self._lock:
            return self.pheromone.sample_forward(x_cm, y_cm, heading_rad)

    def tick(self) -> None:
        with self._lock:
            if self._paused:
                return
            now = time.time()
            # Decouple pheromone update cadence from render cadence so the UI
            # can run smoothly without changing the field dynamics.
            if now - self._last_decay >= self.cfg.pheromone_decay_period_s:
                self.pheromone.update()
                self._last_decay = now

    def clear_pheromone(self) -> None:
        with self._lock:
            self.pheromone.clear()

    def toggle_paused(self) -> bool:
        with self._lock:
            self._paused = not self._paused
            return self._paused

    def toggle_trails(self) -> bool:
        with self._lock:
            self._show_trails = not self._show_trails
            return self._show_trails

    def toggle_pheromone(self) -> bool:
        with self._lock:
            self._show_pheromone = not self._show_pheromone
            return self._show_pheromone

    def toggle_targets(self) -> bool:
        with self._lock:
            self._show_targets = not self._show_targets
            return self._show_targets

    def flash_control(self, control_id: str, duration_s: float = 0.3) -> None:
        with self._lock:
            self._control_flash_until[control_id] = time.time() + max(0.0, duration_s)

    def snapshot(self) -> WorldSnapshot:
        with self._lock:
            now = time.time()
            flashed_controls = sorted(
                control_id
                for control_id, until in self._control_flash_until.items()
                if until > now
            )
            self._control_flash_until = {
                control_id: until
                for control_id, until in self._control_flash_until.items()
                if until > now
            }
            stale_ids = self.robot_registry.stale_ids(self.cfg.robot_stale_after_s)
            telemetry = {
                **self.robot_registry.telemetry(self.cfg.robot_stale_after_s),
                **self.pheromone.telemetry(),
                "paused": self._paused,
                "show_trails": self._show_trails,
                "show_pheromone": self._show_pheromone,
                "show_targets": self._show_targets,
                "stale_ids": stale_ids,
                "flashed_controls": flashed_controls,
            }
            return WorldSnapshot(
                robots=self.robot_registry.snapshot(),
                # Snapshot copies keep the renderer read-only and avoid holding
                # locks while PyGame is drawing.
                trails=self.trails.snapshot() if self._show_trails else {},
                pheromone_grid=self.pheromone.grid.copy() if self._show_pheromone else np.zeros_like(self.pheromone.grid),
                telemetry=telemetry,
            )
