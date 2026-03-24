from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from command_center.config import CommandCenterConfig
from command_center.core.pheromone_field import PheromoneField
from command_center.core.robot_registry import RobotRegistry
from command_center.core.trail_store import TrailStore


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
        self._last_decay = time.time()

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

    def snapshot(self) -> WorldSnapshot:
        with self._lock:
            telemetry = {
                **self.robot_registry.telemetry(),
                **self.pheromone.telemetry(),
                "paused": self._paused,
                "show_trails": self._show_trails,
                "show_pheromone": self._show_pheromone,
                "stale_ids": self.robot_registry.stale_ids(self.cfg.robot_stale_after_s),
            }
            return WorldSnapshot(
                robots=self.robot_registry.snapshot(),
                # Snapshot copies keep the renderer read-only and avoid holding
                # locks while PyGame is drawing.
                trails=self.trails.snapshot() if self._show_trails else {},
                pheromone_grid=self.pheromone.grid.copy() if self._show_pheromone else np.zeros_like(self.pheromone.grid),
                telemetry=telemetry,
            )
