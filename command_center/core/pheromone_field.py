from __future__ import annotations

import math

import numpy as np

from command_center.config import CommandCenterConfig


class PheromoneField:
    """Digital pheromone grid aligned with the simulator's sampling contract."""

    def __init__(self, cfg: CommandCenterConfig) -> None:
        self.cfg = cfg
        self.width_cm = cfg.world_width_cm
        self.height_cm = cfg.world_height_cm
        self.x_min_cm = -cfg.half_width_cm
        self.y_min_cm = -cfg.half_height_cm
        self.cell_size_cm = cfg.cell_size_cm
        self.grid_w = int(math.floor(self.width_cm / self.cell_size_cm)) + 1
        self.grid_h = int(math.floor(self.height_cm / self.cell_size_cm)) + 1
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def clear(self) -> None:
        self.grid.fill(0.0)

    def coordinate_to_cell(self, x_cm: float, y_cm: float) -> tuple[int, int]:
        gx = int(np.clip((x_cm - self.x_min_cm) // self.cell_size_cm, 0, self.grid_w - 1))
        gy = int(np.clip((y_cm - self.y_min_cm) // self.cell_size_cm, 0, self.grid_h - 1))
        return gx, gy

    def deposit(self, x_cm: float, y_cm: float, amount: float) -> tuple[int, int]:
        gx, gy = self.coordinate_to_cell(x_cm, y_cm)
        self.grid[gy, gx] += float(amount)
        return gx, gy

    def decay_step(self) -> None:
        self.grid *= self.cfg.decay_factor

    def diffuse_step(self) -> None:
        diff = self.cfg.diffuse_rate
        if diff <= 0.0:
            return
        grid = self.grid
        rolled = (
            np.roll(grid, 1, axis=0)
            + np.roll(grid, -1, axis=0)
            + np.roll(grid, 1, axis=1)
            + np.roll(grid, -1, axis=1)
        ) * 0.25
        self.grid = ((1.0 - diff) * grid + diff * rolled).astype(np.float32)

    def update(self) -> None:
        self.decay_step()
        self.diffuse_step()

    def sample_forward(self, x_cm: float, y_cm: float, heading_rad: float) -> np.ndarray:
        samples = []
        for i in range(self.cfg.pheromone_samples):
            dist = (i + 1) * self.cfg.agent_radius_cm * 1.5
            sx = x_cm + math.cos(heading_rad) * dist
            sy = y_cm + math.sin(heading_rad) * dist
            gx, gy = self.coordinate_to_cell(sx, sy)
            samples.append(self.grid[gy, gx])
        arr = np.array(samples, dtype=np.float32)
        if arr.max() > 0:
            arr = arr / (arr.max() + 1e-6)
        return arr

    def telemetry(self) -> dict[str, float]:
        total = float(self.grid.sum())
        max_val = float(self.grid.max()) if self.grid.size else 0.0
        return {
            "pheromone_total": total,
            "pheromone_max": max_val,
            "grid_w": float(self.grid_w),
            "grid_h": float(self.grid_h),
        }

