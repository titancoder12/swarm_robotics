from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from env.config import SwarmConfig
from robot.messages import SensorPacket


@dataclass
class ObservationBuilder:
    """Convert real-world sensor packets into policy-ready observation vectors."""

    cfg: SwarmConfig

    def obs_dim(self) -> int:
        return (
            self.cfg.lidar_rays
            + 2
            + 2
            + 2
            + 1
            + self.cfg.pheromone_samples
        )

    def build(self, packet: SensorPacket) -> np.ndarray:
        """Build one normalized observation vector matching simulator layout."""

        lidar = self._normalize_ranges(packet.ranges_m)
        target = self._normalize_xy(packet.target_vector_body)
        neighbor = self._normalize_xy(packet.neighbor_vector_body)
        heading = np.array(
            [math.sin(packet.imu_yaw), math.cos(packet.imu_yaw)],
            dtype=np.float32,
        )
        speed = np.array(
            [np.clip(packet.speed_mps / self.cfg.max_speed, -1.0, 1.0)],
            dtype=np.float32,
        )
        pheromone = self._normalize_pheromone(packet.pheromone_samples)
        obs = np.concatenate([lidar, target, neighbor, heading, speed, pheromone]).astype(np.float32)
        if obs.shape[0] != self.obs_dim():
            raise ValueError(f"Observation size mismatch: expected {self.obs_dim()}, got {obs.shape[0]}.")
        return obs

    def _normalize_ranges(self, ranges_m: list[float]) -> np.ndarray:
        values = list(ranges_m[: self.cfg.lidar_rays])
        while len(values) < self.cfg.lidar_rays:
            values.append(self.cfg.lidar_max_range)
        arr = np.array(values, dtype=np.float32)
        arr = np.clip(arr / self.cfg.lidar_max_range, 0.0, 1.0)
        return arr * 2.0 - 1.0

    def _normalize_xy(self, vec: list[float]) -> np.ndarray:
        if len(vec) < 2:
            vec = [0.0, 0.0]
        scale = max(self.cfg.lidar_max_range, 1.0)
        arr = np.array(vec[:2], dtype=np.float32) / scale
        return np.clip(arr, -1.0, 1.0)

    def _normalize_pheromone(self, values: list[float]) -> np.ndarray:
        samples = list(values[: self.cfg.pheromone_samples])
        while len(samples) < self.cfg.pheromone_samples:
            samples.append(0.0)
        arr = np.array(samples, dtype=np.float32)
        if arr.size == 0:
            return arr
        max_abs = float(np.max(np.abs(arr)))
        if max_abs > 1e-6:
            arr = arr / max_abs
        return np.clip(arr, -1.0, 1.0)

