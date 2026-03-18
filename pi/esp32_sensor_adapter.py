from __future__ import annotations

import time

from env.config import SwarmConfig
from pi.esp32_robot import ESP32Robot
from robot.messages import SensorPacket


class ESP32SensorAdapter:
    """Convert ESP32 scan lines into the generic SensorPacket used by robot/."""

    def __init__(self, cfg: SwarmConfig, robot: ESP32Robot):
        self.cfg = cfg
        self.robot = robot

    def read_sensor_packet(self, duration: float = 0.5) -> SensorPacket:
        scan_points = self.robot.read_sensor_lines(duration=duration)
        ranges_mm = self._bucketize_scan(scan_points)
        ranges_m = [value / 1000.0 for value in ranges_mm]
        return SensorPacket(
            ts_ms=int(time.time() * 1000),
            ranges_m=ranges_m,
            imu_yaw=0.0,
            imu_yaw_rate=0.0,
            speed_mps=0.0,
            target_vector_body=[0.0, 0.0],
            neighbor_vector_body=[0.0, 0.0],
            pheromone_samples=[0.0] * self.cfg.pheromone_samples,
            battery_v=0.0,
            estop=False,
        )

    def _bucketize_scan(self, scan_points: list[dict]) -> list[float]:
        """Convert arbitrary scan angles into fixed lidar-style buckets."""

        max_range_mm = float(self.cfg.lidar_max_range)
        buckets = [max_range_mm for _ in range(self.cfg.lidar_rays)]
        if self.cfg.lidar_rays <= 0:
            return buckets

        angle_span = 180.0
        bucket_width = angle_span / self.cfg.lidar_rays
        for item in scan_points:
            if item.get("type") != "scan":
                continue
            angle = item.get("angle")
            dist_mm = item.get("tof_mm", -1)
            if angle is None or dist_mm is None or dist_mm < 0:
                continue
            index = int(float(angle) / bucket_width)
            index = max(0, min(self.cfg.lidar_rays - 1, index))
            buckets[index] = min(buckets[index], float(dist_mm))
        return buckets

