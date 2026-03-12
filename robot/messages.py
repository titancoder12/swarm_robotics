from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SensorPacket:
    """Normalized sensor payload delivered to the Pi-side runtime."""

    ts_ms: int
    imu_yaw: float = 0.0
    imu_yaw_rate: float = 0.0
    speed_mps: float = 0.0
    ranges_m: List[float] = field(default_factory=list)
    target_vector_body: List[float] = field(default_factory=lambda: [0.0, 0.0])
    neighbor_vector_body: List[float] = field(default_factory=lambda: [0.0, 0.0])
    pheromone_samples: List[float] = field(default_factory=list)
    battery_v: float = 0.0
    estop: bool = False
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass
class ActionCommand:
    """High-level command sent from the Pi to the low-level controller."""

    ts_ms: int
    action_id: int
    throttle: float
    turn: float
    mode: str = "run"


@dataclass
class RuntimeSample:
    """One control-loop sample for logging or offline playback."""

    sensor: SensorPacket
    observation: List[float]
    command: ActionCommand

