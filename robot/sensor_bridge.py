from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import BinaryIO

from robot.messages import SensorPacket


class SensorBridge(ABC):
    """Abstract source of robot sensor packets."""

    @abstractmethod
    def read_sensor_packet(self) -> SensorPacket:
        """Return the latest available sensor packet."""

    def close(self) -> None:
        """Release resources held by the bridge."""


class JsonlSensorBridge(SensorBridge):
    """Read line-delimited JSON packets from a byte stream or serial transport."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def read_sensor_packet(self) -> SensorPacket:
        line = self.stream.readline()
        if not line:
            raise RuntimeError("No sensor data available from transport.")
        payload = json.loads(line.decode("utf-8"))
        return SensorPacket(
            ts_ms=int(payload["ts_ms"]),
            imu_yaw=float(payload.get("imu_yaw", 0.0)),
            imu_yaw_rate=float(payload.get("imu_yaw_rate", 0.0)),
            speed_mps=float(payload.get("speed_mps", 0.0)),
            ranges_m=list(payload.get("ranges_m", [])),
            target_vector_body=list(payload.get("target_vector_body", [0.0, 0.0])),
            neighbor_vector_body=list(payload.get("neighbor_vector_body", [0.0, 0.0])),
            pheromone_samples=list(payload.get("pheromone_samples", [])),
            battery_v=float(payload.get("battery_v", 0.0)),
            estop=bool(payload.get("estop", False)),
            extras=dict(payload.get("extras", {})),
        )

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if callable(close):
            close()


def open_serial_jsonl_bridge(port: str, baudrate: int = 115200, timeout: float = 0.1) -> JsonlSensorBridge:
    """Create a JSONL sensor bridge backed by pyserial."""

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for serial deployment. Install it before using the serial bridge."
        ) from exc
    return JsonlSensorBridge(serial.Serial(port=port, baudrate=baudrate, timeout=timeout))

