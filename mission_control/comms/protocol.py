from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ProtocolError(ValueError):
    """Raised when a line does not match the command-center protocol."""


@dataclass(frozen=True)
class Message:
    kind: str


@dataclass(frozen=True)
class PositionMessage(Message):
    robot_id: str
    x_cm: float
    y_cm: float
    heading_deg: float | None = None


@dataclass(frozen=True)
class PheromoneMessage(Message):
    robot_id: str
    x_cm: float
    y_cm: float
    amount: float


@dataclass(frozen=True)
class SenseMessage(Message):
    robot_id: str
    x_cm: float
    y_cm: float
    heading_deg: float


@dataclass(frozen=True)
class LidarMessage(Message):
    robot_id: str
    ranges_mm: tuple[float, ...]


@dataclass(frozen=True)
class TargetMessage(Message):
    robot_id: str
    x_cm: float
    y_cm: float
    confidence: float


@dataclass(frozen=True)
class StatusMessage(Message):
    robot_id: str
    control_mode: str
    action_id: int
    throttle: float
    turn: float
    deposit: bool
    camera_found: bool
    camera_distance_m: float
    camera_angle_deg: float
    front_min_mm: float
    left_min_mm: float
    right_min_mm: float
    serial_ok: bool
    serial_cmd: str
    serial_reply: str


def parse_line(line: str) -> Message:
    # Keep the wire format deliberately small and line-oriented so it can be
    # mirrored easily on the robot side with serial.readline()-style loops.
    parts = [part.strip() for part in line.strip().split(",")]
    if not parts or not parts[0]:
        raise ProtocolError("empty line")

    kind = parts[0].upper()
    if kind == "POS":
        if len(parts) not in (4, 5):
            raise ProtocolError(f"POS expects 4 or 5 fields, got {len(parts)}")
        heading_deg = None if len(parts) == 4 else float(parts[4])
        return PositionMessage(kind="POS", robot_id=parts[1], x_cm=float(parts[2]), y_cm=float(parts[3]), heading_deg=heading_deg)

    if kind == "PHER":
        if len(parts) != 5:
            raise ProtocolError(f"PHER expects 5 fields, got {len(parts)}")
        return PheromoneMessage(kind="PHER", robot_id=parts[1], x_cm=float(parts[2]), y_cm=float(parts[3]), amount=float(parts[4]))

    if kind == "SENSE":
        if len(parts) != 5:
            raise ProtocolError(f"SENSE expects 5 fields, got {len(parts)}")
        return SenseMessage(kind="SENSE", robot_id=parts[1], x_cm=float(parts[2]), y_cm=float(parts[3]), heading_deg=float(parts[4]))

    if kind == "LIDAR":
        if len(parts) < 3:
            raise ProtocolError(f"LIDAR expects at least 3 fields, got {len(parts)}")
        try:
            ranges_mm = tuple(float(part) for part in parts[2:])
        except ValueError as exc:
            raise ProtocolError(f"LIDAR contains a non-numeric range: {exc}") from exc
        return LidarMessage(kind="LIDAR", robot_id=parts[1], ranges_mm=ranges_mm)

    if kind == "TARGET":
        if len(parts) != 5:
            raise ProtocolError(f"TARGET expects 5 fields, got {len(parts)}")
        return TargetMessage(
            kind="TARGET",
            robot_id=parts[1],
            x_cm=float(parts[2]),
            y_cm=float(parts[3]),
            confidence=float(parts[4]),
        )

    if kind == "STATUS":
        if len(parts) != 16:
            raise ProtocolError(f"STATUS expects 16 fields, got {len(parts)}")
        return StatusMessage(
            kind="STATUS",
            robot_id=parts[1],
            control_mode=parts[2],
            action_id=int(parts[3]),
            throttle=float(parts[4]),
            turn=float(parts[5]),
            deposit=parts[6] == "1",
            camera_found=parts[7] == "1",
            camera_distance_m=float(parts[8]),
            camera_angle_deg=float(parts[9]),
            front_min_mm=float(parts[10]),
            left_min_mm=float(parts[11]),
            right_min_mm=float(parts[12]),
            serial_ok=parts[13] == "1",
            serial_cmd=parts[14],
            serial_reply=parts[15],
        )

    raise ProtocolError(f"unknown message type {kind}")


def format_pheromone_response(robot_id: str, samples: np.ndarray) -> str:
    # Always emit exactly 3 pheromone values so the reply can be copied
    # directly into the current 23-D observation vector slots.
    values = [f"{float(value):.6f}" for value in samples[:3]]
    while len(values) < 3:
        values.append("0.000000")
    return ",".join(["PHER_RESP", str(robot_id), *values])
