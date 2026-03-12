from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import BinaryIO

from env.config import SwarmConfig
from robot.messages import ActionCommand


class ActionBridge(ABC):
    """Abstract sink for high-level action commands."""

    @abstractmethod
    def send(self, command: ActionCommand) -> None:
        """Transmit an action command to the low-level controller."""

    def close(self) -> None:
        """Release resources held by the bridge."""


class JsonlActionBridge(ActionBridge):
    """Write line-delimited JSON commands to a byte stream or serial transport."""

    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def send(self, command: ActionCommand) -> None:
        payload = {
            "ts_ms": command.ts_ms,
            "mode": command.mode,
            "action_id": command.action_id,
            "throttle": command.throttle,
            "turn": command.turn,
        }
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        self.stream.write(line.encode("utf-8"))
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()

    def close(self) -> None:
        close = getattr(self.stream, "close", None)
        if callable(close):
            close()


class CommandMapper:
    """Map discrete policy actions into high-level robot motion commands."""

    def __init__(self, cfg: SwarmConfig):
        self.cfg = cfg
        self.action_table = self._build_action_table()

    def build_command(self, action_id: int, mode: str = "run") -> ActionCommand:
        if action_id < 0 or action_id >= len(self.action_table):
            raise ValueError(f"Unsupported action id {action_id}.")
        throttle, turn = self.action_table[action_id]
        return ActionCommand(
            ts_ms=int(time.time() * 1000),
            action_id=action_id,
            throttle=float(throttle),
            turn=float(turn),
            mode=mode,
        )

    def _build_action_table(self) -> list[tuple[float, float]]:
        throttle_vals = [-1.0, 0.0, 1.0]
        turn_vals = [-1.0, 0.0, 1.0]
        table = []
        for throttle in throttle_vals:
            for turn in turn_vals:
                table.append((throttle, turn))
        return table


def open_serial_action_bridge(port: str, baudrate: int = 115200, timeout: float = 0.1) -> JsonlActionBridge:
    """Create a JSONL action bridge backed by pyserial."""

    try:
        import serial
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is required for serial deployment. Install it before using the serial bridge."
        ) from exc
    return JsonlActionBridge(serial.Serial(port=port, baudrate=baudrate, timeout=timeout))

