from __future__ import annotations

from dataclasses import dataclass

from pi.esp32_robot import ESP32Robot
from robot.messages import ActionCommand


@dataclass
class ESP32MotionConfig:
    """Mapping from generic policy actions into the existing ESP32 command set."""

    turn_step_deg: int = 25
    move_distance_mm: int = 100
    reverse_distance_mm: int = 60
    forward_angle_deg: int = 0
    reverse_angle_deg: int = 180


class ESP32ActionBridge:
    """Translate ActionCommand objects into the current ESP32 serial protocol."""

    def __init__(self, robot: ESP32Robot, motion_cfg: ESP32MotionConfig):
        self.robot = robot
        self.motion_cfg = motion_cfg

    def send(self, command: ActionCommand) -> None:
        if command.mode == "stop":
            self.robot.stop()
            return

        if command.throttle == 0.0 and command.turn == 0.0:
            self.robot.stop()
            return

        if command.turn != 0.0:
            turn_angle = int(round(command.turn * self.motion_cfg.turn_step_deg))
            self.robot.turn(turn_angle)

        if command.throttle > 0.0:
            self.robot.move(self.motion_cfg.forward_angle_deg, self.motion_cfg.move_distance_mm)
            return

        if command.throttle < 0.0:
            self.robot.move(self.motion_cfg.reverse_angle_deg, self.motion_cfg.reverse_distance_mm)
            return

        self.robot.stop()

    def close(self) -> None:
        self.robot.close()

