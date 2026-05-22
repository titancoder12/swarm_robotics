#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Iterable


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"


STEP_RE = re.compile(
    r"step=(?P<step>\d+).*?"
    r"control_mode=(?P<mode>\w+).*?"
    r"action=(?P<action>-?\d+)\s+"
    r"throttle=(?P<throttle>[+-]?\d+(?:\.\d+)?)\s+"
    r"turn=(?P<turn>[+-]?\d+(?:\.\d+)?)\s+"
    r"deposit=(?P<deposit>\d+).*?"
    r"heading_deg=(?P<heading>[+-]?\d+(?:\.\d+)?)"
)

CAMERA_FOUND_RE = re.compile(
    r"camera found=(?P<found>True|False)\s+distance_m=(?P<distance>[+-]?\d+(?:\.\d+)?)\s+angle_deg=(?P<angle>[+-]?\d+(?:\.\d+)?)"
)
CAMERA_TARGET_RE = re.compile(
    r"camera target found area=(?P<area>[+-]?\d+(?:\.\d+)?) .*? distance_m=(?P<distance>[+-]?\d+(?:\.\d+)?) angle_deg=(?P<angle>[+-]?\d+(?:\.\d+)?)"
)
BLE_LIDAR_RE = re.compile(r"BLE notify -> LIDAR,[^,]+,(?P<values>.+)$")
SERIAL_OUT_RE = re.compile(r"serial -> (?P<command>.+)$")
SERIAL_IN_RE = re.compile(r"serial <- (?P<reply>.+)$")


ACTION_LABELS: dict[int, str] = {
    0: "reverse-left",
    1: "reverse-left-deposit",
    2: "reverse-straight",
    3: "reverse-straight-deposit",
    4: "reverse-right",
    5: "reverse-right-deposit",
    6: "stop-left",
    7: "stop-left-deposit",
    8: "stop",
    9: "stop-deposit",
    10: "stop-right",
    11: "stop-right-deposit",
    12: "forward-left",
    13: "forward-left-deposit",
    14: "forward-straight",
    15: "forward-straight-deposit",
    16: "forward-right",
    17: "forward-right-deposit",
}


@dataclass
class MonitorState:
    camera_found: bool = False
    camera_distance_m: float = 0.0
    camera_angle_deg: float = 0.0
    camera_area: float | None = None
    lidar_mm: list[float] = field(default_factory=list)
    last_serial_out: str = ""
    last_serial_in: str = ""


def color(text: str, code: str, *, enable: bool) -> str:
    if not enable:
        return text
    return f"{code}{text}{ANSI_RESET}"


def parse_lidar_values(payload: str) -> list[float]:
    try:
        return [float(part.strip()) for part in payload.split(",")]
    except ValueError:
        return []


def infer_obstacle(lidar_mm: list[float]) -> tuple[str, str]:
    valid = [v for v in lidar_mm if v > 0]
    if not valid:
        return "no lidar data", "unknown"

    min_dist = min(valid)
    center_slice = lidar_mm[max(0, len(lidar_mm) // 2 - 1) : min(len(lidar_mm), len(lidar_mm) // 2 + 2)]
    center_valid = [v for v in center_slice if v > 0]
    center_min = min(center_valid) if center_valid else min_dist

    if center_min < 120:
        return f"blocked ahead ({center_min:.0f} mm)", "blocked"
    if center_min < 300:
        return f"close obstacle ahead ({center_min:.0f} mm)", "close"
    if min_dist < 500:
        return f"near obstacle nearby ({min_dist:.0f} mm)", "near"
    return f"path mostly clear (nearest {min_dist:.0f} mm)", "clear"


def infer_camera(state: MonitorState) -> tuple[str, str]:
    if not state.camera_found:
        return "camera sees no target", "none"
    return (
        f"camera sees target at {state.camera_distance_m:.2f} m, {state.camera_angle_deg:+.1f} deg",
        "target",
    )


def infer_action(action_id: int, throttle: float, turn: float, deposit: int) -> str:
    label = ACTION_LABELS.get(action_id, f"action_{action_id}")
    motion = []
    if throttle > 0:
        motion.append("forward")
    elif throttle < 0:
        motion.append("reverse")
    else:
        motion.append("stop")
    if turn < 0:
        motion.append("turn left")
    elif turn > 0:
        motion.append("turn right")
    else:
        motion.append("no turn")
    if deposit:
        motion.append("deposit pheromone")
    return f"{label}: " + ", ".join(motion)


def print_summary(state: MonitorState, line: str, *, use_color: bool) -> None:
    match = STEP_RE.search(line)
    if not match:
        return

    step = int(match.group("step"))
    action_id = int(match.group("action"))
    throttle = float(match.group("throttle"))
    turn = float(match.group("turn"))
    deposit = int(match.group("deposit"))
    heading = float(match.group("heading"))
    mode = match.group("mode")

    obstacle_text, obstacle_level = infer_obstacle(state.lidar_mm)
    camera_text, camera_level = infer_camera(state)
    action_text = infer_action(action_id, throttle, turn, deposit)

    obstacle_color = {
        "blocked": ANSI_RED,
        "close": ANSI_YELLOW,
        "near": ANSI_YELLOW,
        "clear": ANSI_GREEN,
        "unknown": ANSI_DIM,
    }[obstacle_level]
    camera_color = ANSI_CYAN if camera_level == "target" else ANSI_DIM
    action_color = ANSI_MAGENTA if throttle != 0 or turn != 0 else ANSI_WHITE

    print(
        color(f"Step {step}", ANSI_BOLD + ANSI_BLUE, enable=use_color)
        + f"  mode={mode}  heading={heading:.1f} deg"
    )
    print("  " + color("Sense:", ANSI_BOLD, enable=use_color) + " " + color(obstacle_text, obstacle_color, enable=use_color))
    print("  " + color("Camera:", ANSI_BOLD, enable=use_color) + " " + color(camera_text, camera_color, enable=use_color))
    print("  " + color("Action:", ANSI_BOLD, enable=use_color) + " " + color(action_text, action_color, enable=use_color))
    if state.last_serial_out:
        serial_text = f"last serial {state.last_serial_out}"
        if state.last_serial_in:
            serial_text += f" -> {state.last_serial_in}"
        print("  " + color("Motor:", ANSI_BOLD, enable=use_color) + " " + serial_text)
    print()


def iter_journal_lines(service: str, since: str | None) -> Iterable[str]:
    cmd = ["journalctl", "-u", service, "-f", "-n", "0", "--output=cat"]
    if since:
        cmd.extend(["--since", since])
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            yield line.rstrip("\n")
    finally:
        proc.kill()


def iter_stdin_lines() -> Iterable[str]:
    for line in sys.stdin:
        yield line.rstrip("\n")


def update_state_from_line(state: MonitorState, line: str) -> None:
    camera_match = CAMERA_FOUND_RE.search(line)
    if camera_match:
        state.camera_found = camera_match.group("found") == "True"
        state.camera_distance_m = float(camera_match.group("distance"))
        state.camera_angle_deg = float(camera_match.group("angle"))
        if not state.camera_found:
            state.camera_area = None
        return

    target_match = CAMERA_TARGET_RE.search(line)
    if target_match:
        state.camera_found = True
        state.camera_area = float(target_match.group("area"))
        state.camera_distance_m = float(target_match.group("distance"))
        state.camera_angle_deg = float(target_match.group("angle"))
        return

    lidar_match = BLE_LIDAR_RE.search(line)
    if lidar_match:
        state.lidar_mm = parse_lidar_values(lidar_match.group("values"))
        return

    serial_out = SERIAL_OUT_RE.search(line)
    if serial_out:
        state.last_serial_out = serial_out.group("command")
        return

    serial_in = SERIAL_IN_RE.search(line)
    if serial_in:
        state.last_serial_in = serial_in.group("reply")
        return

    if "lidar=" in line:
        try:
            payload = line.split("lidar=", 1)[1].split(" pheromone=", 1)[0]
            parsed = ast.literal_eval(payload)
            if isinstance(parsed, list):
                # This is the normalized preview, keep only if BLE mm lidar is absent.
                if not state.lidar_mm:
                    state.lidar_mm = [float(v) for v in parsed]
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Follow swarm-robot.service logs and print a human-readable robot state summary.")
    parser.add_argument("--service", default="swarm-robot.service", help="systemd service name to follow")
    parser.add_argument("--stdin", action="store_true", help="read log lines from stdin instead of journalctl")
    parser.add_argument("--since", default=None, help="optional journalctl --since value, e.g. '5 min ago'")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state = MonitorState()
    use_color = not args.no_color and sys.stdout.isatty()

    lines = iter_stdin_lines() if args.stdin else iter_journal_lines(args.service, args.since)

    try:
        for line in lines:
            update_state_from_line(state, line)
            print_summary(state, line, use_color=use_color)
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
