from __future__ import annotations

import argparse
import math
import random
import socket
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fake robot client for local command-center testing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--robot-id", default="robot_0")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--world-width-cm", type=float, default=900.0)
    parser.add_argument("--world-height-cm", type=float, default=600.0)
    parser.add_argument("--step-cm", type=float, default=12.0)
    parser.add_argument("--max-turn-deg", type=float, default=30.0)
    parser.add_argument("--lidar-max-range-mm", type=float, default=1600.0)
    return parser


def _wrap_degrees(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def _ray_distance_mm(
    x_cm: float,
    y_cm: float,
    heading_deg: float,
    world_half_width_cm: float,
    world_half_height_cm: float,
    max_range_mm: float,
) -> float:
    angle_rad = math.radians(heading_deg)
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    candidates_cm = []
    if abs(dx) > 1e-6:
        tx = (world_half_width_cm - x_cm) / dx
        if tx > 0:
            y_at_tx = y_cm + tx * dy
            if -world_half_height_cm <= y_at_tx <= world_half_height_cm:
                candidates_cm.append(tx)
        tx = (-world_half_width_cm - x_cm) / dx
        if tx > 0:
            y_at_tx = y_cm + tx * dy
            if -world_half_height_cm <= y_at_tx <= world_half_height_cm:
                candidates_cm.append(tx)
    if abs(dy) > 1e-6:
        ty = (world_half_height_cm - y_cm) / dy
        if ty > 0:
            x_at_ty = x_cm + ty * dx
            if -world_half_width_cm <= x_at_ty <= world_half_width_cm:
                candidates_cm.append(ty)
        ty = (-world_half_height_cm - y_cm) / dy
        if ty > 0:
            x_at_ty = x_cm + ty * dx
            if -world_half_width_cm <= x_at_ty <= world_half_width_cm:
                candidates_cm.append(ty)
    if not candidates_cm:
        return max_range_mm
    return min(max_range_mm, min(candidates_cm) * 10.0)


def _bounded_random_walk_step(
    x_cm: float,
    y_cm: float,
    heading_deg: float,
    step_cm: float,
    max_turn_deg: float,
    world_half_width_cm: float,
    world_half_height_cm: float,
) -> tuple[float, float, float]:
    # Wander locally, but add a soft steering bias back toward the arena center
    # as the fake robot approaches the world edges.
    turn_jitter_deg = random.uniform(-max_turn_deg, max_turn_deg)
    center_heading_deg = math.degrees(math.atan2(-y_cm, -x_cm)) if abs(x_cm) + abs(y_cm) > 1e-6 else heading_deg
    edge_pressure = max(abs(x_cm) / max(world_half_width_cm, 1e-6), abs(y_cm) / max(world_half_height_cm, 1e-6))
    if edge_pressure > 0.65:
        steer_to_center = _wrap_degrees(center_heading_deg - heading_deg)
        turn_jitter_deg += 0.45 * steer_to_center
    next_heading_deg = (heading_deg + turn_jitter_deg) % 360.0
    next_x_cm = x_cm + math.cos(math.radians(next_heading_deg)) * step_cm
    next_y_cm = y_cm + math.sin(math.radians(next_heading_deg)) * step_cm

    if abs(next_x_cm) > world_half_width_cm or abs(next_y_cm) > world_half_height_cm:
        bounce_heading_deg = center_heading_deg % 360.0
        next_heading_deg = bounce_heading_deg
        next_x_cm = x_cm + math.cos(math.radians(next_heading_deg)) * step_cm
        next_y_cm = y_cm + math.sin(math.radians(next_heading_deg)) * step_cm

    next_x_cm = max(-world_half_width_cm, min(world_half_width_cm, next_x_cm))
    next_y_cm = max(-world_half_height_cm, min(world_half_height_cm, next_y_cm))
    return next_x_cm, next_y_cm, next_heading_deg


def _randomize_lidar_mm(base_ranges_mm: list[float], max_range_mm: float) -> list[float]:
    if not base_ranges_mm:
        return []
    values = []
    occluder_center = random.randrange(len(base_ranges_mm))
    occluder_half_width = random.choice((0, 1, 1, 2))
    occluder_active = random.random() < 0.35
    occluder_depth_mm = random.uniform(180.0, 700.0)
    for idx, base in enumerate(base_ranges_mm):
        jitter_mm = random.uniform(-60.0, 60.0)
        wobble_mm = random.uniform(-35.0, 35.0) * math.sin(idx * 0.9 + random.uniform(-0.6, 0.6))
        value = base + jitter_mm + wobble_mm
        if occluder_active and abs(idx - occluder_center) <= occluder_half_width:
            value = min(value, occluder_depth_mm + random.uniform(-40.0, 40.0))
        values.append(max(80.0, min(max_range_mm, value)))
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    period = 1.0 / max(args.rate_hz, 1e-6)
    world_half_width_cm = args.world_width_cm * 0.5
    world_half_height_cm = args.world_height_cm * 0.5
    x_cm = 0.0
    y_cm = 0.0
    heading_deg = random.uniform(0.0, 360.0)

    with socket.create_connection((args.host, args.port), timeout=3.0) as sock:
        file = sock.makefile("r", encoding="utf-8", newline="\n")
        for step in range(args.steps):
            x_cm, y_cm, heading_deg = _bounded_random_walk_step(
                x_cm=x_cm,
                y_cm=y_cm,
                heading_deg=heading_deg,
                step_cm=args.step_cm,
                max_turn_deg=args.max_turn_deg,
                world_half_width_cm=world_half_width_cm,
                world_half_height_cm=world_half_height_cm,
            )

            pos_line = f"POS,{args.robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}\n"
            sock.sendall(pos_line.encode("utf-8"))

            lidar_values = []
            for ray_idx in range(9):
                ray_heading_deg = heading_deg + (ray_idx - 4) * (180.0 / 8.0)
                lidar_values.append(
                    _ray_distance_mm(
                        x_cm=x_cm,
                        y_cm=y_cm,
                        heading_deg=ray_heading_deg,
                        world_half_width_cm=world_half_width_cm,
                        world_half_height_cm=world_half_height_cm,
                        max_range_mm=args.lidar_max_range_mm,
                    )
                )
            lidar_values = _randomize_lidar_mm(lidar_values, args.lidar_max_range_mm)
            lidar_line = "LIDAR," + args.robot_id + "," + ",".join(f"{value:.1f}" for value in lidar_values) + "\n"
            sock.sendall(lidar_line.encode("utf-8"))

            if step % 3 == 0:
                amount = 1.0 + 0.25 * random.random()
                pher_line = f"PHER,{args.robot_id},{x_cm:.2f},{y_cm:.2f},{amount:.3f}\n"
                sock.sendall(pher_line.encode("utf-8"))

            sense_line = f"SENSE,{args.robot_id},{x_cm:.2f},{y_cm:.2f},{heading_deg:.2f}\n"
            sock.sendall(sense_line.encode("utf-8"))
            response = file.readline().strip()
            print(f"step={step} pos=({x_cm:.1f},{y_cm:.1f}) heading={heading_deg:.1f} resp={response}")
            time.sleep(period)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
