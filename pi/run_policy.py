from __future__ import annotations

import argparse
import time

from AntSwarmFirmware.ant import ESP32Robot
from env.config import SwarmConfig
from robot.messages import SensorPacket
from robot.observation_builder import ObservationBuilder
from robot.policy_runner import PolicyRunner


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--port", type=str, default="/dev/serial0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--scan-duration", type=float, default=0.5)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lidar-max-range-mm", type=float, default=1600.0)
    parser.add_argument("--max-speed-mps", type=float, default=0.5)
    parser.add_argument("--turn-step-deg", type=int, default=25)
    parser.add_argument("--move-distance-mm", type=int, default=100)
    parser.add_argument("--reverse-distance-mm", type=int, default=60)
    return parser.parse_args(argv)


def bucketize_scan(cfg: SwarmConfig, scan_points: list[dict]) -> list[float]:
    max_range_mm = float(cfg.lidar_max_range * 1000.0)
    buckets = [max_range_mm for _ in range(cfg.lidar_rays)]
    if cfg.lidar_rays <= 0:
        return buckets

    angle_span = 180.0
    bucket_width = angle_span / cfg.lidar_rays
    for item in scan_points:
        if item.get("type") != "scan":
            continue
        angle = item.get("angle")
        dist_mm = item.get("tof_mm", -1)
        if angle is None or dist_mm is None or dist_mm < 0:
            continue
        index = int(float(angle) / bucket_width)
        index = max(0, min(cfg.lidar_rays - 1, index))
        buckets[index] = min(buckets[index], float(dist_mm))
    return buckets


def read_sensor_packet(robot: ESP32Robot, cfg: SwarmConfig, duration: float) -> SensorPacket:
    scan_points = robot.read_sensor_lines(duration=duration)
    ranges_m = [value / 1000.0 for value in bucketize_scan(cfg, scan_points)]
    return SensorPacket(
        ts_ms=int(time.time() * 1000),
        ranges_m=ranges_m,
        imu_yaw=0.0,
        imu_yaw_rate=0.0,
        speed_mps=0.0,
        target_vector_body=[0.0, 0.0],
        neighbor_vector_body=[0.0, 0.0],
        pheromone_samples=[0.0] * cfg.pheromone_samples,
        battery_v=0.0,
        estop=False,
    )


def execute_action(
    robot: ESP32Robot,
    action_id: int,
    turn_step_deg: int,
    move_distance_mm: int,
    reverse_distance_mm: int,
) -> None:
    throttle_vals = (-1.0, 0.0, 1.0)
    turn_vals = (-1.0, 0.0, 1.0)
    action_table = [(throttle, turn) for throttle in throttle_vals for turn in turn_vals]

    if action_id < 0 or action_id >= len(action_table):
        raise ValueError(f"Unsupported action id {action_id}.")

    throttle, turn = action_table[action_id]

    if throttle == 0.0 and turn == 0.0:
        robot.stop()
        return

    if turn != 0.0:
        robot.turn(int(round(turn * turn_step_deg)))

    if throttle > 0.0:
        robot.move(0, move_distance_mm)
        return

    if throttle < 0.0:
        robot.move(180, reverse_distance_mm)
        return

    robot.stop()


def main(argv=None):
    args = parse_args(argv)
    cfg = SwarmConfig(
        lidar_max_range=args.lidar_max_range_mm / 1000.0,
        max_speed=args.max_speed_mps,
    )
    robot = ESP32Robot(port=args.port, baudrate=args.baudrate)
    observation_builder = ObservationBuilder(cfg)
    policy = PolicyRunner(cfg, checkpoint_dir=args.checkpoint_dir, shared_policy=args.shared_policy)
    policy.load(observation_builder.obs_dim())

    period_s = 1.0 / max(args.hz, 1e-6)
    step = 0
    robot.connect()

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            t0 = time.perf_counter()
            packet = read_sensor_packet(robot, cfg, duration=args.scan_duration)
            observation = observation_builder.build(packet)
            action_id = policy.predict(observation)
            execute_action(
                robot,
                action_id=action_id,
                turn_step_deg=args.turn_step_deg,
                move_distance_mm=args.move_distance_mm,
                reverse_distance_mm=args.reverse_distance_mm,
            )
            step += 1

            elapsed = time.perf_counter() - t0
            sleep_time = period_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
