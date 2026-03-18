from __future__ import annotations

import argparse
import time

from env.config import SwarmConfig
from pi.esp32_action_bridge import ESP32ActionBridge, ESP32MotionConfig
from pi.esp32_robot import ESP32Robot
from pi.esp32_sensor_adapter import ESP32SensorAdapter
from robot.action_bridge import CommandMapper
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


def main(argv=None):
    args = parse_args(argv)
    cfg = SwarmConfig(
        lidar_max_range=args.lidar_max_range_mm / 1000.0,
        max_speed=args.max_speed_mps,
    )
    robot = ESP32Robot(port=args.port, baudrate=args.baudrate)
    sensor_adapter = ESP32SensorAdapter(cfg, robot)
    observation_builder = ObservationBuilder(cfg)
    policy = PolicyRunner(cfg, checkpoint_dir=args.checkpoint_dir, shared_policy=args.shared_policy)
    policy.load(observation_builder.obs_dim())
    command_mapper = CommandMapper(cfg)
    action_bridge = ESP32ActionBridge(
        robot,
        ESP32MotionConfig(
            turn_step_deg=args.turn_step_deg,
            move_distance_mm=args.move_distance_mm,
            reverse_distance_mm=args.reverse_distance_mm,
        ),
    )

    period_s = 1.0 / max(args.hz, 1e-6)
    step = 0
    robot.connect()

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            t0 = time.perf_counter()
            packet = sensor_adapter.read_sensor_packet(duration=args.scan_duration)
            observation = observation_builder.build(packet)
            action_id = policy.predict(observation)
            command = command_mapper.build_command(action_id=action_id)
            action_bridge.send(command)
            step += 1

            elapsed = time.perf_counter() - t0
            sleep_time = period_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        action_bridge.close()


if __name__ == "__main__":
    main()
