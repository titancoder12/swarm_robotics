from __future__ import annotations

import argparse
import sys
import time

from env.config import SwarmConfig
from robot.action_bridge import CommandMapper, JsonlActionBridge, open_serial_action_bridge
from robot.observation_builder import ObservationBuilder
from robot.policy_runner import PolicyRunner
from robot.sensor_bridge import JsonlSensorBridge, open_serial_jsonl_bridge


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--sensor-port", type=str, default="")
    parser.add_argument("--command-port", type=str, default="")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--max-steps", type=int, default=0)
    return parser.parse_args(argv)


def _build_sensor_bridge(args):
    if args.sensor_port:
        return open_serial_jsonl_bridge(args.sensor_port, baudrate=args.baudrate)
    return JsonlSensorBridge(sys.stdin.buffer)


def _build_action_bridge(args):
    if args.command_port:
        return open_serial_action_bridge(args.command_port, baudrate=args.baudrate)
    return JsonlActionBridge(sys.stdout.buffer)


def main(argv=None):
    args = parse_args(argv)
    cfg = SwarmConfig()
    observation_builder = ObservationBuilder(cfg)
    policy = PolicyRunner(cfg, checkpoint_dir=args.checkpoint_dir, shared_policy=args.shared_policy)
    policy.load(observation_builder.obs_dim())
    command_mapper = CommandMapper(cfg)

    sensor_bridge = _build_sensor_bridge(args)
    action_bridge = _build_action_bridge(args)

    period_s = 1.0 / max(args.hz, 1e-6)
    step = 0

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            t0 = time.perf_counter()
            packet = sensor_bridge.read_sensor_packet()
            if packet.estop:
                command = command_mapper.build_command(action_id=4, mode="stop")
            else:
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
        sensor_bridge.close()
        action_bridge.close()


if __name__ == "__main__":
    main()
