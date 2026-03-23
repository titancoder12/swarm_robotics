from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "checkpoints"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ant.py should be in the same dir.
from ant import ESP32Robot
from models.q_network import QNetwork


@dataclass
class PolicyConfig:
    num_actions: int = 9
    lidar_rays: int = 9
    lidar_max_range_m: float = 2.0
    max_speed_mps: float = 0.5
    pheromone_samples: int = 3
    obs_include_nest_direction: bool = True
    obs_include_food_presence: bool = True
    obs_include_carrying: bool = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a trained policy through the AntSwarmFirmware serial robot interface."
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--shared-policy", action="store_true")
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--scan-duration", type=float, default=0.5)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--lidar-max-range-mm", type=float, default=1600.0)
    parser.add_argument("--max-speed-mps", type=float, default=0.5)
    parser.add_argument("--turn-step-deg", type=int, default=25)
    parser.add_argument("--move-distance-mm", type=int, default=100)
    parser.add_argument("--reverse-distance-mm", type=int, default=60)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def obs_dim(cfg: PolicyConfig) -> int:
    return (
        cfg.lidar_rays
        + 2
        + (2 if cfg.obs_include_nest_direction else 0)
        + 2
        + 2
        + 1
        + (1 if cfg.obs_include_food_presence else 0)
        + (1 if cfg.obs_include_carrying else 0)
        + cfg.pheromone_samples
    )


def bucketize_scan(cfg: PolicyConfig, scan_points: list[dict]) -> list[float]:
    max_range_mm = float(cfg.lidar_max_range_m * 1000.0)
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


def normalize_ranges(cfg: PolicyConfig, ranges_m: list[float]) -> np.ndarray:
    values = list(ranges_m[: cfg.lidar_rays])
    while len(values) < cfg.lidar_rays:
        values.append(cfg.lidar_max_range_m)
    arr = np.array(values, dtype=np.float32)
    arr = np.clip(arr / cfg.lidar_max_range_m, 0.0, 1.0)
    return arr * 2.0 - 1.0


def normalize_xy(cfg: PolicyConfig, vec: list[float]) -> np.ndarray:
    if len(vec) < 2:
        vec = [0.0, 0.0]
    scale = max(cfg.lidar_max_range_m, 1.0)
    arr = np.array(vec[:2], dtype=np.float32) / scale
    return np.clip(arr, -1.0, 1.0)


def normalize_pheromone(cfg: PolicyConfig, values: list[float]) -> np.ndarray:
    samples = list(values[: cfg.pheromone_samples])
    while len(samples) < cfg.pheromone_samples:
        samples.append(0.0)
    arr = np.array(samples, dtype=np.float32)
    if arr.size == 0:
        return arr
    max_abs = float(np.max(np.abs(arr)))
    if max_abs > 1e-6:
        arr = arr / max_abs
    return np.clip(arr, -1.0, 1.0)


def build_observation(cfg: PolicyConfig, scan_points: list[dict]) -> np.ndarray:
    ranges_m = [value / 1000.0 for value in bucketize_scan(cfg, scan_points)]
    lidar = normalize_ranges(cfg, ranges_m)
    target = normalize_xy(cfg, [0.0, 0.0])
    nest = normalize_xy(cfg, [0.0, 0.0])
    neighbor = normalize_xy(cfg, [0.0, 0.0])
    heading = np.array([0.0, 1.0], dtype=np.float32)
    speed = np.array([0.0], dtype=np.float32)
    food_presence = np.array([0.0], dtype=np.float32)
    carrying = np.array([0.0], dtype=np.float32)
    pheromone = normalize_pheromone(cfg, [0.0] * cfg.pheromone_samples)

    parts = [lidar, target]
    if cfg.obs_include_nest_direction:
        parts.append(nest)
    parts.extend([neighbor, heading, speed])
    if cfg.obs_include_food_presence:
        parts.append(food_presence)
    if cfg.obs_include_carrying:
        parts.append(carrying)
    parts.append(pheromone)

    observation = np.concatenate(parts).astype(np.float32)
    if observation.shape[0] != obs_dim(cfg):
        raise ValueError(f"Observation size mismatch: expected {obs_dim(cfg)}, got {observation.shape[0]}.")
    return observation


def load_policy(
    checkpoint_dir: Path,
    shared_policy: bool,
    observation_dim: int,
    num_actions: int,
    debug: bool = False,
) -> QNetwork:
    model = QNetwork(observation_dim, num_actions)
    checkpoint_name = "shared.pt" if shared_policy else "agent_0.pt"

    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir

    checkpoint_path = checkpoint_dir / checkpoint_name

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Repo root: {REPO_ROOT}\n"
            f"Checkpoint dir arg: {checkpoint_dir}"
        )

    if debug:
        print(
            f"[debug] loading checkpoint={checkpoint_path} "
            f"obs_dim={observation_dim} num_actions={num_actions}",
            flush=True,
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_action(model: QNetwork, observation: np.ndarray) -> tuple[int, np.ndarray]:
    with torch.no_grad():
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        q_values = model(obs_tensor)
        action_id = int(torch.argmax(q_values, dim=1).item())
        return action_id, q_values.squeeze(0).cpu().numpy()


def execute_action(
    robot: ESP32Robot,
    action_id: int,
    turn_step_deg: int,
    move_distance_mm: int,
    reverse_distance_mm: int,
) -> tuple[float, float]:
    throttle_vals = (-1.0, 0.0, 1.0)
    turn_vals = (-1.0, 0.0, 1.0)
    action_table = [(throttle, turn) for throttle in throttle_vals for turn in turn_vals]

    if action_id < 0 or action_id >= len(action_table):
        raise ValueError(f"Unsupported action id {action_id}.")

    throttle, turn = action_table[action_id]

    if throttle == 0.0 and turn == 0.0:
        robot.stop()
        return throttle, turn

    if turn != 0.0:
        robot.turn(int(round(turn * turn_step_deg)))

    if throttle > 0.0:
        robot.move(0, move_distance_mm)
        return throttle, turn

    if throttle < 0.0:
        robot.move(180, reverse_distance_mm)
        return throttle, turn

    robot.stop()
    return throttle, turn


def main(argv=None):
    args = parse_args(argv)
    cfg = PolicyConfig(
        lidar_max_range_m=args.lidar_max_range_mm / 1000.0,
        max_speed_mps=args.max_speed_mps,
    )

    robot = ESP32Robot(port=args.port, baudrate=args.baudrate)
    policy = load_policy(
        checkpoint_dir=args.checkpoint_dir,
        shared_policy=args.shared_policy,
        observation_dim=obs_dim(cfg),
        num_actions=cfg.num_actions,
        debug=args.debug,
    )

    period_s = 1.0 / max(args.hz, 1e-6)
    step = 0

    if args.debug:
        print(
            f"[debug] startup port={args.port} baudrate={args.baudrate} "
            f"hz={args.hz} scan_duration={args.scan_duration} "
            f"max_steps={args.max_steps} lidar_max_range_mm={args.lidar_max_range_mm}",
            flush=True,
        )

    robot.connect()

    if args.debug:
        print("[debug] robot connected", flush=True)

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            start = time.perf_counter()
            scan_points = robot.read_sensor_lines(duration=args.scan_duration)
            observation = build_observation(cfg, scan_points)
            action_id, q_values = predict_action(policy, observation)
            throttle, turn = execute_action(
                robot,
                action_id=action_id,
                turn_step_deg=args.turn_step_deg,
                move_distance_mm=args.move_distance_mm,
                reverse_distance_mm=args.reverse_distance_mm,
            )
            step += 1

            elapsed = time.perf_counter() - start
            sleep_time = period_s - elapsed

            if args.debug:
                valid_scan_count = sum(
                    1
                    for item in scan_points
                    if item.get("type") == "scan" and item.get("tof_mm", -1) >= 0
                )
                lidar_preview = np.round(observation[: cfg.lidar_rays], 3).tolist()
                q_values_preview = np.round(q_values, 3).tolist()
                print(
                    f"[debug] step={step} scans={len(scan_points)} valid_scans={valid_scan_count} "
                    f"action={action_id} throttle={throttle:+.1f} turn={turn:+.1f} "
                    f"elapsed={elapsed:.3f}s sleep={max(sleep_time, 0.0):.3f}s",
                    flush=True,
                )
                print(
                    f"[debug] lidar={lidar_preview} q_values={q_values_preview}",
                    flush=True,
                )

            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if args.debug:
            print("[debug] closing robot connection", flush=True)
        robot.close()


if __name__ == "__main__":
    main()
