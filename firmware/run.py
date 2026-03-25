from __future__ import annotations

import argparse
import math
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
from firmware.bluetooth import (
    DEFAULT_BLE_NOTIFY_CHAR_UUID,
    DEFAULT_BLE_WRITE_CHAR_UUID,
    CommandCenterBLEClient,
)
from models.q_network import QNetwork


@dataclass
class PolicyConfig:
    # Keep these aligned with the observation contract the policy was trained
    # on. If the simulator-side observation layout changes, the robot runtime
    # needs the same shape and ordering or checkpoint inference will drift.
    num_actions: int = 18
    lidar_rays: int = 9
    lidar_max_range_m: float = 2.0
    max_speed_mps: float = 0.5
    agent_radius_cm: float = 7.0
    pheromone_samples: int = 3
    pheromone_sample_spacing_scale: float = 1.5
    obs_include_nest_direction: bool = True
    obs_include_food_presence: bool = True
    obs_include_carrying: bool = True
    observation_history_steps: int = 3


@dataclass
class PoseEstimate:
    # This is only a dead-reckoned local estimate derived from the commands we
    # send. It is useful for nest direction and Mission Control pheromone queries,
    # but it is not a fused localization solution.
    x_mm: float = 0.0
    y_mm: float = 0.0
    heading_deg: float = 0.0


def pose_to_cm(pose: PoseEstimate) -> tuple[float, float]:
    """Convert the local dead-reckoned pose from mm to Mission Control cm."""
    return pose.x_mm / 10.0, pose.y_mm / 10.0


def parse_args(argv=None):
    # Keep runtime, robot-link, and command-center transport flags together so
    # the deployment surface is visible from one place.
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
    parser.add_argument("--initial-heading-deg", type=float, default=0.0)
    parser.add_argument("--robot-id", type=str, default="robot_0")
    parser.add_argument("--agent-radius-cm", type=float, default=7.0)
    parser.add_argument("--cc-ble-enable", action="store_true")
    parser.add_argument("--cc-ble-address", type=str, default="")
    parser.add_argument("--cc-ble-device-name", type=str, default="")
    parser.add_argument("--cc-ble-write-char-uuid", type=str, default=DEFAULT_BLE_WRITE_CHAR_UUID)
    parser.add_argument("--cc-ble-notify-char-uuid", type=str, default=DEFAULT_BLE_NOTIFY_CHAR_UUID)
    parser.add_argument("--cc-ble-timeout", type=float, default=0.5)
    parser.add_argument("--cc-deposit-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cc-pheromone-deposit-amount", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args(argv)


def obs_dim(cfg: PolicyConfig) -> int:
    return single_obs_dim(cfg) * cfg.observation_history_steps


def single_obs_dim(cfg: PolicyConfig) -> int:
    # Mirror the simulator's feature count exactly. The trained MLP expects a
    # fixed-length vector; a mismatch here means the checkpoint is unusable.
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
    # The ESP32 scan stream is an unordered list of angle/distance samples.
    # Collapse it into the fixed 9-ray front-arc representation the policy was
    # trained on by taking the minimum reading that lands in each bucket.
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
    # Training used lidar normalized to [-1, 1], not [0, 1]. Far readings land
    # near +1 while close obstacles land near -1.
    values = list(ranges_m[: cfg.lidar_rays])
    while len(values) < cfg.lidar_rays:
        values.append(cfg.lidar_max_range_m)
    arr = np.array(values, dtype=np.float32)
    arr = np.clip(arr / cfg.lidar_max_range_m, 0.0, 1.0)
    return arr * 2.0 - 1.0


def normalize_xy(cfg: PolicyConfig, vec: list[float]) -> np.ndarray:
    # Relative vectors are expressed as fractions of the configured sensing
    # scale so simulator and robot inference stay on a comparable numeric range.
    if len(vec) < 2:
        vec = [0.0, 0.0]
    scale = max(cfg.lidar_max_range_m, 1.0)
    arr = np.array(vec[:2], dtype=np.float32) / scale
    return np.clip(arr, -1.0, 1.0)


def normalize_pheromone(cfg: PolicyConfig, values: list[float]) -> np.ndarray:
    # Mission Control already returns three local pheromone samples, but we still
    # normalize defensively so malformed or out-of-range values do not blow up
    # the observation.
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


def pheromone_awareness_radius_cm(cfg: PolicyConfig) -> float:
    # This matches the simulator's forward sampling geometry:
    # sample_i distance = (i + 1) * agent_radius * 1.5.
    return float(cfg.pheromone_samples) * float(cfg.agent_radius_cm) * float(cfg.pheromone_sample_spacing_scale)


def build_observation(
    cfg: PolicyConfig,
    scan_points: list[dict],
    pose: PoseEstimate,
    pheromone_values: list[float] | tuple[float, ...] | None = None,
    speed_mps: float = 0.0,
) -> np.ndarray:
    return build_observation_history(
        cfg,
        [build_single_observation(cfg, scan_points, pose, pheromone_values=pheromone_values, speed_mps=speed_mps)],
    )


def build_single_observation(
    cfg: PolicyConfig,
    scan_points: list[dict],
    pose: PoseEstimate,
    pheromone_values: list[float] | tuple[float, ...] | None = None,
    speed_mps: float = 0.0,
) -> np.ndarray:
    # The observation layout mirrors the simulator contract, but on hardware we
    # currently synthesize only the channels we can estimate locally or fetch
    # from the command center.
    ranges_m = [value / 1000.0 for value in bucketize_scan(cfg, scan_points)]
    lidar = normalize_ranges(cfg, ranges_m)
    # These channels are still placeholders in the current robot runtime
    # because the Pi does not yet estimate target and neighbor state.
    target = normalize_xy(cfg, [0.0, 0.0])
    # Starting at the nest means the nest-relative vector is just the negative
    # of our current dead-reckoned pose.
    nest = normalize_xy(cfg, [-(pose.x_mm / 1000.0), -(pose.y_mm / 1000.0)])
    neighbor = normalize_xy(cfg, [0.0, 0.0])
    # The simulator observation encodes heading as sin/cos rather than a raw
    # angle to avoid discontinuities around 360 -> 0 wrap-around.
    heading_rad = math.radians(pose.heading_deg)
    heading = np.array([math.sin(heading_rad), math.cos(heading_rad)], dtype=np.float32)
    speed = np.array([np.clip(speed_mps / max(cfg.max_speed_mps, 1e-6), -1.0, 1.0)], dtype=np.float32)
    food_presence = np.array([0.0], dtype=np.float32)
    carrying = np.array([0.0], dtype=np.float32)
    pheromone = normalize_pheromone(cfg, list(pheromone_values or ([0.0] * cfg.pheromone_samples)))

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
    if observation.shape[0] != single_obs_dim(cfg):
        raise ValueError(f"Observation size mismatch: expected {single_obs_dim(cfg)}, got {observation.shape[0]}.")
    return observation


def build_observation_history(cfg: PolicyConfig, frames: list[np.ndarray]) -> np.ndarray:
    """Flatten a short observation history into the model input vector."""
    if not frames:
        raise ValueError("At least one observation frame is required.")
    normalized_frames = [np.asarray(frame, dtype=np.float32) for frame in frames[-cfg.observation_history_steps :]]
    while len(normalized_frames) < cfg.observation_history_steps:
        normalized_frames.insert(0, normalized_frames[0].copy())
    observation = np.concatenate(normalized_frames, axis=0).astype(np.float32)
    if observation.shape[0] != obs_dim(cfg):
        raise ValueError(f"Observation history size mismatch: expected {obs_dim(cfg)}, got {observation.shape[0]}.")
    return observation


def load_policy(
    checkpoint_dir: Path,
    shared_policy: bool,
    observation_dim: int,
    num_actions: int,
    debug: bool = False,
) -> QNetwork:
    # The hardware runtime only supports the custom PyTorch checkpoint format
    # used by the repo's QNetwork definition.
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
    # Inference is greedy argmax over Q-values; there is no exploration term in
    # this runtime path.
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
) -> tuple[float, float, bool]:
    # Rebuild the same discrete action table used in the simulator so action id
    # -> (throttle, turn) semantics stay stable between training and deployment.
    throttle_vals = (-1.0, 0.0, 1.0)
    turn_vals = (-1.0, 0.0, 1.0)
    action_table = [
        (throttle, turn, bool(deposit))
        for throttle in throttle_vals
        for turn in turn_vals
        for deposit in (0, 1)
    ]

    if action_id < 0 or action_id >= len(action_table):
        raise ValueError(f"Unsupported action id {action_id}.")

    throttle, turn, deposit = action_table[action_id]

    if throttle == 0.0 and turn == 0.0:
        robot.stop()
        return throttle, turn, deposit

    if turn != 0.0:
        robot.turn(int(round(turn * turn_step_deg)))

    if throttle > 0.0:
        robot.move(0, move_distance_mm)
        return throttle, turn, deposit

    if throttle < 0.0:
        robot.move(180, reverse_distance_mm)
        return throttle, turn, deposit

    robot.stop()
    return throttle, turn, deposit


def update_pose_estimate(
    pose: PoseEstimate,
    throttle: float,
    turn: float,
    turn_step_deg: int,
    move_distance_mm: int,
    reverse_distance_mm: int,
) -> tuple[PoseEstimate, float]:
    # Match execute_action(...): discrete turn happens before translation, so
    # the local pose estimate uses the post-turn heading for displacement.
    next_heading_deg = pose.heading_deg + float(turn) * float(turn_step_deg)

    if throttle > 0.0:
        distance_mm = float(move_distance_mm)
    elif throttle < 0.0:
        distance_mm = -float(reverse_distance_mm)
    else:
        distance_mm = 0.0

    heading_rad = math.radians(next_heading_deg)
    next_x_mm = pose.x_mm + distance_mm * math.cos(heading_rad)
    next_y_mm = pose.y_mm + distance_mm * math.sin(heading_rad)
    return PoseEstimate(x_mm=next_x_mm, y_mm=next_y_mm, heading_deg=next_heading_deg), distance_mm


def main(argv=None):
    args = parse_args(argv)
    # Runtime config is deliberately small and local: enough to build the
    # observation vector and interpret movement commands, without importing the
    # whole simulator environment stack.
    cfg = PolicyConfig(
        lidar_max_range_m=args.lidar_max_range_mm / 1000.0,
        max_speed_mps=args.max_speed_mps,
        agent_radius_cm=args.agent_radius_cm,
    )
    awareness_radius_cm = pheromone_awareness_radius_cm(cfg)
    mission_control_link = None
    if args.cc_ble_enable:
        # The BLE helper owns the line-oriented POS / SENSE / PHER exchange
        # with desktop Mission Control.
        mission_control_link = CommandCenterBLEClient(
            address=args.cc_ble_address,
            device_name=args.cc_ble_device_name,
            write_char_uuid=args.cc_ble_write_char_uuid,
            notify_char_uuid=args.cc_ble_notify_char_uuid,
            timeout_s=args.cc_ble_timeout,
            debug=args.debug,
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
    pose = PoseEstimate(heading_deg=args.initial_heading_deg)
    # Speed is reconstructed from the commanded step size and loop frequency,
    # not measured from wheel odometry.
    speed_mps = 0.0
    obs_history: list[np.ndarray] = []

    if args.debug:
        print(
            f"[debug] startup port={args.port} baudrate={args.baudrate} "
            f"hz={args.hz} scan_duration={args.scan_duration} "
            f"max_steps={args.max_steps} lidar_max_range_mm={args.lidar_max_range_mm} "
            f"initial_heading_deg={args.initial_heading_deg} robot_id={args.robot_id} "
            f"cc_ble_enable={args.cc_ble_enable} pheromone_awareness_radius_cm={awareness_radius_cm:.1f} "
            f"cc_deposit_enable={args.cc_deposit_enable}",
            flush=True,
        )

    robot.connect()

    if args.debug:
        print("[debug] robot connected", flush=True)

    if mission_control_link is not None:
        # Publish the origin/nest pose immediately so Mission Control has a
        # consistent starting point before the first SENSE request arrives.
        x_cm, y_cm = pose_to_cm(pose)
        mission_control_link.send_position(args.robot_id, x_cm, y_cm, pose.heading_deg)

    try:
        while args.max_steps <= 0 or step < args.max_steps:
            # One control iteration:
            # 1. read local scan data
            # 2. optionally query remote pheromone state from Mission Control
            # 3. assemble the observation vector
            # 4. run greedy Q inference
            # 5. execute the selected movement
            # 6. update local dead-reckoned state and optional pheromone deposit
            start = time.perf_counter()
            scan_points = robot.read_sensor_lines(duration=args.scan_duration)
            lidar_ranges_mm = bucketize_scan(cfg, scan_points)
            pheromone_values = (0.0, 0.0, 0.0)
            if mission_control_link is not None:
                # Publish the latest dead-reckoned pose before the query so
                # Mission Control samples pheromone against the same nest-relative
                # displacement estimate the robot uses locally.
                x_cm, y_cm = pose_to_cm(pose)
                mission_control_link.send_position(args.robot_id, x_cm, y_cm, pose.heading_deg)
                mission_control_link.send_lidar(args.robot_id, lidar_ranges_mm)
                # Mission Control is the source of truth for the digital pheromone
                # field, so the runtime pulls the latest 3-sample slice right
                # before inference.
                pheromone_values = mission_control_link.sense_pheromone(args.robot_id, x_cm, y_cm, pose.heading_deg)

            current_frame = build_single_observation(
                cfg,
                scan_points,
                pose=pose,
                pheromone_values=pheromone_values,
                speed_mps=speed_mps,
            )
            obs_history.append(current_frame)
            obs_history = obs_history[-cfg.observation_history_steps :]
            observation = build_observation_history(cfg, obs_history)
            action_id, q_values = predict_action(policy, observation)
            throttle, turn, deposit = execute_action(
                robot,
                action_id=action_id,
                turn_step_deg=args.turn_step_deg,
                move_distance_mm=args.move_distance_mm,
                reverse_distance_mm=args.reverse_distance_mm,
            )
            pose, commanded_distance_mm = update_pose_estimate(
                pose,
                throttle=throttle,
                turn=turn,
                turn_step_deg=args.turn_step_deg,
                move_distance_mm=args.move_distance_mm,
                reverse_distance_mm=args.reverse_distance_mm,
            )
            speed_mps = commanded_distance_mm / 1000.0 / period_s
            if mission_control_link is not None:
                # Send the post-action pose as soon as the dead-reckoned
                # displacement update is applied so Mission Control tracks the
                # robot's current position rather than only the previous step.
                x_cm, y_cm = pose_to_cm(pose)
                mission_control_link.send_position(args.robot_id, x_cm, y_cm, pose.heading_deg)
            if (
                mission_control_link is not None
                and args.cc_deposit_enable
                and deposit
            ):
                # Digital pheromone placement is now policy-driven. The action
                # id carries a deposit bit, so the robot only emits PHER when
                # the model explicitly selects a depositing action.
                mission_control_link.deposit_pheromone(
                    args.robot_id,
                    pose.x_mm / 10.0,
                    pose.y_mm / 10.0,
                    args.cc_pheromone_deposit_amount,
                )
            step += 1

            elapsed = time.perf_counter() - start
            sleep_time = period_s - elapsed

            if args.debug:
                # The debug stream is intentionally high-signal: enough to see
                # pose, timing, scan health, pheromone input, and network output
                # without dumping the full scan payload every loop.
                valid_scan_count = sum(
                    1
                    for item in scan_points
                    if item.get("type") == "scan" and item.get("tof_mm", -1) >= 0
                )
                lidar_preview = np.round(observation[: cfg.lidar_rays], 3).tolist()
                q_values_preview = np.round(q_values, 3).tolist()
                print(
                    f"[debug] step={step} scans={len(scan_points)} valid_scans={valid_scan_count} "
                    f"action={action_id} throttle={throttle:+.1f} turn={turn:+.1f} deposit={int(deposit)} "
                    f"heading_deg={pose.heading_deg:.1f} commanded_distance_mm={commanded_distance_mm:.1f} "
                    f"x_mm={pose.x_mm:.1f} y_mm={pose.y_mm:.1f} "
                    f"elapsed={elapsed:.3f}s sleep={max(sleep_time, 0.0):.3f}s",
                    flush=True,
                )
                print(
                    f"[debug] lidar={lidar_preview} pheromone={list(np.round(pheromone_values, 3))} q_values={q_values_preview}",
                    flush=True,
                )

            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if args.debug:
            print("[debug] closing robot connection", flush=True)
        if mission_control_link is not None:
            mission_control_link.close()
        robot.close()


if __name__ == "__main__":
    main()
