from __future__ import annotations

import argparse
import json
import math
import random
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
from ant import ESP32Robot, RobotConnectionError
from algorithms.mappo.inference import load_actor
from firmware.bluetooth import (
    DEFAULT_BLE_SERVICE_UUID,
    DEFAULT_BLE_NOTIFY_CHAR_UUID,
    DEFAULT_BLE_WRITE_CHAR_UUID,
    CommandCenterBLEPeripheral,
)
from firmware.camera import CameraDetection, CameraTargetDetector
from firmware.command_center_client import CommandCenterTCPClient
from firmware.relay_client import CommandCenterRelayClient
from models.q_network import QNetwork
from policy_debug import make_policy_debug_config, print_policy_debug, should_debug_policy


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


@dataclass
class LoadedPolicy:
    kind: str
    model: object
    device: torch.device
    hidden_state: torch.Tensor | None = None
    checkpoint_dir: Path | None = None


def pose_to_cm(pose: PoseEstimate) -> tuple[float, float]:
    """Convert the local dead-reckoned pose from mm to Mission Control cm."""
    return pose.x_mm / 10.0, pose.y_mm / 10.0


def project_camera_target_to_world_cm(pose: PoseEstimate, detection: CameraDetection) -> tuple[float, float]:
    distance_cm = max(0.0, float(detection.distance_m) * 100.0)
    absolute_heading_rad = math.radians(pose.heading_deg) + float(detection.angle_rad)
    return (
        pose.x_mm / 10.0 + math.cos(absolute_heading_rad) * distance_cm,
        pose.y_mm / 10.0 + math.sin(absolute_heading_rad) * distance_cm,
    )


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
    parser.add_argument("--cc-tcp-enable", action="store_true")
    parser.add_argument("--cc-tcp-host", type=str, default="")
    parser.add_argument("--cc-tcp-port", type=int, default=8765)
    parser.add_argument("--cc-tcp-timeout", type=float, default=1.0)
    parser.add_argument("--cc-relay-url", type=str, default="")
    parser.add_argument("--cc-relay-session", type=str, default="")
    parser.add_argument("--cc-relay-timeout", type=float, default=1.0)
    parser.add_argument("--cc-ble-enable", action="store_true")
    parser.add_argument(
        "--cc-ble-address",
        type=str,
        default="",
        help="Deprecated in current reversed-role BLE mode; the robot advertises and Mission Control connects.",
    )
    parser.add_argument("--cc-ble-device-name", type=str, default="")
    parser.add_argument("--cc-ble-service-uuid", type=str, default=DEFAULT_BLE_SERVICE_UUID)
    parser.add_argument("--cc-ble-write-char-uuid", type=str, default=DEFAULT_BLE_WRITE_CHAR_UUID)
    parser.add_argument("--cc-ble-notify-char-uuid", type=str, default=DEFAULT_BLE_NOTIFY_CHAR_UUID)
    parser.add_argument("--cc-ble-timeout", type=float, default=0.5)
    parser.add_argument("--cc-deposit-enable", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cc-pheromone-deposit-amount", type=float, default=1.0)
    parser.add_argument("--control-mode", choices=("policy", "heuristic", "hybrid"), default="policy")
    parser.add_argument("--heuristic-random-turn-prob", type=float, default=0.35)
    parser.add_argument("--heuristic-rng-seed", type=int, default=0)
    parser.add_argument("--camera-enable", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-horizontal-fov-deg", type=float, default=62.0)
    parser.add_argument("--camera-target-width-cm", type=float, default=6.0)
    parser.add_argument("--camera-min-area-px", type=int, default=100)
    parser.add_argument("--camera-detector-mode", choices=("apriltag", "hsv"), default="apriltag")
    parser.add_argument("--camera-apriltag-family", type=str, default="DICT_APRILTAG_25h9")
    parser.add_argument("--camera-apriltag-id", type=int, default=0)
    parser.add_argument("--camera-hsv-lower", type=str, default="35,70,70")
    parser.add_argument("--camera-hsv-upper", type=str, default="100,255,255")
    parser.add_argument("--camera-hold-time-s", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-policy", action="store_true")
    parser.add_argument("--debug-policy-agents", type=str, default="")
    parser.add_argument("--debug-policy-max-steps", type=int, default=0)
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


def summarize_lidar_sectors(lidar_ranges_mm: list[float]) -> tuple[float, float, float]:
    if not lidar_ranges_mm:
        return 0.0, 0.0, 0.0

    def sector_min(indices: tuple[int, ...]) -> float:
        values = [
            float(lidar_ranges_mm[index])
            for index in indices
            if 0 <= index < len(lidar_ranges_mm) and float(lidar_ranges_mm[index]) > 0.0
        ]
        return min(values) if values else 0.0

    left_min_mm = sector_min((0, 1, 2))
    front_min_mm = sector_min((3, 4, 5))
    right_min_mm = sector_min((6, 7, 8))
    return front_min_mm, left_min_mm, right_min_mm


def recover_robot_connection(
    robot: ESP32Robot,
    scan_duration: float,
    debug: bool,
    attempts: int = 2,
) -> bool:
    for attempt in range(1, attempts + 1):
        try:
            robot.close()
        except Exception:
            pass
        time.sleep(0.5)
        try:
            robot.connect()
            if debug:
                print(f"[debug] robot reconnect attempt={attempt} connected", flush=True)
            stream_ready = robot.wait_for_stream_ready(timeout=max(3.0, scan_duration + 1.0))
            if debug:
                print(f"[debug] robot reconnect attempt={attempt} stream_ready={stream_ready}", flush=True)
            if stream_ready:
                return True
        except RobotConnectionError as exc:
            if debug:
                print(f"[debug] robot reconnect attempt={attempt} failed: {type(exc).__name__}: {exc}", flush=True)
        except Exception as exc:
            if debug:
                print(f"[debug] robot reconnect attempt={attempt} unexpected failure: {type(exc).__name__}: {exc!r}", flush=True)
    return False


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
    camera_detection: CameraDetection | None = None,
    pheromone_values: list[float] | tuple[float, ...] | None = None,
    speed_mps: float = 0.0,
) -> np.ndarray:
    # The observation layout mirrors the simulator contract, but on hardware we
    # currently synthesize only the channels we can estimate locally or fetch
    # from the command center.
    ranges_m = [value / 1000.0 for value in bucketize_scan(cfg, scan_points)]
    lidar = normalize_ranges(cfg, ranges_m)
    if camera_detection is not None and camera_detection.found:
        target = np.array(
            [
                np.clip(camera_detection.distance_m / max(cfg.lidar_max_range_m, 1e-6), 0.0, 1.0),
                np.clip(camera_detection.angle_rad / math.pi, -1.0, 1.0),
            ],
            dtype=np.float32,
        )
        food_presence = np.array([1.0], dtype=np.float32)
    else:
        target = np.zeros(2, dtype=np.float32)
        food_presence = np.array([0.0], dtype=np.float32)
    # Starting at the nest means the nest-relative vector is just the negative
    # of our current dead-reckoned pose.
    nest = normalize_xy(cfg, [-(pose.x_mm / 1000.0), -(pose.y_mm / 1000.0)])
    neighbor = normalize_xy(cfg, [0.0, 0.0])
    # The simulator observation encodes heading as sin/cos rather than a raw
    # angle to avoid discontinuities around 360 -> 0 wrap-around.
    heading_rad = math.radians(pose.heading_deg)
    heading = np.array([math.sin(heading_rad), math.cos(heading_rad)], dtype=np.float32)
    speed = np.array([np.clip(speed_mps / max(cfg.max_speed_mps, 1e-6), -1.0, 1.0)], dtype=np.float32)
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


def resolve_checkpoint_dir(checkpoint_dir: Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = REPO_ROOT / checkpoint_dir
    # The newer robot runtime defaults live under checkpoints/mappo_g/latest.
    # If the caller points at the top-level checkpoints/ folder, prefer that
    # canonical latest MAPPO checkpoint over the legacy DQN layout.
    mappo_latest_dir = checkpoint_dir / "mappo_g" / "latest"
    if (checkpoint_dir / "actor.pt").exists() or (checkpoint_dir / "metadata.json").exists():
        return checkpoint_dir
    if (checkpoint_dir / "shared.pt").exists() or (checkpoint_dir / "agent_0.pt").exists():
        return checkpoint_dir
    if mappo_latest_dir.is_dir() and (mappo_latest_dir / "actor.pt").exists():
        return mappo_latest_dir
    return checkpoint_dir


def detect_checkpoint_format(checkpoint_dir: Path) -> str:
    metadata_path = checkpoint_dir / "metadata.json"
    actor_path = checkpoint_dir / "actor.pt"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception:
            metadata = {}
        if str(metadata.get("algorithm", "")).lower() == "recurrent_mappo_gru":
            return "mappo_gru"
    if actor_path.exists():
        return "mappo_gru"
    if (checkpoint_dir / "shared.pt").exists() or (checkpoint_dir / "agent_0.pt").exists():
        return "dqn"
    raise FileNotFoundError(
        f"Unsupported checkpoint directory: {checkpoint_dir}\n"
        "Expected either a MAPPO actor checkpoint set (actor.pt / metadata.json) "
        "or legacy DQN checkpoint files (shared.pt / agent_0.pt)."
    )


def load_policy(
    checkpoint_dir: Path,
    shared_policy: bool,
    observation_dim: int,
    num_actions: int,
    debug: bool = False,
) -> LoadedPolicy:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_dir)
    checkpoint_format = detect_checkpoint_format(checkpoint_dir)

    if checkpoint_format == "mappo_gru":
        metadata_path = checkpoint_dir / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        checkpoint_obs_dim = int(metadata.get("obs_dim", observation_dim))
        checkpoint_action_dim = int(metadata.get("action_dim", num_actions))
        if checkpoint_obs_dim != observation_dim:
            raise ValueError(
                f"MAPPO checkpoint obs_dim mismatch: metadata says {checkpoint_obs_dim}, "
                f"but firmware builds {observation_dim}."
            )
        if checkpoint_action_dim != num_actions:
            raise ValueError(
                f"MAPPO checkpoint action_dim mismatch: metadata says {checkpoint_action_dim}, "
                f"but firmware expects {num_actions}."
            )
        if debug:
            print(
                f"[debug] loading MAPPO actor checkpoint_dir={checkpoint_dir} "
                f"obs_dim={observation_dim} num_actions={num_actions}",
                flush=True,
            )
        actor, device = load_actor(str(checkpoint_dir), observation_dim, num_actions, device="cpu")
        hidden_state = actor.initial_hidden(1, device)
        return LoadedPolicy(
            kind="mappo_gru",
            model=actor,
            device=device,
            hidden_state=hidden_state,
            checkpoint_dir=checkpoint_dir,
        )

    # Legacy DQN fallback for older hardware checkpoints.
    model = QNetwork(observation_dim, num_actions)
    checkpoint_name = "shared.pt" if shared_policy else "agent_0.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Repo root: {REPO_ROOT}\n"
            f"Checkpoint dir arg: {checkpoint_dir}"
        )
    if debug:
        print(
            f"[debug] loading DQN checkpoint={checkpoint_path} "
            f"obs_dim={observation_dim} num_actions={num_actions}",
            flush=True,
        )
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return LoadedPolicy(kind="dqn", model=model, device=torch.device("cpu"), checkpoint_dir=checkpoint_dir)


def predict_action(policy: LoadedPolicy, observation: np.ndarray) -> tuple[int, np.ndarray]:
    # Inference is greedy argmax over policy outputs. For DQN these outputs are
    # Q-values; for recurrent MAPPO they are action logits plus recurrent state.
    obs_tensor = torch.tensor(observation, dtype=torch.float32, device=policy.device).unsqueeze(0)
    with torch.no_grad():
        if policy.kind == "mappo_gru":
            if policy.hidden_state is None:
                raise RuntimeError("MAPPO policy hidden state is uninitialized.")
            done_mask = torch.ones((1,), dtype=torch.float32, device=policy.device)
            logits, next_hidden = policy.model(obs_tensor, policy.hidden_state, done_mask)
            policy.hidden_state = next_hidden
            action_id = int(torch.argmax(logits, dim=1).item())
            return action_id, logits.squeeze(0).cpu().numpy()

        q_values = policy.model(obs_tensor)
        action_id = int(torch.argmax(q_values, dim=1).item())
        return action_id, q_values.squeeze(0).cpu().numpy()


def action_id_from_controls(throttle: float, turn: float, deposit: bool) -> int:
    throttle_vals = (-1.0, 0.0, 1.0)
    turn_vals = (-1.0, 0.0, 1.0)
    action_table = [
        (candidate_throttle, candidate_turn, bool(candidate_deposit))
        for candidate_throttle in throttle_vals
        for candidate_turn in turn_vals
        for candidate_deposit in (0, 1)
    ]
    target = (float(throttle), float(turn), bool(deposit))
    for index, candidate in enumerate(action_table):
        if candidate == target:
            return index
    raise ValueError(f"Unsupported control triple {target}.")


def choose_heuristic_action(
    cfg: PolicyConfig,
    lidar_ranges_mm: list[float],
    camera_detection: CameraDetection | None,
    deposit_default: bool = True,
    rng: random.Random | None = None,
    random_turn_prob: float = 0.35,
) -> tuple[int, np.ndarray]:
    front_indices = (3, 4, 5)
    left_indices = (0, 1, 2)
    right_indices = (6, 7, 8)
    front_min = min(lidar_ranges_mm[index] for index in front_indices)
    left_min = min(lidar_ranges_mm[index] for index in left_indices)
    right_min = min(lidar_ranges_mm[index] for index in right_indices)

    obstacle_emergency_mm = 90.0
    obstacle_close_mm = 260.0
    obstacle_caution_mm = 520.0
    obstacle_clear_mm = 800.0
    forward_wiggle_prob = 0.40
    camera_angle_deg = math.degrees(camera_detection.angle_rad) if camera_detection is not None else 0.0
    camera_found = bool(camera_detection and camera_detection.found)
    strong_camera_lock = bool(camera_found and has_strong_camera_lock(camera_detection))
    random_turn_prob = float(np.clip(random_turn_prob, 0.0, 1.0))

    def choose_turn_sign() -> float:
        return 1.0

    throttle = 0.0
    turn = 0.0

    if front_min <= obstacle_emergency_mm:
        throttle = -1.0
        turn = 0.0
    elif strong_camera_lock:
        throttle = 1.0 if front_min >= obstacle_close_mm else 0.0
        turn = 0.0
    else:
        if front_min >= obstacle_caution_mm:
            throttle = 1.0
            turn = 0.0
        elif front_min >= obstacle_close_mm:
            throttle = 1.0
            turn = 0.0
        else:
            throttle = 1.0
            turn = 0.0

    action_id = action_id_from_controls(throttle, turn, deposit_default)
    score_vector = np.full((cfg.num_actions,), -1.0, dtype=np.float32)
    score_vector[action_id] = 1.0
    return action_id, score_vector


def should_use_heuristic_override(
    lidar_ranges_mm: list[float],
    camera_detection: CameraDetection | None,
) -> bool:
    # In hybrid mode, prefer the forward-biased heuristic over the learned
    # policy. The policy still remains available via --policy-control.
    return True


def has_strong_camera_lock(camera_detection: CameraDetection | None) -> bool:
    if camera_detection is None or not camera_detection.found:
        return False
    if camera_detection.confidence < 0.003:
        return False
    if camera_detection.bbox_w < 24 or camera_detection.bbox_h < 24:
        return False
    center_x = camera_detection.bbox_x + (camera_detection.bbox_w * 0.5)
    if center_x < 64.0 or center_x > (640.0 - 64.0):
        return False
    return True


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
    camera_detector = None
    if args.camera_enable:
        try:
            lower = tuple(int(part.strip()) for part in args.camera_hsv_lower.split(","))
            upper = tuple(int(part.strip()) for part in args.camera_hsv_upper.split(","))
            if len(lower) != 3 or len(upper) != 3:
                raise ValueError("camera HSV bounds must each contain exactly 3 comma-separated integers")
            camera_detector = CameraTargetDetector(
                camera_index=args.camera_index,
                width=args.camera_width,
                height=args.camera_height,
                horizontal_fov_deg=args.camera_horizontal_fov_deg,
                target_width_cm=args.camera_target_width_cm,
                min_area_px=args.camera_min_area_px,
                detector_mode=args.camera_detector_mode,
                apriltag_family=args.camera_apriltag_family,
                apriltag_id=args.camera_apriltag_id,
                hsv_lower=lower,
                hsv_upper=upper,
                hold_time_s=args.camera_hold_time_s,
                debug=args.debug,
            )
        except Exception as exc:
            if args.debug:
                print(f"[debug] camera detector initialization failed: {type(exc).__name__}: {exc!r}", flush=True)
            camera_detector = None
    if args.cc_ble_enable and args.cc_ble_address and args.debug:
        print(
            "[debug] ignoring --cc-ble-address; in current BLE mode the robot is the peripheral "
            "and Mission Control connects as the client",
            flush=True,
        )
    mission_control_link = None
    if args.cc_relay_url:
        mission_control_link = CommandCenterRelayClient(
            relay_url=args.cc_relay_url,
            session=args.cc_relay_session or args.robot_id,
            timeout_s=args.cc_relay_timeout,
            debug=args.debug,
        )
    elif args.cc_tcp_enable or args.cc_tcp_host:
        mission_control_link = CommandCenterTCPClient(
            host=args.cc_tcp_host or "127.0.0.1",
            port=args.cc_tcp_port,
            timeout_s=args.cc_tcp_timeout,
            debug=args.debug,
        )
    elif args.cc_ble_enable:
        # In reversed-role BLE mode the robot advertises the UART-like
        # peripheral and Mission Control connects as the client/central.
        mission_control_link = CommandCenterBLEPeripheral(
            device_name=args.cc_ble_device_name or args.robot_id,
            service_uuid=args.cc_ble_service_uuid,
            write_char_uuid=args.cc_ble_write_char_uuid,
            notify_char_uuid=args.cc_ble_notify_char_uuid,
            timeout_s=args.cc_ble_timeout,
            debug=args.debug,
        )

    robot = ESP32Robot(port=args.port, baudrate=args.baudrate, debug=args.debug)
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
    debug_policy_cfg = make_policy_debug_config(args.debug_policy, args.debug_policy_agents, args.debug_policy_max_steps)
    heuristic_rng = random.Random(args.heuristic_rng_seed if args.heuristic_rng_seed != 0 else time.time_ns())

    if args.debug:
        print(
            f"[debug] startup port={args.port} baudrate={args.baudrate} "
            f"hz={args.hz} scan_duration={args.scan_duration} "
            f"max_steps={args.max_steps} lidar_max_range_mm={args.lidar_max_range_mm} "
            f"initial_heading_deg={args.initial_heading_deg} robot_id={args.robot_id} "
            f"cc_relay_url={args.cc_relay_url or '-'} "
            f"cc_relay_session={args.cc_relay_session or args.robot_id} "
            f"cc_tcp_enable={args.cc_tcp_enable or bool(args.cc_tcp_host)} "
            f"cc_tcp_host={args.cc_tcp_host or '-'} cc_tcp_port={args.cc_tcp_port} "
            f"cc_ble_enable={args.cc_ble_enable} pheromone_awareness_radius_cm={awareness_radius_cm:.1f} "
            f"cc_deposit_enable={args.cc_deposit_enable}",
            flush=True,
        )

    robot.connect()

    if args.debug:
        print("[debug] robot connected", flush=True)

    stream_ready = robot.wait_for_stream_ready(timeout=max(3.0, args.scan_duration + 1.0))
    if args.debug:
        print(f"[debug] robot stream_ready={stream_ready}", flush=True)
    if not stream_ready:
        if args.debug:
            print("[debug] robot stream not ready after first connect; retrying serial startup once", flush=True)
        stream_ready = recover_robot_connection(robot, args.scan_duration, args.debug, attempts=1)
        if args.debug:
            print(f"[debug] robot stream_ready_after_retry={stream_ready}", flush=True)

    if camera_detector is not None:
        camera_ready = camera_detector.warmup(timeout_s=4.0)
        if args.debug:
            print(f"[debug] camera_ready={camera_ready}", flush=True)

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
            try:
                scan_points = robot.read_sensor_lines(duration=args.scan_duration)
                lidar_ranges_mm = bucketize_scan(cfg, scan_points)
                camera_detection = camera_detector.detect() if camera_detector is not None else None
                pheromone_values = (0.0, 0.0, 0.0)
                if mission_control_link is not None:
                    # Publish the latest dead-reckoned pose before the query so
                    # Mission Control samples pheromone against the same nest-relative
                    # displacement estimate the robot uses locally.
                    x_cm, y_cm = pose_to_cm(pose)
                    mission_control_link.send_position(args.robot_id, x_cm, y_cm, pose.heading_deg)
                    mission_control_link.send_lidar(args.robot_id, lidar_ranges_mm)
                    if camera_detection is not None and camera_detection.found:
                        target_x_cm, target_y_cm = project_camera_target_to_world_cm(pose, camera_detection)
                        mission_control_link.send_target(args.robot_id, target_x_cm, target_y_cm, camera_detection.confidence)
                    # Mission Control is the source of truth for the digital pheromone
                    # field, so the runtime pulls the latest 3-sample slice right
                    # before inference.
                    pheromone_values = mission_control_link.sense_pheromone(args.robot_id, x_cm, y_cm, pose.heading_deg)

                current_frame = build_single_observation(
                    cfg,
                    scan_points,
                    pose=pose,
                    camera_detection=camera_detection,
                    pheromone_values=pheromone_values,
                    speed_mps=speed_mps,
                )
                obs_history.append(current_frame)
                obs_history = obs_history[-cfg.observation_history_steps :]
                observation = build_observation_history(cfg, obs_history)
                if args.control_mode == "heuristic":
                    action_id, q_values = choose_heuristic_action(
                        cfg,
                        lidar_ranges_mm=lidar_ranges_mm,
                        camera_detection=camera_detection,
                        deposit_default=args.cc_deposit_enable,
                        rng=heuristic_rng,
                        random_turn_prob=args.heuristic_random_turn_prob,
                    )
                else:
                    policy_action_id, policy_q_values = predict_action(policy, observation)
                    if args.control_mode == "hybrid" and should_use_heuristic_override(
                        lidar_ranges_mm=lidar_ranges_mm,
                        camera_detection=camera_detection,
                    ):
                        action_id, q_values = choose_heuristic_action(
                            cfg,
                            lidar_ranges_mm=lidar_ranges_mm,
                            camera_detection=camera_detection,
                            deposit_default=args.cc_deposit_enable,
                            rng=heuristic_rng,
                            random_turn_prob=args.heuristic_random_turn_prob,
                        )
                    else:
                        action_id, q_values = policy_action_id, policy_q_values
                if args.control_mode in ("policy", "hybrid") and should_debug_policy(debug_policy_cfg, step, 0, args.robot_id):
                    print_policy_debug(
                        step=step,
                        agent_index=0,
                        agent_id=args.robot_id,
                        policy_label="mappo_gru" if policy.kind == "mappo_gru" else ("shared" if args.shared_policy else "agent_0"),
                        mode="greedy",
                        output_name="policy_logits" if policy.kind == "mappo_gru" else "q_values",
                        output_values=q_values,
                        action=action_id,
                        num_actions=cfg.num_actions,
                    )
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
                    if hasattr(mission_control_link, "send_status"):
                        front_min_mm, left_min_mm, right_min_mm = summarize_lidar_sectors(lidar_ranges_mm)
                        camera_found = bool(camera_detection and camera_detection.found)
                        camera_distance_m = camera_detection.distance_m if camera_detection is not None else 0.0
                        camera_angle_deg = math.degrees(camera_detection.angle_rad) if camera_detection is not None else 0.0
                        mission_control_link.send_status(
                            robot_id=args.robot_id,
                            control_mode=args.control_mode,
                            action_id=action_id,
                            throttle=throttle,
                            turn=turn,
                            deposit=deposit,
                            camera_found=camera_found,
                            camera_distance_m=camera_distance_m,
                            camera_angle_deg=camera_angle_deg,
                            front_min_mm=front_min_mm,
                            left_min_mm=left_min_mm,
                            right_min_mm=right_min_mm,
                            serial_ok=robot.last_response_ok,
                            serial_cmd=robot.last_command,
                            serial_reply=robot.last_response_raw,
                        )
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
            except RobotConnectionError as exc:
                if args.debug:
                    print(f"[debug] serial connection error during control step: {exc}", flush=True)
                recovered = recover_robot_connection(robot, args.scan_duration, args.debug, attempts=2)
                if not recovered:
                    raise
                obs_history.clear()
                speed_mps = 0.0
                continue

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
                    f"control_mode={args.control_mode} "
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
                if camera_detector is not None:
                    found = bool(camera_detection and camera_detection.found)
                    print(
                        f"[debug] camera found={found} "
                        f"distance_m={(camera_detection.distance_m if camera_detection else 0.0):.3f} "
                        f"angle_deg={(math.degrees(camera_detection.angle_rad) if camera_detection else 0.0):.1f}",
                        flush=True,
                    )

            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if args.debug:
            print("[debug] closing robot connection", flush=True)
        if camera_detector is not None:
            camera_detector.close()
        if mission_control_link is not None:
            mission_control_link.close()
        robot.close()


if __name__ == "__main__":
    main()
