"""
Simple policy probing script for the custom DQN checkpoints in this repo.

Usage examples:

    python train/policy_probe.py --checkpoint-dir checkpoints --shared-policy --case target_ahead
    python train/policy_probe.py --checkpoint-dir checkpoints --case wall_ahead --agent-index 0
    python train/policy_probe.py --checkpoint-dir checkpoints --list-cases

This is intentionally a small playground script:
- it only supports the custom PyTorch checkpoints used by this repo
- it uses the current 23-dimensional observation layout
- it prints observations and Q-values in a beginner-friendly way
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.q_network import QNetwork


# This script is meant to be easy to edit.
# If you want a different default test case, change this line.
DEFAULT_CASE = "target_ahead"


# The observation order below matches the current implementation in env/swarm_env.py:
# _get_obs() builds:
#   lidar
#   target vector
#   nest vector
#   nearest-neighbor vector
#   heading sin/cos
#   speed
#   food presence
#   carrying food
#   pheromone samples
OBS_NAMES = [
    "lidar_0",
    "lidar_1",
    "lidar_2",
    "lidar_3",
    "lidar_4",
    "lidar_5",
    "lidar_6",
    "lidar_7",
    "lidar_8",
    "target_dx_body_norm",
    "target_dy_body_norm",
    "nest_dx_body_norm",
    "nest_dy_body_norm",
    "neighbor_dx_body_norm",
    "neighbor_dy_body_norm",
    "heading_sin",
    "heading_cos",
    "speed_norm",
    "food_presence",
    "carrying_food",
    "pheromone_sample_0",
    "pheromone_sample_1",
    "pheromone_sample_2",
]


# The action order below matches _build_action_table() in env/swarm_env.py.
ACTION_NAMES = [
    "reverse_left",
    "reverse_straight",
    "reverse_right",
    "idle_left",
    "idle",
    "idle_right",
    "forward_left",
    "forward_straight",
    "forward_right",
]

ACTION_EXPLANATIONS = [
    "move backward while turning left",
    "move backward in a straight line",
    "move backward while turning right",
    "turn left in place / with zero throttle",
    "stay neutral with zero throttle and zero turn",
    "turn right in place / with zero throttle",
    "move forward while turning left",
    "move forward in a straight line",
    "move forward while turning right",
]


# These cases are intentionally hand-written and easy to modify.
# Values are realistic for the current observation format:
# - lidar and pheromone samples in [0, 1]
# - relative vectors in [-1, 1]
# - heading is sin/cos(theta)
# - speed is normalized by max_speed
# - food_presence and carrying_food are binary
TEST_CASES = {
    "target_ahead": {
        "description": "Target is directly ahead, space is open, the agent is not carrying food, and there is no pheromone trail.",
        "values": [
            # lidar_0 ... lidar_8
            0.95, 0.95, 0.95, 0.98, 1.00, 0.98, 0.95, 0.95, 0.95,
            # target_dx_body_norm, target_dy_body_norm
            0.60, 0.00,
            # nest_dx_body_norm, nest_dy_body_norm
            -0.60, 0.00,
            # neighbor_dx_body_norm, neighbor_dy_body_norm
            0.00, 0.00,
            # heading_sin, heading_cos
            0.00, 1.00,
            # speed_norm
            0.10,
            # food_presence, carrying_food
            1.00, 0.00,
            # pheromone_sample_0 ... pheromone_sample_2
            0.00, 0.00, 0.00,
        ],
    },
    "wall_ahead": {
        "description": "A wall or obstacle is very close in front, the target is also ahead, and the policy may need to avoid a collision.",
        "values": [
            0.90, 0.70, 0.40, 0.18, 0.08, 0.18, 0.40, 0.70, 0.90,
            0.45, 0.00,
            -0.80, 0.00,
            0.00, 0.00,
            0.00, 1.00,
            0.30,
            1.00, 0.00,
            0.00, 0.00, 0.00,
        ],
    },
    "pheromone_trail": {
        "description": "No immediate food cue is present, but there is a strong pheromone trail increasing straight ahead.",
        "values": [
            1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
            0.00, 0.00,
            -0.50, 0.10,
            0.10, -0.10,
            0.00, 1.00,
            0.00,
            0.00, 0.00,
            0.20, 0.65, 1.00,
        ],
    },
    "carrying_to_nest": {
        "description": "The agent is carrying food, the nest is ahead and slightly to the left, and the policy should favor returning to the nest.",
        "values": [
            0.90, 0.95, 1.00, 1.00, 1.00, 0.95, 0.90, 0.85, 0.80,
            0.00, 0.00,
            0.70, 0.18,
            0.15, -0.05,
            0.00, 1.00,
            0.20,
            0.00, 1.00,
            0.10, 0.20, 0.35,
        ],
    },
    "open_space": {
        "description": "Open space, no nearby target, no pheromone, no nearby neighbor. This is a neutral exploration case.",
        "values": [
            1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00,
            0.00, 0.00,
            -0.80, 0.00,
            0.00, 0.00,
            0.00, 1.00,
            0.00,
            0.00, 0.00,
            0.00, 0.00, 0.00,
        ],
    },
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory containing shared.pt or agent_<i>.pt")
    parser.add_argument("--shared-policy", action="store_true", help="Load shared.pt instead of agent_<i>.pt")
    parser.add_argument("--agent-index", type=int, default=0, help="Which agent checkpoint to load when not using --shared-policy")
    parser.add_argument("--case", type=str, default="", help="Name of the test case to run")
    parser.add_argument("--list-cases", action="store_true", help="Print available test cases and exit")
    return parser.parse_args(argv)


def load_checkpoint_metadata(checkpoint_dir: str) -> dict:
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(checkpoint_dir: str, shared_policy: bool, agent_index: int, obs_dim: int, action_dim: int) -> QNetwork:
    model = QNetwork(obs_dim, action_dim)
    if shared_policy:
        checkpoint_path = os.path.join(checkpoint_dir, "shared.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"agent_{agent_index}.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def print_available_cases() -> None:
    print("Available test cases:")
    for case_name, case in TEST_CASES.items():
        print(f"- {case_name}: {case['description']}")


def print_observation(obs_values):
    print("Observation values:")
    for name, value in zip(OBS_NAMES, obs_values):
        print(f"  {name:24s} {value: .4f}")


def print_action_values(q_values):
    print("\nModel outputs (Q-values):")
    for index, (name, explanation, q_value) in enumerate(zip(ACTION_NAMES, ACTION_EXPLANATIONS, q_values)):
        print(f"  {index:2d}  {name:18s}  {q_value: .6f}   ({explanation})")


def explain_choice(obs_values, chosen_index: int) -> str:
    front_lidar = obs_values[4]
    target_dx = obs_values[9]
    target_dy = obs_values[10]
    nest_dx = obs_values[11]
    nest_dy = obs_values[12]
    food_presence = obs_values[18]
    carrying_food = obs_values[19]
    pheromone_values = obs_values[20:23]

    action_name = ACTION_NAMES[chosen_index]
    action_text = ACTION_EXPLANATIONS[chosen_index]

    if carrying_food > 0.5:
        return (
            f"The agent is carrying food, and the nest vector is ({nest_dx:.2f}, {nest_dy:.2f}) in body coordinates. "
            f"The model chooses {action_name}, which means it wants to {action_text}."
        )
    if food_presence > 0.5 or abs(target_dx) > 0.05 or abs(target_dy) > 0.05:
        return (
            f"The target vector is ({target_dx:.2f}, {target_dy:.2f}) in body coordinates, so the model likely sees food ahead or nearby. "
            f"It chooses {action_name}, meaning it wants to {action_text}."
        )
    if max(pheromone_values) > 0.05:
        return (
            f"The pheromone samples are {pheromone_values}, so the model sees a trail in front of it. "
            f"It chooses {action_name}, which means it wants to {action_text}."
        )
    if front_lidar < 0.2:
        return (
            f"The front lidar value is only {front_lidar:.2f}, so something is very close ahead. "
            f"The model chooses {action_name}, meaning it wants to {action_text}."
        )
    return (
        f"This is mostly an open-space or neutral observation. "
        f"The model chooses {action_name}, which means it wants to {action_text}."
    )


def main():
    args = parse_args()

    if args.list_cases:
        print_available_cases()
        return

    case_name = args.case or DEFAULT_CASE
    if case_name not in TEST_CASES:
        raise ValueError(f"Unknown case '{case_name}'. Use --list-cases to see valid names.")

    case = TEST_CASES[case_name]
    obs_values = case["values"]

    if len(obs_values) != len(OBS_NAMES):
        raise ValueError(
            f"Test case '{case_name}' has length {len(obs_values)}, but the current observation spec expects {len(OBS_NAMES)} values."
        )

    metadata = load_checkpoint_metadata(args.checkpoint_dir)
    if metadata:
        saved_obs_dim = metadata.get("obs_dim")
        if saved_obs_dim is not None and int(saved_obs_dim) != len(OBS_NAMES):
            raise ValueError(
                f"Checkpoint metadata says obs_dim={saved_obs_dim}, but this script is built for obs_dim={len(OBS_NAMES)}."
            )

    obs_tensor = torch.tensor(obs_values, dtype=torch.float32).unsqueeze(0)
    model = load_model(
        checkpoint_dir=args.checkpoint_dir,
        shared_policy=args.shared_policy,
        agent_index=args.agent_index,
        obs_dim=len(OBS_NAMES),
        action_dim=len(ACTION_NAMES),
    )

    with torch.no_grad():
        q_values = model(obs_tensor).squeeze(0).cpu().tolist()

    chosen_index = int(torch.argmax(torch.tensor(q_values)).item())

    print(f"Selected case: {case_name}")
    print(f"Case description: {case['description']}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Checkpoint mode: {'shared policy' if args.shared_policy else f'agent_{args.agent_index} checkpoint'}")
    print(f"Observation length: {len(obs_values)}")
    print(f"Action space: Discrete({len(ACTION_NAMES)})")
    print()

    print_observation(obs_values)
    print_action_values(q_values)

    print("\nChosen action:")
    print(f"  index: {chosen_index}")
    print(f"  name:  {ACTION_NAMES[chosen_index]}")
    print(f"  meaning: {ACTION_EXPLANATIONS[chosen_index]}")

    print("\nWhat this means:")
    print(f"- {explain_choice(obs_values, chosen_index)}")


if __name__ == "__main__":
    main()
