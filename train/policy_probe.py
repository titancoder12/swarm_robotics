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

# Add the repo root to Python's import path so this script can be run directly
# from the command line as `python train/policy_probe.py`.
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
OBS_NAMES_23 = [
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

# Legacy observation order used by older checkpoints from this repo.
# Older custom checkpoints were trained before nest direction, local food
# presence, and carrying-food state were added to the environment. Those
# checkpoints still use the same 9-action output, but their input layer expects
# only 19 observation values.
OBS_NAMES_19 = [
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
    "neighbor_dx_body_norm",
    "neighbor_dy_body_norm",
    "heading_sin",
    "heading_cos",
    "speed_norm",
    "pheromone_sample_0",
    "pheromone_sample_1",
    "pheromone_sample_2",
]


# The action order below matches _build_action_table() in env/swarm_env.py.
# The model does not output text labels directly. It outputs one scalar Q-value
# per discrete action index, so we keep the human-readable names and
# explanations here for printing.
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
#
# The purpose of these cases is not to be exhaustive or physically perfect.
# They are "teaching examples" that make it easy to ask questions like:
# - "What does the policy do when food is straight ahead?"
# - "What does it do when the front lidar is very small?"
# - "Does it respond to a pheromone trail?"
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
    """Parse CLI flags for selecting a checkpoint and a hand-written test case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory containing shared.pt or agent_<i>.pt")
    parser.add_argument("--shared-policy", action="store_true", help="Load shared.pt instead of agent_<i>.pt")
    parser.add_argument("--agent-index", type=int, default=0, help="Which agent checkpoint to load when not using --shared-policy")
    parser.add_argument("--case", type=str, default="", help="Name of the test case to run")
    parser.add_argument("--list-cases", action="store_true", help="Print available test cases and exit")
    return parser.parse_args(argv)


def load_checkpoint_metadata(checkpoint_dir: str) -> dict:
    """Read optional checkpoint metadata.json if it exists.

    The probing script does not rely only on metadata, because metadata can
    become stale if a checkpoint directory contains mixed files from older and
    newer runs. The actual checkpoint tensor shapes are treated as the final
    source of truth.
    """
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(checkpoint_dir: str, shared_policy: bool, agent_index: int, obs_dim: int, action_dim: int) -> QNetwork:
    """Load a custom DQN model from disk.

    This helper mirrors the custom checkpoint layout used elsewhere in the repo:
    - shared policy: `shared.pt`
    - independent policy: `agent_<i>.pt`

    The current `main()` path loads the state dict directly so it can inspect
    tensor shapes before constructing the model, but this helper is still useful
    as a small reference for the normal load path.
    """
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
    """Print the names and descriptions of all built-in observation cases."""
    print("Available test cases:")
    for case_name, case in TEST_CASES.items():
        print(f"- {case_name}: {case['description']}")


def print_observation(obs_names, obs_values):
    """Pretty-print one observation vector with a label for each component."""
    print("Observation values:")
    for name, value in zip(obs_names, obs_values):
        print(f"  {name:24s} {value: .4f}")


def print_action_values(q_values):
    """Pretty-print the model's raw Q-values for all 9 discrete actions."""
    print("\nModel outputs (Q-values):")
    for index, (name, explanation, q_value) in enumerate(zip(ACTION_NAMES, ACTION_EXPLANATIONS, q_values)):
        print(f"  {index:2d}  {name:18s}  {q_value: .6f}   ({explanation})")


def explain_choice(obs_names, obs_values, chosen_index: int) -> str:
    """Generate a short plain-language interpretation of the chosen action.

    This function is intentionally heuristic. It does not explain the true inner
    reasoning of the neural network. Instead, it looks at a few salient input
    features and produces a simple story that helps a beginner connect:
    observation -> chosen action.

    The logic is written against observation names rather than fixed indices so
    it works for both the current 23-D layout and the legacy 19-D layout.
    Missing features default to 0.0.
    """
    # Build a name -> value mapping so the rest of the function can refer to
    # observation components symbolically instead of by raw index.
    obs_map = dict(zip(obs_names, obs_values))

    # Pull out a few features that are most useful for a short explanation.
    # These are not the only features the model uses; they are just the ones
    # we choose to mention in the printed explanation.
    front_lidar = obs_map.get("lidar_4", 0.0)
    target_dx = obs_map.get("target_dx_body_norm", 0.0)
    target_dy = obs_map.get("target_dy_body_norm", 0.0)
    nest_dx = obs_map.get("nest_dx_body_norm", 0.0)
    nest_dy = obs_map.get("nest_dy_body_norm", 0.0)
    food_presence = obs_map.get("food_presence", 0.0)
    carrying_food = obs_map.get("carrying_food", 0.0)
    pheromone_values = [
        obs_map.get("pheromone_sample_0", 0.0),
        obs_map.get("pheromone_sample_1", 0.0),
        obs_map.get("pheromone_sample_2", 0.0),
    ]

    action_name = ACTION_NAMES[chosen_index]
    action_text = ACTION_EXPLANATIONS[chosen_index]

    # Prioritize carrying-food explanations first, because returning to the nest
    # is usually the most important task-state cue once the agent already has food.
    if carrying_food > 0.5:
        return (
            f"The agent is carrying food, and the nest vector is ({nest_dx:.2f}, {nest_dy:.2f}) in body coordinates. "
            f"The model chooses {action_name}, which means it wants to {action_text}."
        )
    # If food is locally present or the nearest-target vector is non-zero, talk
    # about the target cue next.
    if food_presence > 0.5 or abs(target_dx) > 0.05 or abs(target_dy) > 0.05:
        return (
            f"The target vector is ({target_dx:.2f}, {target_dy:.2f}) in body coordinates, so the model likely sees food ahead or nearby. "
            f"It chooses {action_name}, meaning it wants to {action_text}."
        )
    # If pheromone is present, talk about trail-following before obstacle
    # avoidance. This ordering keeps the explanation short and single-purpose.
    if max(pheromone_values) > 0.05:
        return (
                f"The pheromone samples are {tuple(round(v, 2) for v in pheromone_values)}, so the model sees a trail in front of it. "
                f"It chooses {action_name}, which means it wants to {action_text}."
        )
    # A very small center lidar value is a strong sign that something is close
    # directly ahead.
    if front_lidar < 0.2:
        return (
            f"The front lidar value is only {front_lidar:.2f}, so something is very close ahead. "
            f"The model chooses {action_name}, meaning it wants to {action_text}."
        )
    # Fallback explanation for mostly neutral observations.
    return (
        f"This is mostly an open-space or neutral observation. "
        f"The model chooses {action_name}, which means it wants to {action_text}."
    )


def main():
    """Run the policy probe from the command line."""
    args = parse_args()

    # `--list-cases` is a quick discovery mode so users can see the built-in
    # playground scenarios without loading a checkpoint.
    if args.list_cases:
        print_available_cases()
        return

    # Pick the requested case, or fall back to the editable default near the top
    # of the file.
    case_name = args.case or DEFAULT_CASE
    if case_name not in TEST_CASES:
        raise ValueError(f"Unknown case '{case_name}'. Use --list-cases to see valid names.")

    case = TEST_CASES[case_name]
    obs_values = case["values"]

    # All built-in test cases are authored in the current 23-D observation
    # format. If we later load a legacy 19-D checkpoint, we will derive a
    # compatible 19-D slice from this 23-D case.
    if len(obs_values) != len(OBS_NAMES_23):
        raise ValueError(
            f"Test case '{case_name}' has length {len(obs_values)}, but the current observation spec expects {len(OBS_NAMES_23)} values."
        )

    metadata = load_checkpoint_metadata(args.checkpoint_dir)

    # Resolve which actual checkpoint file we are probing.
    checkpoint_path = os.path.join(args.checkpoint_dir, "shared.pt" if args.shared_policy else f"agent_{args.agent_index}.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load the raw state dict first so we can inspect tensor shapes before we
    # instantiate the model. This is the key step that lets the script adapt to
    # either current 23-D checkpoints or older 19-D checkpoints.
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_obs_dim = int(state_dict["net.0.weight"].shape[1])
    checkpoint_action_dim = int(state_dict["net.4.weight"].shape[0])

    # The script only knows how to explain this repo's current discrete 9-action
    # interface. If the checkpoint output width differs, the action labels would
    # no longer be trustworthy.
    if checkpoint_action_dim != len(ACTION_NAMES):
        raise ValueError(
            f"Checkpoint outputs {checkpoint_action_dim} actions, but this script only knows how to explain {len(ACTION_NAMES)} actions."
        )

    if checkpoint_obs_dim == 23:
        # Current environment contract: use the full observation vector exactly
        # as written in TEST_CASES.
        obs_names = OBS_NAMES_23
        obs_values_to_use = obs_values
        obs_note = "using the current 23-dimensional observation layout"
    elif checkpoint_obs_dim == 19:
        # Older checkpoints were trained before nest direction, local food
        # presence, and carrying-food state were added. To probe them with the
        # same hand-written cases, drop those newer components and keep only the
        # subset the old model actually expects.
        obs_names = OBS_NAMES_19
        obs_values_to_use = [
            *obs_values[0:11],   # lidar + target
            *obs_values[13:18],  # neighbor + heading + speed
            *obs_values[20:23],  # pheromone
        ]
        obs_note = "using the legacy 19-dimensional observation layout (nest/food/carrying features are dropped)"
    else:
        raise ValueError(
            f"Unsupported checkpoint obs_dim={checkpoint_obs_dim}. This probing script currently supports only 19-D and 23-D checkpoints."
        )

    # Metadata is useful for display, but checkpoint tensor shapes are treated
    # as authoritative. If they disagree, print a warning rather than failing.
    if metadata:
        saved_obs_dim = metadata.get("obs_dim")
        if saved_obs_dim is not None and int(saved_obs_dim) != checkpoint_obs_dim:
            print(
                f"Warning: metadata.json says obs_dim={saved_obs_dim}, but the checkpoint tensor shape says obs_dim={checkpoint_obs_dim}."
            )

    # Now that we know the input/output sizes, build the model, load weights,
    # and switch to eval mode.
    model = QNetwork(checkpoint_obs_dim, checkpoint_action_dim)
    model.load_state_dict(state_dict)
    model.eval()

    # PyTorch models expect a batch dimension, so convert:
    #   [obs_dim] -> [1, obs_dim]
    obs_tensor = torch.tensor(obs_values_to_use, dtype=torch.float32).unsqueeze(0)

    # Run one forward pass with gradients disabled. The network output is one
    # Q-value per discrete action.
    with torch.no_grad():
        q_values = model(obs_tensor).squeeze(0).cpu().tolist()

    # Pick the greedy action exactly the same way the evaluator/demo choose
    # actions at inference time: argmax over Q-values.
    chosen_index = int(torch.argmax(torch.tensor(q_values)).item())

    # Print a compact summary before dumping the detailed observation and action
    # tables.
    print(f"Selected case: {case_name}")
    print(f"Case description: {case['description']}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint mode: {'shared policy' if args.shared_policy else f'agent_{args.agent_index} checkpoint'}")
    print(f"Observation length used: {len(obs_values_to_use)}")
    print(f"Observation note: {obs_note}")
    print(f"Action space: Discrete({checkpoint_action_dim})")
    print()

    print_observation(obs_names, obs_values_to_use)
    print_action_values(q_values)

    print("\nChosen action:")
    print(f"  index: {chosen_index}")
    print(f"  name:  {ACTION_NAMES[chosen_index]}")
    print(f"  meaning: {ACTION_EXPLANATIONS[chosen_index]}")

    print("\nWhat this means:")
    print(f"- {explain_choice(obs_names, obs_values_to_use, chosen_index)}")


if __name__ == "__main__":
    main()
