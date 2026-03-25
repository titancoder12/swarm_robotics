from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PolicyDebugConfig:
    enabled: bool = False
    agent_filters: frozenset[str] = field(default_factory=frozenset)
    max_steps: int = 0


def make_policy_debug_config(enabled: bool, agent_spec: str = "", max_steps: int = 0) -> PolicyDebugConfig:
    """Build a policy-debug config from CLI-style inputs."""
    filters = frozenset(token.strip() for token in agent_spec.split(",") if token.strip())
    return PolicyDebugConfig(enabled=bool(enabled), agent_filters=filters, max_steps=max(0, int(max_steps)))


def should_debug_policy(cfg: PolicyDebugConfig, step: int, agent_index: int, agent_id: str) -> bool:
    """Return whether policy debug output should be printed for this agent/step."""
    if not cfg.enabled:
        return False
    if cfg.max_steps > 0 and step >= cfg.max_steps:
        return False
    if not cfg.agent_filters:
        return True
    aliases = {str(agent_index), agent_id, f"agent_{agent_index}"}
    return any(alias in cfg.agent_filters for alias in aliases)


def action_meaning(action_id: int, num_actions: int) -> str:
    """Return a human-readable label for the discrete action id when known."""
    if num_actions not in (9, 18):
        return f"action_{action_id}"

    throttle_vals = (-1.0, 0.0, 1.0)
    turn_vals = (-1.0, 0.0, 1.0)
    deposit_vals = (0, 1) if num_actions == 18 else (0,)
    table = [
        (throttle, turn, deposit)
        for throttle in throttle_vals
        for turn in turn_vals
        for deposit in deposit_vals
    ]
    if action_id < 0 or action_id >= len(table):
        return f"action_{action_id}"

    throttle, turn, deposit = table[action_id]
    if throttle == 0.0 and turn == 0.0:
        motion = "stop"
    elif throttle == 0.0:
        motion = "turn_left" if turn > 0.0 else "turn_right"
    else:
        move = "forward" if throttle > 0.0 else "reverse"
        if turn > 0.0:
            motion = f"{move}_left"
        elif turn < 0.0:
            motion = f"{move}_right"
        else:
            motion = move

    if num_actions == 18:
        suffix = "drop" if deposit else "keep"
        return f"{motion}_{suffix}"
    return motion


def _format_values(values) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    rounded = [round(float(v), 3) for v in arr.tolist()]
    return str(rounded)


def print_policy_debug(
    *,
    step: int,
    agent_index: int,
    agent_id: str,
    action: int,
    num_actions: int,
    output_name: str | None = None,
    output_values=None,
    mode: str | None = None,
    epsilon: float | None = None,
    prev_reward: float | None = None,
    prev_done: bool | None = None,
    policy_label: str | None = None,
) -> None:
    """Print one human-readable policy debug line for a single agent decision."""
    parts = ["[POLICY_DEBUG]", f"step={step}", f"agent={agent_id}", f"index={agent_index}"]
    if policy_label:
        parts.append(f"policy={policy_label}")
    if epsilon is not None:
        parts.append(f"epsilon={epsilon:.3f}")
    if mode is not None:
        parts.append(f"mode={mode}")
    if prev_reward is not None:
        parts.append(f"prev_reward={prev_reward:.3f}")
    if prev_done is not None:
        parts.append(f"prev_done={int(bool(prev_done))}")
    if output_name is not None and output_values is not None:
        parts.append(f"{output_name}={_format_values(output_values)}")
    parts.append(f"action={action} ({action_meaning(action, num_actions)})")
    print(" ".join(parts), flush=True)
