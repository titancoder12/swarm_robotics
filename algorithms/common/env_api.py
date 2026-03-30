from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnvSpaces:
    """Compact description of environment spaces used by training code."""

    obs_dim: int
    action_dim: int
    state_dim: int


def extract_env_spaces(env: Any) -> EnvSpaces:
    """Read local observation, action, and centralized state dimensions from an env."""
    agent = env.possible_agents[0]
    obs_space = env.observation_space(agent)
    action_space = env.action_space(agent)
    state_space = env.state_space()
    return EnvSpaces(
        obs_dim=int(obs_space.shape[0]),
        action_dim=int(action_space.n),
        state_dim=int(state_space.shape[0]),
    )

