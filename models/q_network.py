from __future__ import annotations

import torch.nn as nn


class QNetwork(nn.Module):
    """Small MLP mapping observations to Q-values for each discrete action."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        """Return Q-values for each action."""
        return self.net(x)