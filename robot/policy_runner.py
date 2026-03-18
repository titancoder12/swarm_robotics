from __future__ import annotations

import os

import numpy as np
import torch

from env.config import SwarmConfig
from models.q_network import QNetwork


class PolicyRunner:
    """Load a trained custom DQN checkpoint and run deterministic inference."""

    def __init__(self, cfg: SwarmConfig, checkpoint_dir: str, shared_policy: bool = True, device: str = "cpu"):
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.shared_policy = shared_policy
        self.device = torch.device(device)
        self.model: QNetwork | None = None

    def load(self, obs_dim: int) -> None:
        """Load the policy checkpoint for the expected observation size."""

        model = QNetwork(obs_dim, self.cfg.num_actions).to(self.device)
        if self.shared_policy:
            path = os.path.join(self.checkpoint_dir, "shared.pt")
        else:
            path = os.path.join(self.checkpoint_dir, "agent_0.pt")
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        self.model = model

    def predict(self, observation: np.ndarray) -> int:
        """Return the greedy discrete action index for one observation."""

        if self.model is None:
            raise RuntimeError("Policy model not loaded. Call load() first.")
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_vals = self.model(obs_tensor)
            return int(torch.argmax(q_vals, dim=1).item())
