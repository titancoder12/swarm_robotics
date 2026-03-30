from __future__ import annotations

import torch
import torch.nn as nn


class SharedGRUActor(nn.Module):
    """Shared recurrent actor that maps local observations to action logits."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def initial_hidden(self, batch_size: int, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, obs, hidden_state, mask=None):
        """Run one recurrent actor step.

        obs: [B, obs_dim]
        hidden_state: [1, B, H]
        mask: [B] where 0 resets hidden state before this step
        """
        if mask is not None:
            hidden_state = hidden_state * mask.view(1, -1, 1)
        x = self.obs_encoder(obs)
        x, next_hidden = self.gru(x.unsqueeze(1), hidden_state)
        logits = self.policy_head(x.squeeze(1))
        return logits, next_hidden


class CentralizedGRUCritic(nn.Module):
    """Recurrent centralized critic over the training-time global state."""

    def __init__(self, state_dim: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
        )
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def initial_hidden(self, batch_size: int, device):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)

    def forward(self, state, hidden_state, mask=None):
        """Run one recurrent critic step.

        state: [B, state_dim]
        hidden_state: [1, B, H]
        mask: [B] where 0 resets hidden state before this step
        """
        if mask is not None:
            hidden_state = hidden_state * mask.view(1, -1, 1)
        x = self.state_encoder(state)
        x, next_hidden = self.gru(x.unsqueeze(1), hidden_state)
        value = self.value_head(x.squeeze(1)).squeeze(-1)
        return value, next_hidden
