from __future__ import annotations

import os

import torch

from algorithms.mappo.networks import SharedGRUActor


def load_actor(checkpoint_dir: str, obs_dim: int, action_dim: int, device: str = "cpu"):
    """Load a recurrent MAPPO actor checkpoint for decentralized inference."""
    resolved_device = torch.device(device)
    payload = torch.load(os.path.join(checkpoint_dir, "actor.pt"), map_location=resolved_device)
    actor = SharedGRUActor(obs_dim, action_dim, hidden_size=int(payload["hidden_size"])).to(resolved_device)
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    return actor, resolved_device


def select_action(actor: SharedGRUActor, obs, hidden_state, done_mask, deterministic: bool = True):
    """Run one decentralized recurrent actor step."""
    with torch.no_grad():
        logits, next_hidden = actor(obs, hidden_state, done_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
    return action, next_hidden
