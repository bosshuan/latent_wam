"""Action flow head (doc §2.4 action branch).

Per-embodiment decoder: maps the DiT hidden of each action token to a velocity
``v_theta^a`` in that embodiment's action space (GR00T-style category weights, so
no global zero-padded action vector — CLAUDE.md §4). Trained from scratch (Wan
has no action branch). Velocity follows the PROJECT flow convention (CLAUDE.md §2
invariant 1).

Robot-only: the DiT only ever routes ``action_valid=1`` rows here (the action
tokens are structurally omitted for video — CLAUDE.md §2.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.adapters.action import CategorySpecificMLP


class ActionFlowHead(nn.Module):
    """``hidden [B, T, hidden] + embodiment_id [B] -> velocity [B, T, action_dim]``."""

    def __init__(
        self, num_embodiments: int, hidden_dim: int, action_dim: int
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.decoder = CategorySpecificMLP(
            num_embodiments, hidden_dim, hidden_dim, action_dim
        )

    def forward(self, hidden: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        return self.decoder(hidden, embodiment_id)
