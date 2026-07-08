"""Per-embodiment proprio/state adapter (doc §2.4; CLAUDE.md §3).

The ``condition_adapter`` is deliberately split: **text** goes through the shared
cross-attention (all embodiments), while **proprio/state** goes through this
per-embodiment adapter — they carry different semantics and must not be mixed in
one adapter. Proprio ``[B, state_dim] -> [B, hidden]`` becomes a single state
token prepended/added to the conditioning, routed by ``embodiment_id`` via the
GR00T-style category weights.

Like the action adapter, this is robot-only: video rows (``action_valid=0``) have
``proprio=None`` and never select a state adapter (CLAUDE.md §2.3 / §10).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.adapters.action import CategorySpecificMLP


class StateAdapter(nn.Module):
    """``proprio [B, state_dim] -> state token [B, 1, hidden]`` per embodiment."""

    def __init__(
        self, num_embodiments: int, state_dim: int, hidden_dim: int
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.mlp = CategorySpecificMLP(num_embodiments, state_dim, hidden_dim, hidden_dim)

    def forward(self, proprio: torch.Tensor, embodiment_id: torch.Tensor) -> torch.Tensor:
        # proprio: [B, state_dim] -> add a length-1 token axis for the MLP, then
        # return [B, 1, hidden].
        x = proprio.unsqueeze(1)  # [B,1,state_dim]
        return self.mlp(x, embodiment_id)  # [B,1,hidden]
