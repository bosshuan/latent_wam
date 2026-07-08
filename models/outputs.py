"""Output containers for the unified DiT (doc §2.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class WAMOutput:
    """Joint predictor output.

    ``latent_velocity`` : [B, T_fut, N, latent_dim] — v_theta^z on the future grid.
    ``action_velocity`` : [B, T_fut, n_act, action_dim] or ``None`` for video
                          (action tokens structurally omitted — CLAUDE.md §2.3).
    ``value``           : [B, T_fut, num_bins] or ``None`` (Stage A stub, λ_v=0).
    ``latent_hidden``   : [B, T_fut, N, hidden] post-backbone hidden of the Z
                          tokens, kept so the counterfactual path / monitors can
                          recompute x̂1 without a second backbone pass when reused.
    """

    latent_velocity: torch.Tensor
    action_velocity: Optional[torch.Tensor] = None
    value: Optional[torch.Tensor] = None
    latent_hidden: Optional[torch.Tensor] = None
    # M5 interface stub; populated by the M8 closed-loop KV cache (see kv_cache.py).
    kv_cache: Optional[object] = None
