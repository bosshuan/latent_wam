"""Conditioning adapter — text (shared) + proprio routing (doc §2.4).

CLAUDE.md §3: the conditioning is split so **text goes through the shared
cross-attention** (umT5-XXL frozen, all embodiments share it — wired for real in
M4) while **proprio/state goes through a per-embodiment adapter**
(``StateAdapter``). This module is the thin owner that keeps both and exposes the
text-condition tokens the DiT blocks cross-attend.

In M3 the text encoder is not yet attached (that is M4), so this adapter is a
**stub** that simply projects a pre-cached/random text embedding
``[B, L_txt, D_txt] -> [B, L_txt, hidden]`` and supplies a learned null-text token
for CFG / caption-less video (CLAUDE.md §3 "无 caption 视频用 null text"). The
project keeps the text path embodiment-agnostic; only the state path is
per-embodiment.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.adapters.state import StateAdapter


class ConditionAdapter(nn.Module):
    """Holds the shared text projection + the per-embodiment state adapter."""

    def __init__(
        self,
        hidden_dim: int,
        num_embodiments: int,
        state_dim: int,
        text_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_dim = text_dim
        # Shared text projection (stub for the frozen umT5 cross-attn input, M4).
        if text_dim > 0:
            self.text_proj = nn.Linear(text_dim, hidden_dim)
        else:
            self.text_proj = None
        # Learned null-text token (CFG dropout / caption-less video).
        self.null_text = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.state_adapter = StateAdapter(num_embodiments, state_dim, hidden_dim)

    def text_tokens(
        self, text_embedding: Optional[torch.Tensor], batch_size: int, device
    ) -> torch.Tensor:
        """Project cached text embeddings, or fall back to the null-text token.

        Returns ``[B, L, hidden]`` (``L=1`` for the null fallback).
        """
        if text_embedding is None or self.text_proj is None:
            return self.null_text.to(device).expand(batch_size, -1, -1)
        return self.text_proj(text_embedding)

    def state_token(
        self, proprio: Optional[torch.Tensor], embodiment_id: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """``[B, 1, hidden]`` per-embodiment state token, or ``None`` for video."""
        if proprio is None:
            return None
        return self.state_adapter(proprio, embodiment_id)
