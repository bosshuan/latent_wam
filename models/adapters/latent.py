"""V-JEPA latent input adapter (doc §2.4).

Replaces Wan's VAE patch-embed: eats a 384-dim *codec* token on the 12×12 grid
and lifts it to the DiT hidden width. Trained from scratch (CLAUDE.md §3 — the
Wan VAE patch-embed is never loaded). The SAME projection is reused for the
clean *context* latent and the noisy *future* latent; the two are distinguished
downstream by their modality embedding (see ``latent_tokenizer.py``), not by
separate weights, so context and prediction share one feature space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VJEPALatentInputAdapter(nn.Module):
    """``[B, T, N, latent_dim] -> [B, T, N, hidden]`` per-token MLP."""

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
