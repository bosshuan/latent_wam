"""V-JEPA latent flow head (doc §2.4 / §2.5).

Replaces Wan's pixel/VAE velocity head: maps the DiT hidden of each future latent
token to a velocity ``v_theta^z`` in the 384-d VJ-RAE latent space. Trained from
scratch — the Wan pixel head is NEVER loaded into this head (CLAUDE.md §3 / §2.2).

Timestep conditioning is AdaLN-style (a la Wan/DiT): the per-sample flow timestep
``t_z`` modulates the pre-output LayerNorm via (shift, scale). The predicted
velocity follows the PROJECT convention — ``x̂1 = x_t + (1-t) v`` — so it is the
*data minus noise* direction, opposite Wan's sign (CLAUDE.md §2 invariant 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.adapters.action import SinusoidalPositionalEncoding


class VJEPALatentFlowHead(nn.Module):
    """``hidden [B,...,hidden] + t_z [B] -> velocity [B,...,latent_dim]``."""

    def __init__(self, hidden_dim: int, latent_dim: int, init_std: float = 1.0e-3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.t_embed = SinusoidalPositionalEncoding(hidden_dim)
        # produce (shift, scale) for AdaLN from the timestep embedding
        self.t_mod = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))
        self.proj = nn.Linear(hidden_dim, latent_dim)
        # Near-zero init keeps the first prediction close to identity flow while
        # preserving a measurable counterfactual action signal for S_a probes.
        if init_std > 0.0:
            nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)
            nn.init.zeros_(self.proj.bias)
        else:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor, t_z: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, N, hidden]; t_z: [B]
        t_emb = self.t_embed(t_z).to(device=hidden.device, dtype=hidden.dtype)
        shift, scale = self.t_mod(t_emb).chunk(2, dim=-1)  # [B,hidden] each
        # broadcast modulation over the (T, N) token axes
        while shift.ndim < hidden.ndim:
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        h = self.norm(hidden) * (1 + scale) + shift
        return self.proj(h)
