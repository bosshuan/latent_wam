"""Multi-embodiment action adapter (doc §2.4 action branch; CLAUDE.md §4).

Ported — *re-written, not copied* — from DreamZero
``modules/wan_video_dit_action_casual_chunk.py``
(``CategorySpecificLinear`` / ``CategorySpecificMLP`` /
``MultiEmbodimentActionEncoder``), NVIDIA Source Code License-NC, academic
non-commercial. We keep the GR00T-style per-category weights (one weight slab per
embodiment) so each robot's action space gets its own projection without a global
zero-padded vector (CLAUDE.md §4 "never silently zero-fill into a global vector").

Used ONLY for the action/state branch and the action decoder — the shared DiT
blocks, latent head, and text cross-attention stay embodiment-agnostic.

This adapter must never be fed a fabricated action: the DiT structurally omits the
action tokens for ``action_valid=0`` rows (CLAUDE.md §2.3), so this module only
ever sees genuine robot rows whose ``embodiment_id`` is a real (>=0, non-NEW) id.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategorySpecificLinear(nn.Module):
    """Per-category affine: ``y = x @ W[cat] + b[cat]``.

    Source: DreamZero ``CategorySpecificLinear`` (rewritten). ``W`` is
    ``[num_categories, in, out]``; ``cat_ids`` selects one slab per row via a
    batched matmul.
    """

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_categories = num_categories
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The action vector must survive the three-layer action encoder before
        # modality/chunk/step embeddings are added. A tiny 0.02 normal init made
        # true and counterfactual action tokens nearly identical in S_a probes.
        for cat in range(self.num_categories):
            nn.init.xavier_uniform_(self.W[cat])
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, in]; cat_ids: [B] -> y: [B, T, out]
        selected_W = self.W[cat_ids]  # [B, in, out]
        selected_b = self.b[cat_ids]  # [B, out]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Two ``CategorySpecificLinear`` with a ReLU (DreamZero, rewritten)."""

    def __init__(
        self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int
    ) -> None:
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal embedding of a (flow) timestep scalar -> ``dim`` vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalPositionalEncoding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] or [B, T] -> [..., dim]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().unsqueeze(-1) * freqs  # [..., half]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class MultiEmbodimentActionEncoder(nn.Module):
    """Encode ``(action, flow-timestep)`` -> hidden token, per embodiment.

    Source: DreamZero ``MultiEmbodimentActionEncoder`` (rewritten). The flow
    timestep here is the PROJECT-convention ``t_a`` (``t=0`` noise, ``t=1`` data),
    NOT Wan's sigma. ``W1: d->w``, then concat a sinusoidal timestep embedding and
    fuse with ``W2: 2w->w`` (SiLU), then ``W3: w->w``.
    """

    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self, actions: torch.Tensor, timesteps: torch.Tensor, cat_ids: torch.Tensor
    ) -> torch.Tensor:
        """actions: [B, T, action_dim]; timesteps: [B] (per-sample flow t_a);
        cat_ids: [B] -> [B, T, hidden_size].
        """
        a_emb = self.W1(actions, cat_ids)  # [B,T,w]
        # broadcast the per-sample timestep over the T action steps
        tau = self.pos_encoding(timesteps).to(a_emb.dtype)  # [B,w]
        tau = tau.unsqueeze(1).expand(-1, a_emb.shape[1], -1)  # [B,T,w]
        x = torch.cat([a_emb, tau], dim=-1)  # [B,T,2w]
        x = F.silu(self.W2(x, cat_ids))
        return self.W3(x, cat_ids)  # [B,T,w]
