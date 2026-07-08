"""Value head — STAGE A STUB (doc §3 / CLAUDE.md §2.6).

Value is a deterministic *readout*, NOT a flow-generated modality: it never enters
the flow likelihood and its gradient must not perturb the latent/action
distribution. In Stage A ``λ_v = 0`` so this head is present (for shape/plumbing
and the optional joint value-token ablation) but contributes nothing to the loss;
the real value training is M6 (Stage B).

The value token ``V`` is a read-only sink in the attention mask: it reads the
joint latent/action hidden but no ``Z``/``A``/``C`` query reads it back
(``attention_mask.py``), so even the ablation "joint value token" cannot change
the action distribution (doc §3.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """``hidden [B, ..., hidden] -> value [B, ..., num_bins]`` (distributional-ready).

    Defaults to a scalar (``num_bins=1``). Stage A keeps ``λ_v=0`` so this is a
    stub; ``num_bins>1`` leaves room for the M6 distributional bins/quantiles.
    """

    def __init__(self, hidden_dim: int, num_bins: int = 1) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)
