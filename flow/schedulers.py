"""Timestep schedulers for the unified latent-action flow (doc §2.6).

Two regimes:
  * **coupled** (training early): ``t_a = t_z`` — a single timestep per sample
    shared by the latent and action modalities, to pull latent/action alignment.
  * **decoupled** (DreamZero-Flash style, later): the latent gets *higher noise*
    (timesteps skewed toward ``t=0`` = noise) while the action is uniform, so at
    inference the action converges in few solver steps while the latent uses more.

All timesteps live in ``(0, 1)`` under the project convention (``t=0`` noise,
``t=1`` data — CLAUDE.md §2 invariant 1). Pure / seedable so tests are
deterministic; no global RNG state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class TimestepScheduler:
    """Samples ``(t_z, t_a)`` per batch under the project flow convention.

    ``coupled=True`` ties the two; ``coupled=False`` skews the latent toward the
    noise end (``t->0``) by exponentiating a uniform with ``latent_noise_power``
    (> 1 biases toward 0). Action stays uniform.
    """

    coupled: bool = True
    latent_noise_power: float = 2.0
    eps: float = 1e-3

    def _uniform(self, batch: int, device, generator: Optional[torch.Generator]) -> torch.Tensor:
        u = torch.rand(batch, device=device, generator=generator)
        # clamp away from the exact endpoints so (1-t) and t are both > 0
        return u.clamp(self.eps, 1.0 - self.eps)

    def sample(
        self,
        batch: int,
        device: torch.device | str = "cpu",
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_a = self._uniform(batch, device, generator)
        if self.coupled:
            return t_a.clone(), t_a
        # decoupled: bias the latent toward t=0 (more noise). u**p with p>1
        # concentrates mass near 0.
        u = self._uniform(batch, device, generator)
        t_z = (u**self.latent_noise_power).clamp(self.eps, 1.0 - self.eps)
        return t_z, t_a
