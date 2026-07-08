"""V-JEPA Representation Autoencoder (VJ-RAE) — formerly "codec" in M2.

Compresses frozen V-JEPA multi-level dense features into the flow-space latent
the DiT operates on, then is **frozen** before Stage A joint training so the
target space cannot collapse (CLAUDE.md §2.5, doc §2.3/§4.1).

Pipeline (shapes for gigantic: L=4, D=1664, grid 24×24=576):
    raw features            [B, T, 576, L, D]      (from FrozenVJEPAEncoder)
      FixedFeatureNormalizer (per (L,D) channel, offline robot stats for
                              Stage A)            -> z [B, T, 576, L, D]
      MultiLevelFusion (per-layer LayerNorm -> concat L*D=6656 -> MLP)
                                                  -> [B, T, 576, hidden]
      TokenReducer (2×2 spatial pool 24×24->12×12) -> [B, T, 144, hidden]
      CodecEncoder (MLP -> latent_dim)            -> R [B, T, 144, 384]
    decode mirrors back to the normalized feature space z.

Hard rules wired here:
  * **never whole-frame pool** — TokenReducer keeps the 2D grid + time index
    (CLAUDE.md §2.8); `latent_dim = 384 / token`.
  * `codec_in_dim = L*D` is taken from the encoder (`encoder.codec_in_dim`),
    never hardcoded.
  * Stage-A normalization stats are computed on paired robot video+action data,
    separate from action/state norms, and serialized with a `norm_version`
    (`FixedFeatureNormalizer`).

The action-discriminability probe lives in this module too but is a **training
-only** attachment that is discarded after VJ-RAE training (doc §2.3).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.schemas import LatentGrid, MultiLevelFeatures


# ---------------------------------------------------------------------------
# Fixed feature normalizer (offline stats, frozen at train time)
# ---------------------------------------------------------------------------


class FixedFeatureNormalizer(nn.Module):
    """Per-(layer, channel) standardization with offline-computed stats.

    Stage-A stats are accumulated over paired robot video+action clips. If a
    later Stage-A+ ablation uses actionless video, fit a separate norm_version so
    cached VJ-RAE latents never collide.
    Buffers (no grad) so they serialize and move with the module. Tagged with
    ``norm_version`` so the feature cache / VJ-RAE can invalidate on change.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        norm_version: str = "v0",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.norm_version = norm_version
        self.eps = eps
        self.register_buffer("mean", torch.zeros(num_layers, embed_dim))
        self.register_buffer("var", torch.ones(num_layers, embed_dim))
        self.register_buffer("fitted", torch.zeros(1, dtype=torch.bool))
        # accumulation buffers (not part of the serialized stats contract)
        self.register_buffer("_sum", torch.zeros(num_layers, embed_dim))
        self.register_buffer("_sumsq", torch.zeros(num_layers, embed_dim))
        self.register_buffer("_count", torch.zeros(1, dtype=torch.float64))
        self.requires_grad_(False)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Accumulate stats from a feature batch ``x`` [..., L, D]."""
        if x.shape[-2:] != (self.num_layers, self.embed_dim):
            raise ValueError(
                f"normalizer expects [...,{self.num_layers},{self.embed_dim}]; "
                f"got {tuple(x.shape)}"
            )
        flat = x.reshape(-1, self.num_layers, self.embed_dim).to(
            device=self._sum.device, dtype=torch.float64
        )
        self._sum += flat.sum(dim=0)
        self._sumsq += (flat * flat).sum(dim=0)
        self._count += flat.shape[0]

    @torch.no_grad()
    def finalize(self) -> None:
        n = float(self._count.item())
        if n <= 1:
            raise RuntimeError("normalizer.finalize called before any update")
        mean = self._sum / n
        var = self._sumsq / n - mean * mean
        self.mean.copy_(mean.to(self.mean.dtype))
        self.var.copy_(var.clamp_min(0).to(self.var.dtype))
        self.fitted.fill_(True)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var + self.eps) + self.mean

    def stats_state(self) -> dict:
        return {
            "norm_version": self.norm_version,
            "num_layers": self.num_layers,
            "embed_dim": self.embed_dim,
            "mean": self.mean.clone(),
            "var": self.var.clone(),
        }


# ---------------------------------------------------------------------------
# Fusion / spatial reduction / encode / decode
# ---------------------------------------------------------------------------


class MultiLevelFusion(nn.Module):
    """Per-layer LayerNorm -> concat (L*D) -> MLP -> hidden (decided: MLP)."""

    def __init__(self, num_layers: int, embed_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(num_layers)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(num_layers * embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N, L, D]
        normed = [self.layer_norms[i](x[..., i, :]) for i in range(self.num_layers)]
        cat = torch.cat(normed, dim=-1)  # [B, T, N, L*D]
        return self.mlp(cat)  # [B, T, N, hidden]


class TokenReducer(nn.Module):
    """2×2 spatial avg-pool on the dense grid; keeps 2D grid + time index.

    NEVER whole-frame pools. 24×24 -> 12×12 = 144 tokens / time index.
    """

    def __init__(self, grid_hw: tuple[int, int], pool: int = 2) -> None:
        super().__init__()
        h, w = grid_hw
        if h % pool or w % pool:
            raise ValueError(f"grid {grid_hw} not divisible by pool {pool}")
        self.grid_hw = grid_hw
        self.pool = pool
        self.out_hw = (h // pool, w // pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, N=H*W, *C] -> [B, T, (H/p)*(W/p), *C]
        # Accepts arbitrary trailing dims so one code path pools both the fused
        # hidden [B,T,N,hidden] and the multi-level features [B,T,N,L,D].
        b, t, n = x.shape[:3]
        trailing = tuple(x.shape[3:])
        h, w = self.grid_hw
        if n != h * w:
            raise ValueError(f"token count {n} != grid {h}x{w}")
        p = self.pool
        g = x.reshape(b, t, h, w, *trailing)
        g = g.reshape(b, t, h // p, p, w // p, p, *trailing)
        g = g.mean(dim=(3, 5))  # average each 2x2 block
        return g.reshape(b, t, (h // p) * (w // p), *trailing)


class CodecEncoder(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CodecDecoder(nn.Module):
    """Reconstruct the POOLED normalized multi-level features from the latent.

    DECISION (user Q1): the codec is the invertible `pooled-features <-> 384`
    compressor at the **12×12 = 144-token** level only. The 2×2 TokenReducer is a
    fixed, intentional, lossy downsample that the codec does NOT try to invert —
    so the decoder stays at 144 tokens and does **not** upsample back to 24×24.
    Reconstruction target = `pool(normalize(raw features))` [B,T,144,L,D] (a fixed
    target; reconstructing the trained fusion's own output would be circular).
    The discarded high-freq from pooling is excluded from both sides of the loss,
    so the M2 cosine acceptance is measured in this pooled 12×12 feature space.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers * embed_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, T, n_red, latent_dim] -> [B, T, n_red, L, D] (same n_red)
        b, t, n_red, _ = latent.shape
        out = self.net(latent)  # [B, T, n_red, L*D]
        return out.reshape(b, t, n_red, self.num_layers, self.embed_dim)


class VJEPALatentCodec(nn.Module):
    """Backward-compatible class name for the VJ-RAE bundle.

    New code should import ``VJEPRepresentationAutoencoder`` from
    ``models.vj_rae``. The old name remains so previous milestone
    tests/checkpoints keep loading while the repo migrates terminology.
    """

    def __init__(
        self,
        num_layers: int,
        embed_dim: int,
        grid_hw: tuple[int, int],
        hidden_dim: int = 1024,
        latent_dim: int = 384,
        pool: int = 2,
        normalizer: Optional[FixedFeatureNormalizer] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.grid_hw = grid_hw
        self.latent_dim = latent_dim
        self.pool = pool
        self.normalizer = normalizer or FixedFeatureNormalizer(num_layers, embed_dim)
        self.reducer = TokenReducer(grid_hw, pool)
        self.fusion = MultiLevelFusion(num_layers, embed_dim, hidden_dim)
        self.enc = CodecEncoder(hidden_dim, latent_dim)
        self.dec = CodecDecoder(latent_dim, hidden_dim, num_layers, embed_dim)

    @property
    def reduced_grid(self) -> tuple[int, int]:
        return self.reducer.out_hw

    def normalize_features(self, feats: MultiLevelFeatures) -> torch.Tensor:
        return self.normalizer.normalize(feats.features)

    def pooled_target(self, feats: MultiLevelFeatures) -> torch.Tensor:
        """Fixed reconstruction target: pool(normalize(raw)) at 12×12.

        [B,T,576,L,D] -> [B,T,144,L,D]. Fixed (frozen encoder + fixed pool +
        fixed normalizer) so the autoencoder has a real, non-circular signal.
        """
        return self.reducer(self.normalizer.normalize(feats.features))

    def encode(self, feats: MultiLevelFeatures) -> LatentGrid:
        # pool FIRST (the lossy reduction), then fuse + compress at 144 tokens.
        z_pooled = self.pooled_target(feats)  # [B,T,144,L,D]
        h = self.fusion(z_pooled)  # [B,T,144,hidden]
        r = self.enc(h)  # [B,T,144,latent_dim]
        t_tok = feats.token_grid[0]
        rh, rw = self.reduced_grid
        return LatentGrid(latent=r, grid=(t_tok, rh, rw))

    def decode(self, latent: LatentGrid) -> torch.Tensor:
        """Reconstruct the POOLED normalized features [B, T, 144, L, D]."""
        return self.dec(latent.latent)

    def forward(self, feats: MultiLevelFeatures) -> tuple[LatentGrid, torch.Tensor]:
        """Return (latent, reconstruction-of-pooled-features)."""
        latent = self.encode(feats)
        recon = self.decode(latent)
        return latent, recon

    def freeze(self) -> None:
        """Freeze the VJ-RAE after training (CLAUDE.md §2.2 / doc §4.1)."""
        self.eval()
        self.requires_grad_(False)


# ---------------------------------------------------------------------------
# Action-discriminability probe (TRAINING-ONLY; discarded after codec training)
# ---------------------------------------------------------------------------


class ActionDiscriminabilityProbe(nn.Module):
    """g_φ( C(z_{t+1}) - C(z_t) ) -> a_t  (doc §2.3).

    A temporary probe that checks the compressed latent *difference* still
    linearly/MLP-decodes the action. It is attached only during codec training
    and **thrown away** afterwards (never part of the frozen VJ-RAE or the DiT).
    Acceptance (server): post-codec Δr->a accuracy drop vs pre-codec <= threshold.
    """

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, latent_delta: torch.Tensor) -> torch.Tensor:
        """latent_delta: [..., latent_dim] (spatially pooled C(z_{t+1})-C(z_t))."""
        return self.net(latent_delta)


def pooled_latent_delta(latent: torch.Tensor) -> torch.Tensor:
    """Mean-pool the spatial grid and difference consecutive time tokens.

    latent [B, T, N, C] -> delta [B, T-1, C]. Used to feed the probe.
    """
    pooled = latent.mean(dim=2)  # [B, T, C]
    return pooled[:, 1:] - pooled[:, :-1]  # [B, T-1, C]


# Preferred new name; kept here as an alias to avoid a disruptive class move.
VJEPRepresentationAutoencoder = VJEPALatentCodec
