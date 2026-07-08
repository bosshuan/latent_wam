"""Tiny CPU stand-in for the V-JEPA gigantic backbone (tests only).

Mirrors just the attributes/forward contract ``FrozenVJEPAEncoder`` reads:
  * dim attrs: ``embed_dim, num_heads, patch_size, tubelet_size, img_height,
    img_width`` and ``blocks`` (len => depth).
  * layer registry: ``hierarchical_layers``, ``out_layers_distillation`` (the
    encoder overrides the latter from config), ``norms_block``.
  * ``return_hierarchical`` flag; forward returns the per-layer-LayerNorm'd
    concat ``[B, S, L*embed_dim]`` when set, deterministically.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Large stride so (t * _TAG_TIME + layer) is unambiguous for small layer counts.
_TAG_TIME = 100


class TinyVJEPABackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 16,
        depth: int = 4,
        num_heads: int = 2,
        patch_size: int = 16,
        tubelet_size: int = 2,
        img_size: int = 384,
        hierarchical_layers: tuple[int, ...] = (1, 3),
        tagged: bool = False,
    ) -> None:
        super().__init__()
        # When ``tagged``, forward returns a deterministic tensor whose values
        # encode (time_index, layer_index) so a test can verify the encoder's
        # reshape places each (t, layer) block where the real layout dictates
        # (time-major sequence, layer-major channels) — content, not just shape.
        self.tagged = tagged
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.img_height = img_size
        self.img_width = img_size
        # only len() is read for depth
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(depth)])
        self.hierarchical_layers = list(hierarchical_layers)
        self.out_layers_distillation = list(hierarchical_layers)
        self.return_hierarchical = False

        # deterministic patch embed (tubelet x patch x patch) -> embed_dim
        torch.manual_seed(0)
        self.patch_embed = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.norms_block = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(len(self.hierarchical_layers))]
        )

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        # vid: [B, 3, T, H, W]
        b = vid.shape[0]
        t_tok = vid.shape[2] // self.tubelet_size
        gh = vid.shape[3] // self.patch_size
        gw = vid.shape[4] // self.patch_size
        n = gh * gw
        e = self.embed_dim

        if self.tagged:
            # value at (time t, layer i) = t * TAG_TIME + i, broadcast over the
            # N spatial tokens and the D channels of that layer's block. Mirrors
            # the real layout: sequence time-major (t outer), channels
            # layer-major (i outer). Always hierarchical.
            n_layers = len(self.out_layers_distillation)
            out = torch.zeros(b, t_tok * n, n_layers * e)
            for t in range(t_tok):
                for i in range(n_layers):
                    out[:, t * n : (t + 1) * n, i * e : (i + 1) * e] = (
                        t * _TAG_TIME + i
                    )
            return out

        x = self.patch_embed(vid)  # [B, embed, T_tok, gh, gw]
        b, e, t_tok, gh, gw = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, T_tok*gh*gw, embed]
        if self.return_hierarchical:
            outs = [
                self.norms_block[i](tokens)
                for i in range(len(self.out_layers_distillation))
            ]
            return torch.cat(outs, dim=-1)  # [B, S, L*embed]
        return self.norms_block[-1](tokens)


def make_tiny_encoder(extract_layers=(1, 3), tagged=False, **kwargs):
    """Build a FrozenVJEPAEncoder over a tiny mock (no download)."""
    from models.vjepa_encoder import FrozenVJEPAEncoder

    backbone = TinyVJEPABackbone(
        hierarchical_layers=tuple(extract_layers), tagged=tagged, **kwargs
    )
    return FrozenVJEPAEncoder(
        hub_name="tiny-mock",
        extract_layers=tuple(extract_layers),
        backbone=backbone,
        assert_gigantic=False,
    )
