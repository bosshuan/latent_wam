"""Wan2.2 DiT blocks — re-derived (not copied) from Wan2.1/2.2 + DreamZero
``CausalWanAttentionBlock`` (NVIDIA SCL-NC, academic non-commercial).

The submodule names MIRROR the official Wan checkpoint
(``blocks.{i}.self_attn.{q,k,v,o,norm_q,norm_k}``, ``blocks.{i}.cross_attn.*``,
``blocks.{i}.{norm1,norm2,norm3,ffn.0,ffn.2,modulation}``, ``time_embedding.*``,
``time_projection.*``, ``text_embedding.*``) so ``weight_loading.py`` maps Wan keys
to our modules without renaming (CLAUDE.md §3, user M4 point 2/3).

Two project-specific changes (mask only — Q/K/V/O weights still load, user M4
point 3):
  * **bidirectional -> chunk-causal**: the self-attention consumes the M3
    ``attention_mask`` additive mask (the 5 rules), not Wan's blockwise pattern;
  * **3D-RoPE remap**: q/k are rotated by ``rope.apply_rope`` with freqs assembled
    for our ``(chunk-time, 12, 12)`` layout (token order verified == Wan's).

Cross-attention here is **text-only** (t2v style): q/k/v/o + norm_q/norm_k. The 5B
ti2v checkpoint's image cross-attn (``cross_attn.{k_img,v_img,norm_k_img}``) is
intentionally NOT loaded (we condition on V-JEPA latent context in the self-attn
sequence, not on a CLIP image token) — ``weight_loading`` records it as dropped.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import apply_rope


class WanRMSNorm(nn.Module):
    """RMSNorm over the last dim (Wan ``norm_q``/``norm_k``)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight


class WanLayerNorm(nn.LayerNorm):
    """Wan layer norm (affine optional; matches the checkpoint's norm1/2/3)."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Wan timestep sinusoid: position [...] -> [..., dim] (half cos, half sin)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=position.device, dtype=torch.float32) / half
    )
    args = position.float().unsqueeze(-1) * freqs
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class WanSelfAttention(nn.Module):
    """Masked self-attention + 3D-RoPE (names: q,k,v,o,norm_q,norm_k)."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(
        self, x: torch.Tensor, freqs: torch.Tensor, additive_mask: torch.Tensor
    ) -> torch.Tensor:
        b, s, _ = x.shape
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        # [B,n,S,d]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d) + additive_mask
        out = attn.softmax(dim=-1) @ v
        out = out.transpose(1, 2).reshape(b, s, n * d)
        return self.o(out)


class WanT2VCrossAttention(nn.Module):
    """Text cross-attention (names: q,k,v,o,norm_q,norm_k)."""

    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        lc = context.shape[1]
        n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d).transpose(1, 2)
        k = self.norm_k(self.k(context)).view(b, lc, n, d).transpose(1, 2)
        v = self.v(context).view(b, lc, n, d).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, s, n * d)
        return self.o(out)


class WanAttentionBlock(nn.Module):
    """norm1 -> self_attn -> norm3 -> cross_attn -> norm2 -> ffn, with the 6-way
    AdaLN modulation from the per-token timestep embedding (Wan)."""

    def __init__(
        self, dim: int, ffn_dim: int, num_heads: int, qk_norm: bool, cross_attn_norm: bool, eps: float
    ) -> None:
        super().__init__()
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,            # [B, S, 6, dim] per-token timestep modulation
        freqs: torch.Tensor,
        additive_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        # e + per-block modulation, split into 6 [B,S,1,dim] -> squeeze
        parts = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        sh_a, sc_a, g_a, sh_f, sc_f, g_f = (p.squeeze(2) for p in parts)
        y = self.self_attn(self.norm1(x) * (1 + sc_a) + sh_a, freqs, additive_mask)
        x = x + y * g_a
        x = x + self.cross_attn(self.norm3(x), context)
        y = self.ffn(self.norm2(x) * (1 + sc_f) + sh_f)
        x = x + y * g_f
        return x


class WanBackbone(nn.Module):
    """Wan DiT trunk: time/text embeddings + the stack of attention blocks.

    Holds exactly the modules whose Wan weights we LOAD. The VAE patch-embed and
    pixel head live OUTSIDE (and are never loaded): the input/output adapters are
    our from-scratch V-JEPA-latent modules in the parent DiT.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.freq_dim = cfg.freq_dim
        # text projection (umT5 text_dim -> dim) — Wan ``text_embedding``
        self.text_embedding = nn.Sequential(
            nn.Linear(cfg.text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )
        # timestep MLPs — Wan ``time_embedding`` + ``time_projection``
        self.time_embedding = nn.Sequential(nn.Linear(cfg.freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(dim, cfg.ffn_dim, cfg.num_heads, cfg.qk_norm, cfg.cross_attn_norm, cfg.eps)
                for _ in range(cfg.num_layers)
            ]
        )

    def timestep_modulation(self, t_tok: torch.Tensor) -> torch.Tensor:
        """Per-token timestep [B,S] -> modulation [B,S,6,dim]."""
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t_tok))  # [B,S,dim]
        return self.time_projection(e).unflatten(-1, (6, self.cfg.dim))         # [B,S,6,dim]

    def forward(
        self,
        x: torch.Tensor,
        t_tok: torch.Tensor,
        freqs: torch.Tensor,
        additive_mask: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        e = self.timestep_modulation(t_tok)
        for blk in self.blocks:
            x = blk(x, e, freqs, additive_mask, context)
        return x
