"""3D-RoPE remapped to our ``(chunk-time, 12, 12)`` token layout (M4).

Ported — re-derived, not copied — from Wan2.1/2.2 + DreamZero
(``wan2_1_submodule.rope_params_polar`` / ``CausalWanModel._create_freqs`` /
``causal_rope_action_apply_polar``), NVIDIA Source Code License-NC, academic
non-commercial.

WHY THIS FILE IS DANGEROUS (CLAUDE.md §3 / user M4 point 1)
----------------------------------------------------------
M3 used *learned* modality/chunk/spatial embeddings, which do not care about token
order. RoPE DOES: it rotates q/k by a per-token angle derived from the token's
``(t, h, w)`` coordinate, so the coordinate we hand each token MUST line up with
the packing order in ``latent_tokenizer.pack`` and the mask in
``attention_mask.py``. A mismatch silently corrupts position (no error, wrong
attention geometry).

Coordinate-order contract (verified against both sides — see M4_NOTES):
  * Wan ``_create_freqs`` builds the grid as
    ``cat([time.expand(f,h,w), h.expand(f,h,w), w.expand(f,h,w)], -1)
        .reshape(f*h*w, ...)`` — i.e. **time-major, then height, then width
    (``w`` fastest)**, row-major over the 2-D grid.
  * Our tokenizer packs the latent/context blocks as
    ``[B, T, N, H] -> [B, T*N, H]`` with ``N = grid_h * grid_w`` already in
    **row-major (h outer, w inner)** order (the codec ``TokenReducer`` reshapes
    ``[B,T,h,w,...]``). So per-token order is ``(t, h, w)`` with ``w`` fastest —
    IDENTICAL to Wan. We therefore reuse Wan's grid construction unchanged.

Frequency split (Wan, head_dim ``d``): time gets ``d - 4*(d//6)`` dims, height and
width get ``2*(d//6)`` each (complex halves sum to ``d/2``). Context ``C_k`` and
future ``Z_k`` at the same chunk share the same time index, so their relative
rotation reflects ``|k-j|`` (teacher-forcing geometry). Action/value tokens get a
**1-D** RoPE over the chunk-time index (Wan gives the action register its own 1-D
freqs), concatenated in packing order after the C/Z grid freqs.
"""

from __future__ import annotations

import torch

from models.attention_mask import (
    ACTION,
    CONTEXT,
    LATENT,
    STATE,
    VALUE,
    TokenLayout,
)


def rope_freqs_1d(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Complex RoPE multipliers ``[max_seq_len, dim//2]`` (Wan ``rope_params_polar``)."""
    if dim % 2 != 0:
        raise ValueError("rope dim must be even")
    # Rope3D is intentionally not an nn.Module, so these plain tensors are not
    # visited by FSDP's meta-device param_init_fn. Keep the tiny immutable tables
    # on CPU even when the parent model is constructed under torch.device("meta");
    # Rope3D.to() moves them to the rank-local CUDA device before use.
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float64, device="cpu"),
        1.0
        / torch.pow(
            theta,
            torch.arange(0, dim, 2, dtype=torch.float64, device="cpu") / dim,
        ),
    )
    return torch.polar(torch.ones_like(freqs), freqs)  # complex128 [L, dim/2]


class Rope3D:
    """Precomputes the 3-D grid freqs + 1-D action/value freqs and assembles a
    per-token complex freq table for a packed ``[C|Z|A|V]`` sequence.

    Not an ``nn.Module`` — it holds no learnable params (RoPE is parameter-free);
    the tables are plain tensors created on demand / moved with ``.to``.
    """

    def __init__(self, head_dim: int, max_pos: int = 1024, theta: float = 10000.0) -> None:
        self.head_dim = head_dim
        d = head_dim
        # Wan split: time = d - 4*(d//6), h = 2*(d//6), w = 2*(d//6)
        self.dim_t = d - 4 * (d // 6)
        self.dim_h = 2 * (d // 6)
        self.dim_w = 2 * (d // 6)
        # each per-axis dim must be even (rope_freqs_1d needs dim%2==0); Wan only
        # requires head_dim even, which guarantees this.
        if d % 2 != 0 or any(p % 2 for p in (self.dim_t, self.dim_h, self.dim_w)):
            raise ValueError(f"head_dim {head_dim} incompatible with 3-D RoPE split")
        assert (self.dim_t + self.dim_h + self.dim_w) == d
        self.freqs_t = rope_freqs_1d(max_pos, self.dim_t, theta)  # [P, dim_t/2]
        self.freqs_h = rope_freqs_1d(max_pos, self.dim_h, theta)
        self.freqs_w = rope_freqs_1d(max_pos, self.dim_w, theta)
        # 1-D freqs (full head_dim) for the action / value register tokens
        self.freqs_1d = rope_freqs_1d(max_pos, d, theta)            # [P, d/2]

    def to(self, device) -> "Rope3D":
        self.freqs_t = self.freqs_t.to(device)
        self.freqs_h = self.freqs_h.to(device)
        self.freqs_w = self.freqs_w.to(device)
        self.freqs_1d = self.freqs_1d.to(device)
        return self

    # -- grid freqs (Wan _create_freqs, (t,h,w) row-major, w fastest) ----
    def grid_freqs(self, chunks: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Freqs for a block of ``len(chunks)`` chunks over an ``h*w`` grid.

        ``chunks`` is the per-chunk TIME index (so ``C_k`` and ``Z_k`` at the same
        chunk index get the same time rotation). Returns complex
        ``[len(chunks)*h*w, d/2]`` in ``(t, h, w)`` row-major order.
        """
        f = chunks.shape[0]
        device = chunks.device
        t_part = self.freqs_t[chunks].view(f, 1, 1, -1).expand(f, h, w, -1)
        h_part = self.freqs_h[:h].view(1, h, 1, -1).expand(f, h, w, -1).to(device)
        w_part = self.freqs_w[:w].view(1, 1, w, -1).expand(f, h, w, -1).to(device)
        return torch.cat([t_part, h_part, w_part], dim=-1).reshape(f * h * w, -1)

    def reg_freqs(self, positions: torch.Tensor) -> torch.Tensor:
        """1-D freqs (full head_dim) for register (action/value) tokens at the
        given integer ``positions`` (chunk-time index). Returns ``[len, d/2]``."""
        return self.freqs_1d[positions]

    # -- assemble per-token freqs in the packed [C|Z|A|V] order ----------
    def assemble(self, layout: TokenLayout, grid_hw: tuple[int, int]) -> torch.Tensor:
        """Build the ``[S, head_dim/2]`` complex freq table matching the packed
        sequence order. Relies on the tokenizer's modality-contiguous packing
        (``[C | Z | A | V]``) so we can rebuild each block's coordinates from the
        layout's ``chunk_idx`` runs without re-deriving the spatial index.
        """
        gh, gw = grid_hw
        n = gh * gw
        mod = layout.modality
        chunk = layout.chunk_idx
        device = chunk.device
        parts = []

        # C block: contiguous CONTEXT tokens; their chunk ids are 0..T_ctx-1 each
        # repeated n times -> the unique chunks in first-seen order.
        ctx_mask = mod == CONTEXT
        if ctx_mask.any():
            ctx_chunks = chunk[ctx_mask].view(-1, n)[:, 0]  # [T_ctx]
            parts.append(self.grid_freqs(ctx_chunks, gh, gw))

        # Z block
        z_mask = mod == LATENT
        if z_mask.any():
            z_chunks = chunk[z_mask].view(-1, n)[:, 0]       # [T_fut]
            parts.append(self.grid_freqs(z_chunks, gh, gw))

        # A block: 1-D RoPE with a UNIQUE position per action token, enumerated in
        # packing order (chunk-major, step-minor) — i.e. position = i*n_act + step,
        # which is exactly DreamZero's per-block freqs_action slice. This keeps
        # within-chunk step identity AND monotonic cross-chunk temporal order (a
        # later chunk's actions get higher positions). Mirrors Wan giving the
        # action register its own 1-D freqs, concatenated after the video grid.
        a_mask = mod == ACTION
        if a_mask.any():
            parts.append(self.reg_freqs(torch.arange(int(a_mask.sum()), device=device)))

        # V block: one value query per future chunk -> position = future ordinal.
        v_mask = mod == VALUE
        if v_mask.any():
            parts.append(self.reg_freqs(torch.arange(int(v_mask.sum()), device=device)))

        # S block: proprio register -> 1-D RoPE at its chunk-time index (it sits at
        # the current chunk; Wan gives the state register its own ``freqs_state``).
        s_mask = mod == STATE
        if s_mask.any():
            parts.append(self.reg_freqs(chunk[s_mask].to(device)))

        return torch.cat(parts, dim=0).to(device)  # [S, d/2] complex


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` [B, S, n_heads, head_dim] by per-token complex ``freqs`` [S, d/2].

    Mirrors Wan ``rope_apply_polar``: view as complex, multiply, view as real.
    Computed in the input dtype's complex counterpart; freqs broadcast over batch
    and heads.
    """
    b, s, n, d = x.shape
    xc = torch.view_as_complex(x.float().reshape(b, s, n, d // 2, 2))
    fr = freqs.to(xc.dtype).view(1, s, 1, d // 2)
    out = torch.view_as_real(xc * fr).reshape(b, s, n, d)
    return out.to(x.dtype)
