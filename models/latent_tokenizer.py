"""Token packing for the unified DiT (doc §2.2 ``[C, Z, A, V]`` organization).

Takes the *already projected* hidden tokens for each modality and packs them into
one sequence with modality + chunk(time) + spatial position embeddings, and emits
the :class:`~models.attention_mask.TokenLayout` the mask/RoPE consume.

Key contract (CLAUDE.md §2.3): ``action_hidden`` is ``None`` for an actionless
(video) batch — the ``A`` tokens are then **structurally absent** from the
sequence (not built-then-masked). The same holds for the value tokens when the
value stub is disabled.

Time/chunk indexing
-------------------
``context_hidden`` carries clean history latents (``T_ctx`` chunks, indices
``0..T_ctx-1``). The noisy future starts immediately after that history, so
future chunk ``i`` has global index ``T_ctx + i``. This matches the real cached
robot path, where ``latent[:, :history]`` is context and ``latent[:, history:]``
is the future flow target. The attention mask therefore lets future ``Z/A`` read
all clean history context while still forbidding any same-chunk clean target leak
(there is no future clean ``C_k`` in the packed sequence).

Packing order is modality-contiguous ``[C | Z | A | V]`` so unpacking is a plain
slice+reshape; attention order is irrelevant since the mask is index-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from models.attention_mask import (
    ACTION,
    CONTEXT,
    LATENT,
    NUM_MODALITIES,
    STATE,
    VALUE,
    TokenLayout,
)


@dataclass
class PackSlices:
    """Offsets/shapes to unpack the packed sequence back to per-modality grids."""

    t_ctx: int
    t_fut: int
    grid_n: int
    n_act: int
    use_value: bool
    ctx_len: int
    z_len: int
    a_len: int
    v_len: int
    s_len: int = 0  # state register length (0 = video / no proprio)

    @property
    def z_start(self) -> int:
        return self.ctx_len

    @property
    def a_start(self) -> int:
        return self.ctx_len + self.z_len

    @property
    def v_start(self) -> int:
        return self.ctx_len + self.z_len + self.a_len

    @property
    def s_start(self) -> int:
        return self.ctx_len + self.z_len + self.a_len + self.v_len


class LatentActionTokenizer(nn.Module):
    """Pack/unpack ``[C,Z,A,V,S]`` + modality/chunk/spatial embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        max_chunks: int,
        grid_n: int,
        max_actions: int,
        positional: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_n = grid_n
        # ``positional=False`` (M4) hands chunk/spatial/action position to 3-D RoPE
        # inside attention; only the modality embedding stays additive (RoPE has no
        # modality concept). M3 keeps learned positions (positional=True).
        self.positional = positional
        self.modality_embed = nn.Embedding(NUM_MODALITIES, hidden_dim)
        self.chunk_embed = nn.Embedding(max_chunks, hidden_dim)
        self.spatial_embed = nn.Embedding(grid_n, hidden_dim)
        self.action_pos_embed = nn.Embedding(max_actions, hidden_dim)

    # -- helpers ---------------------------------------------------------
    def _mod(self, code: int, device) -> torch.Tensor:
        return self.modality_embed(torch.tensor(code, device=device))

    def pack(
        self,
        context_hidden: torch.Tensor,          # [B, T_ctx, N, H]
        latent_hidden: torch.Tensor,           # [B, T_fut, N, H]
        action_hidden: Optional[torch.Tensor], # [B, T_fut, n_act, H] or None
        value_hidden: Optional[torch.Tensor],  # [B, T_fut, 1, H] or None
        state_hidden: Optional[torch.Tensor] = None,  # [B, 1, H] proprio register
    ) -> tuple[torch.Tensor, TokenLayout, PackSlices]:
        b, t_ctx, n, h = context_hidden.shape
        t_fut = latent_hidden.shape[1]
        device = context_hidden.device
        if n != self.grid_n:
            raise ValueError(f"grid_n mismatch: tokenizer {self.grid_n} vs input {n}")
        if self.positional and t_ctx + t_fut > self.chunk_embed.num_embeddings:
            raise ValueError(
                f"context+future chunks {t_ctx + t_fut} exceed tokenizer max_chunks "
                f"{self.chunk_embed.num_embeddings}"
            )

        fut_global = torch.arange(t_ctx, t_ctx + t_fut, device=device)  # [T_fut]
        pos = self.positional
        z0 = torch.zeros((), device=device)
        spatial = self.spatial_embed(torch.arange(n, device=device)).view(1, 1, n, h) if pos else z0

        # --- context block [B, T_ctx*N, H] ---
        ctx_chunks = (
            self.chunk_embed(torch.arange(t_ctx, device=device)).view(1, t_ctx, 1, h) if pos else z0
        )
        ctx = (context_hidden + self._mod(CONTEXT, device) + ctx_chunks + spatial).reshape(
            b, t_ctx * n, h
        )

        # --- latent (future Z) block [B, T_fut*N, H] ---
        fut_chunks = self.chunk_embed(fut_global).view(1, t_fut, 1, h) if pos else z0
        z = (latent_hidden + self._mod(LATENT, device) + fut_chunks + spatial).reshape(
            b, t_fut * n, h
        )

        blocks = [ctx, z]
        n_act = 0
        a_len = 0
        if action_hidden is not None:
            n_act = action_hidden.shape[2]
            apos = (
                self.action_pos_embed(torch.arange(n_act, device=device)).view(1, 1, n_act, h)
                if pos
                else z0
            )
            a = (action_hidden + self._mod(ACTION, device) + fut_chunks + apos).reshape(
                b, t_fut * n_act, h
            )
            blocks.append(a)
            a_len = t_fut * n_act

        use_value = value_hidden is not None
        v_len = 0
        if use_value:
            v = (value_hidden + self._mod(VALUE, device) + fut_chunks).reshape(
                b, t_fut * value_hidden.shape[2], h
            )
            blocks.append(v)
            v_len = t_fut * value_hidden.shape[2]

        # --- state register block [B, 1, H] ---  (proprio q_l; robot only)
        # Placed LAST so the z/a/v offsets are unchanged; lives at the first
        # future chunk so Z/A of every future chunk can read it causally.
        s_len = 0
        cur_chunk = t_ctx  # current time l = first future chunk
        if state_hidden is not None:
            s_tok = state_hidden + self._mod(STATE, device)  # [B,1,H]
            blocks.append(s_tok)
            s_len = state_hidden.shape[1]

        seq = torch.cat(blocks, dim=1)  # [B, S, H]

        layout = self._build_layout(
            t_ctx, t_fut, n, n_act, use_value, s_len, cur_chunk, fut_global, device
        )
        slices = PackSlices(
            t_ctx=t_ctx, t_fut=t_fut, grid_n=n, n_act=n_act, use_value=use_value,
            ctx_len=t_ctx * n, z_len=t_fut * n, a_len=a_len, v_len=v_len, s_len=s_len,
        )
        return seq, layout, slices

    def _build_layout(
        self, t_ctx, t_fut, n, n_act, use_value, s_len, cur_chunk, fut_global, device
    ) -> TokenLayout:
        chunk_parts = []
        mod_parts = []

        # context: chunk j repeated N times, modality CONTEXT
        ctx_chunk = torch.arange(t_ctx, device=device).repeat_interleave(n)
        chunk_parts.append(ctx_chunk)
        mod_parts.append(torch.full((t_ctx * n,), CONTEXT, device=device, dtype=torch.long))

        # latent: future global chunk repeated N times, modality LATENT
        chunk_parts.append(fut_global.repeat_interleave(n))
        mod_parts.append(torch.full((t_fut * n,), LATENT, device=device, dtype=torch.long))

        if n_act > 0:
            chunk_parts.append(fut_global.repeat_interleave(n_act))
            mod_parts.append(torch.full((t_fut * n_act,), ACTION, device=device, dtype=torch.long))

        if use_value:
            chunk_parts.append(fut_global.repeat_interleave(1))
            mod_parts.append(torch.full((t_fut,), VALUE, device=device, dtype=torch.long))

        if s_len > 0:
            # state register at the current (first future) chunk
            chunk_parts.append(torch.full((s_len,), cur_chunk, device=device, dtype=torch.long))
            mod_parts.append(torch.full((s_len,), STATE, device=device, dtype=torch.long))

        return TokenLayout(
            chunk_idx=torch.cat(chunk_parts).long(),
            modality=torch.cat(mod_parts).long(),
        )

    # -- unpack ----------------------------------------------------------
    def unpack(
        self, seq: torch.Tensor, slices: PackSlices
    ) -> dict[str, Optional[torch.Tensor]]:
        b, _, h = seq.shape
        z = seq[:, slices.z_start : slices.z_start + slices.z_len].reshape(
            b, slices.t_fut, slices.grid_n, h
        )
        out: dict[str, Optional[torch.Tensor]] = {"latent": z, "action": None, "value": None}
        if slices.a_len > 0:
            out["action"] = seq[:, slices.a_start : slices.a_start + slices.a_len].reshape(
                b, slices.t_fut, slices.n_act, h
            )
        if slices.use_value:
            out["value"] = seq[:, slices.v_start : slices.v_start + slices.v_len].reshape(
                b, slices.t_fut, slices.v_len // slices.t_fut, h
            )
        return out
