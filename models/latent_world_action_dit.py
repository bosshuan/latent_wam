"""Tiny unified Latent-World-Action DiT (Milestone 3).

A *small, CPU-runnable* stand-in for the Wan2.2-TI2V-5B backbone (which is loaded
in M4). It exercises the full Stage-A wiring so the invariants can be unit-tested
without the 5B weights:

  * codec-latent input adapter + clean *context* / noisy *future* latents;
  * GR00T-style multi-embodiment action branch, **structurally omitted** for
    actionless video (CLAUDE.md §2.3 — never build-then-mask);
  * ``[C,Z,A,V]`` packing + chunk-causal attention (chunk-internal Z/A
    bidirectional, cross-chunk causal, value read-only sink);
  * latent / action flow heads (PROJECT velocity convention) + value stub;
  * a **counterfactual forward path** (re-forward with a permuted action under
    the SAME noise/timestep) for ``L_cf`` and the ``S_a`` monitor (§2.7).

Backbone blocks are DiT blocks with per-token timestep modulation (``t_z`` on Z
tokens, ``t_a`` on A tokens, clean tokens get t=0) + a text cross-attention stub.
The residual gates use a tiny warm start instead of exact AdaLN-Zero so the
smoke model can immediately exercise the action-token -> latent-token path;
Wan-backed training uses ``wan_blocks.py``. 3D-RoPE remapping is deferred to M4;
here position is a learned modality/chunk/spatial embedding
(``latent_tokenizer.py``).

Homogeneity: this tiny forward requires the action branch to be either fully
present (robot batch) or fully absent (video batch, ``noisy_action=None``). The
M5 sampler delivers homogeneous batches; a mixed batch must be split upstream so
the omission stays *structural* rather than a mask over fabricated rows.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow.interpolation import predict_x1
from models.adapters.action import (
    MultiEmbodimentActionEncoder,
    SinusoidalPositionalEncoding,
)
from models.adapters.condition import ConditionAdapter
from models.adapters.latent import VJEPALatentInputAdapter
from models.attention_mask import (
    ACTION,
    LATENT,
    build_chunk_attention_mask,
    to_additive,
)
from models.heads.action_flow import ActionFlowHead
from models.heads.latent_flow import VJEPALatentFlowHead
from models.heads.value import ValueHead
from models.latent_tokenizer import LatentActionTokenizer
from models.outputs import WAMOutput


# ---------------------------------------------------------------------------
# Backbone primitives
# ---------------------------------------------------------------------------


class _SelfAttention(nn.Module):
    """Masked multi-head self-attention (additive mask, no dropout -> the leakage
    test is exact)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv(x).reshape(b, s, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3,B,heads,S,hd]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,heads,S,S]
        attn = attn + additive_mask  # [S,S] broadcasts over (B, heads)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o(out)


class _CrossAttention(nn.Module):
    """Text/condition cross-attention (q from tokens, kv from condition tokens)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.o = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        lc = cond.shape[1]
        q = self.q(x).reshape(b, s, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(cond).reshape(b, lc, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, s, self.heads * self.head_dim)
        return self.o(out)


class _DiTBlock(nn.Module):
    """Tiny DiT block with per-token timestep modulation + text cross-attn.

    The final AdaLN linear is otherwise zero-initialized, but the attention/MLP
    gates may receive a small positive bias. Exact zero gates are stable, but in
    this tiny smoke model they also make ``S_a`` exactly zero at initialization,
    which hides whether action tokens can influence latent tokens at all.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 2.0,
        gate_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = _SelfAttention(dim, heads)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross = _CrossAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        # 6 per-token modulation signals from the conditioning embed:
        # shift/scale/gate for attention, then shift/scale/gate for MLP.
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)
        if gate_init != 0.0:
            with torch.no_grad():
                self.ada[-1].bias[2 * dim : 3 * dim].fill_(gate_init)
                self.ada[-1].bias[5 * dim : 6 * dim].fill_(gate_init)

    def forward(
        self,
        x: torch.Tensor,
        cond_embed: torch.Tensor,    # [B,S,H] per-token timestep embedding
        additive_mask: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        sh_msa, sc_msa, g_msa, sh_mlp, sc_mlp, g_mlp = self.ada(cond_embed).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc_msa) + sh_msa
        x = x + g_msa * self.attn(h, additive_mask)
        x = x + self.cross(self.norm_cross(x), text)
        h = self.norm2(x) * (1 + sc_mlp) + sh_mlp
        x = x + g_mlp * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# DiT
# ---------------------------------------------------------------------------


class LatentWorldActionDiT(nn.Module):
    def __init__(
        self,
        latent_dim: int = 384,
        action_dim: int = 7,
        hidden_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        num_embodiments: int = 4,
        grid_n: int = 144,
        max_chunks: int = 16,
        max_actions: int = 8,
        state_dim: int = 0,
        text_dim: int = 0,
        value_bins: int = 1,
        adaln_gate_init: float = 0.05,
        action_token_scale: float = 1.0,
        action_latent_bridge_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.grid_n = grid_n
        self.action_token_scale = float(action_token_scale)
        self.action_latent_bridge_scale = float(action_latent_bridge_scale)

        # input adapters
        self.latent_adapter = VJEPALatentInputAdapter(latent_dim, hidden_dim)
        self.action_encoder = MultiEmbodimentActionEncoder(action_dim, hidden_dim, num_embodiments)
        self.action_to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.condition = ConditionAdapter(hidden_dim, num_embodiments, max(state_dim, 1), text_dim)
        self.value_query = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

        # packing + backbone
        self.tokenizer = LatentActionTokenizer(hidden_dim, max_chunks, grid_n, max_actions)
        self.t_embed = SinusoidalPositionalEncoding(hidden_dim)
        self.blocks = nn.ModuleList(
            [_DiTBlock(hidden_dim, heads, gate_init=adaln_gate_init) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        # heads
        self.latent_head = VJEPALatentFlowHead(hidden_dim, latent_dim)
        self.action_head = ActionFlowHead(num_embodiments, hidden_dim, action_dim)
        self.value_head = ValueHead(hidden_dim, value_bins)

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        context_latent: torch.Tensor,            # [B, T_ctx, N, latent_dim]
        noisy_latent: torch.Tensor,              # [B, T_fut, N, latent_dim]
        latent_timestep: torch.Tensor,           # [B] t_z
        action_timestep: Optional[torch.Tensor] = None,  # [B] t_a (robot)
        noisy_action: Optional[torch.Tensor] = None,     # [B, T_fut, n_act, A] or None
        action_valid: Optional[torch.Tensor] = None,     # [B] bool
        embodiment_id: Optional[torch.Tensor] = None,    # [B] long
        proprio: Optional[torch.Tensor] = None,          # [B, state_dim]
        text_embedding: Optional[torch.Tensor] = None,   # [B, L_txt, text_dim]
        text: Optional[list[str]] = None,                # accepted for Wan-compatible train_step
        use_value: bool = False,
    ) -> WAMOutput:
        b = context_latent.shape[0]
        device = context_latent.device
        has_action = noisy_action is not None
        if has_action:
            self._assert_homogeneous_action(action_valid, embodiment_id)
            if action_timestep is None:
                raise ValueError("noisy_action provided without action_timestep")

        # --- project inputs to hidden ---
        ctx_h = self.latent_adapter(context_latent)   # [B,T_ctx,N,H]
        z_h = self.latent_adapter(noisy_latent)       # [B,T_fut,N,H]
        t_fut = noisy_latent.shape[1]

        a_h = None
        if has_action:
            n_act = noisy_action.shape[2]
            # encode each future chunk's action steps; flatten (T_fut, n_act) -> tokens
            flat = noisy_action.reshape(b, t_fut * n_act, self.action_dim)
            a_h = self.action_encoder(flat, action_timestep, embodiment_id) * self.action_token_scale
            a_h = a_h.reshape(b, t_fut, n_act, self.hidden_dim)
            if self.action_latent_bridge_scale != 0.0:
                # Explicit residual action conditioner: same-chunk action summary
                # is added to every future latent token before the joint DiT. This
                # keeps action-content visible despite hundreds of dense latent
                # keys in the chunk-causal attention context.
                a_summary = self.action_to_latent(a_h.mean(dim=2)).unsqueeze(2)
                z_h = z_h + self.action_latent_bridge_scale * a_summary

        v_h = None
        if use_value:
            v_h = self.value_query.to(device).expand(b, t_fut, 1, self.hidden_dim)

        # --- proprio -> in-sequence STATE register (DreamZero-faithful: read by
        # Z/A/V of same-or-later chunk, reads only itself, omitted for video). NOT
        # a cross-attn condition — cross-attn stays pure text (all-embodiment).
        # Gated on PROPRIO presence, NOT has_action, so a no-action Δ_cond monitor
        # forward keeps proprio and isolates the ACTION contribution (doc §2.7). ---
        state_h = None
        if proprio is not None:
            if embodiment_id is None or bool((embodiment_id < 0).any()):
                raise ValueError(
                    "proprio/state register requires a real embodiment_id (>=0); "
                    "video rows must pass proprio=None (CLAUDE.md §10/§2.3)."
                )
            state_h = self.condition.state_token(proprio, embodiment_id)

        # --- pack ---
        seq, layout, slices = self.tokenizer.pack(ctx_h, z_h, a_h, v_h, state_hidden=state_h)

        # --- per-token timestep conditioning embed ---
        cond_embed = self._token_timestep_embed(
            layout, latent_timestep, action_timestep, b
        ).to(dtype=seq.dtype)

        # --- condition (text shared cross-attn ONLY; proprio is in-sequence) ---
        cond_tokens = self.condition.text_tokens(text_embedding, b, device)

        # --- attention mask (additive) ---
        add_mask = to_additive(build_chunk_attention_mask(layout), dtype=seq.dtype).to(device)

        # --- backbone ---
        x = seq
        for blk in self.blocks:
            x = blk(x, cond_embed, add_mask, cond_tokens)
        x = self.final_norm(x)

        # --- unpack + heads ---
        parts = self.tokenizer.unpack(x, slices)
        z_hidden = parts["latent"]                       # [B,T_fut,N,H]
        latent_velocity = self.latent_head(z_hidden, latent_timestep)

        action_velocity = None
        if has_action:
            a_hidden = parts["action"]                   # [B,T_fut,n_act,H]
            bb, tf, na, hh = a_hidden.shape
            # CategorySpecific* uses bmm -> needs a [B, T, H] input; flatten the
            # (T_fut, n_act) token axes, decode per-embodiment, reshape back.
            av = self.action_head(a_hidden.reshape(bb, tf * na, hh), embodiment_id)
            action_velocity = av.reshape(bb, tf, na, self.action_dim)

        value = None
        if use_value:
            value = self.value_head(parts["value"].squeeze(2))  # [B,T_fut,bins]

        return WAMOutput(
            latent_velocity=latent_velocity,
            action_velocity=action_velocity,
            value=value,
            latent_hidden=z_hidden,
        )

    # -- convenience: clean-latent prediction r̂1 ------------------------
    def predict_clean_latent(self, out: WAMOutput, noisy_latent: torch.Tensor, t_z: torch.Tensor):
        """x̂1 = x_t + (1-t) v (PROJECT convention) on the future latent grid."""
        return predict_x1(noisy_latent, out.latent_velocity, t_z)

    # -- helpers ---------------------------------------------------------
    def _assert_homogeneous_action(self, action_valid, embodiment_id) -> None:
        if action_valid is not None and not bool(action_valid.all()):
            raise ValueError(
                "noisy_action provided for a non-homogeneous batch — split the "
                "video/robot rows upstream so action tokens are STRUCTURALLY "
                "omitted for video (CLAUDE.md §2.3), not masked over fabricated rows."
            )
        if embodiment_id is None:
            raise ValueError("action branch requires embodiment_id")
        if bool((embodiment_id < 0).any()):
            raise ValueError(
                "action branch got an INVALID/unspecified embodiment_id (<0); video "
                "rows must not reach the action adapter (CLAUDE.md §10)."
            )

    def _token_timestep_embed(self, layout, t_z, t_a, b) -> torch.Tensor:
        device = t_z.device
        s = layout.seq_len
        t_tok = torch.zeros(b, s, device=device, dtype=t_z.dtype)
        is_latent = (layout.modality == LATENT).to(device)
        t_tok[:, is_latent] = t_z.unsqueeze(1)
        if t_a is not None:
            is_action = (layout.modality == ACTION).to(device)
            t_tok[:, is_action] = t_a.unsqueeze(1)
        return self.t_embed(t_tok)  # [B,S,H]
