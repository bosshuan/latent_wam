"""Wan2.2-5B-backed Latent-World-Action DiT (Milestone 4).

Same Stage-A wiring as M3, but the tiny stand-in backbone is replaced by the real
Wan2.2-TI2V-5B trunk (``WanBackbone``, weights loaded via ``weight_loading.py``):

  * **input**: Wan VAE patch-embed dropped -> ``VJEPALatentInputAdapter`` (384-d
    codec token, from scratch); **output**: Wan pixel head dropped ->
    ``VJEPALatentFlowHead`` (from scratch). Neither Wan VAE/pixel weight is loaded.
  * **position**: 3-D RoPE remapped to ``(chunk-time, 12, 12)`` (``rope.py``),
    coordinate order verified identical to the tokenizer pack order — replaces
    M3's learned chunk/spatial embeds (tokenizer ``positional=False``); only the
    modality embedding stays additive.
  * **attention**: Wan's bidirectional video attention -> our chunk-causal mask
    (M3's 5 rules); only the mask changes, Q/K/V/O weights still load.
  * **text**: frozen umT5-XXL (``text_encoder.py``) -> Wan ``text_embedding`` ->
    Wan text cross-attention (loaded). CFG ``text_cfg_dropout`` + null-text.
  * **action**: GR00T multi-embodiment branch (from scratch), structurally omitted
    for video (CLAUDE.md §2.3); per-embodiment proprio state token via the state
    adapter (appended to the cross-attn context).

Geometry comes from ``WanConfig`` (read from the official config — never hardcoded).
M4 tests use tiny dims + mock weight loading on CPU; the real 5B load is server-side.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from flow.interpolation import predict_x1
from models.adapters.action import MultiEmbodimentActionEncoder
from models.adapters.latent import VJEPALatentInputAdapter
from models.adapters.state import StateAdapter
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
from models.rope import Rope3D
from models.text_encoder import FrozenUMT5TextEncoder
from models.wan_blocks import WanBackbone


class WanLatentWorldActionDiT(nn.Module):
    def __init__(
        self,
        cfg,                       # WanConfig (geometry read from official config)
        latent_dim: int = 384,
        action_dim: int = 7,
        num_embodiments: int = 4,
        grid_hw: tuple[int, int] = (12, 12),
        max_chunks: int = 16,
        max_actions: int = 8,
        state_dim: int = 0,
        value_bins: int = 1,
        text_seq_len: int = 8,
        action_token_scale: float = 1.0,
        action_latent_bridge_scale: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        dim = cfg.dim
        self.dim = dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.grid_hw = grid_hw
        self.action_token_scale = float(action_token_scale)
        self.action_latent_bridge_scale = float(action_latent_bridge_scale)
        grid_n = grid_hw[0] * grid_hw[1]
        self.grid_n = grid_n

        # from-scratch input/output (replace Wan VAE patch-embed / pixel head)
        self.latent_adapter = VJEPALatentInputAdapter(latent_dim, dim)
        self.latent_head = VJEPALatentFlowHead(dim, latent_dim)

        # from-scratch action branch + proprio state adapter
        self.action_encoder = MultiEmbodimentActionEncoder(action_dim, dim, num_embodiments)
        self.action_to_latent = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.action_head = ActionFlowHead(num_embodiments, dim, action_dim)
        self.state_adapter = StateAdapter(num_embodiments, max(state_dim, 1), dim)
        self.value_query = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.value_head = ValueHead(dim, value_bins)

        # packing (RoPE owns position -> positional=False), Wan trunk, RoPE, text
        self.tokenizer = LatentActionTokenizer(dim, max_chunks, grid_n, max_actions, positional=False)
        self.backbone = WanBackbone(cfg)
        self.rope = Rope3D(cfg.head_dim)
        self.text_encoder = FrozenUMT5TextEncoder(cfg.text_dim, seq_len=text_seq_len)

    # -- forward ---------------------------------------------------------
    def forward(
        self,
        context_latent: torch.Tensor,
        noisy_latent: torch.Tensor,
        latent_timestep: torch.Tensor,
        action_timestep: Optional[torch.Tensor] = None,
        noisy_action: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,
        embodiment_id: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        text: Optional[list[str]] = None,
        use_value: bool = False,
        cfg_dropout: float = 0.0,
        kv_cache: Optional[object] = None,
    ) -> WAMOutput:
        b = context_latent.shape[0]
        device = context_latent.device
        has_action = noisy_action is not None
        if kv_cache is not None and not getattr(kv_cache, "is_empty", lambda: True)():
            # Stage A is always a full teacher-forced forward; the incremental KV
            # path is M8 (closed-loop). Fail loud rather than silently ignore.
            from models.kv_cache import KVCacheNotImplemented

            raise KVCacheNotImplemented(
                "non-empty KV cache passed to a Stage-A forward — the incremental "
                "decode path is implemented in M8 (closed-loop), see kv_cache.py."
            )
        if has_action:
            self._assert_homogeneous_action(action_valid, embodiment_id)
            if action_timestep is None:
                raise ValueError("noisy_action provided without action_timestep")

        ctx_h = self.latent_adapter(context_latent)
        z_h = self.latent_adapter(noisy_latent)
        t_fut = noisy_latent.shape[1]

        a_h = None
        if has_action:
            n_act = noisy_action.shape[2]
            flat = noisy_action.reshape(b, t_fut * n_act, self.action_dim)
            a_h = self.action_encoder(flat, action_timestep, embodiment_id)
            a_h = (a_h * self.action_token_scale).reshape(b, t_fut, n_act, self.dim)
            if self.action_latent_bridge_scale != 0.0:
                a_summary = self.action_to_latent(a_h.mean(dim=2)).unsqueeze(2)
                z_h = z_h + self.action_latent_bridge_scale * a_summary

        v_h = self.value_query.to(device).expand(b, t_fut, 1, self.dim) if use_value else None

        # proprio -> in-sequence STATE register (read by Z/A/V of same-or-later
        # chunk; reads only itself; omitted for video) — DreamZero-faithful, NOT a
        # cross-attn condition (cross-attn stays pure text / all-embodiment-shared).
        # Gated on PROPRIO presence, NOT on has_action: a no-action forward (the
        # Δ_cond collapse monitor, doc §2.7) must keep proprio so Δ_cond isolates
        # the ACTION contribution — dropping state here would let strong proprio
        # mask an action collapse (false negative). Video has proprio=None (§2.3).
        state_h = None
        if proprio is not None:
            if embodiment_id is None or bool((embodiment_id < 0).any()):
                raise ValueError(
                    "proprio/state register requires a real embodiment_id (>=0); "
                    "video rows must pass proprio=None (CLAUDE.md §10/§2.3)."
                )
            state_h = self.state_adapter(proprio, embodiment_id)  # [B,1,dim]

        seq, layout, slices = self.tokenizer.pack(ctx_h, z_h, a_h, v_h, state_hidden=state_h)

        # 3-D RoPE freqs for the packed layout + chunk-causal additive mask
        self.rope.to(device)
        freqs = self.rope.assemble(layout, self.grid_hw)
        add_mask = to_additive(build_chunk_attention_mask(layout), dtype=seq.dtype).to(device)

        # per-token timestep (t_z on Z, t_a on A, 0 on C/V/S) for Wan AdaLN
        t_tok = self._token_timestep(layout, latent_timestep, action_timestep, b)

        # text cross-attention context: frozen umT5 -> Wan text_embedding (text ONLY)
        texts = text if text is not None else [""] * b
        text_emb = self.text_encoder.encode_with_cfg(
            texts, training=self.training, p=cfg_dropout, device=device
        )
        context = self.backbone.text_embedding(text_emb)  # [B, L, dim]

        x = self.backbone(seq, t_tok, freqs, add_mask, context)

        parts = self.tokenizer.unpack(x, slices)
        z_hidden = parts["latent"]
        latent_velocity = self.latent_head(z_hidden, latent_timestep)

        action_velocity = None
        if has_action:
            a_hidden = parts["action"]
            bb, tf, na, hh = a_hidden.shape
            av = self.action_head(a_hidden.reshape(bb, tf * na, hh), embodiment_id)
            action_velocity = av.reshape(bb, tf, na, self.action_dim)

        value = self.value_head(parts["value"].squeeze(2)) if use_value else None

        return WAMOutput(
            latent_velocity=latent_velocity,
            action_velocity=action_velocity,
            value=value,
            latent_hidden=z_hidden,
        )

    def predict_clean_latent(self, out: WAMOutput, noisy_latent: torch.Tensor, t_z: torch.Tensor):
        return predict_x1(noisy_latent, out.latent_velocity, t_z)

    # -- helpers ---------------------------------------------------------
    def _assert_homogeneous_action(self, action_valid, embodiment_id) -> None:
        if action_valid is not None and not bool(action_valid.all()):
            raise ValueError(
                "noisy_action provided for a non-homogeneous batch — split video/"
                "robot rows upstream (structural Ak omission, CLAUDE.md §2.3)."
            )
        if embodiment_id is None:
            raise ValueError("action branch requires embodiment_id")
        if bool((embodiment_id < 0).any()):
            raise ValueError("action branch got an INVALID embodiment_id (<0) (CLAUDE.md §10).")

    def _token_timestep(self, layout, t_z, t_a, b) -> torch.Tensor:
        device = t_z.device
        s = layout.seq_len
        t_tok = torch.zeros(b, s, device=device, dtype=t_z.dtype)
        t_tok[:, (layout.modality == LATENT).to(device)] = t_z.unsqueeze(1)
        if t_a is not None:
            t_tok[:, (layout.modality == ACTION).to(device)] = t_a.unsqueeze(1)
        return t_tok
