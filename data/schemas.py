"""Unified data containers for Stage A (Latent Flow-WAM).

This module defines the single batch type that flows through the whole Stage A
pipeline (``TrajectoryBatch``) plus the two encoder/codec tensor containers
(``MultiLevelFeatures``, ``LatentGrid``).

Hard invariants enforced/encoded here (CLAUDE.md §2):
  * Actionless video samples carry ``action_valid=0`` and **must not** have
    fabricated action tensors. ``actions``/``proprio`` are ``None`` for such
    samples; a missing-action mask ``m_a`` is derived from ``action_valid`` and
    never from random data (§2.3).
  * Action labels are padded inside an ``action_schema`` to that schema's max
    dim and tracked with ``action_pad_mask`` — never silently zero-filled into a
    global vector (§4 multi-embodiment rule).

M1 -> M3 ACTION-TOKEN CONTRACT (binding; read before touching the DiT)
---------------------------------------------------------------------
The preferred batching is *homogeneous*: the sampler produces either an
all-video batch (``actions=None``) or an all-robot batch (every row valid), so
no inert action rows exist. If a *mixed* batch ever occurs, the ``actions``
tensor here is a dense rectangle only for storage convenience; the zero rows for
``action_valid=0`` samples are NOT a representation the model may consume.

In M3 the DiT/tokenizer MUST **structurally omit the ``Ak`` action tokens** for
``action_valid=0`` rows (gather only the valid rows into the action branch) —
**not** build action tokens for every row and rely on masking ("build-then-mask"
is forbidden). Consequently a zero-filled action row must NEVER:
  * contribute to the action flow-matching loss (gated by ``m_a`` *and* omitted
    from the action token set), nor
  * enter the latent-prediction conditioning path (no pseudo-action ever
    conditions ``Zk``).
``action_pad_mask`` remains the *temporal/feature* pad mask for the genuinely
present robot rows; it is not a substitute for the structural omission above.

No torch ops run at import time; this is pure container plumbing so it imports
cheaply on CPU for the unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# Encoder / codec tensor containers
# ---------------------------------------------------------------------------


@dataclass
class MultiLevelFeatures:
    """Frozen V-JEPA dense features for a clip.

    ``features`` keeps the dense 2D token grid + time index *explicitly* — we
    never flatten the spatial grid away or whole-frame pool (CLAUDE.md §2.8).

    Shapes (B = batch, T_tok = time tokens = frames // tubelet,
    N = H*W spatial tokens, L = len(extract_layers), D = encoder embed_dim):
        features  : [B, T_tok, N, L, D]      (e.g. gigantic -> [B, T, 576, 4, 1664])
        token_grid: (T_tok, H, W)            (e.g. (T, 24, 24))
    """

    features: torch.Tensor
    token_grid: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.features.ndim != 5:
            raise ValueError(
                f"MultiLevelFeatures.features must be 5D [B,T,N,L,D]; "
                f"got shape {tuple(self.features.shape)}"
            )
        t_tok, h, w = self.token_grid
        b, ft, n, _l, _d = self.features.shape
        if ft != t_tok:
            raise ValueError(
                f"token_grid time dim {t_tok} != features T_tok {ft}"
            )
        if n != h * w:
            raise ValueError(
                f"token_grid spatial {h}x{w}={h * w} != features N {n}"
            )

    @property
    def num_layers(self) -> int:
        return self.features.shape[3]

    @property
    def embed_dim(self) -> int:
        return self.features.shape[4]


@dataclass
class LatentGrid:
    """Codec output: compressed flow-space latent on the reduced 2D grid.

    Shapes (after 2x2 pool the gigantic 24x24 grid -> 12x12 = 144 tokens):
        latent: [B, T_tok, N_red, C]   (e.g. [B, T, 144, 384])
        grid  : (T_tok, H_red, W_red)  (e.g. (T, 12, 12))
    """

    latent: torch.Tensor
    grid: tuple[int, int, int]

    def __post_init__(self) -> None:
        if self.latent.ndim != 4:
            raise ValueError(
                f"LatentGrid.latent must be 4D [B,T,N,C]; "
                f"got shape {tuple(self.latent.shape)}"
            )
        t_tok, h, w = self.grid
        _b, ft, n, _c = self.latent.shape
        if ft != t_tok or n != h * w:
            raise ValueError(
                f"grid {self.grid} inconsistent with latent shape "
                f"{tuple(self.latent.shape)}"
            )

    @property
    def latent_dim(self) -> int:
        return self.latent.shape[3]


# ---------------------------------------------------------------------------
# Unified trajectory batch
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryBatch:
    """The single batch type for Stage A (video + robot mixed).

    A batch is a *homogeneous* group produced by the sampler/collate: all
    samples share an ``action_schema_id`` time-bucket so per-embodiment adapters
    and the counterfactual loss can operate within one schema (CLAUDE.md §2.9).

    Field groups
    ------------
    Vision (always present):
        pixels        : [B, T, 3, H, W] raw frames (history+future), or None if
                        features are served straight from cache.
        latent_target : [B, T_tok, N_red, C] codec latent target. None in M1
                        (encoder/codec not wired into the loop until M5); filled
                        once codec is frozen.
        token_grid    : (T_tok, H, W) grid that produced the features.

    Action / proprio (robot only; ``None`` for video):
        actions       : [B, T_chunk, A_pad] future action chunk, padded inside
                        schema; ``None`` when ``action_valid`` is all-zero.
        action_pad_mask : [B, T_chunk, A_pad] 1 where a real action dim/step
                        exists, 0 for padding. Used as flow-loss key-padding.
        proprio       : [B, S_pad] current proprioception/state, schema-padded.
        state_pad_mask: [B, S_pad] validity of proprio dims.

    Labels / routing (always present, per-sample):
        action_valid    : [B] bool/0-1 m_a mask. video=0, robot=1.
        embodiment_id   : [B] long, indexes CategorySpecific adapters.
        action_schema_id: [B] long, schema (pad width + semantics) id.
        view_id         : [B] long, camera/view id.
        fps_meta        : [B] float, source control fps (from info.json), kept
                        so temporal alignment is auditable per sample.

    Text:
        text          : list[str] length B, raw caption/instruction. Video w/o
                        caption uses the null-text sentinel (``NULL_TEXT``).
        text_embedding: [B, L_txt, D_txt] cached umT5 embedding, or None until
                        the text encoder is wired (M4).

    Cache provenance:
        dataset_id    : list[str] length B, scanned dataset id. Empty strings are
                        allowed for synthetic/unit-test batches.
        episode_index : [B] long, source episode id when known, otherwise -1.
        frame_start   : [B] long, inclusive source frame/window start.
        frame_end     : [B] long, exclusive source frame/window end.
        sample_index  : [B] long, source dataset global sample index.
    """

    # vision
    pixels: Optional[torch.Tensor]
    token_grid: tuple[int, int, int]
    latent_target: Optional[torch.Tensor] = None

    # action / proprio  (None for actionless video — never fabricated)
    actions: Optional[torch.Tensor] = None
    action_pad_mask: Optional[torch.Tensor] = None
    proprio: Optional[torch.Tensor] = None
    state_pad_mask: Optional[torch.Tensor] = None

    # labels / routing
    action_valid: torch.Tensor = None  # type: ignore[assignment]
    embodiment_id: torch.Tensor = None  # type: ignore[assignment]
    action_schema_id: torch.Tensor = None  # type: ignore[assignment]
    view_id: Optional[torch.Tensor] = None
    fps_meta: Optional[torch.Tensor] = None

    # text
    text: list[str] = field(default_factory=list)
    text_embedding: Optional[torch.Tensor] = None

    # cache provenance
    dataset_id: list[str] = field(default_factory=list)
    episode_index: Optional[torch.Tensor] = None
    frame_start: Optional[torch.Tensor] = None
    frame_end: Optional[torch.Tensor] = None
    sample_index: Optional[torch.Tensor] = None

    # -- invariants -------------------------------------------------------
    def __post_init__(self) -> None:
        if self.action_valid is None:
            raise ValueError("TrajectoryBatch requires action_valid mask (m_a).")
        b = int(self.action_valid.shape[0])

        # §2.3: an all-actionless batch must not carry action tensors.
        any_action = bool(self.action_valid.any().item())
        if not any_action and self.actions is not None:
            raise ValueError(
                "action_valid is all-zero but `actions` tensor was provided — "
                "actionless video must NOT fabricate action tokens (CLAUDE.md §2.3)."
            )
        # If actions exist, the pad mask must accompany them.
        if self.actions is not None and self.action_pad_mask is None:
            raise ValueError("`actions` provided without `action_pad_mask`.")

        if len(self.text) not in (0, b):
            raise ValueError(
                f"text list length {len(self.text)} != batch size {b}"
            )
        if self.dataset_id and len(self.dataset_id) != b:
            raise ValueError(
                f"dataset_id list length {len(self.dataset_id)} != batch size {b}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.action_valid.shape[0])

    def has_any_action(self) -> bool:
        return bool(self.action_valid.any().item())

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        """Move all tensor fields to ``device`` (non-tensor fields untouched)."""

        def mv(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return None if x is None else x.to(device)

        return TrajectoryBatch(
            pixels=mv(self.pixels),
            token_grid=self.token_grid,
            latent_target=mv(self.latent_target),
            actions=mv(self.actions),
            action_pad_mask=mv(self.action_pad_mask),
            proprio=mv(self.proprio),
            state_pad_mask=mv(self.state_pad_mask),
            action_valid=mv(self.action_valid),
            embodiment_id=mv(self.embodiment_id),
            action_schema_id=mv(self.action_schema_id),
            view_id=mv(self.view_id),
            fps_meta=mv(self.fps_meta),
            text=list(self.text),
            text_embedding=mv(self.text_embedding),
            dataset_id=list(self.dataset_id),
            episode_index=mv(self.episode_index),
            frame_start=mv(self.frame_start),
            frame_end=mv(self.frame_end),
            sample_index=mv(self.sample_index),
        )


# Sentinel used for video clips without a caption (unified with action_valid=0).
NULL_TEXT = ""
