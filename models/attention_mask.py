"""Chunk-causal attention mask (CLAUDE.md §2.7 / doc §2.2).

Token modalities per time chunk: ``C`` clean context latent, ``Z`` noisy future
latent (144 tokens / chunk on the 12×12 grid), ``A`` noisy action token(s), ``V``
optional value query. The mask encodes exactly these information boundaries:

  * **cross-chunk causal** — no token ever reads a *future* chunk;
  * **chunk-internal Z/A bidirectional** — within one chunk ``Z`` and ``A`` see
    each other (and themselves), but reading the noisy tokens of *other* chunks
    is NOT allowed; future chunks condition only on *clean context*;
  * **clean-context is strictly earlier** — a noisy token at chunk ``k`` reads
    context ``C_j`` only for ``j < k``. Reading ``C_k`` (its own chunk's clean
    latent) would leak the flow-matching target, so it is forbidden;
  * **context is a pure causal encoder** — a ``C`` query reads only ``C`` keys
    (``j <= k``); it never sees the noisy ``Z``/``A`` (so the clean context
    representation cannot depend on the noise being denoised);
  * **value is a read-only sink** — ``V`` may read ``C``/``Z``/``A``/``S``, but no
    other query may read a ``V`` key (value never perturbs the latent/action
    distribution — doc §3.1).
  * **proprio is an in-sequence read-only conditioning register** (``S``) — it
    mirrors DreamZero, where the state register is part of the self-attention
    sequence (``wan_video_dit_action_casual_chunk.py:710-714/769-773`` put
    ``noisy_state[i]`` in the Z/A block k/v context, and ``_process_state_blocks``
    ``:626-659`` lets state attend only itself). So ``Z``/``A``/``V`` read the
    same-or-earlier ``S`` (current proprio ``q_l`` conditions the future), ``S``
    reads only itself (its representation never depends on the noisy tokens), and
    ``C`` never reads ``S``. Omitted for video (no fabricated state).

``tests/test_attention_mask.py`` asserts these as *information* boundaries
(perturbing a key leaves every non-attending query's output bit-identical), not
just shapes.

Convention for the returned boolean mask: ``mask[q, k] == True`` means query ``q``
is ALLOWED to attend key ``k``. Use :func:`to_additive` for an SDPA-style float
mask (0 where allowed, ``-inf`` where masked).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Modality codes (kept as plain ints so they live in a long tensor alongside the
# chunk index with no python-object overhead).
CONTEXT = 0  # C — clean context latent
LATENT = 1   # Z — noisy future latent (predicted)
ACTION = 2   # A — noisy action token (robot only; omitted for video)
VALUE = 3    # V — value query (read-only sink; Stage A stub)
STATE = 4    # S — proprio register (read-only conditioning; robot only)

NUM_MODALITIES = 5
MODALITY_NAMES = {CONTEXT: "C", LATENT: "Z", ACTION: "A", VALUE: "V", STATE: "S"}


@dataclass
class TokenLayout:
    """Per-token ``(chunk_idx, modality)`` metadata for one sample's sequence.

    The tokenizer emits this; the mask builder and (later) the 3D-RoPE remap
    consume it. ``chunk_idx`` is the time-chunk index (history chunks first, then
    future chunks); ``modality`` is one of the codes above. Both are 1-D ``[S]``
    long tensors with the same length as the packed token sequence.
    """

    chunk_idx: torch.Tensor  # [S] long
    modality: torch.Tensor   # [S] long

    def __post_init__(self) -> None:
        if self.chunk_idx.shape != self.modality.shape or self.chunk_idx.ndim != 1:
            raise ValueError(
                "chunk_idx and modality must be matching 1-D tensors; got "
                f"{tuple(self.chunk_idx.shape)} vs {tuple(self.modality.shape)}"
            )

    @property
    def seq_len(self) -> int:
        return int(self.chunk_idx.shape[0])


def build_chunk_attention_mask(layout: TokenLayout) -> torch.Tensor:
    """Boolean ``[S, S]`` mask (``True`` = query may attend key) per the rules in
    the module docstring. Pure index logic, no learned params.

    Each KEY belongs to exactly one modality category, so the mask is assembled by
    selecting, per key column, the rule for that key's modality (no order
    ambiguity). ``cq``/``ck`` are the query/key chunk indices; ``mq``/``mk`` their
    modalities, broadcast to ``[S, S]``.
    """
    c = layout.chunk_idx
    m = layout.modality
    s = c.shape[0]

    cq = c.view(s, 1).expand(s, s)  # query chunk
    ck = c.view(1, s).expand(s, s)  # key chunk
    mq = m.view(s, 1).expand(s, s)
    mk = m.view(1, s).expand(s, s)

    is_noisy_q = (mq == LATENT) | (mq == ACTION) | (mq == VALUE)

    # key = noisy Z/A: read only within the same chunk by Z/A/V (not C, not S).
    noisy_key = is_noisy_q & (ck == cq)

    # key = clean context: C-query reads it causally (j<=k); a noisy query reads it
    # STRICTLY earlier (no own-chunk target leak); S/V... S never, V strict.
    ctx_key = torch.where(
        mq == CONTEXT, ck <= cq, ((mq == LATENT) | (mq == ACTION) | (mq == VALUE)) & (ck < cq)
    )

    # key = value: read-only sink, only a value query, j<=k.
    value_key = (mq == VALUE) & (ck <= cq)

    # key = state: in-sequence conditioning register. Read by Z/A/V/S of the
    # same-or-earlier chunk; never by context. State-as-query (handled here too)
    # only matches a state key -> state reads only itself.
    state_key = ((mq == LATENT) | (mq == ACTION) | (mq == VALUE) | (mq == STATE)) & (ck <= cq)

    allow = torch.zeros(s, s, dtype=torch.bool, device=c.device)
    allow = torch.where(mk == CONTEXT, ctx_key, allow)
    allow = torch.where((mk == LATENT) | (mk == ACTION), noisy_key, allow)
    allow = torch.where(mk == VALUE, value_key, allow)
    allow = torch.where(mk == STATE, state_key, allow)
    return allow


def to_additive(mask: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Boolean allow-mask -> additive float mask (0 allowed, -inf masked)."""
    add = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    add.masked_fill_(~mask, float("-inf"))
    return add


def describe_mask(layout: TokenLayout) -> str:
    """Human-readable per-token "who can I attend" summary (debug / notes)."""
    allow = build_chunk_attention_mask(layout)
    lines = []
    for q in range(layout.seq_len):
        keys = [k for k in range(layout.seq_len) if allow[q, k]]
        tag = f"{MODALITY_NAMES[int(layout.modality[q])]}{int(layout.chunk_idx[q])}"
        ktags = ",".join(
            f"{MODALITY_NAMES[int(layout.modality[k])]}{int(layout.chunk_idx[k])}" for k in keys
        )
        lines.append(f"  {tag:>4} (#{q}) <- {ktags}")
    return "\n".join(lines)
