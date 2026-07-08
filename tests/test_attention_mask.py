"""M3 attention-mask tests (CLAUDE.md §2.7 / invariant 7).

These assert *information boundaries*, not just shapes: building the boolean mask
with the documented allow/deny pattern, then proving via a one-layer attention
that perturbing a key leaves every NON-attending query's output bit-identical
(and that an attending query DOES change). This is the "change a token -> the
positions that shouldn't see it stay fixed" guarantee.

Tiny layout (chunks 0,1; grid_n=1; one action token; value on):
    idx 0 = C0 (context, chunk 0)
    idx 1 = C1 (context, chunk 1)
    idx 2 = Z1 (latent,  chunk 1)
    idx 3 = A1 (action,  chunk 1)
    idx 4 = V1 (value,   chunk 1)
"""

from __future__ import annotations

import math

import torch

from models.attention_mask import (
    ACTION,
    CONTEXT,
    LATENT,
    STATE,
    VALUE,
    TokenLayout,
    build_chunk_attention_mask,
    to_additive,
)


def _layout():
    chunk = torch.tensor([0, 1, 1, 1, 1])
    mod = torch.tensor([CONTEXT, CONTEXT, LATENT, ACTION, VALUE])
    return TokenLayout(chunk_idx=chunk, modality=mod)


def _layout_with_state():
    # C0,C1,Z1,A1,V1,S1  (proprio register at the current/future chunk 1)
    chunk = torch.tensor([0, 1, 1, 1, 1, 1])
    mod = torch.tensor([CONTEXT, CONTEXT, LATENT, ACTION, VALUE, STATE])
    return TokenLayout(chunk_idx=chunk, modality=mod)


def test_mask_allow_pattern():
    allow = build_chunk_attention_mask(_layout())
    C0, C1, Z1, A1, V1 = 0, 1, 2, 3, 4

    # Z1 reads earlier clean context C0, itself, and A1 (Z/A bidirectional);
    # NOT its own clean target C1, NOT the value sink V1.
    assert allow[Z1, C0] and allow[Z1, Z1] and allow[Z1, A1]
    assert not allow[Z1, C1]   # own-chunk clean target = leakage -> denied
    assert not allow[Z1, V1]   # value is read-only sink

    # A1 mirrors Z1 (bidirectional within the chunk)
    assert allow[A1, Z1] and allow[A1, C0] and not allow[A1, C1] and not allow[A1, V1]

    # context is a pure causal encoder over context only
    assert allow[C0, C0] and not allow[C0, C1] and not allow[C0, Z1]
    assert allow[C1, C0] and allow[C1, C1] and not allow[C1, Z1]

    # value reads the joint hidden (C0, Z1, A1, itself) but not its own target C1
    assert allow[V1, Z1] and allow[V1, A1] and allow[V1, C0] and allow[V1, V1]
    assert not allow[V1, C1]

    # no future leakage anywhere (no chunk-0 query reads a chunk-1 key)
    assert not allow[C0, Z1] and not allow[C0, A1]


def _attn_layer(x, add_mask, seed=0):
    """Deterministic one-head self-attention + residual (mask under test)."""
    torch.manual_seed(seed)
    s, d = x.shape[-2:]
    wq, wk, wv = (torch.randn(d, d) for _ in range(3))
    q, k, v = x @ wq, x @ wk, x @ wv
    attn = (q @ k.transpose(-2, -1)) / math.sqrt(d) + add_mask
    out = attn.softmax(dim=-1) @ v
    return x + out  # residual: output[q] depends only on x[q] + attended keys


def test_perturbing_key_only_moves_attending_queries():
    layout = _layout()
    allow = build_chunk_attention_mask(layout)
    add = to_additive(allow)
    torch.manual_seed(1)
    x = torch.randn(1, layout.seq_len, 6)

    base = _attn_layer(x, add)

    Z1, V1 = 2, 4
    # (a) perturb Z1 -> A1 (attends Z1) moves, C0 (doesn't) stays
    xp = x.clone()
    xp[:, Z1] += 5.0
    pert = _attn_layer(xp, add)
    A1, C0 = 3, 0
    assert not torch.allclose(pert[:, A1], base[:, A1])  # A1 sees Z1
    assert torch.allclose(pert[:, C0], base[:, C0], atol=1e-6)  # C0 cannot

    # (b) perturb V1 -> NOTHING in Z1/A1 changes (value is a read-only sink)
    xv = x.clone()
    xv[:, V1] += 5.0
    pv = _attn_layer(xv, add)
    assert torch.allclose(pv[:, Z1], base[:, Z1], atol=1e-6)
    assert torch.allclose(pv[:, A1], base[:, A1], atol=1e-6)
    # but V1 itself does read Z1: perturbing Z1 must move V1
    assert not torch.allclose(pert[:, V1], base[:, V1])


def test_state_register_boundaries():
    """Proprio STATE register (DreamZero-faithful): read by Z/A/V of the
    same-or-later chunk, reads ONLY itself, never touched by context, value
    sink still intact."""
    allow = build_chunk_attention_mask(_layout_with_state())
    C0, C1, Z1, A1, V1, S1 = 0, 1, 2, 3, 4, 5

    # Z/A/V read the state register (current proprio conditions the future)
    assert allow[Z1, S1] and allow[A1, S1] and allow[V1, S1]
    # state reads only itself — not Z/A/C/V
    assert allow[S1, S1]
    assert not allow[S1, Z1] and not allow[S1, A1] and not allow[S1, C0] and not allow[S1, V1]
    # context never reads state
    assert not allow[C0, S1] and not allow[C1, S1]
    # the value sink survives the extra modality
    assert not allow[Z1, V1] and allow[V1, Z1]
    # diagonal still complete (no fully-masked row)
    assert bool(allow.diagonal().all())


def test_no_query_is_fully_masked():
    # every token attends at least itself (diagonal allowed) -> no all -inf rows
    allow = build_chunk_attention_mask(_layout())
    assert bool(allow.diagonal().all())
    assert bool(allow.any(dim=1).all())
