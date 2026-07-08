"""M4 3D-RoPE tests (user M4 point 1 — the silent position-corruption guard).

Verifies:
  * the Wan frequency split (time = d-4*(d//6), h/w = 2*(d//6)) sums to head_dim;
  * grid freqs are ordered ``(t, h, w)`` row-major (w fastest) — IDENTICAL to the
    tokenizer's pack order, so a token's RoPE coordinate matches its sequence slot;
  * ``assemble`` lines up with the packed ``[C|Z|A|V]`` layout (length + per-block
    correspondence);
  * the RoPE relative-position property: rotated q·k depends only on (pos_i-pos_j).
"""

from __future__ import annotations

import torch

from models.attention_mask import ACTION, LATENT, STATE
from models.latent_tokenizer import LatentActionTokenizer
from models.rope import Rope3D, apply_rope


def test_freq_split_sums_to_head_dim():
    rope = Rope3D(head_dim=12)  # d//6=2 -> t=4, h=4, w=4
    assert (rope.dim_t, rope.dim_h, rope.dim_w) == (4, 4, 4)
    assert rope.dim_t + rope.dim_h + rope.dim_w == 12
    # complex tables: half dims
    assert rope.freqs_t.shape[1] == 2 and rope.freqs_1d.shape[1] == 6


def test_grid_freqs_thw_row_major_order():
    """Token index ordering must be t-major, then h, then w (w fastest)."""
    rope = Rope3D(head_dim=12)
    chunks = torch.tensor([0, 1])
    fr = rope.grid_freqs(chunks, h=2, w=2)  # [2*2*2=8, 6]
    assert tuple(fr.shape) == (8, 6)
    # token k corresponds to (t,h,w) = (k//4, (k%4)//2, k%2). The time component
    # (first dim_t/2=2 complex slots) must be equal within a time slab and differ
    # across time slabs.
    t_part = fr[:, :2]
    assert torch.allclose(t_part[0], t_part[3])      # both t=0 (k=0..3)
    assert torch.allclose(t_part[4], t_part[7])      # both t=1 (k=4..7)
    assert not torch.allclose(t_part[0], t_part[4])  # t=0 vs t=1 differ
    # within a time slab the h component (next 2 complex slots) splits at k%4<2
    h_part = fr[:, 2:4]
    assert torch.allclose(h_part[0], h_part[1])      # h=0 (k=0,1)
    assert not torch.allclose(h_part[0], h_part[2])  # h=0 vs h=1


def test_assemble_matches_packed_layout():
    tok = LatentActionTokenizer(hidden_dim=12, max_chunks=8, grid_n=4, max_actions=4, positional=False)
    b, t_ctx, t_fut, n, n_act = 1, 3, 2, 4, 2
    ctx = torch.zeros(b, t_ctx, n, 12)
    z = torch.zeros(b, t_fut, n, 12)
    a = torch.zeros(b, t_fut, n_act, 12)
    seq, layout, slices = tok.pack(ctx, z, a, None)

    rope = Rope3D(head_dim=12)
    fr = rope.assemble(layout, grid_hw=(2, 2))
    assert fr.shape[0] == layout.seq_len == seq.shape[1]
    # the context block's freqs equal a direct grid_freqs over chunks 0..T_ctx-1
    ctx_freqs = rope.grid_freqs(torch.arange(t_ctx), 2, 2)
    assert torch.allclose(fr[: slices.ctx_len], ctx_freqs)
    # the Z block equals grid_freqs over the future chunks after history context.
    z_freqs = rope.grid_freqs(torch.arange(t_ctx, t_ctx + t_fut), 2, 2)
    assert torch.allclose(fr[slices.z_start : slices.z_start + slices.z_len], z_freqs)


def test_tokenizer_future_chunks_start_after_history_context():
    tok = LatentActionTokenizer(hidden_dim=12, max_chunks=8, grid_n=4, max_actions=4, positional=False)
    b, t_ctx, t_fut, n, n_act = 1, 3, 2, 4, 2
    ctx = torch.zeros(b, t_ctx, n, 12)
    z = torch.zeros(b, t_fut, n, 12)
    a = torch.zeros(b, t_fut, n_act, 12)
    state = torch.zeros(b, 1, 12)
    _seq, layout, _slices = tok.pack(ctx, z, a, None, state_hidden=state)

    z_chunks = layout.chunk_idx[layout.modality == LATENT].view(t_fut, n)[:, 0]
    a_chunks = layout.chunk_idx[layout.modality == ACTION].view(t_fut, n_act)[:, 0]
    s_chunks = layout.chunk_idx[layout.modality == STATE]
    expected = torch.arange(t_ctx, t_ctx + t_fut)
    assert torch.equal(z_chunks, expected)
    assert torch.equal(a_chunks, expected)
    assert s_chunks.tolist() == [t_ctx]


def test_rope_relative_position_property():
    """<rope(x, a), rope(y, b)> depends only on (a-b) (rotary invariance)."""
    rope = Rope3D(head_dim=8)
    torch.manual_seed(0)
    x = torch.randn(1, 1, 1, 8)
    y = torch.randn(1, 1, 1, 8)

    def dot_at(pa, pb):
        fa = rope.freqs_1d[torch.tensor([pa])]
        fb = rope.freqs_1d[torch.tensor([pb])]
        rx = apply_rope(x, fa)
        ry = apply_rope(y, fb)
        return (rx * ry).sum().item()

    # same relative offset (3-1)=(5-3)=2 -> equal dot products
    assert abs(dot_at(1, 3) - dot_at(3, 5)) < 1e-4
    assert abs(dot_at(2, 6) - dot_at(0, 4)) < 1e-4


def test_apply_rope_shape_and_identity_at_zero():
    rope = Rope3D(head_dim=8)
    x = torch.randn(2, 5, 3, 8)
    fr = rope.freqs_1d[torch.zeros(5, dtype=torch.long)]  # position 0 -> angle 0
    out = apply_rope(x, fr)
    assert out.shape == x.shape
    # rotating by angle 0 is the identity
    assert torch.allclose(out, x, atol=1e-5)
