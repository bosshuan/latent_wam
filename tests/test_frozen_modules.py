"""M1 frozen-module test: V-JEPA encoder must carry zero gradient.

Guards CLAUDE.md §2.2 (frozen encoder). The encoder runs under no_grad, so a
downstream trainable head can backprop without ever touching encoder params.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tests._mock_vjepa import make_tiny_encoder


def test_encoder_params_frozen():
    enc = make_tiny_encoder()
    assert all(not p.requires_grad for p in enc.parameters()), (
        "all encoder params must have requires_grad=False"
    )


def test_encoder_no_grad_after_downstream_backward():
    enc = make_tiny_encoder()
    head = nn.Linear(enc.embed_dim, 1)  # trainable downstream consumer

    pixels = torch.randn(2, 2, 3, enc.img_size, enc.img_size)
    feats = enc(pixels).features  # [B, T_tok, N, L, D], detached (no_grad)
    assert feats.requires_grad is False

    loss = head(feats.mean(dim=(1, 2, 3))).sum()
    loss.backward()

    # encoder got no gradient; head did
    for p in enc.parameters():
        assert p.grad is None
    assert any(p.grad is not None for p in head.parameters())


# --- M2: codec must be frozen after training -----------------------------


def test_codec_frozen_after_freeze():
    from models.latent_codec import VJEPALatentCodec

    codec = VJEPALatentCodec(
        num_layers=2, embed_dim=8, grid_hw=(4, 4), hidden_dim=16, latent_dim=12
    )
    assert any(p.requires_grad for p in codec.parameters())  # trainable before
    codec.freeze()
    assert all(not p.requires_grad for p in codec.parameters())  # frozen after
    assert not codec.training


# --- M4: frozen umT5-XXL text encoder ------------------------------------


def test_umt5_text_encoder_frozen():
    from models.text_encoder import FrozenUMT5TextEncoder

    enc = FrozenUMT5TextEncoder(text_dim=16, seq_len=4)
    assert all(not p.requires_grad for p in enc.parameters())
    assert not enc.training
    # stays eval even if a parent .train() is called
    enc.train()
    assert not enc.training


def test_umt5_no_grad_and_caches():
    from models.text_encoder import FrozenUMT5TextEncoder

    enc = FrozenUMT5TextEncoder(text_dim=16, seq_len=4)
    head = nn.Linear(16, 1)  # trainable downstream consumer
    emb = enc.encode(["pick up the cube", "pick up the cube"])
    assert emb.requires_grad is False
    # deterministic + cached: same string -> identical embedding
    assert torch.equal(emb[0], emb[1])

    loss = head(emb.mean(dim=(1,))).sum()
    loss.backward()
    for p in enc.parameters():
        assert p.grad is None
    assert any(p.grad is not None for p in head.parameters())
