"""M1 shape-flow tests for the frozen encoder.

Tensor-shape walkthrough (tiny mock: embed_dim=16, patch=16, tubelet=2,
img=384, extract_layers=(1,3) => L=2):
    pixels [B=2, T=4, 3, 384, 384]
      -> permute -> [2, 3, 4, 384, 384]
      -> patch_embed (k=stride=(2,16,16)) -> [2, 16, T_tok=2, 24, 24]
      -> flatten+transpose -> tokens [2, T_tok*24*24 = 2*576 = 1152, 16]
      -> hierarchical concat (L=2) -> [2, 1152, 32]
      -> view -> features [2, T_tok=2, N=576, L=2, D=16]
    token_grid = (2, 24, 24).
"""

from __future__ import annotations

import torch

from tests._mock_vjepa import make_tiny_encoder


def test_encoder_forward_shapes():
    enc = make_tiny_encoder(extract_layers=(1, 3))
    b, t = 2, 4
    pixels = torch.randn(b, t, 3, enc.img_size, enc.img_size)
    out = enc(pixels)

    t_tok = t // enc.tubelet
    gh, gw = enc.grid_hw
    assert out.token_grid == (t_tok, gh, gw)
    assert tuple(out.features.shape) == (b, t_tok, gh * gw, enc.num_layers, enc.embed_dim)
    assert out.num_layers == 2
    assert out.embed_dim == enc.embed_dim


def test_encoder_derived_dims():
    enc = make_tiny_encoder(extract_layers=(1, 3), embed_dim=16)
    # codec_in_dim is computed, never hardcoded
    assert enc.codec_in_dim == enc.num_layers * enc.embed_dim == 2 * 16
    assert enc.grid_hw == (enc.img_size // enc.patch_size,) * 2


def test_encoder_deterministic_feature():
    enc = make_tiny_encoder()
    pixels = torch.randn(1, 2, 3, enc.img_size, enc.img_size)
    a = enc(pixels).features
    b = enc(pixels).features
    assert torch.equal(a, b)  # fixed input -> deterministic feature


def test_reshape_content_layout():
    """Content (not just shape): each (t, layer) block lands where the real
    vjepa2 layout dictates — time-major sequence, layer-major channels.

    A wrong reshape (e.g. swapping the (t_tok, n) or (l, d) split, or assuming
    space-major / layer-minor) reorders content without changing shape and would
    silently feed the codec scrambled features; this test turns that red.
    """
    enc = make_tiny_encoder(extract_layers=(1, 3), embed_dim=8, tagged=True)
    b, t = 2, 4
    pixels = torch.randn(b, t, 3, enc.img_size, enc.img_size)
    feats = enc(pixels).features  # [B, t_tok, N, L, D]
    t_tok = t // enc.tubelet
    for ti in range(t_tok):
        for li in range(enc.num_layers):
            expected = ti * 100 + li
            block = feats[:, ti, :, li, :]
            assert torch.all(block == expected), (
                f"reshape misplaced (t={ti}, layer={li}): expected all=={expected}"
            )


def test_encoder_rejects_wrong_resolution():
    enc = make_tiny_encoder()
    bad = torch.randn(1, 2, 3, 256, 256)  # not native 384
    try:
        enc(bad)
    except ValueError:
        return
    raise AssertionError("encoder should reject non-native resolution")


# --- M2 codec shape flow -------------------------------------------------
# tiny codec: L=2, D=8, grid 4x4=16 -> pool2 -> 2x2=4 tokens, latent_dim=12
#   feats [B,T,16,2,8] -pool-> [B,T,4,2,8] -fuse/enc-> latent [B,T,4,12]
#   -dec-> [B,T,4,2,8]  (POOLED target; decoder stays at 4 tokens, no upsample)


def _tiny_codec():
    from models.latent_codec import VJEPALatentCodec

    return VJEPALatentCodec(
        num_layers=2, embed_dim=8, grid_hw=(4, 4), hidden_dim=16, latent_dim=12, pool=2
    )


def _tiny_feats(b=2, t=3):
    from data.schemas import MultiLevelFeatures

    return MultiLevelFeatures(features=torch.randn(b, t, 16, 2, 8), token_grid=(t, 4, 4))


def test_codec_encode_decode_shapes():
    codec = _tiny_codec()
    feats = _tiny_feats()
    latent = codec.encode(feats)
    assert tuple(latent.latent.shape) == (2, 3, 4, 12)
    assert latent.grid == (3, 2, 2)
    recon = codec.decode(latent)
    # reconstruction is the POOLED feature space (4 tokens), NOT 24x24 (16)
    assert tuple(recon.shape) == (2, 3, 4, 2, 8)
    assert tuple(codec.pooled_target(feats).shape) == (2, 3, 4, 2, 8)


def test_token_reducer_keeps_grid_not_wholeframe():
    from models.latent_codec import TokenReducer

    red = TokenReducer((4, 4), pool=2)
    # distinct constant per spatial token => 2x2 block means must differ
    x = torch.arange(16).float().reshape(1, 1, 16, 1)
    out = red(x)
    assert tuple(out.shape) == (1, 1, 4, 1)  # 4 tokens, NOT 1 (no whole-frame pool)
    assert out.unique().numel() == 4          # spatial structure preserved
