"""M4 weight-loading tests (user M4 point 2 — mock Wan ckpt, no 5B download).

Builds a tiny model, fabricates a Wan-style state_dict from its own backbone keys
(so shapes/names match and actually load) plus fake VAE / pixel-head / CLIP /
image-cross-attn keys, then asserts:
  * Wan trunk (self_attn/cross_attn/ffn/time/text) is LOADED (values applied);
  * VAE patch-embed, pixel head, CLIP img_emb, image cross-attn are DROPPED
    (never loaded) — the §2.2 invariant;
  * from-scratch modules (latent_adapter / latent_head / action branch) are MISSING;
  * no forbidden weight reached the load set.
"""

from __future__ import annotations

import torch

from models.wan_config import WanConfig
from models.wan_latent_world_action_dit import WanLatentWorldActionDiT
from models.weight_loading import load_wan_backbone, remap_wan_key


def _tiny_model():
    cfg = WanConfig(dim=24, num_layers=2, num_heads=2, ffn_dim=48, freq_dim=16, text_dim=16)
    return WanLatentWorldActionDiT(
        cfg, latent_dim=8, action_dim=4, num_embodiments=3, grid_hw=(2, 2),
        max_chunks=8, max_actions=4, state_dim=5, text_seq_len=4,
    )


def _mock_wan_state_dict(model):
    """Wan-style keys (strip 'backbone.') with distinctive values + fake VAE/head."""
    sd = {}
    for k, v in model.backbone.state_dict().items():
        sd[k] = torch.full_like(v, 0.123)  # distinctive -> verifies it loaded
    # forbidden keys that MUST be dropped:
    sd["patch_embedding.weight"] = torch.randn(24, 48, 1, 2, 2)   # VAE patch-embed
    sd["patch_embedding.bias"] = torch.randn(24)
    sd["head.head.weight"] = torch.randn(48, 24)                  # pixel velocity head
    sd["head.modulation"] = torch.randn(1, 2, 24)
    sd["img_emb.proj.0.weight"] = torch.randn(24, 1280)           # CLIP i2v
    sd["blocks.0.cross_attn.k_img.weight"] = torch.randn(24, 24)  # image cross-attn
    sd["blocks.0.cross_attn.v_img.weight"] = torch.randn(24, 24)
    sd["vae.encoder.conv.weight"] = torch.randn(4, 4)             # VAE
    return sd


def test_remap_drops_forbidden_keeps_trunk():
    assert remap_wan_key("blocks.0.self_attn.q.weight") == ("backbone.blocks.0.self_attn.q.weight", None)
    assert remap_wan_key("text_embedding.0.weight") == ("backbone.text_embedding.0.weight", None)
    assert remap_wan_key("patch_embedding.weight")[0] is None
    assert remap_wan_key("head.head.weight")[1] == "pixel_head"
    assert remap_wan_key("img_emb.proj.0.weight")[1] == "clip_img_emb"
    assert remap_wan_key("blocks.0.cross_attn.k_img.weight")[1] == "image_cross_attn"
    assert remap_wan_key("vae.encoder.conv.weight")[1] == "vae"


def test_load_applies_trunk_and_drops_vae():
    model = _tiny_model()
    wan_sd = _mock_wan_state_dict(model)
    report = load_wan_backbone(model, wan_sd, verbose=False)

    # trunk loaded + values actually applied
    assert "backbone.blocks.0.self_attn.q.weight" in report.loaded
    assert "backbone.text_embedding.0.weight" in report.loaded
    assert torch.allclose(
        model.backbone.blocks[0].self_attn.q.weight,
        torch.full_like(model.backbone.blocks[0].self_attn.q.weight, 0.123),
    )

    # VAE / pixel head / CLIP / image cross-attn dropped, never loaded
    for cat in ("vae_patch_embed", "pixel_head", "clip_img_emb", "image_cross_attn", "vae"):
        assert cat in report.dropped and len(report.dropped[cat]) >= 1
    for tgt in report.loaded:
        assert "patch_embedding" not in tgt and not tgt.startswith("backbone.head.")
        assert ".k_img" not in tgt and "vae" not in tgt

    # no unexpected (trunk names match exactly)
    assert report.unexpected == []


def test_from_scratch_modules_are_missing():
    model = _tiny_model()
    report = load_wan_backbone(model, _mock_wan_state_dict(model), verbose=False)
    miss = " ".join(report.missing)
    for pref in ("latent_adapter", "latent_head", "action_encoder", "action_to_latent", "action_head"):
        assert pref in miss, f"{pref} should be missing (from scratch), got: {miss[:200]}"
