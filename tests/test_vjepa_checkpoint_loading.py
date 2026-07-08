from __future__ import annotations

import torch

from models.vjepa_encoder import (
    FrozenVJEPAEncoder,
    _clean_backbone_key,
    _select_checkpoint_state_dict,
)
from tests._mock_vjepa import TinyVJEPABackbone


def test_select_checkpoint_prefers_requested_target_encoder():
    target = {"module.backbone.patch_embed.weight": torch.ones(1)}
    ckpt = {
        "encoder": {"patch_embed.weight": torch.zeros(1)},
        "target_encoder": target,
    }
    assert _select_checkpoint_state_dict(ckpt, "target_encoder") is target


def test_select_checkpoint_accepts_plain_state_dict():
    state = {"module.target_encoder.backbone.patch_embed.weight": torch.ones(1)}
    assert _select_checkpoint_state_dict(state, "target_encoder") is state


def test_clean_backbone_key_strips_nested_prefixes():
    cleaned = _clean_backbone_key(
        {
            "module.target_encoder.backbone.patch_embed.weight": torch.ones(1),
            "encoder.blocks.0.weight": torch.ones(1),
        }
    )
    assert "patch_embed.weight" in cleaned
    assert "blocks.0.weight" in cleaned


def test_encoder_rejects_pretrained_and_checkpoint_path():
    backbone = TinyVJEPABackbone()
    try:
        FrozenVJEPAEncoder(
            hub_name="tiny-mock",
            pretrained=True,
            checkpoint_path="/tmp/weights.pt",
            backbone=backbone,
            assert_gigantic=False,
        )
    except ValueError:
        return
    raise AssertionError("checkpoint_path and pretrained=True should be rejected")
