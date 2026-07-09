"""M4 Wan-backed DiT tests: forward shapes, structural action omission, text
cross-attn / CFG, counterfactual — all on tiny dims (no 5B).

Tiny WanConfig: dim=24, layers=2, heads=2 (head_dim=12), ffn=48, text_dim=16.
Inputs: B=2, T_ctx=3, T_fut=2, grid 2x2 (n=4), n_act=2, latent=8, action=4.
"""

from __future__ import annotations

import torch

from models.heads.latent_flow import VJEPALatentFlowHead
from models.wan_config import WanConfig
from models.wan_blocks import WanBackbone
from models.wan_latent_world_action_dit import WanLatentWorldActionDiT


def _model():
    cfg = WanConfig(dim=24, num_layers=2, num_heads=2, ffn_dim=48, freq_dim=16, text_dim=16)
    return WanLatentWorldActionDiT(
        cfg, latent_dim=8, action_dim=4, num_embodiments=3, grid_hw=(2, 2),
        max_chunks=8, max_actions=4, state_dim=5, text_seq_len=4,
    )


def _robot(b=2, t_ctx=3, t_fut=2, n=4, n_act=2):
    return dict(
        context_latent=torch.randn(b, t_ctx, n, 8),
        noisy_latent=torch.randn(b, t_fut, n, 8),
        latent_timestep=torch.rand(b),
        action_timestep=torch.rand(b),
        noisy_action=torch.randn(b, t_fut, n_act, 4),
        action_valid=torch.ones(b, dtype=torch.bool),
        embodiment_id=torch.tensor([0, 1]),
        proprio=torch.randn(b, 5),
        text=["pick up the cube", "stack the blocks"],
    )


def test_timestep_embeddings_follow_module_dtype_without_autocast():
    """FSDP bf16 parameters must not receive the fp32 sinusoid directly."""
    cfg = WanConfig(
        dim=24,
        num_layers=1,
        num_heads=2,
        ffn_dim=48,
        freq_dim=16,
        text_dim=16,
    )
    backbone = WanBackbone(cfg).double()
    modulation = backbone.timestep_modulation(torch.rand(2, 5, dtype=torch.float32))
    assert modulation.dtype == torch.float64

    latent_head = VJEPALatentFlowHead(hidden_dim=24, latent_dim=8).double()
    velocity = latent_head(
        torch.randn(2, 2, 4, 24, dtype=torch.float64),
        torch.rand(2, dtype=torch.float32),
    )
    assert velocity.dtype == torch.float64


def test_robot_forward_shapes():
    torch.manual_seed(0)
    out = _model()(use_value=True, **_robot())
    assert tuple(out.latent_velocity.shape) == (2, 2, 4, 8)
    assert tuple(out.action_velocity.shape) == (2, 2, 2, 4)
    assert tuple(out.value.shape) == (2, 2, 1)


def test_video_forward_omits_action_and_zero_grad():
    torch.manual_seed(0)
    m = _model()
    out = m(
        context_latent=torch.randn(2, 3, 4, 8),
        noisy_latent=torch.randn(2, 2, 4, 8),
        latent_timestep=torch.rand(2),
        noisy_action=None,
        action_valid=torch.zeros(2, dtype=torch.bool),
        text=["", ""],  # caption-less -> null text
    )
    assert out.action_velocity is None
    out.latent_velocity.pow(2).mean().backward()
    action_params = (
        list(m.action_encoder.parameters())
        + list(m.action_to_latent.parameters())
        + list(m.action_head.parameters())
    )
    assert all(p.grad is None for p in action_params)
    assert any(p.grad is not None for p in m.latent_head.parameters())


def test_text_encoder_frozen_no_grad_into_dit():
    torch.manual_seed(0)
    m = _model()
    out = m(**_robot())
    out.latent_velocity.pow(2).mean().backward()
    # umT5 stand-in is frozen: no grad even though text feeds cross-attn
    assert all(p.grad is None for p in m.text_encoder.parameters())
    # but the Wan text_embedding projection (trainable) did get grad
    assert any(p.grad is not None for p in m.backbone.text_embedding.parameters())


def test_cfg_dropout_is_noop_in_eval():
    torch.manual_seed(0)
    m = _model().eval()
    inp = _robot()
    a = m(cfg_dropout=0.9, **inp).latent_velocity
    b = m(cfg_dropout=0.9, **inp).latent_velocity
    assert torch.allclose(a, b)  # eval -> no dropout -> deterministic


def test_counterfactual_changes_prediction():
    torch.manual_seed(0)
    m = _model()
    inp = _robot()
    r_a = m.predict_clean_latent(m(**inp), inp["noisy_latent"], inp["latent_timestep"])
    inp2 = dict(inp)
    inp2["noisy_action"] = inp["noisy_action"].flip(0)
    r_b = m.predict_clean_latent(m(**inp2), inp["noisy_latent"], inp["latent_timestep"])
    assert not torch.allclose(r_a, r_b, atol=1e-5)


def test_proprio_is_insequence_register():
    """Proprio enters as an in-sequence STATE register read by Z: changing it
    changes the latent prediction, and the state adapter gets gradient (it is on
    the self-attention path, not a cross-attn bypass)."""
    torch.manual_seed(0)
    m = _model()
    inp = _robot()
    out_a = m(**inp)
    r_a = m.predict_clean_latent(out_a, inp["noisy_latent"], inp["latent_timestep"])

    inp2 = dict(inp)
    inp2["proprio"] = inp["proprio"] + 3.0  # different physical state
    r_b = m.predict_clean_latent(m(**inp2), inp["noisy_latent"], inp["latent_timestep"])
    assert not torch.allclose(r_a, r_b, atol=1e-5)

    out_a.latent_velocity.pow(2).mean().backward()
    assert any(p.grad is not None for p in m.state_adapter.parameters())


def test_state_register_kept_in_no_action_forward():
    """Δ_cond decoupling (doc §2.7): a no-action forward (noisy_action=None) must
    still consume proprio — the state register is gated on proprio, NOT has_action,
    so the no-action monitor isolates the ACTION contribution rather than dropping
    state too. Asserts proprio changes the output AND the state adapter is on the
    no-action forward's gradient graph."""
    torch.manual_seed(0)
    m = _model()
    b, t_ctx, t_fut, n = 2, 3, 2, 4
    common = dict(
        context_latent=torch.randn(b, t_ctx, n, 8),
        noisy_latent=torch.randn(b, t_fut, n, 8),
        latent_timestep=torch.rand(b),
        noisy_action=None,                       # <- no action branch
        action_valid=torch.zeros(b, dtype=torch.bool),
        embodiment_id=torch.tensor([0, 1]),
        text=["pick", "place"],
    )
    proprio = torch.randn(b, 5)
    out_with = m(proprio=proprio, **common)
    out_without = m(proprio=None, **common)
    # proprio is consumed even with the action branch absent
    assert not torch.allclose(out_with.latent_velocity, out_without.latent_velocity, atol=1e-5)
    assert out_with.action_velocity is None  # still no action output

    # the state adapter is on the no-action forward's graph
    out_with.latent_velocity.pow(2).mean().backward()
    assert any(p.grad is not None for p in m.state_adapter.parameters())
    # ...and the action branch got NO gradient (it was structurally absent)
    action_params = list(m.action_encoder.parameters()) + list(m.action_to_latent.parameters())
    assert all(p.grad is None for p in action_params)


def test_proprio_with_invalid_embodiment_fails_loud():
    """A state register with an unspecified embodiment (<0) must fail loud, not
    silently index the wrong per-embodiment adapter (CLAUDE.md §10)."""
    torch.manual_seed(0)
    m = _model()
    try:
        m(
            context_latent=torch.randn(2, 3, 4, 8),
            noisy_latent=torch.randn(2, 2, 4, 8),
            latent_timestep=torch.rand(2),
            noisy_action=None,
            embodiment_id=torch.tensor([-1, -1]),  # invalid
            proprio=torch.randn(2, 5),
            text=["", ""],
        )
    except ValueError:
        return
    raise AssertionError("proprio + embodiment_id<0 must raise")


def test_mixed_batch_rejected():
    m = _model()
    inp = _robot()
    inp["action_valid"] = torch.tensor([True, False])
    try:
        m(**inp)
    except ValueError:
        return
    raise AssertionError("mixed action_valid with noisy_action must be rejected")
