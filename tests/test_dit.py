"""M3 tiny-DiT tests: shape flow, structural action omission, overfit, CF path.

Tiny dims (CPU, seconds):
    latent_dim=8, action_dim=4, hidden=16, depth=2, heads=2, emb=3,
    grid_n=2, B=2, T_ctx=3, T_fut=2, n_act=2, state_dim=5

Shape walkthrough (robot batch):
    context [2,3,2,8] -adapter-> [2,3,2,16]
    noisy_z [2,2,2,8] -adapter-> [2,2,2,16]
    noisy_a [2,2,2,4] -enc-> [2,2,2,16]
    pack [C|Z|A|V] -> seq [2, 3*2 + 2*2 + 2*2 + 2*1 = 16, 16]
    heads -> v_z [2,2,2,8], v_a [2,2,2,4], value [2,2,1]
"""

from __future__ import annotations

import torch

from flow.interpolation import make_noisy
from flow.losses import flow_matching_loss
from models.latent_world_action_dit import LatentWorldActionDiT


def _dit(**kw):
    cfg = dict(
        latent_dim=8, action_dim=4, hidden_dim=16, depth=2, heads=2,
        num_embodiments=3, grid_n=2, max_chunks=8, max_actions=4, state_dim=5,
        text_dim=0, value_bins=1,
    )
    cfg.update(kw)
    return LatentWorldActionDiT(**cfg)


def _robot_inputs(b=2, t_ctx=3, t_fut=2, n=2, n_act=2):
    return dict(
        context_latent=torch.randn(b, t_ctx, n, 8),
        noisy_latent=torch.randn(b, t_fut, n, 8),
        latent_timestep=torch.rand(b),
        action_timestep=torch.rand(b),
        noisy_action=torch.randn(b, t_fut, n_act, 4),
        action_valid=torch.ones(b, dtype=torch.bool),
        embodiment_id=torch.tensor([0, 1]),
        proprio=torch.randn(b, 5),
    )


def test_robot_forward_shapes():
    torch.manual_seed(0)
    dit = _dit()
    out = dit(use_value=True, **_robot_inputs())
    assert tuple(out.latent_velocity.shape) == (2, 2, 2, 8)
    assert tuple(out.action_velocity.shape) == (2, 2, 2, 4)
    assert tuple(out.value.shape) == (2, 2, 1)


def test_video_forward_omits_action():
    torch.manual_seed(0)
    dit = _dit()
    out = dit(
        context_latent=torch.randn(2, 3, 2, 8),
        noisy_latent=torch.randn(2, 2, 2, 8),
        latent_timestep=torch.rand(2),
        noisy_action=None,
        action_valid=torch.zeros(2, dtype=torch.bool),
    )
    assert tuple(out.latent_velocity.shape) == (2, 2, 2, 8)
    assert out.action_velocity is None  # Ak structurally omitted


def test_video_batch_action_params_zero_grad():
    """CLAUDE.md §2.3 / M3 acceptance: a video-only batch leaves every action
    branch parameter with NO gradient (omitted, not masked)."""
    torch.manual_seed(0)
    dit = _dit()
    out = dit(
        context_latent=torch.randn(2, 3, 2, 8),
        noisy_latent=torch.randn(2, 2, 2, 8),
        latent_timestep=torch.rand(2),
        noisy_action=None,
        action_valid=torch.zeros(2, dtype=torch.bool),
    )
    out.latent_velocity.pow(2).mean().backward()

    action_modules = (
        list(dit.action_encoder.parameters())
        + list(dit.action_to_latent.parameters())
        + list(dit.action_head.parameters())
    )
    assert all(p.grad is None for p in action_modules), (
        "action branch must receive zero gradient on a video-only batch"
    )
    # the latent path did get gradient
    assert any(p.grad is not None for p in dit.latent_head.parameters())


def test_mixed_batch_rejected():
    dit = _dit()
    inp = _robot_inputs()
    inp["action_valid"] = torch.tensor([True, False])  # mixed + actions => illegal
    try:
        dit(**inp)
    except ValueError:
        return
    raise AssertionError("mixed action_valid with noisy_action must be rejected")


def test_counterfactual_forward_changes_prediction():
    """Swapping the conditioning action (same noise/timestep) must change the
    predicted clean latent r̂1 — the forward path L_cf/S_a rely on."""
    torch.manual_seed(0)
    dit = _dit()
    inp = _robot_inputs()
    out_a = dit(**inp)
    r1_a = dit.predict_clean_latent(out_a, inp["noisy_latent"], inp["latent_timestep"])

    inp_b = dict(inp)
    inp_b["noisy_action"] = inp["noisy_action"].flip(0)  # permute actions across the batch
    out_b = dit(**inp_b)
    r1_b = dit.predict_clean_latent(out_b, inp["noisy_latent"], inp["latent_timestep"])

    assert not torch.allclose(r1_a, r1_b, atol=1e-5)


def test_action_latent_bridge_amplifies_counterfactual_path():
    torch.manual_seed(0)
    base = _dit(action_latent_bridge_scale=0.0)
    bridged = _dit(action_latent_bridge_scale=1.0)
    bridged.load_state_dict(base.state_dict(), strict=False)
    inp = _robot_inputs()

    def diff(model):
        out_a = model(**inp)
        r1_a = model.predict_clean_latent(out_a, inp["noisy_latent"], inp["latent_timestep"])
        inp_b = dict(inp)
        inp_b["noisy_action"] = inp["noisy_action"].flip(0)
        out_b = model(**inp_b)
        r1_b = model.predict_clean_latent(out_b, inp["noisy_latent"], inp["latent_timestep"])
        return (r1_a - r1_b).abs().mean()

    assert diff(bridged) > diff(base)


def test_single_batch_overfit():
    """Sanity: the tiny DiT can drive a flow-matching loss down on one fixed
    batch (gradients connected end-to-end)."""
    torch.manual_seed(0)
    dit = _dit()
    inp = _robot_inputs()
    # fixed target latent + its noised input
    x1 = torch.randn(2, 2, 2, 8)
    t = torch.full((2,), 0.5)
    x_t, _x0, u = make_noisy(x1, t, noise=torch.randn(2, 2, 2, 8))
    inp["noisy_latent"] = x_t
    inp["latent_timestep"] = t

    opt = torch.optim.Adam(dit.parameters(), lr=5e-3)
    losses = []
    for _ in range(100):
        opt.zero_grad()
        out = dit(**inp)
        loss = flow_matching_loss(out.latent_velocity, u)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], f"loss did not drop: {losses[0]:.3f}->{losses[-1]:.3f}"
