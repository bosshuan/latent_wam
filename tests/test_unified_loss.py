"""M5 L_A assembly tests (doc §4.2): structural m_a, term presence, weighting."""

from __future__ import annotations

import torch

from flow.losses import UnifiedLossWeights, assemble_unified_loss


def _vz():
    v = torch.randn(2, 2, 4, 8, requires_grad=True)
    u = torch.randn(2, 2, 4, 8)
    r1h = torch.randn(2, 2, 4, 8)
    r1 = torch.randn(2, 2, 4, 8)
    return v, u, r1h, r1


def test_video_loss_has_no_action_terms():
    v, u, r1h, r1 = _vz()
    w = UnifiedLossWeights()
    terms = assemble_unified_loss(v, u, r1h, r1, w)  # v_a=None -> video
    assert "z_fm" in terms and "clean" in terms and "total" in terms
    assert "a_fm" not in terms and "cf" not in terms and "roll" not in terms
    terms["total"].backward()
    assert v.grad is not None


def test_robot_loss_adds_action_and_cf():
    v, u, r1h, r1 = _vz()
    w = UnifiedLossWeights()
    v_a = torch.randn(2, 2, 2, 4)
    u_a = torch.randn(2, 2, 2, 4)
    a_mask = torch.ones(2, 2, 2, dtype=torch.bool)
    l_cf = torch.tensor(0.3)
    terms = assemble_unified_loss(v, u, r1h, r1, w, v_a=v_a, u_a=u_a, a_mask=a_mask, l_cf=l_cf)
    assert "a_fm" in terms and "cf" in terms
    # cf term is weighted by lambda_cf
    assert torch.allclose(terms["cf"], w.lambda_cf * l_cf)


def test_weights_scale_terms():
    v, u, r1h, r1 = _vz()
    w0 = UnifiedLossWeights(lambda_z=1.0)
    w1 = UnifiedLossWeights(lambda_z=3.0)
    t0 = assemble_unified_loss(v, u, r1h, r1, w0)["z_fm"]
    t1 = assemble_unified_loss(v, u, r1h, r1, w1)["z_fm"]
    assert torch.allclose(t1, 3.0 * t0)


def test_l2_token_distance_scales_per_token():
    from flow.losses import flow_distance

    a = torch.zeros(2, 3, 4)
    b = torch.ones(2, 3, 4)
    # Per token L2 over four channels is 2, then averaged over tokens.
    assert torch.allclose(flow_distance(a, b, kind="l2_token"), torch.full((2,), 2.0))


def test_rollout_term_included_when_passed():
    v, u, r1h, r1 = _vz()
    w = UnifiedLossWeights(lambda_roll=2.0)
    terms = assemble_unified_loss(v, u, r1h, r1, w, l_roll=torch.tensor(0.5))
    assert "roll" in terms and torch.allclose(terms["roll"], torch.tensor(1.0))
