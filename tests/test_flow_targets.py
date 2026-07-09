"""M3 flow-target tests (CLAUDE.md §2 invariant 1 — the #1 silent-bug guard).

Two layers:
  * algebra of the PROJECT convention (t=0 noise, t=1 data) — interpolate /
    velocity_target / predict_x1 must round-trip exactly;
  * an end-to-end solver round-trip: pure noise integrated by the Euler solver
    with the *exact* velocity must land on the data sample, and a SIGN-FLIPPED
    velocity (the Wan/DreamZero convention smuggled in) must NOT — so a second
    sign flip in the solver/loss turns this red instead of passing silently.
"""

from __future__ import annotations

import torch

from flow.interpolation import (
    interpolate,
    make_noisy,
    predict_x1,
    velocity_target,
)
from flow.losses import clean_target_loss, flow_matching_loss
from flow.schedulers import TimestepScheduler
from flow.solver import euler_solve


def test_interpolation_endpoints():
    x0 = torch.randn(3, 5)  # noise
    x1 = torch.randn(3, 5)  # data
    # t=0 -> noise, t=1 -> data
    assert torch.allclose(interpolate(x0, x1, torch.zeros(3)), x0, atol=1e-6)
    assert torch.allclose(interpolate(x0, x1, torch.ones(3)), x1, atol=1e-6)


def test_predict_x1_inverts_velocity():
    x0 = torch.randn(4, 8)
    x1 = torch.randn(4, 8)
    t = torch.rand(4)
    x_t = interpolate(x0, x1, t)
    u = velocity_target(x0, x1)
    # x̂1 = x_t + (1-t) u  must recover x1 exactly when v == u
    assert torch.allclose(predict_x1(x_t, u, t), x1, atol=1e-5)


def test_make_noisy_consistency():
    x1 = torch.randn(2, 6)
    t = torch.rand(2)
    noise = torch.randn(2, 6)
    x_t, x0, u = make_noisy(x1, t, noise=noise)
    assert torch.equal(x0, noise)
    assert torch.allclose(u, x1 - noise, atol=1e-6)
    assert torch.allclose(x_t, interpolate(noise, x1, t), atol=1e-6)


def test_solver_roundtrip_exact_velocity():
    torch.manual_seed(0)
    x0 = torch.randn(5, 7)  # pure noise
    x1 = torch.randn(5, 7)  # data target
    u = x1 - x0             # exact constant velocity

    def vfield(x, t):
        return u

    # exact for ANY step count (constant velocity); use few steps
    out = euler_solve(vfield, x0, num_steps=4)
    assert torch.allclose(out, x1, atol=1e-5)


def test_solver_signflip_breaks_roundtrip():
    torch.manual_seed(0)
    x0 = torch.randn(5, 7)
    x1 = torch.randn(5, 7)
    u = x1 - x0

    def wrong(x, t):
        return -(u)  # Wan/DreamZero sign — must NOT reach the data end

    out = euler_solve(wrong, x0, num_steps=4)
    assert not torch.allclose(out, x1, atol=1e-2)
    # in fact it integrates the wrong way, to x0 - u = 2*x0 - x1
    assert torch.allclose(out, x0 - u, atol=1e-5)


def test_flow_matching_loss_zero_at_target():
    v = torch.randn(2, 3, 4)
    assert flow_matching_loss(v, v).item() == 0.0


def test_masked_flow_loss_matches_unmasked_mean_over_channels():
    """An all-valid step mask must not multiply MSE by the action dimension."""
    pred = torch.zeros(2, 3, 4, 5)
    target = torch.ones_like(pred)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)

    unmasked = flow_matching_loss(pred, target)
    masked = flow_matching_loss(pred, target, mask)
    assert unmasked.item() == 1.0
    assert masked.item() == 1.0


def test_clean_target_loss_zero_when_equal():
    r = torch.randn(2, 4, 8)
    terms = clean_target_loss(r, r)
    assert terms["total"].item() < 1e-6


def test_scheduler_coupled_vs_decoupled():
    g = torch.Generator().manual_seed(0)
    coupled = TimestepScheduler(coupled=True)
    t_z, t_a = coupled.sample(16, generator=g)
    assert torch.equal(t_z, t_a)  # shared timestep

    decoupled = TimestepScheduler(coupled=False, latent_noise_power=2.0)
    t_z2, t_a2 = decoupled.sample(4096, generator=g)
    assert not torch.equal(t_z2, t_a2)
    # latent biased toward the noise end (t->0): lower mean than uniform action
    assert t_z2.mean() < t_a2.mean()
    assert (t_z2 > 0).all() and (t_z2 < 1).all()
