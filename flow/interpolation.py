"""Flow interpolation — PROJECT-WIDE convention (CLAUDE.md §2 invariant 1).

The whole project fixes ``t=0 -> noise``, ``t=1 -> data``:

    x_t = (1 - t) * x0 + t * x1          # x0 = noise, x1 = data
    u   = x1 - x0                         # constant-in-t velocity target
    x̂1  = x_t + (1 - t) * v_theta(x_t, t)

This is the OPPOSITE of Wan/DreamZero's ``FlowMatchScheduler`` whose
``sample = (1-sigma)*data + sigma*noise`` makes large timestep = noise and
target = ``noise - sample``. With ``sigma <-> (1-t)`` the two velocities differ
by a sign. Every ported scheduler/solver/loss MUST be re-derived to the
convention here; ``tests/test_flow_targets.py`` guards both the algebra and an
end-to-end round-trip (pure noise integrated by the solver must land on data).

No module-level torch state; all functions are pure so the unit tests run on CPU
in milliseconds.
"""

from __future__ import annotations

from typing import Optional

import torch


def expand_t(t: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a per-sample timestep ``t`` [B] to broadcast over an ``ndim`` tensor.

    ``t`` may be a python float / 0-d tensor (shared across the batch) or a 1-D
    ``[B]`` tensor. Returns ``[B, 1, ..., 1]`` (ndim dims) so it multiplies a
    ``[B, ...]`` data tensor elementwise.
    """
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    if t.ndim == 0:
        return t.reshape(*([1] * ndim))
    if t.ndim != 1:
        raise ValueError(f"timestep must be scalar or 1-D [B]; got shape {tuple(t.shape)}")
    return t.reshape(t.shape[0], *([1] * (ndim - 1)))


def interpolate(x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * x0 + t * x1  (x0 = noise, x1 = data)."""
    te = expand_t(t, x1.ndim).to(x1.dtype).to(x1.device)
    return (1.0 - te) * x0 + te * x1


def velocity_target(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """u = x1 - x0  (the *data minus noise* direction; constant in t)."""
    return x1 - x0


def predict_x1(x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x̂1 = x_t + (1 - t) * v  — invert the velocity to the clean (data) sample."""
    te = expand_t(t, x_t.ndim).to(x_t.dtype).to(x_t.device)
    return x_t + (1.0 - te) * v


def make_noisy(
    x1: torch.Tensor,
    t: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward-noise a data sample for flow-matching training.

    Returns ``(x_t, x0, u)`` where ``x0`` is the sampled noise and ``u = x1 - x0``
    is the velocity regression target. Reusing this for the counterfactual path
    (same ``noise`` + ``t``, different ``x1``) keeps the only changing variable
    the conditioning action.
    """
    if noise is None:
        noise = torch.randn(x1.shape, dtype=x1.dtype, device=x1.device, generator=generator)
    x_t = interpolate(noise, x1, t)
    u = velocity_target(noise, x1)
    return x_t, noise, u
